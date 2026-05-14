//! GPU-resident simulation state.
//!
//! Mirrors `biosim4_core::SimulationState`'s mutable fields as wgpu storage
//! buffers. The CPU side packs SimulationState into these buffers at fast-
//! forward start, the compute shaders mutate them across N generations
//! without CPU intervention, and we read them back at FF end (or at
//! generation boundaries) to run CPU-side selection + reproduction.
//!
//! The layout decisions:
//!   - **Agent struct is 256-byte aligned AoS**. The fields fit in a clean
//!     vec4-aligned packing; storing as struct-of-arrays would mean more
//!     bind-group entries with no measurable benefit at this size.
//!   - **Grid + signals use `atomic<u32>`**. Move drains need CAS to resolve
//!     two agents racing for the same cell; signal emits need atomic add.
//!   - **Move/death queues are append-only with an atomic head counter**.
//!     Capacity equals the population cap (every agent could push at most
//!     once per step).
//!
//! Sensor/action **id mapping**: the simulation registry hands out
//! `enabled_idx` numbers that depend on which sensors/actions are
//! currently enabled. The shader uses fixed GPU-side ids (the
//! `SENSOR_*` / `ACTION_*` constants in `step.wgsl`). When we pack the
//! connection list at FF start we translate `enabled_idx → gpu_id` and
//! store the GPU id directly in `Connection.source_idx` / `sink_idx`.
//! If any enabled registry entry has no GPU mapping the FF caller falls
//! back to the CPU path (see `support::check_supported`).

use std::sync::Arc;

use biosim4_core::{
    agent::AgentId,
    genome::gene::{SINK_ACTION, SOURCE_SENSOR},
    grid::{BARRIER, EMPTY},
    sim_state::SimulationState,
    types::Coord,
};
use bytemuck::{Pod, Zeroable};

use super::context::GpuContext;

pub const MAX_NEURONS: usize = 32;
pub const WORKGROUP: u32 = 64;

// ── Shader-mirroring structs ───────────────────────────────────────────────

#[repr(C, align(16))]
#[derive(Pod, Zeroable, Clone, Copy, Debug, Default)]
pub struct AgentGpu {
    pub loc: [i32; 2],
    pub last_move: [i32; 2],
    pub heading: u32,
    pub age: u32,
    pub osc_period: u32,
    pub long_probe_dist: u32,
    pub responsiveness: f32,
    pub energy: f32,
    pub alive: u32,
    pub color: u32,
    pub challenge_bits: u32,
    pub rng_state: u32,
    pub _pad1: [u32; 2],
    pub memory: [f32; 4],
    pub neuron_outputs: [f32; MAX_NEURONS],
    pub _pad2: [f32; 12],
}

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy, Debug, Default)]
pub struct AgentNetGpu {
    pub conn_start: u32,
    pub conn_end: u32,
    pub neuron_count: u32,
    pub driven_mask: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy, Debug, Default)]
pub struct ConnectionGpu {
    pub source_idx: u32, // sensor GPU-id or neuron index
    pub sink_idx: u32,   // action GPU-id or neuron index
    pub weight: f32,
    pub flags: u32,      // bit 0: sensor src, bit 1: action sink
}

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy, Debug, Default)]
pub struct ParamsGpu {
    pub num_population: u32,
    pub sim_step: u32,
    pub generation: u32,
    pub size_x: u32,
    pub size_y: u32,
    pub steps_per_generation: u32,
    pub _sensor_count: u32,  // unused by shader, kept for layout symmetry
    pub action_count: u32,
    pub pop_radius: f32,
    pub rng_seed_lo: u32,
    pub rng_seed_hi: u32,
    pub short_probe_distance: u32,
    // Energy / signal toggles. The shader reads these on every step.
    pub signal_layers: u32,
    pub food_regen_rate: f32,
    pub energy_per_step_cost: f32,
    pub _pad: u32,
}

// ── GPU sensor / action id tables (mirror of step.wgsl constants) ─────────

/// CPU registry id → GPU sensor constant. Sensors not in this table cause
/// the FF caller to fall back to CPU.
pub fn gpu_sensor_for(id: &str) -> Option<u32> {
    Some(match id {
        "loc_x" => 0,
        "loc_y" => 1,
        "boundary_dist_x" => 2,
        "boundary_dist_y" => 3,
        "boundary_dist" => 4,
        "last_move_dir_x" => 5,
        "last_move_dir_y" => 6,
        "osc1" => 7,
        "age" => 8,
        "random" => 9,
        "memory_0" => 10,
        "memory_1" => 11,
        "memory_2" => 12,
        "memory_3" => 13,
        "barrier_fwd" => 14,
        "barrier_lr" => 15,
        "population" => 16,
        "population_fwd" => 17,
        "population_lr" => 18,
        "kill_barrier_fwd" => 19,
        "signal0" => 20,
        "signal0_fwd" => 21,
        "signal0_lr" => 22,
        "signal1" => 23,
        "signal1_fwd" => 24,
        "signal1_lr" => 25,
        "signal2" => 26,
        "signal2_fwd" => 27,
        "signal2_lr" => 28,
        "longprobe_pop_fwd" => 29,
        "longprobe_bar_fwd" => 30,
        "genetic_sim_fwd" => 31,
        "energy_level" => 32,
        "food_here" => 33,
        "food_fwd" => 34,
        "food_lr" => 35,
        _ => return None,
    })
}

pub fn gpu_action_for(id: &str) -> Option<u32> {
    Some(match id {
        "move_x" => 0,
        "move_y" => 1,
        "move_forward" => 2,
        "move_rl" => 3,
        "move_random" => 4,
        "move_reverse" => 5,
        "move_left" => 6,
        "move_right" => 7,
        "move_east" => 8,
        "move_west" => 9,
        "move_north" => 10,
        "move_south" => 11,
        "set_responsiveness" => 12,
        "set_oscillator_period" => 13,
        "set_longprobe_dist" => 14,
        "emit_signal0" => 15,
        "write_memory_0" => 16,
        "write_memory_1" => 17,
        "write_memory_2" => 18,
        "write_memory_3" => 19,
        "kill_forward" => 20,
        "emit_signal1" => 21,
        "emit_signal2" => 22,
        _ => return None,
    })
}

/// Cached lookup tables built once at FF start for fast translation during
/// connection packing.
pub struct IdMaps {
    /// For each enabled sensor in registry order, the GPU sensor id (None
    /// means unsupported).
    pub sensor: Vec<Option<u32>>,
    pub action: Vec<Option<u32>>,
    /// Number of action slots the shader writes per agent. Capped by
    /// the largest gpu_action_for + 1; if no actions are mapped we use 1.
    pub action_count: u32,
}

impl IdMaps {
    pub fn build(state: &SimulationState) -> Self {
        let mut sensor = Vec::new();
        for (_idx, s, _enabled) in state.sensors.iter() {
            sensor.push(gpu_sensor_for(s.id()));
        }
        let mut action = Vec::new();
        let mut max_id = 0u32;
        for (_idx, a, _enabled) in state.actions.iter() {
            let g = gpu_action_for(a.id());
            if let Some(g) = g { max_id = max_id.max(g + 1); }
            action.push(g);
        }
        Self {
            sensor,
            action,
            action_count: max_id.max(1),
        }
    }

    /// True when every enabled sensor + action has a GPU mapping. The
    /// caller uses this to gate GPU FF vs CPU fallback.
    pub fn all_supported(&self, state: &SimulationState) -> bool {
        for (idx, _s, enabled) in state.sensors.iter() {
            if enabled && self.sensor.get(idx as usize).copied().flatten().is_none() {
                return false;
            }
        }
        for (idx, _a, enabled) in state.actions.iter() {
            if enabled && self.action.get(idx as usize).copied().flatten().is_none() {
                return false;
            }
        }
        true
    }
}

// ── Buffer container ───────────────────────────────────────────────────────

pub struct GpuState {
    ctx: GpuContext,

    // Capacity for which buffers are sized.
    pub population: u32,
    pub size_x: u32,
    pub size_y: u32,
    pub action_count: u32,
    pub max_conn_total: u32,
    pub signal_layers: u32,
    pub max_genome_len: u32,

    // GPU storage buffers
    pub buf_agents: wgpu::Buffer,
    pub buf_agent_nets: wgpu::Buffer,
    pub buf_connections: wgpu::Buffer,
    pub buf_grid: wgpu::Buffer,
    pub buf_signals: wgpu::Buffer,
    pub buf_action_levels: wgpu::Buffer,
    pub buf_move_queue_count: wgpu::Buffer,
    pub buf_move_queue: wgpu::Buffer,
    pub buf_death_queue_count: wgpu::Buffer,
    pub buf_death_queue: wgpu::Buffer,
    pub buf_params: wgpu::Buffer,
    pub buf_food: wgpu::Buffer,
    pub buf_genome_data: wgpu::Buffer,
    pub buf_genome_offsets: wgpu::Buffer,

    // Readback staging (MAP_READ | COPY_DST) sized for agents only.
    pub buf_readback_agents: wgpu::Buffer,
}

impl GpuState {
    pub fn new(ctx: GpuContext, state: &SimulationState, maps: &IdMaps) -> Self {
        let device = &ctx.device;

        // Use id_capacity (population + 1) so agent_id == population fits.
        let population = state.config.population + 1;
        let size_x = state.config.size_x as u32;
        let size_y = state.config.size_y as u32;
        let action_count = maps.action_count;
        // Cap connections per agent at config.genome_max_length.
        let max_conn_per_agent = state.config.genome_max_length as u32;
        let max_conn_total = population.saturating_mul(max_conn_per_agent).max(1);
        let signal_layers = state.signals.layer_count() as u32;
        // Genome similarity only samples 20 genes max. We still allocate
        // genome_max_length per slot so the offset table is trivial; this is
        // a few hundred KB at typical settings.
        let max_genome_len = state.config.genome_max_length as u32;

        let agent_bytes = population as u64 * std::mem::size_of::<AgentGpu>() as u64;
        let net_bytes   = population as u64 * std::mem::size_of::<AgentNetGpu>() as u64;
        let conn_bytes  = max_conn_total as u64 * std::mem::size_of::<ConnectionGpu>() as u64;
        let grid_bytes  = size_x as u64 * size_y as u64 * 4;
        let signal_bytes = signal_layers.max(1) as u64 * size_x as u64 * size_y as u64 * 4;
        let actions_bytes = population as u64 * action_count as u64 * 4;
        let move_q_bytes = population as u64 * (std::mem::size_of::<u32>() * 6) as u64; // MoveReq = 6 u32
        let death_q_bytes = population as u64 * 4;
        let food_bytes = size_x as u64 * size_y as u64 * 4;
        let genome_data_bytes = (population as u64 * max_genome_len as u64).max(1) * 4;
        let genome_off_bytes = population as u64 * 2 * 4;

        let make = |label: &'static str, size: u64, rw: bool| {
            let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            if rw {
                usage |= wgpu::BufferUsages::COPY_SRC;
            }
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let buf_agents = make("gpu agents", agent_bytes, true);
        let buf_agent_nets = make("gpu agent_nets", net_bytes, false);
        let buf_connections = make("gpu connections", conn_bytes, false);
        let buf_grid = make("gpu grid", grid_bytes, true);
        let buf_signals = make("gpu signals", signal_bytes, true);
        let buf_action_levels = make("gpu action_levels", actions_bytes, false);
        let buf_move_queue_count = make("gpu move_q_count", 4, true);
        let buf_move_queue = make("gpu move_q", move_q_bytes, false);
        let buf_death_queue_count = make("gpu death_q_count", 4, true);
        let buf_death_queue = make("gpu death_q", death_q_bytes, false);
        let buf_food = make("gpu food", food_bytes, false);
        let buf_genome_data = make("gpu genome_data", genome_data_bytes, false);
        let buf_genome_offsets = make("gpu genome_offsets", genome_off_bytes, false);

        let buf_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu params"),
            size: std::mem::size_of::<ParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buf_readback_agents = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu readback_agents"),
            size: agent_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            ctx,
            population,
            size_x,
            size_y,
            action_count,
            max_conn_total,
            signal_layers,
            max_genome_len,
            buf_agents,
            buf_agent_nets,
            buf_connections,
            buf_grid,
            buf_signals,
            buf_action_levels,
            buf_move_queue_count,
            buf_move_queue,
            buf_death_queue_count,
            buf_death_queue,
            buf_params,
            buf_food,
            buf_genome_data,
            buf_genome_offsets,
            buf_readback_agents,
        }
    }

    // ── Upload: pack CPU state into GPU buffers ───────────────────────────

    /// Upload everything at FF start (and after each generation rollover —
    /// agent rewiring changes the connection list).
    pub fn upload_all(&self, state: &SimulationState, maps: &IdMaps) {
        self.upload_agents(state);
        self.upload_nets(state, maps);
        self.upload_grid(state);
        self.upload_signals(state);
        self.upload_food(state);
        self.upload_genomes(state);
    }

    fn upload_agents(&self, state: &SimulationState) {
        let mut agents = vec![AgentGpu::default(); self.population as usize];
        for id in 1..self.population {
            let Some(agent) = state.population.get(id as AgentId) else { continue };
            let last_move = agent.last_move_dir.as_normalized_coord();
            let mut neuron_outputs = [0.0f32; MAX_NEURONS];
            for (i, n) in agent.nnet.neurons.iter().enumerate() {
                if i >= MAX_NEURONS { break; }
                neuron_outputs[i] = n.output;
            }
            agents[id as usize] = AgentGpu {
                loc: [agent.loc.x as i32, agent.loc.y as i32],
                last_move: [last_move.x as i32, last_move.y as i32],
                heading: agent.heading.0 as u32,
                age: agent.age,
                osc_period: agent.osc_period,
                long_probe_dist: agent.long_probe_dist,
                responsiveness: agent.responsiveness,
                energy: agent.energy,
                alive: if agent.alive { 1 } else { 0 },
                color: pack_color(agent.color),
                challenge_bits: agent.challenge_bits,
                rng_state: (agent.id.wrapping_mul(2654435761)) | 1,
                _pad1: [0; 2],
                memory: agent.memory,
                neuron_outputs,
                _pad2: [0.0; 12],
            };
        }
        self.ctx.queue.write_buffer(&self.buf_agents, 0, bytemuck::cast_slice(&agents));
    }

    fn upload_nets(&self, state: &SimulationState, maps: &IdMaps) {
        let cap = self.population as usize;
        let mut nets = vec![AgentNetGpu::default(); cap];
        let mut connections: Vec<ConnectionGpu> = Vec::with_capacity(
            state.population.alive_count() * 24,
        );
        for id in 1..self.population {
            let Some(agent) = state.population.get(id as AgentId) else { continue };
            if !agent.alive { continue; }
            let conn_start = connections.len() as u32;
            for g in &agent.nnet.connections {
                let is_sensor_src = g.source_type() == SOURCE_SENSOR;
                let is_action_snk = g.sink_type() == SINK_ACTION;
                // Translate enabled_idx → GPU id (sensor) / GPU id (action).
                // For neuron sources/sinks the index is already a neuron
                // index, used directly.
                let source_idx = if is_sensor_src {
                    maps.sensor
                        .get(g.source_num() as usize)
                        .copied()
                        .flatten()
                        .unwrap_or(u32::MAX)
                } else {
                    g.source_num() as u32
                };
                let sink_idx = if is_action_snk {
                    maps.action
                        .get(g.sink_num() as usize)
                        .copied()
                        .flatten()
                        .unwrap_or(u32::MAX)
                } else {
                    g.sink_num() as u32
                };
                // Skip connections that target an unsupported sensor/action
                // (their `enabled_idx` is in the registry but the GPU
                // doesn't implement them). Functionally equivalent to a
                // zero-weight connection.
                if source_idx == u32::MAX || sink_idx == u32::MAX { continue; }

                let mut flags = 0u32;
                if is_sensor_src { flags |= 1; }
                if is_action_snk { flags |= 2; }
                connections.push(ConnectionGpu {
                    source_idx,
                    sink_idx,
                    weight: g.weight_as_float(),
                    flags,
                });
            }
            let conn_end = connections.len() as u32;
            let mut driven_mask = 0u32;
            for (i, n) in agent.nnet.neurons.iter().enumerate() {
                if i >= 32 { break; }
                if n.driven { driven_mask |= 1 << i; }
            }
            nets[id as usize] = AgentNetGpu {
                conn_start,
                conn_end,
                neuron_count: agent.nnet.neurons.len().min(MAX_NEURONS) as u32,
                driven_mask,
            };
        }
        if connections.len() > self.max_conn_total as usize {
            connections.truncate(self.max_conn_total as usize);
        }
        if !connections.is_empty() {
            self.ctx.queue.write_buffer(
                &self.buf_connections,
                0,
                bytemuck::cast_slice(&connections),
            );
        }
        self.ctx.queue.write_buffer(
            &self.buf_agent_nets,
            0,
            bytemuck::cast_slice(&nets),
        );
    }

    fn upload_grid(&self, state: &SimulationState) {
        let sx = self.size_x as usize;
        let sy = self.size_y as usize;
        let mut grid = vec![EMPTY; sx * sy];
        for y in 0..sy {
            for x in 0..sx {
                let v = state.grid.at(Coord::new(x as i16, y as i16));
                // Same encoding as on the CPU.
                grid[y * sx + x] = v;
            }
        }
        let _ = BARRIER; // referenced for clarity
        self.ctx.queue.write_buffer(&self.buf_grid, 0, bytemuck::cast_slice(&grid));
    }

    fn upload_signals(&self, state: &SimulationState) {
        let sx = self.size_x as usize;
        let sy = self.size_y as usize;
        let layers = self.signal_layers as usize;
        if layers == 0 { return; }
        let mut signals = vec![0u32; layers * sx * sy];
        let avail = state.signals.layer_count() as usize;
        for layer in 0..layers.min(avail) {
            for y in 0..sy {
                for x in 0..sx {
                    let idx = layer * sx * sy + y * sx + x;
                    signals[idx] = state.signals.get(layer as u8, Coord::new(x as i16, y as i16)) as u32;
                }
            }
        }
        self.ctx.queue.write_buffer(&self.buf_signals, 0, bytemuck::cast_slice(&signals));
    }

    fn upload_food(&self, state: &SimulationState) {
        let sx = self.size_x as usize;
        let sy = self.size_y as usize;
        let mut food = vec![0.0f32; sx * sy];
        for y in 0..sy {
            for x in 0..sx {
                food[y * sx + x] = state.food.get(Coord::new(x as i16, y as i16));
            }
        }
        self.ctx.queue.write_buffer(&self.buf_food, 0, bytemuck::cast_slice(&food));
    }

    /// Pack each agent's genome into the flat `genome_data` buffer and the
    /// per-slot offset table. Called on every full upload because reproduction
    /// at generation rollover changes genome lengths and contents.
    fn upload_genomes(&self, state: &SimulationState) {
        let cap = self.population as usize;
        let max_len = self.max_genome_len as usize;
        let mut data = vec![0u32; cap * max_len];
        let mut offsets = vec![0u32; cap * 2];
        for id in 1..self.population {
            let Some(agent) = state.population.get(id as AgentId) else { continue };
            let start = id as usize * max_len;
            let len = agent.genome.len().min(max_len);
            for (i, gene) in agent.genome.iter().take(len).enumerate() {
                data[start + i] = gene.0;
            }
            offsets[id as usize * 2]     = start as u32;
            offsets[id as usize * 2 + 1] = len as u32;
        }
        self.ctx.queue.write_buffer(&self.buf_genome_data, 0, bytemuck::cast_slice(&data));
        self.ctx.queue.write_buffer(&self.buf_genome_offsets, 0, bytemuck::cast_slice(&offsets));
    }

    #[allow(dead_code)] // params are written inline in fast_forward::run_one_generation
    pub fn write_params(&self, params: ParamsGpu) {
        self.ctx.queue.write_buffer(&self.buf_params, 0, bytemuck::bytes_of(&params));
    }

    // ── Download: GPU agent state → CPU SimulationState ───────────────────

    /// Reads back the agents buffer and writes mutable fields back into the
    /// CPU `SimulationState`. The grid is also rebuilt from the agent
    /// positions + alive flags (cheaper than reading the full grid back).
    /// Called at the end of each generation so CPU `spawn_new_generation`
    /// sees the correct survivor set + positions.
    pub fn download_agents(&self, state: &mut SimulationState) {
        // Copy the storage buffer to the readback (MAP_READ) buffer.
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let bytes = self.population as u64 * std::mem::size_of::<AgentGpu>() as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu agents readback"),
        });
        encoder.copy_buffer_to_buffer(&self.buf_agents, 0, &self.buf_readback_agents, 0, bytes);
        queue.submit(Some(encoder.finish()));

        let slice = self.buf_readback_agents.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map_async dropped").expect("map failed");
        let data = slice.get_mapped_range();
        let gpu_agents: &[AgentGpu] = bytemuck::cast_slice(&data);

        // Clear grid first; we'll restamp from agent positions.
        state.grid.zero_fill();
        biosim4_core::barriers::create_barrier(
            &mut state.grid,
            state.config.barrier_type,
        );
        state.reapply_user_barriers();

        // Rebuild population's alive_ids list as we go.
        let alive_only = state.population.alive_ids().to_vec();
        for id in alive_only {
            // Default state — we'll overwrite below.
            if let Some(a) = state.population.get_mut(id) {
                a.alive = false;
            }
        }
        let mut new_alive_ids = Vec::new();

        for id in 1..self.population {
            let g = &gpu_agents[id as usize];
            if let Some(agent) = state.population.get_mut(id as AgentId) {
                let is_alive = g.alive != 0;
                agent.alive = is_alive;
                if !is_alive { continue; }
                agent.loc = Coord::new(g.loc[0] as i16, g.loc[1] as i16);
                agent.heading = biosim4_core::types::Dir((g.heading as u8).into());
                agent.last_move_dir =
                    biosim4_core::types::Dir((infer_heading_from_step(g.last_move) as u8).into());
                agent.age = g.age;
                agent.osc_period = g.osc_period;
                agent.long_probe_dist = g.long_probe_dist;
                agent.responsiveness = g.responsiveness;
                agent.energy = g.energy;
                agent.challenge_bits = g.challenge_bits;
                agent.memory = g.memory;
                for (i, n) in agent.nnet.neurons.iter_mut().enumerate() {
                    if i >= MAX_NEURONS { break; }
                    n.output = g.neuron_outputs[i];
                }
                // Restamp grid at the new location.
                state.grid.set(agent.loc, agent.id);
                new_alive_ids.push(agent.id);
            }
        }

        // Refresh population.alive_ids so subsequent CPU code (especially
        // `spawn_new_generation`) sees the post-GPU survivor set.
        state.population.rebuild_alive_ids();

        drop(data);
        self.buf_readback_agents.unmap();
    }
}

fn pack_color(c: [u8; 3]) -> u32 {
    (c[0] as u32) | ((c[1] as u32) << 8) | ((c[2] as u32) << 16)
}

/// Map a step (dx, dy) ∈ {-1, 0, 1}² to the Compass ordinal that the CPU
/// `Dir` enum uses. Returns 8 (CENTER) for (0, 0).
fn infer_heading_from_step(step: [i32; 2]) -> u32 {
    match step {
        [0, 1] => 0,    // N
        [1, 1] => 1,    // NE
        [1, 0] => 2,    // E
        [1, -1] => 3,   // SE
        [0, -1] => 4,   // S
        [-1, -1] => 5,  // SW
        [-1, 0] => 6,   // W
        [-1, 1] => 7,   // NW
        _ => 8,         // CENTER
    }
}

// Suppress unused Arc-Send concern.
#[allow(dead_code)]
fn _hold(_d: Arc<wgpu::Device>) {}
