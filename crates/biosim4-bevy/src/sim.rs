//! Simulation resource and stepping system.
//!
//! Wraps [`SimulationState`] in a Bevy [`Resource`] and drives it forward at a
//! configurable speed. The parallel feature of `biosim4-core` is enabled via
//! Cargo, so `step_one` internally uses rayon for the per-agent Phase 1
//! computation. We configure the rayon global pool size at startup from
//! [`SimControls::num_threads`].
//!
//! # Per-frame budget
//!
//! `SimControls::speed` is "steps per frame". We cap at 256 steps/frame so a
//! runaway speed setting can't freeze the UI. End-of-generation rollover
//! (selection + reproduction) happens automatically when running.

use bevy::prelude::*;
use biosim4_core::sim_config::SimConfig;
use biosim4_core::sim_state::SimulationState;
use biosim4_core::sim_step::step_one;
use biosim4_core::spawn::spawn_new_generation;
use biosim4_core::analysis::collect_epoch_stats;

/// Maximum simulation steps we'll execute in a single frame in normal
/// (rendered) playback. Above this we'd starve the renderer of frame time.
const MAX_STEPS_PER_FRAME: u32 = 256;

/// Per-frame wall-clock budget for fast-forward mode. We still yield to the
/// renderer this often so the progress modal can update + remain cancellable.
const FF_FRAME_BUDGET: std::time::Duration = std::time::Duration::from_millis(35);

/// History buffer cap. 64 mirrors the React frontend's `HISTORY_CAP`.
const HISTORY_CAP: usize = 64;

/// Reasonable default for the in-app simulation. Smaller than the headless
/// CLI defaults so generations finish quickly enough to see evolution happen.
fn default_config() -> SimConfig {
    SimConfig {
        size_x: 128,
        size_y: 128,
        population: 1000,
        num_threads: 4,
        rng_seed: 12345,
        signal_layers: 1,
        steps_per_generation: 200,
        max_generations: 200,
        point_mutation_rate: 0.005,
        ..SimConfig::default()
    }
}

#[derive(Resource)]
pub struct Sim {
    pub state: SimulationState,
    /// Stashed config JSON used to recreate the simulation on reset. Kept in
    /// sync with `state.config` whenever the UI applies a patch.
    pub config_json: String,
}

impl Sim {
    pub fn new(config: SimConfig) -> Self {
        let json = serde_json::to_string_pretty(&config).unwrap_or_default();
        let state = SimulationState::new(config);
        Self { state, config_json: json }
    }

    /// Convenience: alive count snapshot (cheap — just a length read).
    pub fn alive(&self) -> u32 { self.state.population.alive_count() as u32 }
}

/// Tool currently selected in the floating toolbar.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
    #[default]
    Inspect,
    Barrier,
    KillBarrier,
    Kill,
    Reproduce,
}

impl Tool {
    pub fn label(self) -> &'static str {
        match self {
            Tool::Inspect     => "Inspect",
            Tool::Barrier     => "Barrier",
            Tool::KillBarrier => "Kill Zone",
            Tool::Kill        => "Kill",
            Tool::Reproduce   => "Reproduce",
        }
    }
    pub fn shortcut(self) -> &'static str {
        match self {
            Tool::Inspect     => "I",
            Tool::Barrier     => "B",
            Tool::KillBarrier => "Z",
            Tool::Kill        => "K",
            Tool::Reproduce   => "R",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            Tool::Inspect     => "Click an agent to view its neural network",
            Tool::Barrier     => "Click & drag to paint barriers (right-click to erase)",
            Tool::KillBarrier => "Click & drag to paint kill zones — agents that move into them die (right-click to erase)",
            Tool::Kill        => "Click an agent to kill it instantly",
            Tool::Reproduce   => "Click an agent to spawn a mutated child nearby",
        }
    }
}

/// UI-visible playback + tool state. Mutated by the UI, read by the stepping
/// system and other plugins.
#[derive(Resource)]
pub struct SimControls {
    pub running: bool,
    /// Steps per frame. 1 = 60 sps at 60fps, 8 = 480 sps, etc.
    pub speed: u32,
    /// Pixels per cell — controls the rendered grid scale.
    pub pixel_scale: f32,
    pub tool: Tool,
    pub num_threads: u32,
    pub selected_agent: Option<u32>,
    /// FPS measurement for the top bar (smoothed).
    pub fps: f32,
    /// Tracks the last grid dimensions we rendered against — flipped whenever
    /// `reset` rebuilds the world with new dimensions so the renderer reuploads.
    pub grid_dirty: bool,
    /// Re-fit the camera the next render frame (e.g. on reset).
    pub refit_camera: bool,
    /// Painted-barrier count cache for HUD without crossing through the sim resource.
    pub painted_count: u32,
    /// Try to use the GPU compute backend for fast-forward. Honored only
    /// when (a) `GpuAccel::Ready` and (b) the current registry config
    /// uses only GPU-supported sensors/actions. Otherwise FF transparently
    /// falls back to CPU.
    pub ff_use_gpu: bool,
}

impl Default for SimControls {
    fn default() -> Self {
        Self {
            running: false,
            speed: 4,
            pixel_scale: 4.0,
            tool: Tool::default(),
            num_threads: 4,
            selected_agent: None,
            fps: 0.0,
            grid_dirty: true,
            refit_camera: true,
            painted_count: 0,
            ff_use_gpu: true,
        }
    }
}

/// Per-epoch metrics for the telemetry sparkline.
#[derive(Default, Clone)]
pub struct HistoryPoint {
    pub generation: u32,
    pub survival_rate: f32,
    pub diversity: f32,
    pub alive: u32,
}

/// Active fast-forward run. While `Some`, the stepping system runs full
/// generations in a tight time-bounded loop and the renderer skips texture
/// updates so the GPU isn't burning frames on data nobody is watching.
#[derive(Resource, Default)]
pub struct FastForward {
    pub active: Option<FastForwardState>,
}

#[derive(Clone)]
pub struct FastForwardState {
    pub start_gen: u32,
    pub target_gen: u32,
    pub start_time: std::time::Instant,
    /// Most recent generation we observed — read by the UI for the progress
    /// bar without re-querying the sim resource.
    pub last_gen: u32,
    /// True when this FF run is using the GPU backend. Captured at start
    /// so the modal badge doesn't flicker if the toggle changes mid-run.
    pub using_gpu: bool,
}

impl FastForwardState {
    pub fn done_count(&self) -> u32 { self.last_gen.saturating_sub(self.start_gen) }
    pub fn total(&self) -> u32 { self.target_gen.saturating_sub(self.start_gen) }
    pub fn progress(&self) -> f32 {
        let total = self.total();
        if total == 0 { return 1.0; }
        (self.done_count() as f32 / total as f32).clamp(0.0, 1.0)
    }
    pub fn elapsed(&self) -> std::time::Duration { self.start_time.elapsed() }
    /// Estimated remaining wall time. Returns `None` until at least one
    /// generation has finished (so we have a rate to extrapolate).
    pub fn eta(&self) -> Option<std::time::Duration> {
        let done = self.done_count();
        if done == 0 { return None; }
        let secs_per_gen = self.elapsed().as_secs_f64() / done as f64;
        let remaining = self.total().saturating_sub(done);
        Some(std::time::Duration::from_secs_f64(secs_per_gen * remaining as f64))
    }
}

#[derive(Resource, Default)]
pub struct SimHistory {
    pub points: Vec<HistoryPoint>,
}

impl SimHistory {
    pub fn push(&mut self, p: HistoryPoint) {
        if self.points.len() >= HISTORY_CAP {
            self.points.remove(0);
        }
        self.points.push(p);
    }

    pub fn clear(&mut self) { self.points.clear(); }

    pub fn latest(&self) -> Option<&HistoryPoint> { self.points.last() }
}

/// One-shot command pump: the UI pushes requests here, a single system drains
/// them between frames. Keeps mutation centralized so we don't end up with
/// half-applied state spread across multiple systems.
#[derive(Resource, Default)]
pub struct SimCommandQueue {
    pub items: Vec<SimCommand>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum SimCommand {
    Reset,
    Recreate(SimConfig),
    SetSpeed(u32),
    StepOnce,
    StepGeneration,
    RunEpoch,
    /// `tile = Some(kind)` paints a wall or kill barrier; `tile = None`
    /// erases (force-empty, even if a procedural barrier was here).
    SetBarrier { x: u16, y: u16, tile: Option<biosim4_core::sim_state::BarrierTile> },
    Kill { x: u16, y: u16 },
    Reproduce { x: u16, y: u16 },
    ClearUserBarriers,
    SetThreads(u32),
    SetSensorEnabled(String, bool),
    SetActionEnabled(String, bool),
    SetChallenge(String), // JSON
    PatchConfig(String),  // JSON
    /// Start a fast-forward run targeting `current_gen + n`. While active,
    /// rendering is paused; on completion playback returns to the user's
    /// previous running state.
    FastForward(u32),
    /// Abort an in-flight fast-forward run.
    CancelFastForward,
}

pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        let cfg = default_config();
        let threads = cfg.num_threads;

        // Make the rayon thread pool match the configured thread count. Ignore
        // the error — if the pool was already initialized (e.g. by a test
        // harness in the same process) the old one stays in use.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1) as usize)
            .build_global();

        app
            .insert_resource(Sim::new(cfg.clone()))
            .insert_resource(SimControls {
                num_threads: threads,
                ..Default::default()
            })
            .init_resource::<SimHistory>()
            .init_resource::<SimCommandQueue>()
            .init_resource::<FastForward>()
            .add_systems(Update, (
                update_fps,
                process_commands,
                step_simulation,
            ).chain());
    }
}

fn update_fps(time: Res<Time>, mut controls: ResMut<SimControls>) {
    let dt = time.delta_secs().max(1e-4);
    let instant = 1.0 / dt;
    // EMA smoothing — heavy weight on history so the value doesn't twitch.
    controls.fps = if controls.fps == 0.0 { instant } else { 0.92 * controls.fps + 0.08 * instant };
}

/// Drain the command queue. The UI pushes; we apply.
fn process_commands(
    mut sim: ResMut<Sim>,
    mut controls: ResMut<SimControls>,
    mut history: ResMut<SimHistory>,
    mut queue: ResMut<SimCommandQueue>,
    mut fast_forward: ResMut<FastForward>,
    gpu_accel: Res<crate::gpu::GpuAccel>,
) {
    if queue.items.is_empty() { return; }
    let items = std::mem::take(&mut queue.items);
    for cmd in items {
        match cmd {
            SimCommand::Reset => {
                let cfg = sim.state.config.clone();
                sim.state = SimulationState::new(cfg);
                history.clear();
                controls.selected_agent = None;
                controls.running = false;
                controls.grid_dirty = true;
                controls.refit_camera = true;
                controls.painted_count = 0;
            }
            SimCommand::Recreate(cfg) => {
                if let Ok(json) = serde_json::to_string_pretty(&cfg) {
                    sim.config_json = json;
                }
                rebuild_rayon_pool(cfg.num_threads);
                controls.num_threads = cfg.num_threads;
                sim.state = SimulationState::new(cfg);
                history.clear();
                controls.selected_agent = None;
                controls.running = false;
                controls.grid_dirty = true;
                controls.refit_camera = true;
                controls.painted_count = 0;
            }
            SimCommand::SetSpeed(s) => controls.speed = s.clamp(1, MAX_STEPS_PER_FRAME),
            SimCommand::StepOnce => {
                if !controls.running { single_step(&mut sim, &mut history); }
                controls.grid_dirty = true;
            }
            SimCommand::StepGeneration => {
                if !controls.running { finish_or_advance_generation(&mut sim, &mut history); }
                controls.grid_dirty = true;
            }
            SimCommand::RunEpoch => {
                if !controls.running { run_full_epoch(&mut sim, &mut history); }
                controls.grid_dirty = true;
            }
            SimCommand::SetBarrier { x, y, tile } => {
                set_barrier(&mut sim, x, y, tile);
                controls.painted_count = sim.state.user_barriers.len() as u32;
                controls.grid_dirty = true;
            }
            SimCommand::Kill { x, y } => {
                kill_at(&mut sim, x, y);
                controls.grid_dirty = true;
            }
            SimCommand::Reproduce { x, y } => {
                reproduce_at(&mut sim, x, y);
                controls.grid_dirty = true;
            }
            SimCommand::ClearUserBarriers => {
                sim.state.user_barriers.clear();
                rebuild_procedural_barriers(&mut sim);
                controls.painted_count = 0;
                controls.grid_dirty = true;
            }
            SimCommand::SetThreads(n) => {
                let n = n.max(1);
                controls.num_threads = n;
                sim.state.config.num_threads = n;
                rebuild_rayon_pool(n);
            }
            SimCommand::SetSensorEnabled(id, on) => sim.state.sensors.set_enabled(&id, on),
            SimCommand::SetActionEnabled(id, on) => sim.state.actions.set_enabled(&id, on),
            SimCommand::SetChallenge(json) => {
                if let Err(e) = sim.state.set_challenge(&json) {
                    warn!("set_challenge failed: {e}");
                }
            }
            SimCommand::PatchConfig(json) => {
                if let Err(e) = sim.state.config.patch_json(&json) {
                    warn!("patch_config failed: {e}");
                }
            }
            SimCommand::FastForward(n) => {
                if n == 0 { continue; }
                // Pause rendered playback while we churn through generations.
                controls.running = false;
                let cur = sim.state.generation;

                // Try to initialize the GPU backend if the user requested it.
                // Initialization can fail for two reasons:
                //   1. No GPU adapter / device available (`GpuAccel::Unavailable`).
                //   2. The current registry config uses sensors/actions
                //      not in the GPU support set (returns Err from try_init).
                // Either falls back to CPU.
                let using_gpu = if controls.ff_use_gpu {
                    use crate::gpu::{GpuAccel, GpuFastForward};
                    if let GpuAccel::Ready { ctx, ff } = &*gpu_accel {
                        let mut slot = ff.lock().expect("gpu ff slot");
                        if slot.is_none() {
                            match GpuFastForward::try_init(ctx.clone(), &sim.state) {
                                Ok(gff) => *slot = Some(gff),
                                Err(reason) => {
                                    info!("GPU fast-forward unavailable: {reason}");
                                }
                            }
                        }
                        slot.is_some()
                    } else { false }
                } else { false };

                fast_forward.active = Some(FastForwardState {
                    start_gen:  cur,
                    target_gen: cur + n,
                    start_time: std::time::Instant::now(),
                    last_gen:   cur,
                    using_gpu,
                });
            }
            SimCommand::CancelFastForward => {
                fast_forward.active = None;
                controls.grid_dirty = true;
            }
        }
    }
}

/// Advance the simulation. Two modes: fast-forward (tight loop, rendering
/// suppressed) and normal playback (steps-per-frame from the speed slider).
fn step_simulation(
    mut sim: ResMut<Sim>,
    mut history: ResMut<SimHistory>,
    mut controls: ResMut<SimControls>,
    mut fast_forward: ResMut<FastForward>,
    gpu_accel: Res<crate::gpu::GpuAccel>,
) {
    // ── Fast-forward path ───────────────────────────────────────────────
    if let Some(ff) = fast_forward.active.as_mut() {
        if ff.using_gpu {
            // GPU path: each `run_one_generation` does a full generation
            // on GPU + CPU rollover. Budget one generation per frame so
            // the UI/progress modal can refresh between gens.
            if let crate::gpu::GpuAccel::Ready { ff: gpu_ff_slot, .. } = &*gpu_accel {
                let mut slot = gpu_ff_slot.lock().expect("gpu ff slot");
                // Initial upload — happens once at start of FF. We detect
                // by comparing the cached generation in CPU state vs the
                // last we observed; first call will be at start_gen.
                let already_uploaded = ff.last_gen != ff.start_gen
                    || ff.start_time.elapsed() > std::time::Duration::from_millis(50);
                if let Some(gff) = slot.as_mut() {
                    if !already_uploaded {
                        gff.initial_upload(&sim.state);
                    }
                    let frame_start = std::time::Instant::now();
                    while frame_start.elapsed() < FF_FRAME_BUDGET
                        && sim.state.generation < ff.target_gen
                    {
                        let outcome = gff.run_one_generation(&mut sim.state);
                        ff.last_gen = sim.state.generation;
                        history.push(HistoryPoint {
                            generation: outcome.generation_finished,
                            survival_rate: outcome.survival_rate,
                            diversity: outcome.diversity,
                            alive: outcome.survivors,
                        });
                    }
                }
                if sim.state.generation >= ff.target_gen {
                    fast_forward.active = None;
                    controls.grid_dirty = true;
                    // Drop the GPU FF state so subsequent FF runs allocate
                    // fresh against any config changes.
                    *slot = None;
                }
                return;
            }
        }

        // CPU path (default): step-by-step under a time budget.
        let frame_start = std::time::Instant::now();
        while frame_start.elapsed() < FF_FRAME_BUDGET
            && sim.state.generation < ff.target_gen
        {
            run_full_epoch(&mut sim, &mut history);
            ff.last_gen = sim.state.generation;
        }
        if sim.state.generation >= ff.target_gen {
            fast_forward.active = None;
            controls.grid_dirty = true;
        }
        // While fast-forwarding we skip grid_dirty toggling so the renderer
        // doesn't reupload the texture every frame.
        return;
    }

    // ── Normal playback ─────────────────────────────────────────────────
    if !controls.running { return; }
    let speed = controls.speed.min(MAX_STEPS_PER_FRAME);
    for _ in 0..speed {
        let total = sim.state.config.steps_per_generation;
        if sim.state.sim_step >= total {
            advance_generation(&mut sim, &mut history);
        } else {
            let cur = sim.state.sim_step;
            step_one(&mut sim.state, cur);
            sim.state.sim_step = cur + 1;
        }
    }
    controls.grid_dirty = true;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn rebuild_rayon_pool(_threads: u32) {
    // rayon's global pool is initialize-once. We can't actually rebuild it
    // mid-run — log so the user knows a restart is needed for thread changes
    // to affect Phase 1. The `num_threads` field still flows into per-step
    // chunking inside `phase2_actions_all_parallel`, which honors the new
    // value immediately.
    info!("Note: rayon thread pool is global and immutable; phase 2 chunking will pick up the new thread count, phase 1 will continue to use the original pool.");
}

fn single_step(sim: &mut Sim, history: &mut SimHistory) {
    let total = sim.state.config.steps_per_generation;
    if sim.state.sim_step >= total {
        advance_generation(sim, history);
    } else {
        let cur = sim.state.sim_step;
        step_one(&mut sim.state, cur);
        sim.state.sim_step = cur + 1;
    }
}

/// "Step generation" UX semantics from the React frontend: if we're already
/// parked at the end of a generation, advance (selection + reproduction);
/// otherwise run the remaining steps so we end up parked at the boundary.
fn finish_or_advance_generation(sim: &mut Sim, history: &mut SimHistory) {
    let total = sim.state.config.steps_per_generation;
    if sim.state.sim_step >= total {
        advance_generation(sim, history);
    } else {
        for s in sim.state.sim_step..total {
            step_one(&mut sim.state, s);
        }
        sim.state.sim_step = total;
    }
}

fn run_full_epoch(sim: &mut Sim, history: &mut SimHistory) {
    let total = sim.state.config.steps_per_generation;
    for s in sim.state.sim_step..total {
        step_one(&mut sim.state, s);
    }
    sim.state.sim_step = total;
    advance_generation(sim, history);
}

fn advance_generation(sim: &mut Sim, history: &mut SimHistory) {
    let prev_gen = sim.state.generation;
    let survivors = spawn_new_generation(&mut sim.state);
    let stats = collect_epoch_stats(&mut sim.state, survivors);
    sim.state.sim_step = 0;
    history.push(HistoryPoint {
        generation: prev_gen,
        survival_rate: stats.survival_rate(),
        diversity: stats.diversity,
        alive: survivors,
    });
}

/// Paint or erase a user barrier. `tile = Some(_)` stamps a wall or kill
/// barrier; `tile = None` erases (force-empty). Refuses to overwrite an
/// agent slot — use the kill tool for that.
fn set_barrier(
    sim: &mut Sim,
    x: u16,
    y: u16,
    tile: Option<biosim4_core::sim_state::BarrierTile>,
) {
    use biosim4_core::grid::{BARRIER, EMPTY, KILL_BARRIER};
    use biosim4_core::sim_state::BarrierTile;

    let sx = sim.state.config.size_x;
    let sy = sim.state.config.size_y;
    if x >= sx || y >= sy { return; }
    let loc = biosim4_core::types::Coord::new(x as i16, y as i16);
    let cell = sim.state.grid.at(loc);
    let blocking = cell == BARRIER || cell == KILL_BARRIER;
    let empty = cell == EMPTY;

    let target_val = match tile {
        Some(BarrierTile::Wall)  => BARRIER,
        Some(BarrierTile::Kill)  => KILL_BARRIER,
        Some(BarrierTile::Clear) | None => EMPTY,
    };

    // Only stamp into empty or already-blocking cells; never overwrite
    // an agent slot.
    if !(empty || blocking) { return; }
    sim.state.grid.set(loc, target_val);

    let override_tile = match tile {
        Some(BarrierTile::Wall) => BarrierTile::Wall,
        Some(BarrierTile::Kill) => BarrierTile::Kill,
        Some(BarrierTile::Clear) | None => BarrierTile::Clear,
    };
    sim.state.user_barriers.insert((x as i16, y as i16), override_tile);
}

fn kill_at(sim: &mut Sim, x: u16, y: u16) {
    let sx = sim.state.config.size_x;
    let sy = sim.state.config.size_y;
    if x >= sx || y >= sy { return; }
    let loc = biosim4_core::types::Coord::new(x as i16, y as i16);
    let id = sim.state.grid.at(loc);
    if id == biosim4_core::grid::EMPTY || id == biosim4_core::grid::BARRIER { return; }
    if let Some(a) = sim.state.population.get_mut(id) {
        a.alive = false;
    }
    sim.state.grid.set(loc, biosim4_core::grid::EMPTY);
    sim.state.population.queue_for_death(id);
    sim.state.population.drain_death_queue(&mut sim.state.grid);
}

fn reproduce_at(sim: &mut Sim, x: u16, y: u16) {
    let sx = sim.state.config.size_x;
    let sy = sim.state.config.size_y;
    if x >= sx || y >= sy { return; }
    let parent_loc = biosim4_core::types::Coord::new(x as i16, y as i16);
    let parent_id = sim.state.grid.at(parent_loc);
    if parent_id == biosim4_core::grid::EMPTY || parent_id == biosim4_core::grid::BARRIER { return; }

    let (parent_genome, parent_color) = match sim.state.population.get(parent_id) {
        Some(a) if a.alive => (a.genome.clone(), a.color),
        _ => return,
    };

    let mut candidates: Vec<biosim4_core::types::Coord> = Vec::with_capacity(8);
    for dy in -1..=1i16 {
        for dx in -1..=1i16 {
            if dx == 0 && dy == 0 { continue; }
            let c = biosim4_core::types::Coord::new(parent_loc.x + dx, parent_loc.y + dy);
            if sim.state.grid.is_empty_at(c) { candidates.push(c); }
        }
    }
    if candidates.is_empty() { return; }
    let idx = sim.state.rng.gen_range_usize(0, candidates.len());
    let child_loc = candidates[idx];

    use biosim4_core::genome::ops::{generate_child_genome, ReproductionParams};
    use biosim4_core::genome::neural_net::create_wiring;
    use biosim4_core::agent::Agent;

    let cfg = sim.state.config.clone();
    let parents = vec![parent_genome];
    let repro = ReproductionParams {
        sexual: false,
        choose_by_fitness: false,
        mutation_rate: cfg.point_mutation_rate,
        insertion_deletion_rate: cfg.gene_insertion_deletion_rate,
        deletion_ratio: cfg.deletion_ratio,
        max_len: cfg.genome_max_length,
    };
    let child_genome = generate_child_genome(&parents, &repro, &mut sim.state.rng);
    let nnet = create_wiring(&child_genome, sim.state.wiring_config());
    let id = sim.state.population.next_id();
    let mut child = Agent::new(id, child_loc, child_genome, nnet);
    child.color = parent_color;
    let assigned = sim.state.population.spawn(child);
    sim.state.grid.set(child_loc, assigned);
}

fn rebuild_procedural_barriers(sim: &mut Sim) {
    let sx = sim.state.config.size_x as i16;
    let sy = sim.state.config.size_y as i16;
    for y in 0..sy {
        for x in 0..sx {
            let loc = biosim4_core::types::Coord::new(x, y);
            if sim.state.grid.at(loc) == biosim4_core::grid::BARRIER {
                sim.state.grid.set(loc, biosim4_core::grid::EMPTY);
            }
        }
    }
    biosim4_core::barriers::create_barrier(
        &mut sim.state.grid,
        sim.state.config.barrier_type,
    );
}
