//! WebAssembly bindings for biosim4-rs.
//!
//! Exposes a single `Simulator` class with a step-based API designed to drive
//! a browser frontend (canvas + DOM controls). The simulator is fully
//! deterministic given a fixed `rng_seed`, so the same JSON config produces
//! the same evolution across runs.
//!
//! ## Lifecycle
//!
//! ```js
//! const sim = new Simulator(JSON.stringify({ size_x: 128, size_y: 128, ... }));
//! // configure a survival challenge:
//! sim.set_challenge(JSON.stringify({
//!   active: ["circle"],
//!   composition: "Any",
//!   params: { circle: { cx: 0.5, cy: 0.5, radius: 0.25, weighted: true } }
//! }));
//!
//! // tick the world one step at a time, rendering each frame:
//! for (let i = 0; i < sim.steps_per_generation(); i++) {
//!   sim.step();
//!   ctx.putImageData(new ImageData(new Uint8ClampedArray(sim.get_frame()),
//!                                  sim.size_x(), sim.size_y()), 0, 0);
//! }
//!
//! // end the generation, evolve survivors:
//! const epoch = sim.spawn_next_generation();   // → { generation, survivors, ... }
//! ```
//!
//! ## Custom sensors / actions
//!
//! ```js
//! sim.register_js_sensor("food_smell", "food smell",
//!   (agent, simStep) => myFoodMap[agent.x][agent.y] / 255.0
//! );
//!
//! sim.register_js_action("teleport_home", "teleport home",
//!   (agent, level) => level > 1.0 ? { dx: 0, dy: 0 } : null
//! );
//! ```
//! Newly-registered sensors/actions take effect at the **next** generation —
//! existing agents' neural nets were wired against the previous registry size.

use biosim4_core::{
    agent::AgentSnapshot,
    analysis::collect_epoch_stats,
    genome::gene::{SOURCE_SENSOR, SINK_ACTION},
    sim_config::SimConfig,
    sim_state::SimulationState,
    sim_step::step_one,
    spawn::spawn_new_generation,
    types::Coord,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

mod js_bridge;
mod render;

use js_bridge::{JsAction, JsSensor};

// ── Helpers ───────────────────────────────────────────────────────────────

fn js_err(msg: impl AsRef<str>) -> JsValue {
    JsValue::from_str(msg.as_ref())
}

fn to_js<T: Serialize>(v: &T) -> Result<JsValue, JsValue> {
    // Use plain JS objects for maps so the frontend can do `obj.field`
    // instead of `map.get("field")`.
    let s = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    v.serialize(&s).map_err(|e| js_err(e.to_string()))
}

// ── Snapshot DTOs ─────────────────────────────────────────────────────────

#[derive(Serialize)]
struct SimStats {
    generation: u32,
    sim_step: u32,
    steps_per_generation: u32,
    alive_count: u32,
    population: u32,
    sensor_count: u16,
    action_count: u16,
    challenge_count: u32,
}

#[derive(Serialize)]
struct EpochResult {
    generation: u32,        // generation that just ENDED
    next_generation: u32,   // newly spawned generation
    survivors: u32,
    population: u32,
    diversity: f32,
    survival_rate: f32,
}

#[derive(Serialize)]
struct RegistryEntry {
    index: u16,
    id: String,
    name: String,
}

#[derive(Serialize)]
struct NetNode {
    /// "sensor" | "neuron" | "action"
    kind: &'static str,
    index: u16,
    label: String,
}

#[derive(Serialize)]
struct NetEdge {
    from_kind: &'static str,
    from_index: u16,
    to_kind: &'static str,
    to_index: u16,
    /// –4.0..4.0
    weight: f32,
}

#[derive(Serialize)]
struct NeuronState {
    index: u16,
    output: f32,
    driven: bool,
}

#[derive(Serialize)]
struct NetworkSnapshot {
    id: u32,
    color: [u8; 3],
    age: u32,
    genome_length: usize,
    responsiveness: f32,
    nodes: Vec<NetNode>,
    edges: Vec<NetEdge>,
    neuron_states: Vec<NeuronState>,
}

// ── Simulator ─────────────────────────────────────────────────────────────

#[wasm_bindgen]
pub struct Simulator {
    inner: SimulationState,
    /// Cached config JSON for `reset()` so we re-create with the same params.
    config_json: String,
    /// Cached challenge JSON for `reset()` so we restore the active challenge.
    challenge_json: Option<String>,
    /// Reused render buffer — populated by `render_frame_into` each call so
    /// we don't allocate a fresh ~64 KB Vec on every animation frame.
    frame_buf: Vec<u8>,
    /// Reused signal-layer render buffer.
    signal_buf: Vec<u8>,
}

#[wasm_bindgen]
impl Simulator {
    /// Create a new simulator. Pass an empty string or `"{}"` for defaults.
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<Simulator, JsValue> {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        let trimmed = config_json.trim();
        let (inner, json) = if trimmed.is_empty() || trimmed == "{}" {
            let cfg = SimConfig::default();
            let json = serde_json::to_string(&cfg).map_err(|e| js_err(e.to_string()))?;
            (SimulationState::new(cfg), json)
        } else {
            let state = SimulationState::new_from_json(config_json).map_err(js_err)?;
            (state, config_json.to_string())
        };

        Ok(Self { inner, config_json: json, challenge_json: None, frame_buf: Vec::new(), signal_buf: Vec::new() })
    }

    // ── Time advancement ──────────────────────────────────────────────────

    /// Advance the simulation by one step. Returns the new step index.
    /// When the step would exceed `steps_per_generation`, this is a no-op
    /// and the previous step index is returned — call `spawn_next_generation`
    /// to advance the generation.
    pub fn step(&mut self) -> u32 {
        let total = self.inner.config.steps_per_generation;
        if self.inner.sim_step >= total {
            return self.inner.sim_step;
        }
        let cur = self.inner.sim_step;
        step_one(&mut self.inner, cur);
        self.inner.sim_step = cur + 1;
        self.inner.sim_step
    }

    /// Run all remaining steps in the current generation. Returns the number
    /// of steps actually run.
    pub fn step_generation(&mut self) -> u32 {
        let total = self.inner.config.steps_per_generation;
        let start = self.inner.sim_step;
        if start >= total { return 0; }
        for s in start..total {
            step_one(&mut self.inner, s);
        }
        self.inner.sim_step = total;
        total - start
    }

    /// End the current generation: evaluate survivors, breed the next
    /// generation, reset step counter to 0. Returns an `EpochResult` JSON
    /// object.
    pub fn spawn_next_generation(&mut self) -> Result<JsValue, JsValue> {
        let prev_gen = self.inner.generation;
        let survivors = spawn_new_generation(&mut self.inner);
        let stats = collect_epoch_stats(&mut self.inner, survivors);
        self.inner.sim_step = 0;

        to_js(&EpochResult {
            generation: prev_gen,
            next_generation: self.inner.generation,
            survivors,
            population: stats.population,
            diversity: stats.diversity,
            survival_rate: stats.survival_rate(),
        })
    }

    /// Convenience: run a complete generation then spawn the next. Returns
    /// the same `EpochResult` as `spawn_next_generation`.
    pub fn run_epoch(&mut self) -> Result<JsValue, JsValue> {
        if self.inner.sim_step < self.inner.config.steps_per_generation {
            self.step_generation();
        }
        self.spawn_next_generation()
    }

    /// Reset to a freshly-initialised generation 0 using the (possibly
    /// patched) cached config. **Note (v1):** JS-registered sensors/actions
    /// are *not* preserved — re-register them from the frontend after reset.
    pub fn reset(&mut self) -> Result<(), JsValue> {
        // Drain JS sensors/actions out of the old state and re-register them
        // on the new one — they're still useful after reset.
        let custom_sensors = drain_custom_sensors(&mut self.inner);
        let custom_actions = drain_custom_actions(&mut self.inner);

        self.inner = SimulationState::new_from_json(&self.config_json).map_err(js_err)?;

        for (id, name, cb) in custom_sensors {
            self.inner.sensors.register(Box::new(JsSensor::new(id, name, cb)));
        }
        for (id, name, cb) in custom_actions {
            self.inner.actions.register(Box::new(JsAction::new(id, name, cb)));
        }

        if let Some(ref json) = self.challenge_json {
            let _ = self.inner.set_challenge(json);
        }

        Ok(())
    }

    // ── Rendering ─────────────────────────────────────────────────────────

    pub fn size_x(&self) -> u16 { self.inner.config.size_x }
    pub fn size_y(&self) -> u16 { self.inner.config.size_y }

    /// Flat RGBA buffer of length `size_x * size_y * 4`. Row 0 is the top of
    /// the world (canvas convention). Returns a `Uint8Array` that is a *view*
    /// into wasm linear memory — no copy.
    ///
    /// **Caution**: the view is invalidated by any wasm-side allocation that
    /// triggers memory growth (next call to `step`, `set_challenge`, …).
    /// Callers that retain the buffer across awaits must copy via
    /// `.slice(0)` or `new Uint8Array(view)` first. The frontend's
    /// `ImageData(...).data.set(frame)` reads it synchronously, so it's safe.
    pub fn get_frame(&mut self) -> js_sys::Uint8Array {
        render::render_frame_into(&self.inner, &mut self.frame_buf);
        // SAFETY: the view is only valid until the next wasm allocation;
        // callers are documented to consume it synchronously.
        unsafe { js_sys::Uint8Array::view(&self.frame_buf) }
    }

    /// Same encoding as `get_frame`, but only the requested signal layer
    /// rendered as a tinted alpha mask. Use to composite over `get_frame()`.
    pub fn get_signal_frame(&mut self, layer: u8, r: u8, g: u8, b: u8) -> js_sys::Uint8Array {
        render::render_signal_layer_into(&self.inner, layer, [r, g, b], &mut self.signal_buf);
        unsafe { js_sys::Uint8Array::view(&self.signal_buf) }
    }

    // ── Stats / introspection ─────────────────────────────────────────────

    pub fn generation(&self) -> u32 { self.inner.generation }
    pub fn sim_step(&self) -> u32 { self.inner.sim_step }
    pub fn alive_count(&self) -> u32 { self.inner.population.alive_count() as u32 }
    pub fn steps_per_generation(&self) -> u32 { self.inner.config.steps_per_generation }

    /// One-shot stats blob for HUDs / status bars.
    pub fn get_stats(&self) -> Result<JsValue, JsValue> {
        to_js(&SimStats {
            generation: self.inner.generation,
            sim_step: self.inner.sim_step,
            steps_per_generation: self.inner.config.steps_per_generation,
            alive_count: self.inner.population.alive_count() as u32,
            population: self.inner.config.population,
            sensor_count: self.inner.sensors.count(),
            action_count: self.inner.actions.count(),
            challenge_count: self.inner.challenges.schema_list().as_array().map(|a| a.len() as u32).unwrap_or(0),
        })
    }

    /// `Vec<AgentSnapshot>` for every alive agent. For large populations call
    /// sparingly; prefer `get_frame()` for per-tick rendering.
    pub fn get_agents(&self) -> Result<JsValue, JsValue> {
        let snaps: Vec<AgentSnapshot> = self
            .inner
            .population
            .iter_alive()
            .map(AgentSnapshot::from_agent)
            .collect();
        to_js(&snaps)
    }

    /// Single agent snapshot by id. Returns `null` if id is unknown / dead.
    pub fn get_agent(&self, id: u32) -> JsValue {
        match self.inner.population.get(id) {
            Some(a) if a.alive => {
                let snap = AgentSnapshot::from_agent(a);
                serde_wasm_bindgen::to_value(&snap).unwrap_or(JsValue::NULL)
            }
            _ => JsValue::NULL,
        }
    }

    /// Return the agent id at world coordinate (x, y). Returns 0 if the cell
    /// is empty or out-of-bounds (0 is the reserved invalid id).
    /// Note: y=0 is world-bottom (same convention as the core); the canvas
    /// Y-flip is handled in the frontend.
    pub fn agent_at(&self, x: u16, y: u16) -> u32 {
        let sx = self.inner.config.size_x as i16;
        let sy = self.inner.config.size_y as i16;
        if x as i16 >= sx || y as i16 >= sy {
            return 0;
        }
        let id = self.inner.grid.at(Coord::new(x as i16, y as i16));
        if id == 0 || id == u32::MAX { 0 } else { id }
    }

    // ── World-editing tools (used by the frontend toolbar) ────────────────

    /// Cell type at world (x, y): `"empty"`, `"barrier"`, or `"agent"`. Returns
    /// `"oob"` for out-of-bounds queries.
    pub fn cell_kind(&self, x: u16, y: u16) -> String {
        let sx = self.inner.config.size_x;
        let sy = self.inner.config.size_y;
        if x >= sx || y >= sy { return "oob".to_string(); }
        match self.inner.grid.at(Coord::new(x as i16, y as i16)) {
            biosim4_core::grid::EMPTY => "empty".to_string(),
            biosim4_core::grid::BARRIER => "barrier".to_string(),
            _ => "agent".to_string(),
        }
    }

    /// Set a barrier cell at world (x, y). Returns true on success.
    /// Refuses if the cell is currently occupied by an agent (use `kill_at`
    /// first). Persists across generations: the override is recorded in
    /// `user_barriers` and re-applied after every `spawn_new_generation`.
    pub fn set_barrier(&mut self, x: u16, y: u16, on: bool) -> bool {
        let sx = self.inner.config.size_x;
        let sy = self.inner.config.size_y;
        if x >= sx || y >= sy { return false; }
        let loc = Coord::new(x as i16, y as i16);
        let cell = self.inner.grid.at(loc);
        let ok = match (on, cell) {
            (true, biosim4_core::grid::EMPTY) => {
                self.inner.grid.set(loc, biosim4_core::grid::BARRIER);
                true
            }
            (false, biosim4_core::grid::BARRIER) => {
                self.inner.grid.set(loc, biosim4_core::grid::EMPTY);
                true
            }
            (true, biosim4_core::grid::BARRIER) | (false, biosim4_core::grid::EMPTY) => true,
            _ => false, // cell occupied by an agent
        };
        if ok {
            self.inner.user_barriers.insert((x as i16, y as i16), on);
        }
        ok
    }

    /// Forget all manual barrier edits and rebuild the grid from the
    /// procedural `barrier_type` pattern. Idempotent. Agents are unaffected;
    /// only barrier and empty cells are touched.
    pub fn clear_user_barriers(&mut self) {
        self.inner.user_barriers.clear();
        // Rebuild the procedural barrier on top of the current grid: clear all
        // existing BARRIER cells first, then re-stamp via create_barrier.
        let sx = self.inner.config.size_x as i16;
        let sy = self.inner.config.size_y as i16;
        for y in 0..sy {
            for x in 0..sx {
                let loc = Coord::new(x, y);
                if self.inner.grid.at(loc) == biosim4_core::grid::BARRIER {
                    self.inner.grid.set(loc, biosim4_core::grid::EMPTY);
                }
            }
        }
        biosim4_core::barriers::create_barrier(
            &mut self.inner.grid,
            self.inner.config.barrier_type,
        );
    }

    /// Number of cells the user has manually painted (or cleared) since the
    /// last `clear_user_barriers` / `reset`. Useful for the frontend toolbar.
    pub fn user_barrier_count(&self) -> u32 {
        self.inner.user_barriers.len() as u32
    }

    /// Kill the agent at world (x, y). Returns the dead agent's id, or 0 if
    /// the cell wasn't occupied by an agent. Death is processed immediately
    /// (not queued) so the cell becomes empty before the next step.
    pub fn kill_at(&mut self, x: u16, y: u16) -> u32 {
        let id = self.agent_at(x, y);
        if id == 0 { return 0; }
        if let Some(a) = self.inner.population.get_mut(id) {
            a.alive = false;
        }
        let loc = Coord::new(x as i16, y as i16);
        self.inner.grid.set(loc, biosim4_core::grid::EMPTY);
        // Drop from alive_ids by recomputing — drain_death_queue does this for
        // queued kills; we just call it after queueing.
        self.inner.population.queue_for_death(id);
        self.inner.population.drain_death_queue(&mut self.inner.grid);
        id
    }

    /// Clone the agent at (x, y) and place a child carrying a mutated copy of
    /// its genome into a random empty neighboring cell (8-connected). Returns
    /// the new child's id, or 0 if no parent or no empty neighbor.
    pub fn reproduce_at(&mut self, x: u16, y: u16) -> u32 {
        let parent_id = self.agent_at(x, y);
        if parent_id == 0 { return 0; }

        let (parent_genome, parent_color) = {
            let parent = match self.inner.population.get(parent_id) {
                Some(a) if a.alive => a,
                _ => return 0,
            };
            (parent.genome.clone(), parent.color)
        };

        // Pick an empty 8-connected neighbor at random
        let parent_loc = Coord::new(x as i16, y as i16);
        let mut candidates: Vec<Coord> = Vec::with_capacity(8);
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 { continue; }
                let c = Coord::new(parent_loc.x + dx, parent_loc.y + dy);
                if self.inner.grid.is_empty_at(c) { candidates.push(c); }
            }
        }
        if candidates.is_empty() { return 0; }
        let idx = self.inner.rng.gen_range_usize(0, candidates.len());
        let child_loc = candidates[idx];

        // Generate a mutated child genome
        use biosim4_core::genome::genome::generate_child_genome;
        use biosim4_core::genome::neural_net::create_wiring;
        use biosim4_core::agent::Agent;

        let cfg = self.inner.config.clone();
        let parents = vec![parent_genome];
        let child_genome = generate_child_genome(
            &parents,
            cfg.sexual_reproduction,
            false, // no fitness bias for a single-parent clone
            cfg.point_mutation_rate,
            cfg.gene_insertion_deletion_rate,
            cfg.deletion_ratio,
            cfg.genome_max_length,
            &mut self.inner.rng,
        );

        let nnet = create_wiring(&child_genome, self.inner.wiring_config());
        let id = self.inner.population.next_id();
        let mut child = Agent::new(id, child_loc, child_genome, nnet);
        // Inherit colour so manually-bred lineages stay visually identifiable.
        child.color = parent_color;
        let assigned = self.inner.population.spawn(child);
        self.inner.grid.set(child_loc, assigned);
        assigned
    }

    /// Full neural-network snapshot for a single agent — used by the inspector
    /// popup. Returns `null` if the agent is unknown or dead.
    pub fn get_agent_network(&self, id: u32) -> JsValue {
        let agent = match self.inner.population.get(id) {
            Some(a) if a.alive => a,
            _ => return JsValue::NULL,
        };

        let nnet = &agent.nnet;

        // Build label maps from registries
        let sensor_names: Vec<String> = self.inner.sensors
            .iter()
            .map(|(_, s)| s.name().to_string())
            .collect();
        let action_names: Vec<String> = self.inner.actions
            .iter()
            .map(|(_, a)| a.name().to_string())
            .collect();

        // Collect unique sensor and action indices used in this net
        let mut used_sensors: Vec<u16> = Vec::new();
        let mut used_actions: Vec<u16> = Vec::new();
        let mut used_neurons: Vec<u16> = Vec::new();

        for g in &nnet.connections {
            let src_idx = g.source_num() as u16;
            let snk_idx = g.sink_num() as u16;
            if g.source_type() == SOURCE_SENSOR {
                if !used_sensors.contains(&src_idx) { used_sensors.push(src_idx); }
            } else {
                if !used_neurons.contains(&src_idx) { used_neurons.push(src_idx); }
            }
            if g.sink_type() == SINK_ACTION {
                if !used_actions.contains(&snk_idx) { used_actions.push(snk_idx); }
            } else {
                if !used_neurons.contains(&snk_idx) { used_neurons.push(snk_idx); }
            }
        }
        used_sensors.sort();
        used_actions.sort();
        used_neurons.sort();

        let mut nodes: Vec<NetNode> = Vec::new();
        for &idx in &used_sensors {
            let label = sensor_names.get(idx as usize).cloned()
                .unwrap_or_else(|| format!("S{}", idx));
            nodes.push(NetNode { kind: "sensor", index: idx, label });
        }
        for &idx in &used_neurons {
            nodes.push(NetNode { kind: "neuron", index: idx, label: format!("N{}", idx) });
        }
        for &idx in &used_actions {
            let label = action_names.get(idx as usize).cloned()
                .unwrap_or_else(|| format!("A{}", idx));
            nodes.push(NetNode { kind: "action", index: idx, label });
        }

        let edges: Vec<NetEdge> = nnet.connections.iter().map(|g| NetEdge {
            from_kind: if g.source_type() == SOURCE_SENSOR { "sensor" } else { "neuron" },
            from_index: g.source_num() as u16,
            to_kind: if g.sink_type() == SINK_ACTION { "action" } else { "neuron" },
            to_index: g.sink_num() as u16,
            weight: g.weight_as_float(),
        }).collect();

        let neuron_states: Vec<NeuronState> = nnet.neurons.iter().enumerate()
            .map(|(i, n)| NeuronState {
                index: i as u16,
                output: n.output,
                driven: n.driven,
            })
            .collect();

        let snap = NetworkSnapshot {
            id: agent.id,
            color: agent.color,
            age: agent.age,
            genome_length: agent.genome.len(),
            responsiveness: agent.responsiveness,
            nodes,
            edges,
            neuron_states,
        };

        to_js(&snap).unwrap_or(JsValue::NULL)
    }

    // ── Config / challenges ───────────────────────────────────────────────

    /// Replace the challenge configuration. JSON shape:
    /// ```json
    /// { "active": ["circle"], "composition": "Any", "params": { "circle": {...} } }
    /// ```
    pub fn set_challenge(&mut self, json: &str) -> Result<(), JsValue> {
        self.inner.set_challenge(json).map_err(js_err)?;
        self.challenge_json = Some(json.to_string());
        Ok(())
    }

    /// Returns an array of `{id, name, description, schema}` describing every
    /// registered challenge — feed into a frontend form generator.
    pub fn get_challenge_schemas(&self) -> Result<JsValue, JsValue> {
        let v = self.inner.challenges.schema_list();
        to_js(&v)
    }

    /// Returns JSON array of challenge overlays (circles, rects, points).
    pub fn get_challenge_overlays(&self) -> Result<JsValue, JsValue> {
        let world = self.inner.world();
        let overlays = self.inner.challenges.get_overlays(&world);
        to_js(&overlays)
    }

    /// Patch a subset of config fields without resetting.
    /// Note: dimensional fields (`size_x`, `size_y`, `population`) are best
    /// changed via `new Simulator(...)`; patching them mid-run is allowed but
    /// won't resize existing buffers.
    pub fn patch_config(&mut self, json: &str) -> Result<(), JsValue> {
        self.inner.config.patch_json(json).map_err(|e| js_err(e.to_string()))?;
        // Keep cached config JSON in sync so reset() uses the patched values.
        self.config_json = serde_json::to_string(&self.inner.config)
            .map_err(|e| js_err(e.to_string()))?;
        Ok(())
    }

    pub fn get_config(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner.config)
    }

    // ── Registry introspection ────────────────────────────────────────────

    pub fn list_sensors(&self) -> Result<JsValue, JsValue> {
        let entries: Vec<RegistryEntry> = self
            .inner
            .sensors
            .iter()
            .map(|(i, s)| RegistryEntry {
                index: i,
                id: s.id().to_string(),
                name: s.name().to_string(),
            })
            .collect();
        to_js(&entries)
    }

    pub fn list_actions(&self) -> Result<JsValue, JsValue> {
        let entries: Vec<RegistryEntry> = self
            .inner
            .actions
            .iter()
            .map(|(i, a)| RegistryEntry {
                index: i,
                id: a.id().to_string(),
                name: a.name().to_string(),
            })
            .collect();
        to_js(&entries)
    }

    // ── Custom JS sensors / actions ───────────────────────────────────────

    /// Register a JS function as a new sensor. Takes effect at the next
    /// generation (existing nnets were wired against the previous count).
    pub fn register_js_sensor(&mut self, id: &str, name: &str, callback: js_sys::Function) {
        self.inner
            .sensors
            .register(Box::new(JsSensor::new(id.to_string(), name.to_string(), callback)));
    }

    /// Register a JS function as a new action. Takes effect at the next
    /// generation.
    pub fn register_js_action(&mut self, id: &str, name: &str, callback: js_sys::Function) {
        self.inner
            .actions
            .register(Box::new(JsAction::new(id.to_string(), name.to_string(), callback)));
    }
}

// ── Internals ─────────────────────────────────────────────────────────────

/// V1 limitation: custom JS sensors/actions cannot be migrated across a
/// `reset()` because the boxed `dyn Sensor` can't be safely downcast back to
/// `JsSensor` (no `Any` bound on the trait). Frontend code should
/// re-register them after calling `reset()`.
fn drain_custom_sensors(_state: &mut SimulationState) -> Vec<(String, String, js_sys::Function)> {
    Vec::new()
}

fn drain_custom_actions(_state: &mut SimulationState) -> Vec<(String, String, js_sys::Function)> {
    Vec::new()
}
