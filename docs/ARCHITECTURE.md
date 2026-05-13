# biosim4-rs Architecture

biosim4-rs is a Rust implementation of a genetic neural-net artificial life simulator. Agents evolve over generations on a 2D grid, guided by pluggable sensors, actions, and survival challenges. The workspace targets two deployment surfaces: a native command-line interface (CLI) and a WebAssembly (WASM) module consumed by a React frontend.

---

## Crate Map

| Crate | Role |
|---|---|
| `biosim4-core` | Simulation engine — all genetics, neural nets, environment, stepping, and reproduction logic. Platform-agnostic library. |
| `biosim4-native` | Thin CLI binary. Reads a JSON config, runs the simulation loop, prints stats. Enables the `parallel` feature of core. |
| `biosim4-wasm` | WebAssembly bindings. Wraps `SimulationState` in a JS-callable `Simulator` class; renders frames to RGBA buffers. |

---

## Module Dependency Directed Acyclic Graph (DAG)

Dependencies flow strictly upward. No cycles.

```
analysis  sim_step  spawn
    └────────┬────────┘
          sim_state
    ┌─────────┼──────────────────────┐
 sensors   actions   challenges   breeds
    └─────────┼──────────────────────┘
           registry
    ┌─────────┼────────────────────┐
 population  world  signals_layer  barriers
    └─────────┼────────────────────┘
            agent
       genome/{gene,ops,neural_net}
    ┌─────────┼────────────────────┐
   grid    sim_config     rng
              └──────┬──────────────┘
                   types
```

---

## Generation Lifecycle

One generation proceeds as follows:

1. **World reset** — `population.clear()`, `grid.zero_fill()`, `signals.zero_fill()`.
2. **Barrier placement** — `create_barrier(grid, barrier_type)` stamps the procedural layout; `reapply_user_barriers()` overlays manual overrides on top.
3. **Registry commit** — `sensors.commit_enabled()` and `actions.commit_enabled()` rebuild the `active_map`, reflecting any enable/disable changes queued since the previous generation.
4. **Neural wiring config** — `wiring_config()` returns `{sensor_count, action_count, max_neurons}` from the committed active sets.
5. **Agent spawn** — for each population slot: generate (or reproduce) a genome, compile it into a `NeuralNet` via `create_wiring`, find a random empty grid cell, and place the agent.
6. **Generation start hooks** — `challenges.on_generation_start()` runs (e.g., `SunTracker` repositions its target zone).
7. **Step loop** — `step_one()` executes `steps_per_generation` times:
   - Challenge `on_sim_step` hooks run.
   - Every alive agent is evaluated: sensors feed into the neural net, action levels are computed, actions are queued or executed.
   - Death queue and move queue are drained.
   - Signal layer 0 fades by 1.
8. **Survivor evaluation** — `spawn_new_generation()` evaluates every alive agent against active challenges, builds a survivor pool, and reproduces the next population.

---

## Cross-Cutting Patterns

### Registry pending/commit lifecycle

`SensorRegistry` and `ActionRegistry` maintain two indices: the raw registration order (never changes) and the `active_map` — a dense `Vec<u16>` mapping `enabled_idx → actual_idx`. Genomes reference sensors and actions by `enabled_idx`.

`set_enabled(id, false)` marks a sensor/action as pending-disabled without rebuilding `active_map`. The change takes effect in two stages:

- **Mid-generation**: the registry's `evaluate`/`execute` methods check the disabled set and return `0.0` / skip immediately, so existing agents experience the change without any wiring shift.
- **At generation boundary**: `commit_enabled()` rebuilds `active_map`. New neural nets compiled in `spawn_new_generation` wire against the updated `enabled_count`, so they never contain genes pointing at disabled nodes.

This two-stage approach keeps wiring stable within a generation, preventing mid-run index shifts. It also allows experiments to toggle sensors and actions between generations.

### Deferred move/death queues

Agent actions do not modify the grid or population immediately. Instead:

- Movement requests go into `population.move_queue: Vec<(AgentId, Coord)>`.
- Kill requests go into `population.death_queue: Vec<AgentId>`.

At the end of each `step_one()`, `drain_death_queue()` runs first, then `drain_move_queue()`. Running death first ensures a killed agent's slot is freed before any move tries to enter it. Immediate mutation would corrupt the `alive_ids` snapshot being iterated. It would also cause borrow-checker conflicts between the agent under evaluation and the population as a whole.

### Scratch buffers (StepScratch)

`SimulationState` holds a `StepScratch` with three reusable buffers:

- `alive_ids: Vec<AgentId>` — snapshotted from `population.alive_ids()` at the start of `step_all_agents`; iteration is over this snapshot, not the live list.
- `action_accum: Vec<f32>` — per-agent action level accumulator passed to `feed_forward`.
- `neuron_accum: Vec<f32>` — per-agent neuron accumulator passed to `feed_forward`.

These buffers carry no semantic state between steps. The system clears and resizes them at the start of each use. They eliminate roughly 600K heap allocations per generation at typical parameters (1000 agents × 300 steps × 2 accumulators).

### Raw pointer split in `step_one_agent`

Each agent step requires simultaneous mutable access to `agent.nnet` (to update neuron outputs) and immutable access to the rest of `population` (for sensors that scan neighbors). The Rust borrow checker cannot express this split through safe references on the same `Vec<Option<Agent>>`.

`step_one_agent` uses raw pointers to isolate the two aliasing domains:

- **Phase 1** (sensor eval + feed-forward): `step_one_agent` uses `agent_ptr: *mut Agent` only to reach `agent.nnet`. Sensors receive a `&Agent` view (via `population.get(id)`) that reads `loc`, `heading`, `age`, `osc_period`, `long_probe_dist`, `genome`, `responsiveness`, and `last_move_dir` — never `nnet`. No two live references reach the same memory.
- **Phase 2** (action execution): `agent_ptr` is reused to get `&mut Agent`. `Population` slots are index-stable (append-only `Vec`), so the pointer remains valid. The inline SAFETY comments acknowledge the aliasing tension between `ctx.signals: &mut Signals` and `world.signals: &Signals`. The aliasing is safe in practice because no built-in action reads `world.signals` during Phase 2.

See the inline `SAFETY` comments in `sim_step.rs` for the per-pointer aliasing analysis.

### Genome modulo wiring

A `Gene`'s `source_num` and `sink_num` fields are raw 7-bit values (0..127). `create_wiring` remaps them modulo `sensor_count` / `action_count` / `max_neurons`, so the same genome is valid for any registry configuration. Changing `enabled_count` shifts all wiring semantics for existing nets. `commit_enabled()` therefore runs at generation boundaries — only nets compiled after the commit use the new counts. Nets from the previous generation (now dead) are never re-wired.

### Determinism contract

`SimConfig.rng_seed != 0` produces a fully reproducible simulation: `SimulationState::new` calls `Rng::seeded(rng_seed)`. In `step_one_agent`, each agent's stochastic sensor/action calls use a forked random number generator (RNG): `state.rng.fork(agent_id)`, which XORs the main RNG's next `u64` with the agent's ID. This gives per-agent independent stochasticity without locks and without affecting the main RNG's sequence for spawn and reproduction decisions.

---

## Extension Points

### Adding a sensor

1. Implement `biosim4_core::registry::Sensor` on a `Send + Sync` struct.
2. `evaluate` must return a value in `[0.0, 1.0]`.
3. Call `state.sensors.register(Box::new(MySensor))`.
4. The sensor participates in genome wiring from the next `commit_enabled()`.

### Adding an action

1. Implement `biosim4_core::registry::Action` on a `Send + Sync` struct.
2. `execute` receives the raw neural output level (arbitrary float range).
3. Call `state.actions.register(Box::new(MyAction))`.

### Adding a challenge

1. Implement `biosim4_core::registry::Challenge`.
2. `evaluate` must return `(pass: bool, fitness: f32)` where `fitness` is in `[0.0, 1.0]`.
3. Provide a `params_schema()` JSON Schema object and a `configure(Value)` method.
4. Call `state.challenges.register(Box::new(MyChallenge::default()))`.
5. Set it active via `state.challenges.set_single("my_id", Some(params_json))`.

---

## WASM Surface

The `biosim4-wasm` crate exports a single `Simulator` class via `wasm-bindgen`.

**JavaScript (JS) lifecycle:**
```js
const sim = new Simulator(configJson);       // init + generation 0
sim.set_challenge(challengeConfigJson);       // optional; must be called before stepping
const rgba = sim.get_frame();                 // Uint8Array, size_x*size_y*4 bytes, Y-flipped
sim.step();                                   // advance one simulation step
const stats = sim.spawn_next_generation();    // end of generation; returns JSON EpochStats
```

`get_frame()` produces a row-major red-green-blue-alpha (RGBA) buffer with Y flipped: canvas row 0 is the top of the world, world Y=0 (bottom) is the last canvas row. Pass the buffer directly into `new ImageData(rgba, width, height)`.

Register custom sensors and actions from JS:
```js
sim.register_js_sensor("my_id", "My Sensor", (agentId, worldJson) => 0.5);
sim.register_js_action("my_id", "My Action", (agentId, level, worldJson) => {});
```
