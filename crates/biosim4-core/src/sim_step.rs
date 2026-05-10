//! Per-step simulation execution engine.
//!
//! # Entry points
//!
//! [`step_generation`] runs all steps in one generation.
//! [`step_one`] runs a single step and is also exported for embedders (e.g.
//! the WASM frontend) that drive the simulation incrementally.
//!
//! # Two-phase per-agent design
//!
//! Each agent is ticked in two phases to satisfy Rust's aliasing rules.
//!
//! **Phase 1 — sensor evaluation → neural feed-forward.**
//! Requires `&mut agent.nnet` (to write neuron outputs in-place) alongside
//! `&Population` (for neighbor-scanning sensors). These alias at the type
//! level — both live inside `SimulationState` — so raw pointers are used.
//! Sensors read only `loc`, `heading`, `age`, `osc_period`,
//! `long_probe_dist`, `genome`, `responsiveness`, and `last_move_dir`;
//! they never read `nnet`. No two live references reach the same memory.
//! `sensor_rng` is forked from `state.rng` before any raw pointer is
//! derived, making it a completely independent object.
//!
//! **Phase 2 — action execution.**
//! `action_accum` (written in Phase 1) is read as `&[f32]` via the same raw
//! pointer while `ActionContext` holds `&mut` to `agent`, the move/death
//! queues, `signals`, and `rng`. `action_accum` lives in `scratch`, which is
//! disjoint from all of those fields.
//!
//! # Alive-IDs snapshot
//!
//! `step_all_agents` snapshots `population.alive_ids()` into
//! `scratch.alive_ids` at the start of each step (no allocation — the
//! buffer is reused). The loop walks the snapshot by index rather than
//! holding a borrow on `scratch`, so `step_one_agent(state, id)` can take
//! `&mut state` without borrow-checker conflict. See [`SIMULATION_LOOP.md`]
//! for the full rationale.
//!
//! # Deferred queues
//!
//! Actions push to `population.move_queue` and `population.death_queue`.
//! These are drained at end-of-step in order: deaths first (freeing cells),
//! then moves (entering freed cells). Immediate mutation would corrupt the
//! alive-IDs snapshot being iterated.

use crate::agent::AgentId;
use crate::genome::neural_net::feed_forward;
use crate::registry::action::ActionContext;
use crate::registry::challenge::WorldMut;
use crate::registry::sensor::SensorContext;
use crate::sim_state::SimulationState;
use crate::world::World;

/// Run all simulation steps for one generation, then return.
pub fn step_generation(state: &mut SimulationState) {
    for step in 0..state.config.steps_per_generation {
        step_one(state, step);
    }
}

/// Run a single simulation step at index `step`. Sets `state.sim_step = step`,
/// runs challenge hooks, ticks every alive agent, then drains the deferred
/// queues and fades signals. Exposed so embedders (e.g. the WASM frontend)
/// can drive the simulation step-by-step for incremental rendering.
pub fn step_one(state: &mut SimulationState, step: u32) {
    state.sim_step = step;
    run_challenge_step_hooks(state);
    step_all_agents(state);
    state.population.drain_death_queue(&mut state.grid);
    state.population.drain_move_queue(&mut state.grid);
    state.signals.fade(0);
}

fn run_challenge_step_hooks(state: &mut SimulationState) {
    let mut world_mut = WorldMut {
        grid: &mut state.grid,
        signals: &mut state.signals,
        population: &mut state.population,
        rng: &mut state.rng,
        config: &state.config,
        step: state.sim_step,
        generation: state.generation,
    };
    state.challenges.on_sim_step(&mut world_mut);
}

fn step_all_agents(state: &mut SimulationState) {
    // Snapshot alive_ids into the reusable scratch buffer instead of allocating
    // a fresh Vec each step (200 allocs per generation eliminated).
    state.scratch.alive_ids.clear();
    state.scratch.alive_ids.extend_from_slice(state.population.alive_ids());
    // Walk the snapshot by index — `for id in &state.scratch.alive_ids` would
    // hold an immutable borrow of `state.scratch` for the loop body, which
    // conflicts with `step_one_agent(state, ...)` taking `&mut state`.
    for i in 0..state.scratch.alive_ids.len() {
        let id = state.scratch.alive_ids[i];
        step_one_agent(state, id);
    }
}

fn step_one_agent(state: &mut SimulationState, id: AgentId) {
    if state.population.get(id).is_none_or(|a| !a.alive) {
        return;
    }

    let sim_step = state.sim_step;
    let action_count = state.actions.enabled_count();

    // Phase 1: evaluate sensors → action levels.
    //
    // We need:
    //   - `&mut agent.nnet` (to update neuron outputs in-place — no clone)
    //   - `&Population` (for sensors that scan neighbors)
    //   - `&mut Rng` for stochastic sensors
    //   - `&mut scratch.{action,neuron}_accum` for feed_forward
    //
    // All of these come from `state`, so we use raw pointers to dodge the
    // borrow checker. SAFETY invariants for Phase 1:
    //   - `agent.nnet` is a distinct field from anything sensors read on
    //     `&Agent` (sensors read loc/heading/age/osc_period/long_probe_dist/
    //     genome/responsiveness/last_move_dir — never `nnet`). No aliasing
    //     between `nnet: &mut NeuralNet` and the `&Agent` view sensors receive.
    //   - `pop_ptr` is `*const Population`. The shared `&Population` derived
    //     from it is used only to call `population.get(id)` (immutable). The
    //     mutation path through `agent_ptr` targets `population.agents[id].nnet`,
    //     a distinct sub-field; no two &mut references overlap.
    //   - `scratch.alive_ids` is a separate field from `action_accum` /
    //     `neuron_accum` — all three fields are accessed through disjoint paths.
    //   - `sensor_rng` is forked from `state.rng` before any raw pointer is
    //     created; it is a completely independent RNG object and does not alias
    //     `state.rng`.
    //   - All raw pointers are derived from live, valid references and are used
    //     only within this function scope, so they cannot dangle.
    let agent_ptr: *mut crate::agent::Agent = match state.population.get_mut(id) {
        Some(a) if a.alive => a as *mut _,
        _ => return,
    };
    let nnet: &mut crate::genome::neural_net::NeuralNet =
        unsafe { &mut (*agent_ptr).nnet };

    let mut sensor_rng = state.rng.fork(id as u64);

    let size_x = state.config.size_x;
    let size_y = state.config.size_y;
    let steps_per_gen = state.config.steps_per_generation;
    let generation = state.generation;

    let grid_ptr = &state.grid as *const _;
    let signals_ptr = &state.signals as *const _;
    let pop_ptr = &state.population as *const _;
    let action_accum_ptr: *mut Vec<f32> = &mut state.scratch.action_accum;
    let neuron_accum_ptr: *mut Vec<f32> = &mut state.scratch.neuron_accum;

    let world = World {
        grid: unsafe { &*grid_ptr },
        signals: unsafe { &*signals_ptr },
        population: unsafe { &*pop_ptr },
        size_x,
        size_y,
        steps_per_generation: steps_per_gen,
        generation,
        step: sim_step,
    };

    let action_accum: &mut Vec<f32> = unsafe { &mut *action_accum_ptr };
    let neuron_accum: &mut Vec<f32> = unsafe { &mut *neuron_accum_ptr };

    feed_forward(nnet, action_count, action_accum, neuron_accum, |sensor_idx| {
        let agent_ref = world.population.get(id).unwrap();
        let mut ctx = SensorContext {
            agent: agent_ref,
            world: &world,
            sim_step,
            rng: &mut sensor_rng,
        };
        state.sensors.evaluate(sensor_idx, &mut ctx)
    });

    // Phase 2: execute actions (mutable access to agent, queues, signals)
    if state.population.get(id).is_none_or(|a| !a.alive) {
        return;
    }

    let kill_enable = state.config.kill_enable;

    // Reuse `agent_ptr` from phase 1 (still valid; agent slot didn't move).
    // SAFETY: `agent_ptr` was obtained from `population.get_mut(id)` at the
    // top of this function. `Population` uses a `Vec<Option<Agent>>` that only
    // ever grows (via `spawn`) — slots are never relocated or removed — so the
    // pointer remains valid and non-dangling. The alive re-check above confirms
    // the slot still contains a live agent; no other code path can free or move
    // this slot while we hold `&mut SimulationState`.
    let agent: &mut crate::agent::Agent = unsafe { &mut *agent_ptr };

    // SAFETY notes for the ActionContext construction:
    //
    // (a) `agent` vs `world.population`:
    //     `agent` is a `&mut Agent` into `population.agents[id]`. `world.population`
    //     is a `&Population` (via `pop_ptr`). These overlap at the type level:
    //     `*pop_ptr` logically contains `*agent_ptr`. However:
    //       - No action implementation reads `world.population.agents` directly;
    //         they call `world.grid.at(loc)` to get an AgentId, then stop.
    //         The aliased slot (`agents[id]`) is never accessed through `world`
    //         during Phase 2.
    //       - `ctx.move_queue` and `ctx.death_queue` are `&mut` into
    //         `population.move_queue` / `population.death_queue` — fields
    //         entirely separate from `population.agents`.
    //     Conclusion: no two live references reach the same memory location.
    //
    // (b) `ctx.signals` vs `world.signals`:
    //     `ctx.signals` is `&mut state.signals` and `world.signals` is
    //     `&*signals_ptr = &state.signals`. These are aliased `&mut T` / `&T`
    //     of the same object, which is technically unsound under Stacked Borrows.
    //     It is safe in practice because NO built-in action reads `world.signals`
    //     during Phase 2; `EmitSignal0` writes via `ctx.signals` and reads the
    //     grid (not signals) from `ctx.world`. A future refactor should split
    //     `ActionContext.world` into separate `grid` and `population` refs so
    //     signals can be dropped from the read-only world view in Phase 2.
    //
    // (c) `action_accum_ptr` / `action_levels` vs ActionContext:
    //     `action_accum` lives in `state.scratch`, a field completely disjoint
    //     from `population`, `signals`, and `rng`. Reading it as `&[f32]` while
    //     ActionContext holds `&mut` to the other fields is safe.
    let mut ctx = ActionContext {
        agent,
        world: &world,
        move_queue: &mut state.population.move_queue,
        death_queue: &mut state.population.death_queue,
        signals: &mut state.signals,
        rng: &mut state.rng,
        config_kill_enable: kill_enable,
    };

    // SAFETY: see note (c) above — action_accum is disjoint from all fields
    // held by ActionContext.
    let action_levels: &[f32] = unsafe { &*action_accum_ptr };
    for (action_idx, &level) in action_levels.iter().enumerate() {
        state.actions.execute(action_idx as u16, level, &mut ctx);
    }

    if let Some(agent) = state.population.get_mut(id) {
        agent.age += 1;
    }
}
