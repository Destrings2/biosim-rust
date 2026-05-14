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
//! Each step has two phases for every alive agent:
//!
//! **Phase 1 — sensor evaluation + neural feed-forward.** Pure per-agent
//! computation. Reads world state immutably; writes only to the agent's own
//! `nnet` and to a per-agent action-levels output buffer. Has no observable
//! effect on shared state, so it is safe to run in parallel across agents.
//!
//! **Phase 2 — action execution + age/energy bookkeeping.** Pushes to the
//! shared move/death queues, mutates signals, draws from `state.rng`, and
//! advances per-agent age/energy. Runs sequentially in `alive_ids` order so
//! that rng advancement and queue ordering are deterministic.
//!
//! # Phase 1 rng
//!
//! Each agent's Phase 1 uses an independent rng, seeded from a stateless hash
//! of `(rng_seed, generation, sim_step, agent_id)`. No draws are taken from
//! `state.rng` during Phase 1, which removes the only sequential bottleneck.
//! Phase 1 results are still reproducible across runs with the same seed
//! (the hash is deterministic), but the result no longer matches the
//! advancement pattern of a single-threaded fork-per-agent scheme.
//!
//! # Alive-IDs snapshot
//!
//! `step_all_agents` snapshots `population.alive_ids()` into
//! `scratch.alive_ids` at the start of each step. The loop walks the snapshot
//! by index rather than holding a borrow on `scratch`, so phase 2 can take
//! `&mut state` without borrow-checker conflict.
//!
//! # Deferred queues
//!
//! Actions push to `population.move_queue` and `population.death_queue`.
//! These are drained at end-of-step in order: deaths first (freeing cells),
//! then moves (entering freed cells).

use crate::agent::AgentId;
use crate::genome::neural_net::feed_forward;
use crate::registry::action::ActionContext;
use crate::registry::challenge::WorldMut;
use crate::registry::sensor::SensorContext;
use crate::rng::Rng;
use crate::sim_state::SimulationState;
use crate::types::Coord;
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
    for layer in 0..state.signals.layer_count() {
        state.signals.fade(layer);
    }
    if state.config.enable_energy {
        state.food.regenerate(state.config.food_regen_rate, &state.grid);
    }
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
    // ── Setup ────────────────────────────────────────────────────────────
    //
    // Snapshot alive_ids into the reusable scratch buffer instead of allocating
    // a fresh Vec each step.
    state.scratch.alive_ids.clear();
    state.scratch.alive_ids.extend_from_slice(state.population.alive_ids());
    let n = state.scratch.alive_ids.len();

    // Size the per-agent scratch buffers to match alive_ids.
    state.scratch.per_agent_action_levels.resize_with(n, Vec::new);
    state.scratch.per_agent_neuron_accum.resize_with(n, Vec::new);

    // ── Phase 1 (parallelizable) ────────────────────────────────────────
    phase1_compute_all(state, n);

    // ── Phase 2 (parallelizable) — action execution + age ───────────────
    phase2_actions_all(state, n);

    // ── Phase 2b (sequential) — energy bookkeeping ──────────────────────
    if state.config.enable_energy {
        phase2_energy_sequential(state, n);
    }
}

/// Stateless per-agent sensor-rng seed. Hashes
/// `(rng_seed, generation, sim_step, agent_id)` so each agent's Phase 1 has an
/// independent stream without any draws on `state.rng`.
#[inline]
fn phase1_seed_for(rng_seed: u64, generation: u32, sim_step: u32, id: AgentId) -> u64 {
    // SplitMix64-style mix — fast and well-distributed.
    let mut z = rng_seed
        .wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(generation as u64 + 1))
        .wrapping_add(0xBF58476D1CE4E5B9u64.wrapping_mul(sim_step as u64 + 1))
        .wrapping_add(id as u64);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Run Phase 1 (sensor evaluation + neural feed-forward) for every alive agent.
/// Writes the resulting action levels into `scratch.per_agent_action_levels[i]`
/// and mutates each agent's `nnet` (neuron outputs).
fn phase1_compute_all(state: &mut SimulationState, n: usize) {
    #[cfg(feature = "parallel")]
    {
        let num_threads = state.config.num_threads.max(1) as usize;
        if num_threads > 1 && n > 1 {
            phase1_compute_all_parallel(state, n);
            return;
        }
    }
    phase1_compute_all_sequential(state, n);
}

fn phase1_compute_all_sequential(state: &mut SimulationState, n: usize) {
    let action_count = state.actions.enabled_count();
    let sim_step = state.sim_step;
    let generation = state.generation;
    let size_x = state.config.size_x;
    let size_y = state.config.size_y;
    let steps_per_gen = state.config.steps_per_generation;
    let rng_seed = state.config.rng_seed;

    for i in 0..n {
        let id = state.scratch.alive_ids[i];
        let seed = phase1_seed_for(rng_seed, generation, sim_step, id);

        // SAFETY: split the &mut SimulationState into disjoint sub-borrows
        // via raw pointers. The per-agent scratch slot `i` is unique to this
        // iteration; `agents[id]` is the only slot mutated (its `nnet` field),
        // and the immutable views of grid/signals/food/population read distinct
        // fields. This mirrors the same pattern as before, just per-call.
        let action_levels_ptr: *mut Vec<f32> =
            &mut state.scratch.per_agent_action_levels[i];
        let neuron_accum_ptr: *mut Vec<f32> =
            &mut state.scratch.per_agent_neuron_accum[i];

        // SAFETY: action_levels_ptr and neuron_accum_ptr point to distinct
        // Vec slots in the scratch buffer; no other code reads or writes
        // these slots during this call.
        unsafe {
            phase1_one_agent(
                state,
                id,
                seed,
                action_count,
                sim_step,
                generation,
                size_x,
                size_y,
                steps_per_gen,
                action_levels_ptr,
                neuron_accum_ptr,
            );
        }
    }
}

#[cfg(feature = "parallel")]
fn phase1_compute_all_parallel(state: &mut SimulationState, n: usize) {
    use rayon::prelude::*;

    let action_count = state.actions.enabled_count();
    let sim_step = state.sim_step;
    let generation = state.generation;
    let size_x = state.config.size_x;
    let size_y = state.config.size_y;
    let steps_per_gen = state.config.steps_per_generation;
    let rng_seed = state.config.rng_seed;

    // SAFETY wrapper for the state pointer. Each closure invocation accesses
    // a unique `i` (and the unique `agent_id` it maps to), so the per-agent
    // mutations target disjoint memory:
    //   - scratch.per_agent_action_levels[i]
    //   - scratch.per_agent_neuron_accum[i]
    //   - population.agents[alive_ids[i]].nnet
    // All other state reads (grid, signals, food, population.agents for
    // sensors, sensors registry, config) are immutable. No data races.
    //
    // `as_mut` forces the closure to capture the whole `StatePtr` wrapper
    // (which is `Send + Sync`) rather than partial-capturing the inner raw
    // pointer (which is neither).
    #[derive(Copy, Clone)]
    struct StatePtr(*mut SimulationState);
    impl StatePtr {
        #[allow(clippy::mut_from_ref)]
        unsafe fn as_mut<'a>(&self) -> &'a mut SimulationState {
            unsafe { &mut *self.0 }
        }
    }
    unsafe impl Send for StatePtr {}
    unsafe impl Sync for StatePtr {}
    let state_ptr = StatePtr(state as *mut SimulationState);

    (0..n).into_par_iter().for_each(|i| {
            // SAFETY: see StatePtr SAFETY note above; each closure invocation has
            // a unique `i` and operates on disjoint memory.
            let state: &mut SimulationState = unsafe { state_ptr.as_mut() };
            let id = state.scratch.alive_ids[i];
            let seed = phase1_seed_for(rng_seed, generation, sim_step, id);

            let action_levels_ptr: *mut Vec<f32> =
                &mut state.scratch.per_agent_action_levels[i];
            let neuron_accum_ptr: *mut Vec<f32> =
                &mut state.scratch.per_agent_neuron_accum[i];

            unsafe {
                phase1_one_agent(
                    state,
                    id,
                    seed,
                    action_count,
                    sim_step,
                    generation,
                    size_x,
                    size_y,
                    steps_per_gen,
                    action_levels_ptr,
                    neuron_accum_ptr,
                );
            }
        });
    }

/// Phase 1 for a single agent. Reads world state immutably; mutates only this
/// agent's `nnet` and the per-agent scratch buffers passed in by pointer.
///
/// # Safety
///
/// - `action_levels_out` and `neuron_accum_out` must point to disjoint, live
///   `Vec<f32>` slots that no other thread accesses for the duration of this
///   call.
/// - `state.population.agents[id].nnet` must not be accessed by any other
///   thread for the duration of this call (uphold via unique `id` per call).
/// - Immutable reads of `state.grid`, `state.signals`, `state.food`,
///   `state.population`, and `state.sensors` must not race with any writer.
unsafe fn phase1_one_agent(
    state: &mut SimulationState,
    id: AgentId,
    seed: u64,
    action_count: u16,
    sim_step: u32,
    generation: u32,
    size_x: u16,
    size_y: u16,
    steps_per_gen: u32,
    action_levels_out: *mut Vec<f32>,
    neuron_accum_out: *mut Vec<f32>,
) {
    // Skip dead agents — defensive; alive_ids snapshot should never contain
    // dead slots, but a challenge hook could conceivably kill someone after
    // the snapshot is taken.
    if state.population.get(id).is_none_or(|a| !a.alive) {
        unsafe { (*action_levels_out).clear(); }
        return;
    }

    // SAFETY: we need both `&mut agent.nnet` (to update neuron outputs in
    // place) and `&Population` (for neighbor-scanning sensors). These overlap
    // only at the type level — sensors never read `nnet`, and `nnet` is a
    // distinct sub-field of `agent`. Raw pointers split the borrow.
    let agent_ptr: *mut crate::agent::Agent = match state.population.get_mut(id) {
        Some(a) if a.alive => a as *mut _,
        _ => {
            unsafe { (*action_levels_out).clear(); }
            return;
        }
    };
    let nnet: &mut crate::genome::neural_net::NeuralNet =
        unsafe { &mut (*agent_ptr).nnet };

    let mut sensor_rng = Rng::seeded(seed);

    let grid_ptr = &state.grid as *const _;
    let signals_ptr = &state.signals as *const _;
    let food_ptr = &state.food as *const _;
    let pop_ptr = &state.population as *const _;

    let world = World {
        grid: unsafe { &*grid_ptr },
        signals: unsafe { &*signals_ptr },
        food: unsafe { &*food_ptr },
        population: unsafe { &*pop_ptr },
        size_x,
        size_y,
        steps_per_generation: steps_per_gen,
        generation,
        step: sim_step,
    };

    // SAFETY: action_levels_out and neuron_accum_out are unique per-agent
    // slots in scratch; see function-level SAFETY note.
    let action_accum: &mut Vec<f32> = unsafe { &mut *action_levels_out };
    let neuron_accum: &mut Vec<f32> = unsafe { &mut *neuron_accum_out };

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
}

/// Phase 2 — run action execution + age increment for every alive agent.
/// Parallelizable via `parallel` feature; each thread gets its own
/// move_queue, death_queue, and Rng, merged into the population's queues
/// after the parallel section finishes.
fn phase2_actions_all(state: &mut SimulationState, n: usize) {
    #[cfg(feature = "parallel")]
    {
        let num_threads = state.config.num_threads.max(1) as usize;
        if num_threads > 1 && n > 1 {
            phase2_actions_all_parallel(state, n, num_threads);
            return;
        }
    }
    phase2_actions_all_sequential(state, n);
}

fn phase2_actions_all_sequential(state: &mut SimulationState, n: usize) {
    let sim_step = state.sim_step;
    let generation = state.generation;
    let kill_enable = state.config.kill_enable;
    let size_x = state.config.size_x;
    let size_y = state.config.size_y;
    let steps_per_gen = state.config.steps_per_generation;

    for i in 0..n {
        let id = state.scratch.alive_ids[i];
        if state.population.get(id).is_none_or(|a| !a.alive) { continue; }

        // SAFETY: phase 2 holds `&mut Agent` for the unique id and `&mut`
        // references to the population's queues + state.rng — all disjoint
        // fields. Sensors aren't run here; the world view is read-only.
        let agent_ptr: *mut crate::agent::Agent = match state.population.get_mut(id) {
            Some(a) if a.alive => a as *mut _,
            _ => continue,
        };

        let grid_ptr = &state.grid as *const _;
        let signals_ptr = &state.signals as *const _;
        let food_ptr = &state.food as *const _;
        let pop_ptr = &state.population as *const _;
        let action_levels_ptr: *const Vec<f32> = &state.scratch.per_agent_action_levels[i];

        let world = World {
            grid: unsafe { &*grid_ptr },
            signals: unsafe { &*signals_ptr },
            food: unsafe { &*food_ptr },
            population: unsafe { &*pop_ptr },
            size_x,
            size_y,
            steps_per_generation: steps_per_gen,
            generation,
            step: sim_step,
        };

        let agent: &mut crate::agent::Agent = unsafe { &mut *agent_ptr };

        let mut ctx = ActionContext {
            agent,
            world: &world,
            move_queue: &mut state.population.move_queue,
            death_queue: &mut state.population.death_queue,
            signals: &state.signals,
            rng: &mut state.rng,
            config_kill_enable: kill_enable,
        };

        let action_levels: &[f32] = unsafe { &*action_levels_ptr };
        for (action_idx, &level) in action_levels.iter().enumerate() {
            state.actions.execute(action_idx as u16, level, &mut ctx);
        }

        if let Some(agent) = state.population.get_mut(id) {
            agent.age += 1;
        }
    }
}

#[cfg(feature = "parallel")]
fn phase2_actions_all_parallel(state: &mut SimulationState, n: usize, num_threads: usize) {
    use rayon::prelude::*;

    let sim_step = state.sim_step;
    let generation = state.generation;
    let kill_enable = state.config.kill_enable;
    let size_x = state.config.size_x;
    let size_y = state.config.size_y;
    let steps_per_gen = state.config.steps_per_generation;

    // Per-thread Rng seeds — drawn sequentially from state.rng so each
    // worker gets an independent stream that's reproducible across runs.
    let rng_base: u64 = rand::RngCore::next_u64(&mut state.rng);

    // Build chunks. Each chunk processes a contiguous range of `i` and has
    // its own local move_queue, death_queue, and Rng.
    let chunk_size = n.div_ceil(num_threads.max(1)).max(1);
    let chunk_starts: Vec<usize> = (0..n).step_by(chunk_size).collect();

    #[derive(Copy, Clone)]
    struct StatePtr(*mut SimulationState);
    impl StatePtr {
        #[allow(clippy::mut_from_ref)]
        unsafe fn as_mut<'a>(&self) -> &'a mut SimulationState {
            unsafe { &mut *self.0 }
        }
    }
    unsafe impl Send for StatePtr {}
    unsafe impl Sync for StatePtr {}
    let state_ptr = StatePtr(state as *mut SimulationState);

    // Each chunk returns its local queues; merge sequentially after.
    let chunk_results: Vec<(Vec<(AgentId, Coord)>, Vec<AgentId>)> = chunk_starts
        .into_par_iter()
        .enumerate()
        .map(|(chunk_idx, start)| {
            let end = (start + chunk_size).min(n);

            let mut local_move = Vec::with_capacity(end - start);
            let mut local_death = Vec::with_capacity(8);
            let mut local_rng = Rng::seeded(rng_base.wrapping_add(chunk_idx as u64 + 1));

            // SAFETY: each chunk processes a disjoint range of `i`, mapping
            // to disjoint agent_ids in alive_ids. The state mutations are:
            //   - agents[id].{responsiveness, osc_period, ..., age} — unique
            //     id per chunk iteration
            //   - state.signals (atomic AtomicU8 cells, &Signals)
            //   - local_move, local_death, local_rng — chunk-local
            // No two chunks alias on agent state.
            let state: &mut SimulationState = unsafe { state_ptr.as_mut() };

            for i in start..end {
                let id = state.scratch.alive_ids[i];
                if state.population.get(id).is_none_or(|a| !a.alive) { continue; }

                let agent_ptr: *mut crate::agent::Agent = match state.population.get_mut(id) {
                    Some(a) if a.alive => a as *mut _,
                    _ => continue,
                };

                let grid_ptr = &state.grid as *const _;
                let signals_ptr = &state.signals as *const _;
                let food_ptr = &state.food as *const _;
                let pop_ptr = &state.population as *const _;
                let action_levels_ptr: *const Vec<f32> =
                    &state.scratch.per_agent_action_levels[i];

                let world = World {
                    grid: unsafe { &*grid_ptr },
                    signals: unsafe { &*signals_ptr },
                    food: unsafe { &*food_ptr },
                    population: unsafe { &*pop_ptr },
                    size_x,
                    size_y,
                    steps_per_generation: steps_per_gen,
                    generation,
                    step: sim_step,
                };

                let agent: &mut crate::agent::Agent = unsafe { &mut *agent_ptr };

                let mut ctx = ActionContext {
                    agent,
                    world: &world,
                    move_queue: &mut local_move,
                    death_queue: &mut local_death,
                    signals: &state.signals,
                    rng: &mut local_rng,
                    config_kill_enable: kill_enable,
                };

                let action_levels: &[f32] = unsafe { &*action_levels_ptr };
                for (action_idx, &level) in action_levels.iter().enumerate() {
                    state.actions.execute(action_idx as u16, level, &mut ctx);
                }

                // Age advance — per-agent, safe to do here.
                if let Some(agent) = state.population.get_mut(id) {
                    agent.age += 1;
                }
            }

            (local_move, local_death)
        })
        .collect();

    // Merge chunk-local queues into the population's queues.
    for (moves, deaths) in chunk_results {
        state.population.move_queue.extend(moves);
        state.population.death_queue.extend(deaths);
    }
}

/// Sequential post-step: apply food eating + energy decrement + queue any
/// agents that ran out of energy for death. Runs after all parallel action
/// execution so food cell writes don't need synchronization (each agent
/// touches only its own loc, which is unique per the grid invariant).
fn phase2_energy_sequential(state: &mut SimulationState, n: usize) {
    let cost = state.config.energy_per_step_cost;

    for i in 0..n {
        let id = state.scratch.alive_ids[i];
        let Some(agent) = state.population.get(id) else { continue };
        if !agent.alive { continue; }

        let loc = agent.loc;
        let food_val = state.food.get(loc);
        let absorbed = if food_val > 0.0 {
            let v = food_val.min(cost * 3.0);
            state.food.set(loc, food_val - v);
            v
        } else {
            0.0
        };

        if let Some(agent) = state.population.get_mut(id) {
            agent.energy = (agent.energy + absorbed).clamp(0.0, 1.0);
            agent.energy -= cost;
            if agent.energy <= 0.0 {
                agent.energy = 0.0;
                state.population.death_queue.push(id);
            }
        }
    }
}
