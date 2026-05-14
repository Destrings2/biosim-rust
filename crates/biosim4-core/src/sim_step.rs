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
//! shared move/death queues, mutates signals (atomic), advances per-agent
//! age/energy. Runs in parallel using thread-local move/death queues and a
//! thread-local Rng, merged at the end with no ordering guarantee.
//!
//! # Determinism contract
//!
//! Determinism is conditional on thread count:
//!
//! - **`num_threads == 1`** (or the `parallel` feature off): fully
//!   reproducible at a fixed `rng_seed`. Single-thread runs route every
//!   draw through `state.rng` and the per-agent Phase 1 hash.
//! - **`num_threads > 1`**: intentionally non-deterministic. Phase 2
//!   workers seed thread-local Rngs from system entropy, signal fade /
//!   energy bookkeeping run in parallel, and chunk-local queues merge in
//!   arbitrary work-stealing order. This trade is what makes the parallel
//!   path ~3× faster than 1-thread on common workloads.
//!
//! Phase 1 still uses a stateless `(rng_seed, generation, sim_step,
//! agent_id)` hash regardless of thread count, so **per-agent sensor
//! randomness is always reproducible** — only Phase 2 action draws diverge
//! across thread counts.
//!
//! # Alive-IDs snapshot
//!
//! `step_all_agents` snapshots `population.alive_ids()` into
//! `scratch.alive_ids` at the start of each step. The loop walks the
//! snapshot by index rather than holding a borrow on `scratch`, so phase 2
//! can take `&mut state` without borrow-checker conflict.
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
    fade_signals(state);
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
    // Snapshot alive_ids into the reusable scratch buffer instead of
    // allocating a fresh Vec each step.
    state.scratch.alive_ids.clear();
    state.scratch.alive_ids.extend_from_slice(state.population.alive_ids());
    let n = state.scratch.alive_ids.len();

    state.scratch.per_agent_action_levels.resize_with(n, Vec::new);
    state.scratch.per_agent_neuron_accum.resize_with(n, Vec::new);

    phase1_compute_all(state, n);
    phase2_actions_all(state, n);
    if state.config.enable_energy {
        phase2_energy_all(state, n);
    }
}

// ── Determinism / parallelism dispatch ──────────────────────────────────────

/// Decide whether to run the parallel path. Centralised so all phases agree.
#[inline]
fn use_parallel(state: &SimulationState, n: usize) -> bool {
    #[cfg(feature = "parallel")]
    {
        state.config.num_threads.max(1) as usize > 1 && n > 1
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (state, n);
        false
    }
}

/// Raw-pointer wrapper used to thread `&mut SimulationState` through rayon
/// closures. Each closure invocation operates on a disjoint per-agent slot
/// (or atomic/thread-local field), so the aliasing is sound — see SAFETY
/// notes on each phase's worker body for the per-call argument.
#[cfg(feature = "parallel")]
#[derive(Copy, Clone)]
struct StatePtr(*mut SimulationState);

#[cfg(feature = "parallel")]
impl StatePtr {
    #[allow(clippy::mut_from_ref)]
    unsafe fn as_mut<'a>(&self) -> &'a mut SimulationState {
        unsafe { &mut *self.0 }
    }
}

#[cfg(feature = "parallel")]
unsafe impl Send for StatePtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for StatePtr {}

// ── Phase 1 ────────────────────────────────────────────────────────────────

/// Stateless per-agent sensor-rng seed for Phase 1. Hashes
/// `(rng_seed, generation, sim_step, agent_id)` so each agent's Phase 1 has
/// an independent stream without any draws on `state.rng`. Kept deterministic
/// regardless of thread scheduling so per-agent sensor randomness is
/// reproducible from `(seed, gen, step, id)` alone.
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

fn phase1_compute_all(state: &mut SimulationState, n: usize) {
    let action_count = state.actions.enabled_count();
    let sim_step = state.sim_step;
    let generation = state.generation;
    let size_x = state.config.size_x;
    let size_y = state.config.size_y;
    let steps_per_gen = state.config.steps_per_generation;
    let rng_seed = state.config.rng_seed;

    if use_parallel(state, n) {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let state_ptr = StatePtr(state as *mut SimulationState);

            (0..n).into_par_iter().for_each(|i| {
                // SAFETY: each `i` is unique per closure invocation. Phase 1
                // mutates only `agents[alive_ids[i]].nnet` and the i-th
                // entries of `scratch.per_agent_*`. Everything else is read.
                let state: &mut SimulationState = unsafe { state_ptr.as_mut() };
                let id = state.scratch.alive_ids[i];
                let seed = phase1_seed_for(rng_seed, generation, sim_step, id);
                let lvls = &mut state.scratch.per_agent_action_levels[i] as *mut _;
                let accum = &mut state.scratch.per_agent_neuron_accum[i] as *mut _;
                unsafe {
                    phase1_one_agent(state, id, seed, action_count, sim_step,
                                     generation, size_x, size_y, steps_per_gen,
                                     lvls, accum);
                }
            });
            return;
        }
    }

    for i in 0..n {
        let id = state.scratch.alive_ids[i];
        let seed = phase1_seed_for(rng_seed, generation, sim_step, id);
        let lvls = &mut state.scratch.per_agent_action_levels[i] as *mut _;
        let accum = &mut state.scratch.per_agent_neuron_accum[i] as *mut _;
        unsafe {
            phase1_one_agent(state, id, seed, action_count, sim_step,
                             generation, size_x, size_y, steps_per_gen,
                             lvls, accum);
        }
    }
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
    // Skip dead agents defensively — a challenge hook may have killed
    // someone after the snapshot.
    if state.population.get(id).is_none_or(|a| !a.alive) {
        unsafe { (*action_levels_out).clear(); }
        return;
    }

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

// ── Phase 2: actions + aging ───────────────────────────────────────────────

/// Per-step args we want once at the top so the inner loops don't keep
/// reborrowing them out of `state.config`.
struct Phase2Args {
    sim_step: u32,
    generation: u32,
    kill_enable: bool,
    size_x: u16,
    size_y: u16,
    steps_per_gen: u32,
}

fn phase2_actions_all(state: &mut SimulationState, n: usize) {
    let args = Phase2Args {
        sim_step: state.sim_step,
        generation: state.generation,
        kill_enable: state.config.kill_enable,
        size_x: state.config.size_x,
        size_y: state.config.size_y,
        steps_per_gen: state.config.steps_per_generation,
    };

    if use_parallel(state, n) {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let state_ptr = StatePtr(state as *mut SimulationState);

            // Per-worker accumulator: thread-local move/death queues + Rng.
            // Rayon `fold` calls the init closure once per worker, so each
            // worker pays the Rng/Vec allocation cost once per step (not per
            // agent). The reduce step then merges accumulators in arbitrary
            // work-stealing order — non-deterministic but cheap.
            let merged: (Vec<(AgentId, Coord)>, Vec<AgentId>) = (0..n)
                .into_par_iter()
                .fold(
                    || (Vec::<(AgentId, Coord)>::with_capacity(64),
                        Vec::<AgentId>::with_capacity(8),
                        Rng::from_entropy()),
                    |(mut moves, mut deaths, mut rng), i| {
                        // SAFETY: unique `i` ⇒ unique agent_id ⇒ disjoint
                        // mutation target. Signal writes are atomic. Queue
                        // pushes go into thread-local Vecs.
                        let state: &mut SimulationState = unsafe { state_ptr.as_mut() };
                        unsafe {
                            phase2_one_agent(state, i, &args, &mut moves, &mut deaths, &mut rng);
                        }
                        (moves, deaths, rng)
                    },
                )
                .map(|(m, d, _rng)| (m, d))
                .reduce(
                    || (Vec::new(), Vec::new()),
                    |(mut ma, mut da), (mb, db)| {
                        ma.extend(mb);
                        da.extend(db);
                        (ma, da)
                    },
                );

            state.population.move_queue.extend(merged.0);
            state.population.death_queue.extend(merged.1);
            return;
        }
    }

    // Sequential fallback. Uses `state.rng` directly so single-thread runs
    // stay bit-exact at a fixed seed.
    let state_ptr: *mut SimulationState = state as *mut _;
    for i in 0..n {
        // SAFETY: sequential loop. Each iteration uniquely owns the state
        // mutations it makes; the raw pointer just sidesteps the borrow
        // checker since `phase2_one_agent` needs split borrows of state.
        unsafe {
            let s: &mut SimulationState = &mut *state_ptr;
            let move_q: *mut Vec<(AgentId, Coord)> = &mut s.population.move_queue;
            let death_q: *mut Vec<AgentId> = &mut s.population.death_queue;
            let rng: *mut Rng = &mut s.rng;
            phase2_one_agent(s, i, &args, &mut *move_q, &mut *death_q, &mut *rng);
        }
    }
}

/// Per-agent body shared by the parallel and sequential paths.
///
/// # Safety
///
/// - `agent_levels[i]` must be uniquely owned by this call (parallel mode
///   guarantees this through `i` uniqueness).
/// - `moves`, `deaths`, `rng` are thread-local accumulators in the parallel
///   path and the population's queues in the sequential path; either way,
///   no other concurrent reader/writer touches them during this call.
unsafe fn phase2_one_agent(
    state: &mut SimulationState,
    i: usize,
    args: &Phase2Args,
    moves: &mut Vec<(AgentId, Coord)>,
    deaths: &mut Vec<AgentId>,
    rng: &mut Rng,
) {
    let id = state.scratch.alive_ids[i];
    if state.population.get(id).is_none_or(|a| !a.alive) { return; }

    let agent_ptr: *mut crate::agent::Agent = match state.population.get_mut(id) {
        Some(a) if a.alive => a as *mut _,
        _ => return,
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
        size_x: args.size_x,
        size_y: args.size_y,
        steps_per_generation: args.steps_per_gen,
        generation: args.generation,
        step: args.sim_step,
    };

    let agent: &mut crate::agent::Agent = unsafe { &mut *agent_ptr };

    let mut ctx = ActionContext {
        agent,
        world: &world,
        move_queue: moves,
        death_queue: deaths,
        signals: &state.signals,
        rng,
        config_kill_enable: args.kill_enable,
    };

    let action_levels: &[f32] = unsafe { &*action_levels_ptr };
    for (action_idx, &level) in action_levels.iter().enumerate() {
        state.actions.execute(action_idx as u16, level, &mut ctx);
    }

    if let Some(agent) = state.population.get_mut(id) {
        agent.age += 1;
    }
}

// ── Phase 2b: energy bookkeeping ───────────────────────────────────────────

fn phase2_energy_all(state: &mut SimulationState, n: usize) {
    let cost = state.config.energy_per_step_cost;

    if use_parallel(state, n) {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let state_ptr = StatePtr(state as *mut SimulationState);

            // Each agent only touches its own cell in `food` (grid invariant:
            // one agent per cell). Per-agent energy mutation is unique by
            // agent id. Thread-local death queues are merged at the end.
            let merged_deaths: Vec<AgentId> = (0..n)
                .into_par_iter()
                .fold(
                    || Vec::<AgentId>::with_capacity(8),
                    |mut deaths, i| {
                        let state: &mut SimulationState = unsafe { state_ptr.as_mut() };
                        unsafe {
                            energy_one_agent(state, i, cost, &mut deaths);
                        }
                        deaths
                    },
                )
                .reduce(Vec::new, |mut a, b| { a.extend(b); a });

            state.population.death_queue.extend(merged_deaths);
            return;
        }
    }

    // Sequential fallback.
    let state_ptr: *mut SimulationState = state as *mut _;
    for i in 0..n {
        unsafe {
            let s: &mut SimulationState = &mut *state_ptr;
            let dq: *mut Vec<AgentId> = &mut s.population.death_queue;
            energy_one_agent(s, i, cost, &mut *dq);
        }
    }
}

/// # Safety
///
/// `deaths` must not be aliased by any other thread for the duration of this
/// call. `state.food` is mutated only at the agent's own loc, which is unique
/// per (agent_id, step) by the grid-occupancy invariant.
unsafe fn energy_one_agent(
    state: &mut SimulationState,
    i: usize,
    cost: f32,
    deaths: &mut Vec<AgentId>,
) {
    let id = state.scratch.alive_ids[i];
    let Some(agent) = state.population.get(id) else { return };
    if !agent.alive { return; }

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
            deaths.push(id);
        }
    }
}

// ── Signal fade ────────────────────────────────────────────────────────────

fn fade_signals(state: &mut SimulationState) {
    let layer_count = state.signals.layer_count();
    if layer_count == 0 { return; }

    #[cfg(feature = "parallel")]
    {
        if state.config.num_threads.max(1) > 1 {
            state.signals.fade_all_parallel();
            return;
        }
    }
    for layer in 0..layer_count {
        state.signals.fade(layer);
    }
}
