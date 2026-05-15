//! Per-step simulation execution engine.
//!
//! # Entry points
//!
//! [`step_generation`] runs all steps in one generation.
//! [`step_one`] runs a single step and is also exported for embedders (e.g.
//! the WASM frontend) that drive the simulation incrementally.
//!
//! # Fused per-agent pipeline
//!
//! Each step the per-agent work (Phase 1 sensors+nnet, Phase 2 actions+aging,
//! Phase 2b energy) runs **inside a single rayon par_iter** — one fork/join
//! per step instead of three. Per-agent state stays warm in L1 across the
//! three phases, and the worker keeps its nnet scratch and RNG in
//! `thread_local!` cells so the fold body never allocates.
//!
//! **Phase 1 — sensor evaluation + neural feed-forward.** Reads world state
//! immutably; writes only to the agent's own `nnet` and a worker-local
//! `(action_levels, neuron_accum)` scratch pair.
//!
//! **Phase 2 — action execution + age bookkeeping.** Mutates per-agent
//! fields (responsiveness/osc_period/memory/age), pushes to worker-local
//! move/death queues, increments signal cells atomically.
//!
//! **Phase 2b — energy.** Reads/writes the agent's own food cell, mutates
//! `agent.energy`, queues death on starvation.
//!
//! # Fusion safety contract
//!
//! Fusing the phases is sound because Phase 2 only mutates the agent's
//! **own** fields, and Phase 1 only reads peer state via fields that Phase
//! 2 does *not* touch — peer `loc` / `heading` / `last_move_dir` are
//! updated at end-of-step in `drain_move_queue`, peer `genome` is immutable
//! per generation. Any new sensor that reads peer `responsiveness`,
//! `osc_period`, `memory`, `age`, or `energy` would break this contract
//! and require unfusing.
//!
//! # Determinism contract
//!
//! Determinism is conditional on thread count:
//!
//! - **`num_threads == 1`** (or the `parallel` feature off): fully
//!   reproducible at a fixed `rng_seed`. Single-thread runs route every
//!   draw through `state.rng` and the per-agent Phase 1 hash.
//! - **`num_threads > 1`**: intentionally non-deterministic. Worker RNGs
//!   seed once from system entropy and persist for the process; chunk-local
//!   queues merge in arbitrary work-stealing order.
//!
//! Phase 1 still uses a stateless `(rng_seed, generation, sim_step,
//! agent_id)` hash regardless of thread count, so **per-agent sensor
//! randomness is always reproducible** — only Phase 2 action draws diverge
//! across thread counts.
//!
//! # Alive-IDs snapshot
//!
//! `step_all_agents` snapshots `population.alive_ids()` into
//! `scratch.alive_ids` at the start of each step. The fused loop walks the
//! snapshot by index so the borrow checker doesn't see a `&mut state`
//! conflicting with a `&state.population.alive_ids` borrow.
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

/// Recompute Phase 1 (sensors + neural feedforward) for a single agent
/// against the current world state and return the resulting action levels.
/// Used by inspectors that want to display the latest action output without
/// the hot step loop having to materialise it for every agent.
///
/// Returns `None` if the agent id is unknown or dead.
pub fn inspect_action_levels(state: &mut SimulationState, agent_id: AgentId) -> Option<Vec<f32>> {
    if state.population.get(agent_id).is_none_or(|a| !a.alive) {
        return None;
    }
    let args = StepArgs::from_state(state);
    let seed = phase1_seed_for(args.rng_seed, args.generation, args.sim_step, agent_id);
    let mut action_levels: Vec<f32> = Vec::new();
    let mut neuron_accum: Vec<f32> = Vec::new();
    // SAFETY: this runs synchronously on the caller's thread with exclusive
    // access to `state`; no other phase is running concurrently.
    unsafe {
        phase1_one_agent(state, agent_id, seed, &args, &mut action_levels, &mut neuron_accum);
    }
    Some(action_levels)
}

/// Run a single simulation step at index `step`. Sets `state.sim_step = step`,
/// runs challenge hooks, ticks every alive agent, then drains the deferred
/// queues and fades signals. Exposed so embedders (e.g. the WASM frontend)
/// can drive the simulation step-by-step for incremental rendering.
pub fn step_one(state: &mut SimulationState, step: u32) {
    state.sim_step = step;
    run_challenge_step_hooks(state);
    let queues = step_all_agents(state);
    state.population.drain_death_queue_from(&mut state.grid, queues.deaths);
    state.population.drain_move_queue_from(&mut state.grid, queues.moves);
    fade_signals(state);
    if state.config.enable_energy {
        state.food.regenerate(state.config.food_regen_rate, &state.grid);
    }
}

/// Move and death lists produced by one simulation step.
///
/// `step_all_agents` returns these lists so they can be passed directly to
/// [`drain_move_queue_from`](crate::population::Population::drain_move_queue_from)
/// and [`drain_death_queue_from`](crate::population::Population::drain_death_queue_from),
/// avoiding a redundant extend through the population's internal queues.
pub struct StepQueues {
    /// Pending (agent_id, destination) moves to apply after all agents step.
    pub moves: Vec<(AgentId, Coord)>,
    /// Agents to kill after all agents step.
    pub deaths: Vec<AgentId>,
}

impl StepQueues {
    fn empty() -> Self {
        Self { moves: Vec::new(), deaths: Vec::new() }
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

fn step_all_agents(state: &mut SimulationState) -> StepQueues {
    // Snapshot alive_ids into the reusable scratch buffer instead of
    // allocating a fresh Vec each step.
    state.scratch.alive_ids.clear();
    state.scratch.alive_ids.extend_from_slice(state.population.alive_ids());
    let n = state.scratch.alive_ids.len();
    if n == 0 {
        return StepQueues::empty();
    }

    let args = StepArgs::from_state(state);

    #[cfg(feature = "parallel")]
    {
        if use_parallel(state, n) {
            return step_all_agents_parallel(state, n, &args);
        }
    }
    step_all_agents_sequential(state, n, &args)
}

/// Per-step args snapshotted from `state.config` once. Threading them
/// through avoids re-borrowing config inside the hot fold body.
struct StepArgs {
    sim_step: u32,
    generation: u32,
    rng_seed: u64,
    action_count: u16,
    size_x: u16,
    size_y: u16,
    steps_per_gen: u32,
    kill_enable: bool,
    energy_enabled: bool,
    energy_cost: f32,
    responsiveness_curve_k: f32,
}

impl StepArgs {
    fn from_state(state: &SimulationState) -> Self {
        Self {
            sim_step: state.sim_step,
            generation: state.generation,
            rng_seed: state.config.rng_seed,
            action_count: state.actions.enabled_count(),
            size_x: state.config.size_x,
            size_y: state.config.size_y,
            steps_per_gen: state.config.steps_per_generation,
            kill_enable: state.config.kill_enable,
            energy_enabled: state.config.enable_energy,
            energy_cost: state.config.energy_per_step_cost,
            responsiveness_curve_k: state.config.responsiveness_curve_k_factor,
        }
    }
}

#[cfg(feature = "parallel")]
fn step_all_agents_parallel(state: &mut SimulationState, n: usize, args: &StepArgs) -> StepQueues {
    use rayon::prelude::*;
    let state_ptr = StatePtr(state as *mut SimulationState);

    // One rayon fold for the whole per-agent pipeline (phase1 → phase2 → energy).
    // Going from three par_iter calls to one cuts the fork/join overhead by 3×
    // and lets each agent stay hot in L1 across all phases. Per-worker scratch
    // for nnet buffers lives in a `thread_local!`; the action RNG also lives
    // in a `thread_local!`. `with_min_len` caps the number of fold subgroups
    // so we don't pay the fold-init Vec allocations on tiny chunks.
    let (moves, deaths) = (0..n)
        .into_par_iter()
        .with_min_len(MIN_FOLD_CHUNK)
        .fold(
            || (Vec::<(AgentId, Coord)>::with_capacity(64), Vec::<AgentId>::with_capacity(8)),
            |(mut moves, mut deaths), i| {
                // SAFETY: unique `i` ⇒ unique agent_id ⇒ disjoint mutation
                // target for agent fields (heading/age/memory/etc.) and the
                // agent's own nnet. Cross-agent reads in Phase 1 only touch
                // peer position/heading/genome — fields that Phase 2 does not
                // mutate. Signal writes are atomic. Queue pushes go into
                // worker-local Vecs merged by `reduce`.
                let state = unsafe { state_ptr.as_mut() };
                WORKER_SCRATCH.with(|sc| {
                    WORKER_RNG.with(|r| {
                        let mut scratch = sc.borrow_mut();
                        let mut rng = r.borrow_mut();
                        let (action_levels, neuron_accum) = &mut *scratch;
                        unsafe {
                            step_one_agent(
                                state,
                                i,
                                args,
                                &mut moves,
                                &mut deaths,
                                &mut rng,
                                action_levels,
                                neuron_accum,
                            );
                        }
                    });
                });
                (moves, deaths)
            },
        )
        .reduce(
            || (Vec::new(), Vec::new()),
            |(mut ma, mut da), (mb, db)| {
                ma.extend(mb);
                da.extend(db);
                (ma, da)
            },
        );

    StepQueues { moves, deaths }
}

/// Floor on the number of agents in one rayon fold subgroup. Below this,
/// fork/join + fold-init overhead exceeds the savings from extra parallelism.
#[cfg(feature = "parallel")]
const MIN_FOLD_CHUNK: usize = 32;

fn step_all_agents_sequential(
    state: &mut SimulationState,
    n: usize,
    args: &StepArgs,
) -> StepQueues {
    let mut moves: Vec<(AgentId, Coord)> = Vec::new();
    let mut deaths: Vec<AgentId> = Vec::new();
    let mut action_levels: Vec<f32> = Vec::new();
    let mut neuron_accum: Vec<f32> = Vec::new();
    let state_ptr: *mut SimulationState = state as *mut _;
    for i in 0..n {
        // SAFETY: sequential loop, raw pointer just sidesteps the borrow
        // checker for the split borrows step_one_agent needs.
        unsafe {
            let s: &mut SimulationState = &mut *state_ptr;
            let rng: *mut Rng = &mut s.rng;
            step_one_agent(
                s,
                i,
                args,
                &mut moves,
                &mut deaths,
                &mut *rng,
                &mut action_levels,
                &mut neuron_accum,
            );
        }
    }
    StepQueues { moves, deaths }
}

/// Fused per-agent step body: phase1 → phase2 → energy bookkeeping.
///
/// # Safety
///
/// Same contract as the per-phase helpers it calls: `i` must be unique among
/// concurrent calls so the targeted agent slot is disjoint. The `moves`,
/// `deaths`, `rng`, `action_levels`, `neuron_accum` buffers must all be
/// owned by this call alone.
unsafe fn step_one_agent(
    state: &mut SimulationState,
    i: usize,
    args: &StepArgs,
    moves: &mut Vec<(AgentId, Coord)>,
    deaths: &mut Vec<AgentId>,
    rng: &mut Rng,
    action_levels: &mut Vec<f32>,
    neuron_accum: &mut Vec<f32>,
) {
    let id = state.scratch.alive_ids[i];
    // A challenge step-hook may have killed someone after the alive_ids
    // snapshot — bail before running phase 1.
    if state.population.get(id).is_none_or(|a| !a.alive) {
        action_levels.clear();
        return;
    }

    let seed = phase1_seed_for(args.rng_seed, args.generation, args.sim_step, id);
    unsafe {
        phase1_one_agent(state, id, seed, args, action_levels, neuron_accum);
        phase2_one_agent(state, id, args, action_levels, moves, deaths, rng);
        if args.energy_enabled {
            energy_one_agent(state, id, args.energy_cost, deaths);
        }
    }
}

// ── Determinism / parallelism dispatch ──────────────────────────────────────

/// Decide whether to run the parallel path. Centralised so all phases agree.
#[cfg(feature = "parallel")]
#[inline]
fn use_parallel(state: &SimulationState, n: usize) -> bool {
    state.config.num_threads.max(1) as usize > 1 && n > 1
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

#[cfg(feature = "parallel")]
thread_local! {
    /// Per-worker RNG for Phase 2 action draws. Initialised from entropy on
    /// first use per thread, then reused for the rest of the process.
    /// Replaces a previous design that called `Rng::from_entropy()` inside
    /// every rayon `fold` init — i.e. once per fold subgroup per step.
    static WORKER_RNG: std::cell::RefCell<Rng> = std::cell::RefCell::new(Rng::from_entropy());

    /// Per-worker nnet scratch — `(action_levels, neuron_accum)`. Reused
    /// across every agent the worker processes, so the rayon fold body
    /// allocates these vecs once per worker per process (then keeps the
    /// capacity warm). Replaces a previous design that stored a
    /// `Vec<Vec<f32>>` indexed by alive_id on `state.scratch`.
    static WORKER_SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>)>
        = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

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

/// Phase 1 for a single agent. Reads world state immutably; mutates only
/// this agent's `nnet`. Writes feedforward results into `action_levels`.
///
/// # Safety
///
/// Caller must guarantee that no other thread is reading or writing this
/// agent's `nnet` for the duration of the call, that `action_levels` and
/// `neuron_accum` are uniquely owned by this call, and that `state.grid`,
/// `state.signals`, `state.food`, `state.population`, `state.sensors` are
/// not being mutated by any other thread.
unsafe fn phase1_one_agent(
    state: &mut SimulationState,
    id: AgentId,
    seed: u64,
    args: &StepArgs,
    action_levels: &mut Vec<f32>,
    neuron_accum: &mut Vec<f32>,
) {
    let agent_ptr: *mut crate::agent::Agent = match state.population.get_mut(id) {
        Some(a) if a.alive => a as *mut _,
        _ => {
            action_levels.clear();
            return;
        }
    };
    // SAFETY: `nnet` is a sub-field projection; only this thread owns the
    // unique agent slot named by `id` (caller contract). The read-only
    // `agent_read` reborrow below names a disjoint sub-field set — feed_forward
    // touches only `nnet.*`, and sensors read only `agent.{loc, heading,
    // last_move_dir, oscillator_period, long_probe_dist, responsiveness, age,
    // memory, energy, challenge_bits, genome, genome_color}`. Phase 2 will not
    // run until feed_forward returns, so the agent record is effectively
    // partitioned for this phase.
    let nnet: &mut crate::genome::neural_net::NeuralNet = unsafe { &mut (*agent_ptr).nnet };
    let agent_read: &crate::agent::Agent = unsafe { &*agent_ptr };

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
        size_x: args.size_x,
        size_y: args.size_y,
        steps_per_generation: args.steps_per_gen,
        generation: args.generation,
        step: args.sim_step,
    };

    feed_forward(nnet, args.action_count, action_levels, neuron_accum, |sensor_idx| {
        let mut ctx = SensorContext {
            agent: agent_read,
            world: &world,
            sim_step: args.sim_step,
            rng: &mut sensor_rng,
        };
        state.sensors.evaluate(sensor_idx, &mut ctx)
    });
}

// ── Phase 2: actions + aging ───────────────────────────────────────────────

/// Phase 2 for a single agent. Mutates this agent's fields and pushes to
/// the caller-provided move/death queues.
///
/// # Safety
///
/// Caller must guarantee `id` names a unique agent in this batch (no other
/// thread is mutating that agent), and that `moves`, `deaths`, `rng` are
/// uniquely owned by this call.
unsafe fn phase2_one_agent(
    state: &mut SimulationState,
    id: AgentId,
    args: &StepArgs,
    action_levels: &[f32],
    moves: &mut Vec<(AgentId, Coord)>,
    deaths: &mut Vec<AgentId>,
    rng: &mut Rng,
) {
    let agent_ptr: *mut crate::agent::Agent = match state.population.get_mut(id) {
        Some(a) if a.alive => a as *mut _,
        _ => return,
    };

    let grid_ptr = &state.grid as *const _;
    let signals_ptr = &state.signals as *const _;
    let food_ptr = &state.food as *const _;
    let pop_ptr = &state.population as *const _;

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

    // Resolve set_responsiveness once per agent so the main dispatch loop
    // can cheaply skip it. None when the action is registered-but-disabled
    // or absent entirely.
    let set_resp_idx = state.actions.enabled_index("set_responsiveness");

    // `responsiveness_adjusted` is a placeholder here — overwritten below
    // before any motor action observes it. The movement urge accumulators
    // start fresh at zero for every agent step.
    let mut ctx = ActionContext {
        agent,
        world: &world,
        move_queue: moves,
        death_queue: deaths,
        signals: &state.signals,
        rng,
        config_kill_enable: args.kill_enable,
        responsiveness_adjusted: 0.0,
        move_x_urge: 0.0,
        move_y_urge: 0.0,
    };

    // Run set_responsiveness first so any update to agent.responsiveness is
    // visible to every gated motor action in the same step.
    if let Some(idx) = set_resp_idx {
        let level = action_levels[idx as usize];
        state.actions.execute(idx, level, &mut ctx);
    }

    // Snapshot the (possibly just-updated) responsiveness once per agent so
    // every motor action this step shares the same scale factor.
    ctx.responsiveness_adjusted = crate::registry::action::response_curve(
        ctx.agent.responsiveness,
        args.responsiveness_curve_k,
    );

    for (action_idx, &level) in action_levels.iter().enumerate() {
        if Some(action_idx as u16) == set_resp_idx {
            continue;
        }
        state.actions.execute(action_idx as u16, level, &mut ctx);
    }

    // Combine accumulated movement urges into at most one queued grid step.
    crate::registry::action::resolve_movement(&mut ctx);

    if let Some(agent) = state.population.get_mut(id) {
        agent.age += 1;
    }
}

// ── Phase 2b: energy bookkeeping ───────────────────────────────────────────

/// # Safety
///
/// Caller must guarantee `id` is a unique alive agent for this batch.
/// `state.food` is mutated only at the agent's own loc, which is unique
/// per (agent_id, step) by the grid-occupancy invariant.
unsafe fn energy_one_agent(
    state: &mut SimulationState,
    id: AgentId,
    cost: f32,
    deaths: &mut Vec<AgentId>,
) {
    let Some(agent) = state.population.get(id) else { return };
    if !agent.alive {
        return;
    }

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
    if layer_count == 0 {
        return;
    }

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
