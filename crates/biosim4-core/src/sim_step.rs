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
    if state.population.get(id).map_or(true, |a| !a.alive) {
        return;
    }

    let sim_step = state.sim_step;
    let action_count = state.actions.count();

    // Phase 1: evaluate sensors → action levels.
    //
    // We need:
    //   - `&mut agent.nnet` (to update neuron outputs in-place — no clone)
    //   - `&Population` (for sensors that scan neighbors)
    //   - `&mut Rng` for stochastic sensors
    //   - `&mut scratch.{action,neuron}_accum` for feed_forward
    //
    // All of these come from `state`, so we use raw pointers to dodge the
    // borrow checker. SAFETY:
    //   - `agent.nnet` is a distinct field from anything sensors read on
    //     `&Agent` (sensors only read loc/heading/age/etc., never `nnet`).
    //   - `scratch.alive_ids` is unrelated to action_accum/neuron_accum,
    //     and the action_accum/neuron_accum fields are disjoint pubs.
    //   - The pointers' lifetimes are bounded by this function scope.
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
    if state.population.get(id).map_or(true, |a| !a.alive) {
        return;
    }

    let kill_enable = state.config.kill_enable;

    // Reuse `agent_ptr` from phase 1 (still valid; agent slot didn't move).
    // SAFETY: agent_ptr came from get_mut and the population's storage is
    // index-stable. We re-check `alive` above, so the agent is still valid.
    let agent: &mut crate::agent::Agent = unsafe { &mut *agent_ptr };

    let mut ctx = ActionContext {
        agent,
        world: &world,
        move_queue: &mut state.population.move_queue,
        death_queue: &mut state.population.death_queue,
        signals: &mut state.signals,
        rng: &mut state.rng,
        config_kill_enable: kill_enable,
    };

    // SAFETY: action_accum lives in `state.scratch`, distinct from population
    // queues / signals / rng. We borrow it as &Vec while ActionContext holds
    // disjoint &mut references — no overlap.
    let action_levels: &[f32] = unsafe { &*action_accum_ptr };
    for (action_idx, &level) in action_levels.iter().enumerate() {
        state.actions.execute(action_idx as u16, level, &mut ctx);
    }

    if let Some(agent) = state.population.get_mut(id) {
        agent.age += 1;
    }
}
