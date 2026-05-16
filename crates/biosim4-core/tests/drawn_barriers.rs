//! User-drawn barriers (painted via the GUI barrier tool) must integrate
//! with the same mechanisms that procedural barriers use:
//!
//! - The `near_barrier` challenge reads `grid.barrier_centers`. Drawn cells
//!   must populate that list so the challenge fires near them.
//! - Barrier sensors (`barrier_fwd`, `longprobe_bar_fwd`) must register
//!   drawn `Kill` barriers, not just `Wall` barriers — both are obstacles
//!   to perception.
//!
//! The `set_barrier` paint path lives in the bevy crate; these tests cover
//! the `reapply_user_barriers` path that runs at every generation reset, plus
//! the sensor-side semantics. The two paths share the invariant.

use biosim4_challenges::register_builtin_challenges;
use biosim4_core::{
    agent::{Agent, AgentId},
    food_layer::FoodLayer,
    genome::neural_net::{create_wiring, WiringConfig},
    genome::ops::make_random_genome,
    grid::{Grid, BARRIER, KILL_BARRIER},
    population::Population,
    programmable::ProgrammablePool,
    registry::{ChallengeRegistry, SensorContext, SensorRegistry},
    rng::Rng,
    signals_layer::Signals,
    sim_config::SimConfig,
    sim_state::{BarrierTile, SimulationState},
    types::Coord,
    world::World,
};
use biosim4_sensors::register_builtin_sensors;

fn make_agent(id: AgentId, loc: Coord, cfg: &SimConfig, rng: &mut Rng) -> Agent {
    let g = make_random_genome(cfg, rng);
    let w =
        WiringConfig { sensor_count: 40, action_count: 17, max_neurons: cfg.max_number_neurons };
    let n = create_wiring(&g, w);
    Agent::new(id, loc, g, n)
}

fn world<'a>(
    grid: &'a Grid,
    signals: &'a Signals,
    food: &'a FoodLayer,
    population: &'a Population,
    programmable: &'a ProgrammablePool,
    cfg: &SimConfig,
) -> World<'a> {
    World {
        grid,
        signals,
        food,
        population,
        programmable,
        size_x: cfg.size_x,
        size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0,
        step: 0,
    }
}

fn sensor_idx(reg: &SensorRegistry, id: &str) -> u16 {
    (0..reg.count())
        .find(|&i| reg.id(i) == id)
        .unwrap_or_else(|| panic!("sensor `{id}` not registered"))
}

#[test]
fn reapply_user_barriers_adds_drawn_cells_to_barrier_centers() {
    let cfg = SimConfig { size_x: 32, size_y: 32, barrier_type: 0, ..SimConfig::default() };
    let mut state = SimulationState::new(cfg);

    state.user_barriers.insert((10, 10), BarrierTile::Wall);
    state.user_barriers.insert((20, 20), BarrierTile::Kill);
    state.user_barriers.insert((5, 5), BarrierTile::Clear); // erase — should not become a center

    state.reapply_user_barriers();

    assert!(state.grid.barrier_centers.contains(&Coord::new(10, 10)));
    assert!(state.grid.barrier_centers.contains(&Coord::new(20, 20)));
    assert!(!state.grid.barrier_centers.contains(&Coord::new(5, 5)));
    assert_eq!(state.grid.at(Coord::new(10, 10)), BARRIER);
    assert_eq!(state.grid.at(Coord::new(20, 20)), KILL_BARRIER);
}

#[test]
fn reapply_user_barriers_dedups_centers_across_repeat_calls() {
    // Generation resets call create_barrier (clears centers) followed by
    // reapply_user_barriers. A naive implementation would double-add on the
    // second call within a single gen if create_barrier wasn't run first.
    let cfg = SimConfig { size_x: 32, size_y: 32, barrier_type: 0, ..SimConfig::default() };
    let mut state = SimulationState::new(cfg);
    state.user_barriers.insert((7, 8), BarrierTile::Wall);

    state.reapply_user_barriers();
    state.reapply_user_barriers();

    let count = state.grid.barrier_centers.iter().filter(|c| **c == Coord::new(7, 8)).count();
    assert_eq!(count, 1, "drawn cell should appear exactly once in barrier_centers");
}

#[test]
fn near_barrier_challenge_fires_on_drawn_barrier() {
    let cfg = SimConfig { size_x: 64, size_y: 64, barrier_type: 0, ..SimConfig::default() };
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let programmable = ProgrammablePool::new();

    // Agent at (32, 32). Drawn barrier at (33, 32) — adjacent.
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    grid.set(Coord::new(33, 32), BARRIER);
    grid.barrier_centers.push(Coord::new(33, 32));

    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(0xD00D);
    pop.spawn(make_agent(pop.next_id(), Coord::new(32, 32), &cfg, &mut rng));

    let w = world(&grid, &signals, &food, &pop, &programmable, &cfg);
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    reg.set_single("near_barrier", Some(serde_json::json!({ "radius": 0.1 }))).unwrap();
    let (pass, score) = reg.evaluate(pop.get(1).unwrap(), &w);
    assert!(pass, "agent adjacent to drawn barrier should pass near_barrier");
    assert!(score > 0.0, "non-zero score expected near a drawn barrier (got {score})");

    // Move the barrier far away — agent should now fail.
    let mut grid_far = Grid::new(cfg.size_x, cfg.size_y);
    grid_far.set(Coord::new(0, 0), BARRIER);
    grid_far.barrier_centers.push(Coord::new(0, 0));
    let w2 = world(&grid_far, &signals, &food, &pop, &programmable, &cfg);
    let (pass2, _) = reg.evaluate(pop.get(1).unwrap(), &w2);
    assert!(!pass2, "agent far from the only drawn barrier should fail near_barrier");
}

#[test]
fn near_barrier_works_for_drawn_kill_cells() {
    // Kill cells painted with the kill-barrier tool are still "barriers" for
    // the purpose of the near_barrier challenge.
    let cfg = SimConfig { size_x: 64, size_y: 64, barrier_type: 0, ..SimConfig::default() };
    let mut state = SimulationState::new(cfg);
    state.user_barriers.insert((40, 40), BarrierTile::Kill);
    state.reapply_user_barriers();

    let signals = Signals::new(1, state.config.size_x, state.config.size_y);
    let food = FoodLayer::new(state.config.size_x, state.config.size_y);
    let programmable = ProgrammablePool::new();
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(0xBADC0DE);
    pop.spawn(make_agent(pop.next_id(), Coord::new(41, 40), &state.config, &mut rng));

    let w = world(&state.grid, &signals, &food, &pop, &programmable, &state.config);
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    reg.set_single("near_barrier", Some(serde_json::json!({ "radius": 0.1 }))).unwrap();
    let (pass, _) = reg.evaluate(pop.get(1).unwrap(), &w);
    assert!(pass, "agent next to drawn kill barrier should pass near_barrier");
}

#[test]
fn barrier_fwd_sensor_detects_drawn_kill_barriers() {
    // A drawn kill barrier directly ahead of the agent should register on
    // the general barrier sensor — both walls and hazards are obstacles to
    // perception.
    let cfg = SimConfig { size_x: 32, size_y: 32, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let programmable = ProgrammablePool::new();
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(0xFEED);

    // Agent default last_move_dir is east (+x). Place a kill barrier just
    // ahead of agent at (16, 16).
    let mut agent = make_agent(pop.next_id(), Coord::new(16, 16), &cfg, &mut rng);
    agent.last_move_dir = biosim4_core::types::Dir::new(biosim4_core::types::Compass::E);
    pop.spawn(agent);
    grid.set(Coord::new(17, 16), KILL_BARRIER);

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let bf_idx = sensor_idx(&reg, "barrier_fwd");
    let lpb_idx = sensor_idx(&reg, "longprobe_bar_fwd");

    let w = world(&grid, &signals, &food, &pop, &programmable, &cfg);
    let mut srng = Rng::seeded(0);
    let agent_ref = pop.get(1).unwrap();
    let bf_with_kill = reg.evaluate(
        bf_idx,
        &mut SensorContext { agent: agent_ref, world: &w, sim_step: 0, rng: &mut srng },
    );
    let lpb_with_kill = reg.evaluate(
        lpb_idx,
        &mut SensorContext { agent: agent_ref, world: &w, sim_step: 0, rng: &mut srng },
    );

    // Compare against an empty grid baseline so we don't depend on the
    // exact normalisation formula.
    let empty_grid = Grid::new(cfg.size_x, cfg.size_y);
    let w_empty = world(&empty_grid, &signals, &food, &pop, &programmable, &cfg);
    let bf_empty = reg.evaluate(
        bf_idx,
        &mut SensorContext { agent: agent_ref, world: &w_empty, sim_step: 0, rng: &mut srng },
    );
    let lpb_empty = reg.evaluate(
        lpb_idx,
        &mut SensorContext { agent: agent_ref, world: &w_empty, sim_step: 0, rng: &mut srng },
    );

    assert_ne!(
        bf_with_kill, bf_empty,
        "barrier_fwd should react to a kill barrier ahead (with={bf_with_kill}, empty={bf_empty})"
    );
    assert!(
        lpb_with_kill < lpb_empty,
        "longprobe_bar_fwd should report a closer barrier when a kill barrier is ahead (with={lpb_with_kill}, empty={lpb_empty})"
    );
}
