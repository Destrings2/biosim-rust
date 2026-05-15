//! End-to-end smoke test for the programmable-agent abstraction.
//!
//! Verifies: activating the `wanderers` challenge spawns programmables at
//! gen-start, they actually move during step_one, the pool clears on
//! generation rollover and re-spawns, and the whole loop runs without
//! panicking. This is the integration test the plan called for before any
//! more interesting challenge (predators etc.) plugs into the same pool.

use biosim4_core::{
    registry::ChallengeConfig,
    sim_config::SimConfig,
    sim_state::SimulationState,
    sim_step::{step_generation, step_one},
    spawn::spawn_new_generation,
    types::Coord,
};
use std::collections::HashMap;

mod common;
use common::new_state;

fn small_config() -> SimConfig {
    SimConfig {
        size_x: 32,
        size_y: 32,
        population: 20,
        steps_per_generation: 40,
        max_generations: 3,
        rng_seed: 0xDEAF,
        max_number_neurons: 4,
        ..SimConfig::default()
    }
}

fn activate_wanderers(state: &mut SimulationState, count: u16) {
    let cfg = ChallengeConfig {
        active: vec!["wanderers".to_string()],
        composition: Default::default(),
        params: {
            let mut p = HashMap::new();
            p.insert("wanderers".to_string(), serde_json::json!({ "count": count }));
            p
        },
    };
    state.challenges.apply_config(cfg).expect("wanderers challenge registered");
}

#[test]
fn wanderers_spawn_on_generation_start() {
    let mut state = new_state(small_config());
    assert_eq!(state.programmable.alive_count(), 0, "no programmables before activation");

    activate_wanderers(&mut state, 8);
    // Activation alone doesn't spawn; programmables appear after the next
    // generation rollover when `on_generation_start` runs. Force one.
    spawn_new_generation(&mut state);
    assert_eq!(state.programmable.alive_count(), 8, "wanderers spawned at gen-start");
}

#[test]
fn wanderers_actually_move_during_step() {
    let mut state = new_state(small_config());
    activate_wanderers(&mut state, 6);
    spawn_new_generation(&mut state);

    // Snapshot starting positions.
    let start_locs: Vec<Coord> = state.programmable.iter_alive().map(|p| p.loc).collect();
    assert_eq!(start_locs.len(), 6);

    // Run enough steps that random walks should produce at least one move.
    for s in 0..20 {
        step_one(&mut state, s);
    }

    let end_locs: Vec<Coord> = state.programmable.iter_alive().map(|p| p.loc).collect();
    let movers = start_locs.iter().zip(end_locs.iter()).filter(|(a, b)| a != b).count();
    assert!(
        movers > 0,
        "expected at least one wanderer to have moved after 20 steps (start={start_locs:?}, end={end_locs:?})"
    );
}

#[test]
fn wanderers_pool_resets_each_generation() {
    let mut state = new_state(small_config());
    activate_wanderers(&mut state, 5);

    // Gen 1: spawn, run, check.
    spawn_new_generation(&mut state);
    assert_eq!(state.programmable.alive_count(), 5);
    step_generation(&mut state);
    let count_after_run_1 = state.programmable.alive_count();

    // Gen 2: rollover should clear and re-spawn. Even if some died during gen 1
    // (e.g. stepped on a kill barrier), the new gen must have a fresh 5.
    spawn_new_generation(&mut state);
    assert_eq!(state.programmable.alive_count(), 5, "pool re-populates each gen");
    let _ = count_after_run_1; // documented but not strictly asserted
}

#[test]
fn full_loop_with_wanderers_does_not_panic() {
    let mut state = new_state(small_config());
    activate_wanderers(&mut state, 10);

    for _ in 0..3 {
        step_generation(&mut state);
        let _ = spawn_new_generation(&mut state);
    }

    // After 3 generations the pool must still hold the configured count
    // (since wanderers are always re-spawned at gen-start).
    assert_eq!(state.programmable.alive_count(), 10);
    assert!(state.generation >= 3);
}

#[test]
fn nearest_alien_sensor_falls_to_zero_with_close_programmable() {
    use biosim4_core::registry::SensorContext;
    let mut state = new_state(small_config());
    activate_wanderers(&mut state, 0); // 0 wanderers = empty pool baseline
    spawn_new_generation(&mut state);

    // With empty pool, sensor reads 1.0 (= "nothing nearby").
    let agent = state.population.iter_alive().next().expect("at least one peep");
    let agent_loc = agent.loc;
    let agent_id = agent.id;
    let world = state.world();
    // `SensorRegistry::evaluate` takes the *enabled* index (into the
    // dense active_map), not the registration index. After
    // `apply_feature_enables` some sensors are disabled, so the two
    // diverge. Walk `iter()` and count enabled ones to find the right
    // enabled_idx for our sensor.
    let mut enabled_counter = 0u16;
    let mut sensor_idx = None;
    for (_, sensor, enabled) in state.sensors.iter() {
        if !enabled {
            continue;
        }
        if sensor.id() == "nearest_alien_dist" {
            sensor_idx = Some(enabled_counter);
            break;
        }
        enabled_counter += 1;
    }
    let sensor_idx = sensor_idx.expect("nearest_alien_dist registered and enabled");
    let mut rng = biosim4_core::rng::Rng::seeded(0);
    let mut ctx = SensorContext {
        agent: state.population.get(agent_id).unwrap(),
        world: &world,
        sim_step: 0,
        rng: &mut rng,
    };
    let empty_reading = state.sensors.evaluate(sensor_idx, &mut ctx);
    drop(ctx);
    drop(world);
    assert!((empty_reading - 1.0).abs() < 1e-6, "empty pool: sensor returns 1.0");

    // Now drop a programmable RIGHT NEXT to the agent and re-check — sensor
    // should now read very low (close to 0).
    let neighbour = Coord::new(agent_loc.x.saturating_add(1), agent_loc.y);
    if state.grid.is_empty_at(neighbour) {
        let prog = state.programmable.register_or_get("test_static", || Box::new(StaticProgram));
        state.programmable.spawn(&mut state.grid, prog, 0, neighbour, [255, 0, 0]);
    }
    // The pool's spatial index is normally refreshed once per step before
    // sensors fire (see `sim_step::step_one`). Test reads happen outside
    // that loop, so we have to refresh manually.
    let cfg = state.config.clone();
    state.programmable.refresh_spatial_index(cfg.size_x, cfg.size_y);
    let world = state.world();
    let mut rng = biosim4_core::rng::Rng::seeded(0);
    let mut ctx = SensorContext {
        agent: state.population.get(agent_id).unwrap(),
        world: &world,
        sim_step: 0,
        rng: &mut rng,
    };
    let close_reading = state.sensors.evaluate(sensor_idx, &mut ctx);
    assert!(
        close_reading < empty_reading,
        "close programmable: sensor must drop below the empty baseline (close={close_reading}, empty={empty_reading})"
    );
}

// Tiny inert program — never moves, never dies — used in the sensor test
// where we need a programmable at a fixed cell.
struct StaticProgram;
impl biosim4_core::programmable::Program for StaticProgram {
    fn id(&self) -> &str {
        "test_static"
    }
    fn name(&self) -> &str {
        "Static (test)"
    }
    fn step(
        &self,
        _this: &mut biosim4_core::programmable::Programmable,
        _ctx: &mut biosim4_core::programmable::ProgramContext,
        _out: &mut biosim4_core::programmable::ProgramOutput,
    ) {
    }
}
