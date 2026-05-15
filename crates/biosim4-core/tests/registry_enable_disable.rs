//! Tests for per-sensor and per-action enable / disable.
//!
//! ## Stories covered
//!
//! 1. **Registry unit** — `set_enabled` / `is_enabled` reflect state correctly.
//! 2. **Active-map coherence** — `enabled_count` tracks registered + enabled items.
//! 3. **Mid-generation silencing** — disabled sensor returns 0.0 immediately;
//!    disabled action is silently skipped before the next commit.
//! 4. **Generation boundary** — `commit_enabled` (called inside
//!    `spawn_new_generation`) reduces `wiring_config` counts so new neural
//!    nets reference only enabled sensors/actions.
//! 5. **Re-enable** — re-enabling a sensor restores it to the active map at
//!    the next commit; the count goes back up.
//! 6. **End-to-end** — three generations run to completion without panic when
//!    several sensors and actions are disabled mid-run.
//! 7. **Sensor-output contract preserved** — active sensors still return [0,1]
//!    after some peers are disabled.
//! 8. **All-disabled guard** — disabling every sensor still produces a valid
//!    (if degenerate) simulation run without panicking.

use biosim4_actions::register_builtin_actions;
use biosim4_core::{
    agent::{Agent, AgentId},
    food_layer::FoodLayer,
    genome::{
        neural_net::{create_wiring, WiringConfig},
        ops::make_random_genome,
    },
    grid::Grid,
    population::Population,
    registry::{
        action::{ActionContext, ActionRegistry},
        sensor::{SensorContext, SensorRegistry},
        ChallengeComposition, ChallengeConfig,
    },
    rng::Rng,
    signals_layer::Signals,
    sim_config::SimConfig,
    sim_state::SimulationState,
    sim_step::step_generation,
    spawn::spawn_new_generation,
    types::Coord,
    world::World,
};
use biosim4_sensors::register_builtin_sensors;

mod common;
use common::new_state;

// ── Helpers ───────────────────────────────────────────────────────────────

fn small_cfg() -> SimConfig {
    SimConfig {
        size_x: 32,
        size_y: 32,
        population: 40,
        steps_per_generation: 30,
        max_generations: 3,
        rng_seed: 999,
        max_number_neurons: 3,
        ..SimConfig::default()
    }
}

fn make_test_agent(id: AgentId, loc: Coord, cfg: &SimConfig, rng: &mut Rng) -> Agent {
    let genome = make_random_genome(cfg, rng);
    let wcfg =
        WiringConfig { sensor_count: 21, action_count: 17, max_neurons: cfg.max_number_neurons };
    let nnet = create_wiring(&genome, wcfg);
    Agent::new(id, loc, genome, nnet)
}

fn build_world<'a>(
    grid: &'a Grid,
    signals: &'a Signals,
    food: &'a FoodLayer,
    population: &'a Population,
    cfg: &SimConfig,
) -> World<'a> {
    World {
        grid,
        signals,
        food,
        population,
        size_x: cfg.size_x,
        size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0,
        step: 0,
    }
}

/// Builds a minimal ActionContext using raw-pointer split borrows (same
/// pattern as `sim_step.rs` and `actions_contract.rs`).
///
/// SAFETY: callers must not mutate the underlying agent / signals through
/// the world reference inside the closure.
fn with_action_ctx<R>(
    cfg: &SimConfig,
    grid: &Grid,
    signals: &mut Signals,
    population: &mut Population,
    agent_id: AgentId,
    rng: &mut Rng,
    f: impl FnOnce(&mut ActionContext) -> R,
) -> R {
    let mut move_q: Vec<(AgentId, Coord)> = Vec::new();
    let mut death_q: Vec<AgentId> = Vec::new();

    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let grid_ptr: *const Grid = grid;
    let signals_const_ptr: *const Signals = signals;
    let signals_mut_ptr: *mut Signals = signals;
    let pop_ptr: *const Population = population;
    let agent_ptr: *mut Agent = population.get_mut(agent_id).expect("agent exists");

    let world = World {
        grid: unsafe { &*grid_ptr },
        signals: unsafe { &*signals_const_ptr },
        food: &food,
        population: unsafe { &*pop_ptr },
        size_x: cfg.size_x,
        size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0,
        step: 0,
    };

    let mut ctx = ActionContext {
        agent: unsafe { &mut *agent_ptr },
        world: &world,
        move_queue: &mut move_q,
        death_queue: &mut death_q,
        signals: unsafe { &mut *signals_mut_ptr },
        rng,
        config_kill_enable: false,
        responsiveness_adjusted: biosim4_core::registry::action::response_curve(
            unsafe { &*agent_ptr }.responsiveness,
            cfg.responsiveness_curve_k_factor,
        ),
        move_x_urge: 0.0,
        move_y_urge: 0.0,
    };

    f(&mut ctx)
}

// ── 1. Registry unit ─────────────────────────────────────────────────────

#[test]
fn sensor_is_enabled_by_default_after_registration() {
    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    for (_, s, enabled) in reg.iter() {
        assert!(enabled, "sensor '{}' should be enabled by default", s.id());
    }
}

#[test]
fn action_is_enabled_by_default_after_registration() {
    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    for (_, a, enabled) in reg.iter() {
        assert!(enabled, "action '{}' should be enabled by default", a.id());
    }
}

#[test]
fn sensor_set_enabled_false_reports_disabled() {
    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    reg.set_enabled("loc_x", false);
    assert!(!reg.is_enabled("loc_x"), "loc_x should be disabled");
    // All others must still be enabled
    for (_, s, enabled) in reg.iter() {
        if s.id() != "loc_x" {
            assert!(enabled, "sensor '{}' should still be enabled", s.id());
        }
    }
}

#[test]
fn action_set_enabled_false_reports_disabled() {
    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    reg.set_enabled("move_east", false);
    assert!(!reg.is_enabled("move_east"));
}

#[test]
fn sensor_re_enable_restores_enabled_state() {
    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    reg.set_enabled("loc_y", false);
    assert!(!reg.is_enabled("loc_y"));
    reg.set_enabled("loc_y", true);
    assert!(reg.is_enabled("loc_y"), "loc_y should be re-enabled");
}

// ── 2. Active-map coherence ───────────────────────────────────────────────

#[test]
fn enabled_count_equals_total_count_when_all_enabled() {
    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    assert_eq!(
        reg.enabled_count(),
        reg.count(),
        "all sensors enabled: enabled_count should equal total count"
    );
}

#[test]
fn sensor_enabled_count_decreases_after_commit() {
    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let total = reg.count();

    reg.set_enabled("loc_x", false);
    reg.set_enabled("loc_y", false);
    // Pending — count unchanged until commit
    assert_eq!(reg.enabled_count(), total, "enabled_count must not change before commit");

    reg.commit_enabled();
    assert_eq!(
        reg.enabled_count(),
        total - 2,
        "enabled_count should drop by 2 after committing two disables"
    );
}

#[test]
fn action_enabled_count_decreases_after_commit() {
    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let total = reg.count();

    reg.set_enabled("move_east", false);
    reg.set_enabled("move_west", false);
    reg.commit_enabled();
    assert_eq!(reg.enabled_count(), total - 2, "enabled_count should drop by 2 after commit");
}

#[test]
fn re_enable_and_commit_restores_count() {
    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let total = reg.count();

    reg.set_enabled("loc_x", false);
    reg.commit_enabled();
    assert_eq!(reg.enabled_count(), total - 1);

    reg.set_enabled("loc_x", true);
    reg.commit_enabled();
    assert_eq!(reg.enabled_count(), total, "re-enabling and committing should restore count");
}

#[test]
fn disabling_multiple_and_partial_re_enable_tracks_correctly() {
    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let total = reg.count();

    reg.set_enabled("loc_x", false);
    reg.set_enabled("loc_y", false);
    reg.set_enabled("boundary_dist", false);
    reg.commit_enabled();
    assert_eq!(reg.enabled_count(), total - 3);

    // Re-enable one
    reg.set_enabled("loc_y", true);
    reg.commit_enabled();
    assert_eq!(reg.enabled_count(), total - 2);
}

// ── 3. Mid-generation silencing ───────────────────────────────────────────

#[test]
fn disabled_sensor_returns_zero_before_commit() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(1);
    let mut rng = Rng::seeded(1);

    let agent = make_test_agent(population.next_id(), Coord::new(8, 8), &cfg, &mut rng);
    population.spawn(agent);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let world = build_world(&grid, &signals, &food, &population, &cfg);

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);

    // loc_x at x=8 in a 16-wide world should return 0.5, not 0.0
    let loc_x_enabled_idx = (0..reg.enabled_count())
        .find(|&i| {
            let actual = reg.iter().nth(i as usize).map(|(_, s, _)| s.id().to_string());
            actual.as_deref() == Some("loc_x")
        })
        .unwrap_or(0);

    let agent_ref = population.get(1).unwrap();
    let mut srng = Rng::seeded(0);
    let before = reg.evaluate(
        loc_x_enabled_idx,
        &mut SensorContext { agent: agent_ref, world: &world, sim_step: 0, rng: &mut srng },
    );
    assert!(before > 0.0, "loc_x at center should be non-zero before disable: {}", before);

    // Disable it (pending — no commit). The active_map still points to loc_x,
    // but evaluate() checks the pending disabled set and returns 0.0.
    reg.set_enabled("loc_x", false);
    let after = reg.evaluate(
        loc_x_enabled_idx,
        &mut SensorContext { agent: agent_ref, world: &world, sim_step: 0, rng: &mut srng },
    );
    assert_eq!(after, 0.0, "disabled sensor must return 0.0 immediately (mid-gen)");
}

#[test]
fn disabled_action_is_not_executed_before_commit() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(1);
    let mut rng = Rng::seeded(2);

    let agent = make_test_agent(population.next_id(), Coord::new(5, 5), &cfg, &mut rng);
    let id = population.spawn(agent);
    grid.set(Coord::new(5, 5), id);
    population.get_mut(id).unwrap().responsiveness = 1.0;

    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);

    // Find move_east in the enabled set (it IS enabled, so its index in active_map
    // is just its position among enabled actions — same as registration index until commit)
    let east_enabled_idx = (0..reg.enabled_count())
        .find(|&i| {
            // Before any commit, active_map == [0, 1, 2, ...], so enabled_idx == actual_idx
            reg.id(i) == "move_east"
        })
        .unwrap();

    // Pre-condition: move_east with high level queues a move after resolution.
    let mut arng = Rng::seeded(99);
    let queued_before =
        with_action_ctx(&cfg, &grid, &mut signals, &mut population, id, &mut arng, |ctx| {
            reg.execute(east_enabled_idx, 10.0, ctx);
            biosim4_core::registry::action::resolve_movement(ctx);
            ctx.move_queue.clone()
        });
    assert!(!queued_before.is_empty(), "pre-condition: move_east should queue moves when enabled");

    // Disable move_east (pending — no commit). Its execute is now skipped,
    // so the urge accumulator stays at zero and resolve_movement queues
    // nothing.
    reg.set_enabled("move_east", false);

    let mut arng2 = Rng::seeded(99);
    let queued_after =
        with_action_ctx(&cfg, &grid, &mut signals, &mut population, id, &mut arng2, |ctx| {
            reg.execute(east_enabled_idx, 10.0, ctx);
            biosim4_core::registry::action::resolve_movement(ctx);
            ctx.move_queue.clone()
        });
    assert!(
        queued_after.is_empty(),
        "disabled action must not execute even before commit; got {:?}",
        queued_after
    );
}

// ── 4. Generation boundary — wiring_config reflects enabled set ───────────

#[test]
fn wiring_config_sensor_count_uses_enabled_count_after_commit() {
    let mut state = new_state(small_cfg());

    // Run one generation first so apply_feature_enables fires and establishes
    // the feature-gated baseline (food/energy and signal-layer sensors disabled
    // by default config before any user overrides).
    step_generation(&mut state);
    spawn_new_generation(&mut state);
    let baseline = state.sensors.enabled_count();

    // Now disable two more user-chosen sensors and advance another generation.
    state.sensors.set_enabled("loc_x", false);
    state.sensors.set_enabled("loc_y", false);
    step_generation(&mut state);
    spawn_new_generation(&mut state);

    let wcfg = state.wiring_config();
    assert_eq!(
        wcfg.sensor_count,
        baseline - 2,
        "wiring_config.sensor_count should equal enabled_count after commit"
    );
}

#[test]
fn wiring_config_action_count_uses_enabled_count_after_commit() {
    let mut state = new_state(small_cfg());

    // Establish baseline after apply_feature_enables runs for the first time.
    step_generation(&mut state);
    spawn_new_generation(&mut state);
    let baseline = state.actions.enabled_count();

    state.actions.set_enabled("move_east", false);
    state.actions.set_enabled("move_west", false);
    state.actions.set_enabled("move_north", false);
    step_generation(&mut state);
    spawn_new_generation(&mut state);

    let wcfg = state.wiring_config();
    assert_eq!(
        wcfg.action_count,
        baseline - 3,
        "wiring_config.action_count should equal enabled_count after commit"
    );
}

#[test]
fn new_generation_agents_wired_against_reduced_genome() {
    let mut state = new_state(small_cfg());

    // Disable 5 sensors before the second generation
    state.sensors.set_enabled("loc_x", false);
    state.sensors.set_enabled("loc_y", false);
    state.sensors.set_enabled("age", false);
    state.sensors.set_enabled("osc1", false);
    state.sensors.set_enabled("random", false);
    step_generation(&mut state);
    spawn_new_generation(&mut state); // commits the disabled set

    let expected_sensor_count = state.sensors.enabled_count();

    // Verify every alive agent's nnet was wired against the reduced sensor count.
    // feed_forward resolves sensor indices via the active_map; indices must be
    // in 0..enabled_count or the gene's modulo-based source would alias incorrectly.
    // We check that no gene source index exceeds enabled_count.
    use biosim4_core::genome::gene::SOURCE_SENSOR;
    for agent in state.population.iter_alive() {
        for g in agent.nnet.all_connections() {
            if g.source_type() == SOURCE_SENSOR {
                assert!(
                    (g.source_num() as u16) < expected_sensor_count,
                    "agent {} has a gene referencing sensor #{} but only {} are enabled",
                    agent.id,
                    g.source_num(),
                    expected_sensor_count
                );
            }
        }
    }
}

// ── 5. End-to-end: run multiple generations with partial disables ──────────

#[test]
fn three_generations_with_disabled_sensors_and_actions_no_panic() {
    let mut state = new_state(small_cfg());

    // Apply a challenge so there's selection pressure
    let cc = ChallengeConfig {
        active: vec!["right_half".into()],
        composition: ChallengeComposition::Any,
        params: Default::default(),
    };
    state.challenges.apply_config(cc).unwrap();

    // Disable a mix of sensors and actions before generation 1
    state.sensors.set_enabled("loc_x", false);
    state.sensors.set_enabled("osc1", false);
    state.actions.set_enabled("move_east", false);
    state.actions.set_enabled("emit_signal0", false);

    for _ in 0..3 {
        step_generation(&mut state);
        spawn_new_generation(&mut state);
    }

    assert_eq!(state.generation, 3);
    assert_eq!(
        state.population.alive_count() as u32,
        state.config.population,
        "population should remain full after 3 generations with partial disables"
    );
}

#[test]
fn all_sensors_disabled_simulation_runs_without_panic() {
    // Degenerate case: no sensor input → all outputs ≈ 0 → agents wander randomly
    // (or just stand still). The sim must not panic or overflow.
    let mut state =
        new_state(SimConfig { population: 20, steps_per_generation: 20, ..small_cfg() });

    let sensor_ids: Vec<String> =
        state.sensors.iter().map(|(_, s, _)| s.id().to_string()).collect();
    for id in &sensor_ids {
        state.sensors.set_enabled(id, false);
    }

    step_generation(&mut state);
    spawn_new_generation(&mut state);

    assert_eq!(state.generation, 1, "generation counter should advance normally");
    assert_eq!(state.population.alive_count() as u32, state.config.population);
}

#[test]
fn all_actions_disabled_agents_remain_alive() {
    let mut state =
        new_state(SimConfig { population: 20, steps_per_generation: 20, ..small_cfg() });

    let action_ids: Vec<String> =
        state.actions.iter().map(|(_, a, _)| a.id().to_string()).collect();
    for id in &action_ids {
        state.actions.set_enabled(id, false);
    }

    step_generation(&mut state);
    // Agents cannot move or kill — all should still be alive
    assert_eq!(
        state.population.alive_count() as u32,
        state.config.population,
        "agents with all actions disabled must remain alive (no movement or kills possible)"
    );
}

// ── 6. Sensor output contract preserved with peers disabled ───────────────

#[test]
fn remaining_sensors_still_return_unit_interval_when_some_are_disabled() {
    let cfg = SimConfig {
        size_x: 32,
        size_y: 32,
        population: 4,
        steps_per_generation: 100,
        ..SimConfig::default()
    };
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(3, cfg.size_x, cfg.size_y);
    let mut population = Population::new(cfg.population);
    let mut rng = Rng::seeded(55);

    let agent = make_test_agent(population.next_id(), Coord::new(16, 16), &cfg, &mut rng);
    population.spawn(agent);

    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let world = build_world(&grid, &signals, &food, &population, &cfg);

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);

    // Disable several sensors
    for id in &["loc_x", "loc_y", "age", "osc1", "random"] {
        reg.set_enabled(id, false);
    }
    reg.commit_enabled();

    let agent_ref = population.get(1).unwrap();
    let mut srng = Rng::seeded(0);

    for enabled_idx in 0..reg.enabled_count() {
        let v = reg.evaluate(
            enabled_idx,
            &mut SensorContext { agent: agent_ref, world: &world, sim_step: 5, rng: &mut srng },
        );
        assert!(
            v.is_finite() && (-1e-6..=1.0 + 1e-6).contains(&v),
            "enabled sensor at index {} returned out-of-range value: {}",
            enabled_idx,
            v
        );
    }
}

// ── 7. Determinism preserved across disable/re-enable cycle ───────────────

#[test]
fn disable_and_reenable_before_generation_is_deterministic() {
    // Two identical runs with the same seed: one disables-then-re-enables a
    // sensor before the first spawn; both should produce the same state.
    // Pinned to single-thread because multi-thread runs use entropy-seeded
    // thread-local Rngs and would diverge regardless of the toggle.
    let mut cfg = small_cfg();
    cfg.num_threads = 1;

    let run = |toggle: bool| {
        let mut state = new_state(cfg.clone());
        if toggle {
            state.sensors.set_enabled("loc_x", false);
            state.sensors.set_enabled("loc_x", true); // re-enable before commit
        }
        step_generation(&mut state);
        spawn_new_generation(&mut state);
        state.population.iter_alive().map(|a| (a.id, a.loc, a.age)).collect::<Vec<_>>()
    };

    let r1 = run(false);
    let r2 = run(true);
    assert_eq!(r1, r2, "toggling a sensor off then back on before commit should be a no-op");
}
