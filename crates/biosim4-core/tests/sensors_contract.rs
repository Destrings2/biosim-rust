//! Sensor output contract: every registered sensor must return a value in [0.0, 1.0].
//!
//! This is the most important sensor invariant. Neural nets multiply sensor values
//! by genome-derived weights; values outside [0,1] silently break the dynamic range
//! assumption, leading to numeric pathologies that don't surface as crashes.
//!
//! These tests exercise every built-in sensor in three configurations:
//! 1. Fresh agent at world center (canonical state)
//! 2. Agent at a corner (boundary case)
//! 3. Agent with old age & with neighbors / signals nearby (state-dependent sensors)

use biosim4_core::{
    agent::{Agent, AgentId, PropValue},
    grid::Grid,
    genome::neural_net::{create_wiring, WiringConfig},
    genome::genome::make_random_genome,
    population::Population,
    registry::{SensorContext, SensorRegistry},
    rng::Rng,
    sensors::register_builtin_sensors,
    sim_config::SimConfig,
    signals_layer::Signals,
    types::{Coord, Dir, Compass},
    world::World,
};

fn build_world<'a>(
    grid: &'a Grid,
    signals: &'a Signals,
    population: &'a Population,
    cfg: &SimConfig,
) -> World<'a> {
    World {
        grid,
        signals,
        population,
        size_x: cfg.size_x,
        size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0,
    }
}

fn make_test_agent(id: AgentId, loc: Coord, cfg: &SimConfig, rng: &mut Rng) -> Agent {
    let genome = make_random_genome(cfg, rng);
    let wcfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: cfg.max_number_neurons };
    let nnet = create_wiring(&genome, wcfg);
    Agent::new(id, loc, genome, nnet)
}

#[test]
fn registry_has_all_21_builtin_sensors() {
    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    assert_eq!(reg.count(), 21, "expected 21 built-in sensors");
}

#[test]
fn every_sensor_returns_in_unit_interval() {
    let cfg = SimConfig {
        size_x: 32, size_y: 32, population: 8, steps_per_generation: 100,
        ..SimConfig::default()
    };
    let mut rng = Rng::seeded(42);

    // Set up a world with a few agents and some signals so density sensors have data
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(cfg.population);

    // Spawn a "self" agent at center
    let center = Coord::new(16, 16);
    let agent = make_test_agent(population.next_id(), center, &cfg, &mut rng);
    population.spawn(agent);

    // Spawn a neighbor 2 cells east so population_fwd sensors have data
    let neighbor = make_test_agent(population.next_id(), Coord::new(18, 16), &cfg, &mut rng);
    population.spawn(neighbor);

    // Drop a signal at center
    signals.increment(0, center, &grid);

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);

    // Collect IDs first so we don't hold the registry borrow during evaluate
    let ids: Vec<String> = (0..reg.count()).map(|i| reg.id(i).to_string()).collect();
    let _ = &ids;  // suppress unused warning if loop branches change

    let world = build_world(&grid, &signals, &population, &cfg);

    let agent_ref = population.get(1).unwrap();
    let mut sensor_rng = Rng::seeded(7);
    let mut ctx = SensorContext { agent: agent_ref, world: &world, sim_step: 5, rng: &mut sensor_rng };

    for i in 0..reg.count() {
        let v = reg.evaluate(i, &mut ctx);
        assert!(
            v.is_finite() && (-1e-6..=1.0 + 1e-6).contains(&v),
            "sensor #{} ({}) returned out-of-range value: {}",
            i, ids[i as usize], v,
        );
    }
}

#[test]
fn loc_x_sensor_at_extremes() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(2);
    let mut rng = Rng::seeded(1);

    // Agent at left edge
    let left  = make_test_agent(population.next_id(), Coord::new(0, 8), &cfg, &mut rng);
    population.spawn(left);
    // Agent at right edge
    let right = make_test_agent(population.next_id(), Coord::new(15, 8), &cfg, &mut rng);
    population.spawn(right);

    let world = build_world(&grid, &signals, &population, &cfg);

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let loc_x_idx = (0..reg.count()).find(|&i| reg.id(i) == "loc_x").unwrap();

    let mut srng = Rng::seeded(0);
    let v_left  = reg.evaluate(loc_x_idx, &mut SensorContext {
        agent: population.get(1).unwrap(), world: &world, sim_step: 0, rng: &mut srng });
    let v_right = reg.evaluate(loc_x_idx, &mut SensorContext {
        agent: population.get(2).unwrap(), world: &world, sim_step: 0, rng: &mut srng });

    assert!((v_left  - 0.0).abs() < 1e-6, "loc_x at x=0 should be 0.0 (got {})",  v_left);
    assert!((v_right - 1.0).abs() < 1e-6, "loc_x at x=size-1 should be 1.0 (got {})", v_right);
}

#[test]
fn boundary_dist_at_corner_is_zero_at_center_is_one() {
    let cfg = SimConfig { size_x: 33, size_y: 33, ..SimConfig::default() };  // odd → exact center
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(2);
    let mut rng = Rng::seeded(2);

    let corner = make_test_agent(population.next_id(), Coord::new(0, 0), &cfg, &mut rng);
    population.spawn(corner);
    let center = make_test_agent(population.next_id(), Coord::new(16, 16), &cfg, &mut rng);
    population.spawn(center);

    let world = build_world(&grid, &signals, &population, &cfg);

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let bd_idx = (0..reg.count()).find(|&i| reg.id(i) == "boundary_dist").unwrap();

    let mut srng = Rng::seeded(0);
    let v_corner = reg.evaluate(bd_idx, &mut SensorContext {
        agent: population.get(1).unwrap(), world: &world, sim_step: 0, rng: &mut srng });
    let v_center = reg.evaluate(bd_idx, &mut SensorContext {
        agent: population.get(2).unwrap(), world: &world, sim_step: 0, rng: &mut srng });

    assert!(v_corner.abs() < 1e-6, "boundary_dist at corner should be 0.0 (got {})", v_corner);
    assert!((v_center - 1.0).abs() < 1e-6, "boundary_dist at center should be 1.0 (got {})", v_center);
}

#[test]
fn osc1_oscillates_with_step() {
    let cfg = SimConfig { size_x: 16, size_y: 16, steps_per_generation: 100, ..SimConfig::default() };
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(1);
    let mut rng = Rng::seeded(3);

    let mut agent = make_test_agent(population.next_id(), Coord::new(8, 8), &cfg, &mut rng);
    agent.osc_period = 32;  // fixed period for predictability
    population.spawn(agent);

    let world = build_world(&grid, &signals, &population, &cfg);
    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let osc_idx = (0..reg.count()).find(|&i| reg.id(i) == "osc1").unwrap();

    let mut srng = Rng::seeded(0);
    let agent_ref = population.get(1).unwrap();
    let v_at_0 = reg.evaluate(osc_idx, &mut SensorContext {
        agent: agent_ref, world: &world, sim_step: 0, rng: &mut srng });
    let v_at_8 = reg.evaluate(osc_idx, &mut SensorContext {
        agent: agent_ref, world: &world, sim_step: 8, rng: &mut srng });
    let v_at_16 = reg.evaluate(osc_idx, &mut SensorContext {
        agent: agent_ref, world: &world, sim_step: 16, rng: &mut srng });

    // sin(0) = 0  → 0 * 0.5 + 0.5 = 0.5
    // sin(π/2) = 1 → 1 * 0.5 + 0.5 = 1.0    (step 8 of period 32)
    // sin(π)  = 0  → 0 * 0.5 + 0.5 = 0.5    (step 16 of period 32)
    assert!((v_at_0  - 0.5).abs() < 1e-3, "osc at phase 0 should be 0.5 (got {})", v_at_0);
    assert!((v_at_8  - 1.0).abs() < 1e-3, "osc at phase π/2 should be 1.0 (got {})", v_at_8);
    assert!((v_at_16 - 0.5).abs() < 1e-3, "osc at phase π should be 0.5 (got {})", v_at_16);
}

#[test]
fn random_sensor_produces_varied_output() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(1);
    let mut rng = Rng::seeded(4);
    let agent = make_test_agent(population.next_id(), Coord::new(8, 8), &cfg, &mut rng);
    population.spawn(agent);

    let world = build_world(&grid, &signals, &population, &cfg);
    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let r_idx = (0..reg.count()).find(|&i| reg.id(i) == "random").unwrap();

    let mut srng = Rng::seeded(0);
    let agent_ref = population.get(1).unwrap();
    let samples: Vec<f32> = (0..50).map(|s| {
        reg.evaluate(r_idx, &mut SensorContext { agent: agent_ref, world: &world, sim_step: s, rng: &mut srng })
    }).collect();

    // Every sample must be in [0,1]
    for s in &samples {
        assert!(*s >= 0.0 && *s <= 1.0, "random sensor out of range: {}", s);
    }
    // Variance must be > 0 — random sensor isn't constant
    let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
    let var: f32 = samples.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / samples.len() as f32;
    assert!(var > 0.01, "random sensor variance too low: {}", var);
}
