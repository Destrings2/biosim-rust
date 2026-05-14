//! Sensor tests for state-dependent families that aren't covered by
//! `sensors_contract.rs`: signal layers, memory registers, energy, and food.
//!
//! Each test isolates a single sensor, sets the world state it reads from,
//! and asserts the sensor returns the expected value (or, where the mapping
//! is nontrivial, that it responds in the expected direction).

use biosim4_core::{
    agent::{Agent, AgentId},
    food_layer::FoodLayer,
    grid::Grid,
    genome::neural_net::{create_wiring, WiringConfig},
    genome::ops::make_random_genome,
    population::Population,
    registry::{SensorContext, SensorRegistry},
    rng::Rng,
    sensors::register_builtin_sensors,
    sim_config::SimConfig,
    signals_layer::Signals,
    types::Coord,
    world::World,
};

fn make_agent(id: AgentId, loc: Coord, cfg: &SimConfig, rng: &mut Rng) -> Agent {
    let genome = make_random_genome(cfg, rng);
    let wcfg = WiringConfig { sensor_count: 36, action_count: 17, max_neurons: cfg.max_number_neurons };
    let nnet = create_wiring(&genome, wcfg);
    Agent::new(id, loc, genome, nnet)
}

fn world<'a>(
    grid: &'a Grid, signals: &'a Signals, food: &'a FoodLayer, population: &'a Population,
    cfg: &SimConfig,
) -> World<'a> {
    World {
        grid, signals, food, population,
        size_x: cfg.size_x, size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0, step: 0,
    }
}

fn sensor_idx(reg: &SensorRegistry, id: &str) -> u16 {
    (0..reg.count())
        .find(|&i| reg.id(i) == id)
        .unwrap_or_else(|| panic!("sensor `{id}` not registered"))
}

#[test]
fn signal0_responds_to_local_signal() {
    let cfg = SimConfig { size_x: 32, size_y: 32, ..SimConfig::default() };
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(1);
    let agent = make_agent(pop.next_id(), Coord::new(16, 16), &cfg, &mut rng);
    pop.spawn(agent);

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let idx = sensor_idx(&reg, "signal0");

    let mut srng = Rng::seeded(0);
    let w = world(&grid, &signals, &food, &pop, &cfg);
    let baseline = reg.evaluate(idx, &mut SensorContext {
        agent: pop.get(1).unwrap(), world: &w, sim_step: 0, rng: &mut srng,
    });

    // Drop a strong signal at the agent's cell.
    for _ in 0..50 { signals.increment(0, Coord::new(16, 16), &grid); }

    let w = world(&grid, &signals, &food, &pop, &cfg);
    let after = reg.evaluate(idx, &mut SensorContext {
        agent: pop.get(1).unwrap(), world: &w, sim_step: 0, rng: &mut srng,
    });

    assert!(after > baseline, "signal0 should rise when signal is deposited (baseline={baseline}, after={after})");
    assert!((0.0..=1.0).contains(&after), "signal0 out of unit range: {after}");
}

#[test]
fn signal_sensors_each_layer_isolated() {
    // Verify each Signal{N} sensor reads its own layer.
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(3, cfg.size_x, cfg.size_y);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(2);
    let agent = make_agent(pop.next_id(), Coord::new(8, 8), &cfg, &mut rng);
    pop.spawn(agent);

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);

    // Deposit only on layer 1.
    for _ in 0..30 { signals.increment(1, Coord::new(8, 8), &grid); }

    let w = world(&grid, &signals, &food, &pop, &cfg);
    let mut srng = Rng::seeded(0);
    let agent_ref = pop.get(1).unwrap();

    let s0 = reg.evaluate(sensor_idx(&reg, "signal0"),
        &mut SensorContext { agent: agent_ref, world: &w, sim_step: 0, rng: &mut srng });
    let s1 = reg.evaluate(sensor_idx(&reg, "signal1"),
        &mut SensorContext { agent: agent_ref, world: &w, sim_step: 0, rng: &mut srng });
    let s2 = reg.evaluate(sensor_idx(&reg, "signal2"),
        &mut SensorContext { agent: agent_ref, world: &w, sim_step: 0, rng: &mut srng });

    assert!(s1 > s0, "signal1 should exceed signal0 when only layer 1 has signal (s0={s0}, s1={s1})");
    assert!(s1 > s2, "signal1 should exceed signal2 when only layer 1 has signal (s2={s2}, s1={s1})");
    assert!(s0.abs() < 1e-3, "signal0 should be ~0 with no layer-0 signal (got {s0})");
    assert!(s2.abs() < 1e-3, "signal2 should be ~0 with no layer-2 signal (got {s2})");
}

#[test]
fn memory_sensors_read_agent_memory() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(3);
    let mut agent = make_agent(pop.next_id(), Coord::new(8, 8), &cfg, &mut rng);
    agent.memory = [0.10, 0.42, 0.71, 0.95];
    pop.spawn(agent);

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let w = world(&grid, &signals, &food, &pop, &cfg);
    let mut srng = Rng::seeded(0);
    let agent_ref = pop.get(1).unwrap();

    let m: Vec<f32> = ["memory_0", "memory_1", "memory_2", "memory_3"]
        .iter()
        .map(|id| reg.evaluate(sensor_idx(&reg, id),
            &mut SensorContext { agent: agent_ref, world: &w, sim_step: 0, rng: &mut srng }))
        .collect();

    let expected = [0.10f32, 0.42, 0.71, 0.95];
    for (i, (got, exp)) in m.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "memory_{i}: got {got}, expected {exp}");
    }
}

#[test]
fn energy_level_clamped_to_unit_interval() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let mut pop = Population::new(3);
    let mut rng = Rng::seeded(4);

    // Three agents with energy at 0.0, 0.7, and an out-of-range 1.5.
    for (i, e) in [0.0f32, 0.7, 1.5].iter().enumerate() {
        let mut a = make_agent(pop.next_id(), Coord::new(i as i16, 0), &cfg, &mut rng);
        a.energy = *e;
        pop.spawn(a);
    }

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let w = world(&grid, &signals, &food, &pop, &cfg);
    let mut srng = Rng::seeded(0);
    let idx = sensor_idx(&reg, "energy_level");

    let v0 = reg.evaluate(idx, &mut SensorContext {
        agent: pop.get(1).unwrap(), world: &w, sim_step: 0, rng: &mut srng });
    let v1 = reg.evaluate(idx, &mut SensorContext {
        agent: pop.get(2).unwrap(), world: &w, sim_step: 0, rng: &mut srng });
    let v2 = reg.evaluate(idx, &mut SensorContext {
        agent: pop.get(3).unwrap(), world: &w, sim_step: 0, rng: &mut srng });

    assert!((v0 - 0.0).abs() < 1e-6, "energy=0 → 0.0 (got {v0})");
    assert!((v1 - 0.7).abs() < 1e-6, "energy=0.7 → 0.7 (got {v1})");
    assert!((v2 - 1.0).abs() < 1e-6, "energy=1.5 (out of range) should clamp to 1.0 (got {v2})");
}

#[test]
fn food_here_reads_local_cell() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(5);
    let agent = make_agent(pop.next_id(), Coord::new(4, 4), &cfg, &mut rng);
    pop.spawn(agent);

    food.set(Coord::new(4, 4), 0.6);

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let w = world(&grid, &signals, &food, &pop, &cfg);
    let mut srng = Rng::seeded(0);
    let v = reg.evaluate(sensor_idx(&reg, "food_here"), &mut SensorContext {
        agent: pop.get(1).unwrap(), world: &w, sim_step: 0, rng: &mut srng });
    assert!((v - 0.6).abs() < 1e-6, "food_here should equal cell value (got {v})");
}

#[test]
fn food_fwd_increases_with_food_ahead() {
    let cfg = SimConfig { size_x: 32, size_y: 32, ..SimConfig::default() };
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(6);
    let mut agent = make_agent(pop.next_id(), Coord::new(16, 16), &cfg, &mut rng);
    // Force heading East so the "fwd" axis is well-defined.
    agent.last_move_dir = biosim4_core::types::Dir(biosim4_core::types::Compass::E);
    pop.spawn(agent);

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let idx = sensor_idx(&reg, "food_fwd");

    let w = world(&grid, &signals, &food, &pop, &cfg);
    let mut srng = Rng::seeded(0);
    let baseline = reg.evaluate(idx, &mut SensorContext {
        agent: pop.get(1).unwrap(), world: &w, sim_step: 0, rng: &mut srng });

    // Place food on cells east of the agent.
    for dx in 1..=3 {
        food.set(Coord::new(16 + dx, 16), 1.0);
    }

    let w = world(&grid, &signals, &food, &pop, &cfg);
    let after = reg.evaluate(idx, &mut SensorContext {
        agent: pop.get(1).unwrap(), world: &w, sim_step: 0, rng: &mut srng });

    assert!(after > baseline, "food_fwd should rise when food is placed forward (baseline={baseline}, after={after})");
    assert!((0.0..=1.0).contains(&after), "food_fwd out of unit range: {after}");
}
