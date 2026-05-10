//! Action behavior contract: directional moves queue the right delta, modulators
//! clamp to expected ranges, prob2bool obeys probability, etc.
//!
//! These cover the second architectural pillar — pluggable actions. The most
//! likely class of bugs is a directional action queueing the wrong neighbor cell.

use biosim4_core::{
    actions::{prob2bool, register_builtin_actions, response_curve},
    agent::{Agent, AgentId},
    grid::Grid,
    genome::neural_net::{create_wiring, WiringConfig},
    genome::ops::make_random_genome,
    population::Population,
    registry::{ActionContext, ActionRegistry},
    rng::Rng,
    sim_config::SimConfig,
    signals_layer::Signals,
    types::{Compass, Coord, Dir},
    world::World,
};

fn make_test_agent(id: AgentId, loc: Coord, cfg: &SimConfig, rng: &mut Rng) -> Agent {
    let genome = make_random_genome(cfg, rng);
    let wcfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: cfg.max_number_neurons };
    let nnet = create_wiring(&genome, wcfg);
    Agent::new(id, loc, genome, nnet)
}

/// Helper that builds an ActionContext with raw-pointer split borrows so that
/// `world` (which holds &Signals/&Population) can coexist with the &mut Signals
/// and &mut Agent inside the context. This mirrors how sim_step.rs does it.
///
/// SAFETY: callers must not mutate the same underlying agent or signals through
/// the world reference inside the closure they pass.
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

    let grid_ptr: *const Grid = grid;
    let signals_const_ptr: *const Signals = signals;
    let signals_mut_ptr: *mut Signals = signals;
    let pop_ptr: *const Population = population;
    let agent_ptr: *mut Agent = population.get_mut(agent_id).expect("agent exists");

    // SAFETY: see comment above. We construct one immutable view of signals/population
    // (via World) and one mutable view of signals/agent (via ActionContext). The action
    // implementations only read from world.population and only mutate via the queues
    // and ctx.agent / ctx.signals — they don't write back through the world reference.
    let world = World {
        grid: unsafe { &*grid_ptr },
        signals: unsafe { &*signals_const_ptr },
        population: unsafe { &*pop_ptr },
        size_x: cfg.size_x,
        size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0, step: 0,
    };

    let mut ctx = ActionContext {
        agent: unsafe { &mut *agent_ptr },
        world: &world,
        move_queue: &mut move_q,
        death_queue: &mut death_q,
        signals: unsafe { &mut *signals_mut_ptr },
        rng,
        config_kill_enable: false,
    };

    f(&mut ctx)
}

#[test]
fn registry_has_all_17_builtin_actions() {
    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    assert_eq!(reg.count(), 17, "expected 17 built-in actions");
}

#[test]
fn registry_has_known_action_ids() {
    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let ids: Vec<&str> = (0..reg.count()).map(|i| reg.id(i)).collect();
    for required in [
        "move_x", "move_y", "move_forward", "move_reverse",
        "move_north", "move_south", "move_east", "move_west",
        "move_left", "move_right", "move_random", "move_rl",
        "set_responsiveness", "set_oscillator_period", "set_longprobe_dist",
        "emit_signal0", "kill_forward",
    ] {
        assert!(ids.contains(&required), "missing action id: {required}");
    }
}

#[test]
fn move_east_with_high_level_queues_move_to_x_plus_1() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(2);
    let mut rng = Rng::seeded(11);

    let agent = make_test_agent(population.next_id(), Coord::new(5, 5), &cfg, &mut rng);
    let id = population.spawn(agent);
    grid.set(Coord::new(5, 5), id);

    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let east_idx = (0..reg.count()).find(|&i| reg.id(i) == "move_east").unwrap();

    let mut arng = Rng::seeded(99);
    let queued: Vec<(AgentId, Coord)> = with_action_ctx(
        &cfg, &grid, &mut signals, &mut population, id, &mut arng,
        |ctx| {
            for _ in 0..20 {
                reg.execute(east_idx as u16, 5.0, ctx);
            }
            ctx.move_queue.clone()
        },
    );

    assert!(!queued.is_empty(), "move_east with high level should queue at least one move");
    for (queued_id, target) in &queued {
        assert_eq!(*queued_id, id, "queued move_id should match agent");
        assert_eq!(target.x, 6, "east move should be x+1 from 5");
        assert_eq!(target.y, 5, "east move should preserve y");
    }
}

#[test]
fn move_west_queues_move_to_x_minus_1() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(1);
    let mut rng = Rng::seeded(11);

    let agent = make_test_agent(population.next_id(), Coord::new(5, 5), &cfg, &mut rng);
    let id = population.spawn(agent);
    grid.set(Coord::new(5, 5), id);

    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let idx = (0..reg.count()).find(|&i| reg.id(i) == "move_west").unwrap();
    let mut arng = Rng::seeded(123);
    let queued: Vec<(AgentId, Coord)> = with_action_ctx(
        &cfg, &grid, &mut signals, &mut population, id, &mut arng,
        |ctx| { for _ in 0..20 { reg.execute(idx as u16, 5.0, ctx); } ctx.move_queue.clone() },
    );
    for (_, t) in &queued {
        assert_eq!(t.x, 4, "west move should be x-1");
        assert_eq!(t.y, 5);
    }
}

#[test]
fn move_north_queues_move_to_y_plus_1() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(1);
    let mut rng = Rng::seeded(11);

    let agent = make_test_agent(population.next_id(), Coord::new(5, 5), &cfg, &mut rng);
    let id = population.spawn(agent);
    grid.set(Coord::new(5, 5), id);

    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let idx = (0..reg.count()).find(|&i| reg.id(i) == "move_north").unwrap();
    let mut arng = Rng::seeded(123);
    let queued: Vec<(AgentId, Coord)> = with_action_ctx(
        &cfg, &grid, &mut signals, &mut population, id, &mut arng,
        |ctx| { for _ in 0..20 { reg.execute(idx as u16, 5.0, ctx); } ctx.move_queue.clone() },
    );
    for (_, t) in &queued {
        assert_eq!(t.x, 5);
        assert_eq!(t.y, 6, "north move should be y+1");
    }
}

#[test]
fn move_blocked_by_grid_boundary_does_not_queue() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(1);
    let mut rng = Rng::seeded(12);

    let agent = make_test_agent(population.next_id(), Coord::new(15, 5), &cfg, &mut rng);
    let id = population.spawn(agent);
    grid.set(Coord::new(15, 5), id);

    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let east_idx = (0..reg.count()).find(|&i| reg.id(i) == "move_east").unwrap();
    let mut arng = Rng::seeded(0);
    let queued: Vec<(AgentId, Coord)> = with_action_ctx(
        &cfg, &grid, &mut signals, &mut population, id, &mut arng,
        |ctx| { for _ in 0..50 { reg.execute(east_idx as u16, 10.0, ctx); } ctx.move_queue.clone() },
    );
    assert!(queued.is_empty(), "move_east at right edge must not queue any moves");
}

#[test]
fn move_blocked_by_occupied_cell_does_not_queue() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(2);
    let mut rng = Rng::seeded(13);

    let agent_a = make_test_agent(population.next_id(), Coord::new(5, 5), &cfg, &mut rng);
    let id_a = population.spawn(agent_a);
    grid.set(Coord::new(5, 5), id_a);
    let agent_b = make_test_agent(population.next_id(), Coord::new(6, 5), &cfg, &mut rng);
    let id_b = population.spawn(agent_b);
    grid.set(Coord::new(6, 5), id_b);

    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let east_idx = (0..reg.count()).find(|&i| reg.id(i) == "move_east").unwrap();
    let mut arng = Rng::seeded(0);
    let queued: Vec<(AgentId, Coord)> = with_action_ctx(
        &cfg, &grid, &mut signals, &mut population, id_a, &mut arng,
        |ctx| { for _ in 0..50 { reg.execute(east_idx as u16, 10.0, ctx); } ctx.move_queue.clone() },
    );
    // try_move() filters by is_empty_at — should not queue
    assert!(queued.is_empty(), "move into occupied cell must not queue, got {:?}", queued);
}

#[test]
fn set_responsiveness_clamps_to_unit_interval() {
    let cfg = SimConfig::default();
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(1);
    let mut rng = Rng::seeded(13);
    let agent = make_test_agent(population.next_id(), Coord::new(5, 5), &cfg, &mut rng);
    let id = population.spawn(agent);

    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let idx = (0..reg.count()).find(|&i| reg.id(i) == "set_responsiveness").unwrap();

    let mut arng = Rng::seeded(0);
    let (high, low) = with_action_ctx(
        &cfg, &grid, &mut signals, &mut population, id, &mut arng,
        |ctx| {
            reg.execute(idx as u16, 100.0, ctx);
            let h = ctx.agent.responsiveness;
            reg.execute(idx as u16, -100.0, ctx);
            let l = ctx.agent.responsiveness;
            (h, l)
        },
    );

    assert!((high - 1.0).abs() < 1e-3, "huge level should set responsiveness ≈ 1.0, got {}", high);
    assert!(low.abs() < 1e-3, "very negative level should set responsiveness ≈ 0.0, got {}", low);
}

#[test]
fn set_oscillator_period_produces_positive_period() {
    let cfg = SimConfig::default();
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(1);
    let mut rng = Rng::seeded(0);
    let agent = make_test_agent(population.next_id(), Coord::new(5, 5), &cfg, &mut rng);
    let id = population.spawn(agent);

    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let idx = (0..reg.count()).find(|&i| reg.id(i) == "set_oscillator_period").unwrap();

    let mut arng = Rng::seeded(0);
    with_action_ctx(&cfg, &grid, &mut signals, &mut population, id, &mut arng, |ctx| {
        for &lvl in &[-10.0, -1.0, 0.0, 1.0, 10.0] {
            reg.execute(idx as u16, lvl, ctx);
            assert!(ctx.agent.osc_period > 0, "osc_period must be positive after level {}", lvl);
        }
    });
}

#[test]
fn prob2bool_zero_level_is_almost_never_true() {
    let mut rng = Rng::seeded(0);
    let n = 1000;
    let true_count = (0..n).filter(|_| prob2bool(0.0, &mut rng)).count();
    assert!(true_count < 50, "p2b(0) firing too often: {}/{}", true_count, n);
}

#[test]
fn prob2bool_high_level_is_almost_always_true() {
    let mut rng = Rng::seeded(0);
    let n = 1000;
    let true_count = (0..n).filter(|_| prob2bool(10.0, &mut rng)).count();
    assert!(true_count > 950, "p2b(10) firing too rarely: {}/{}", true_count, n);
}

#[test]
fn response_curve_clamps_to_unit_interval() {
    for r in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0_f32] {
        let v = response_curve(r, 2.0);
        assert!(v.is_finite() && (0.0..=1.0).contains(&v),
                "response_curve({}, 2.0) = {} not in [0,1]", r, v);
    }
}

// ── KillForward integration ───────────────────────────────────────────────

/// Variant of `with_action_ctx` that sets `config_kill_enable = true`.
/// All other behaviour is identical; see the original helper for safety notes.
fn with_kill_ctx<R>(
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

    let grid_ptr: *const Grid = grid;
    let signals_const_ptr: *const Signals = signals;
    let signals_mut_ptr: *mut Signals = signals;
    let pop_ptr: *const Population = population;
    let agent_ptr: *mut Agent = population.get_mut(agent_id).expect("agent exists");

    let world = World {
        grid: unsafe { &*grid_ptr },
        signals: unsafe { &*signals_const_ptr },
        population: unsafe { &*pop_ptr },
        size_x: cfg.size_x,
        size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0, step: 0,
    };

    let mut ctx = ActionContext {
        agent: unsafe { &mut *agent_ptr },
        world: &world,
        move_queue: &mut move_q,
        death_queue: &mut death_q,
        signals: unsafe { &mut *signals_mut_ptr },
        rng,
        config_kill_enable: true,  // ← enabled
    };

    f(&mut ctx)
}

/// KillForward with `config_kill_enable = false` (the default) must never
/// queue a death regardless of level or adjacency.
#[test]
fn kill_forward_disabled_never_queues_death() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(2);
    let mut rng = Rng::seeded(30);

    // Killer at (5,5) facing east
    let killer = make_test_agent(population.next_id(), Coord::new(5, 5), &cfg, &mut rng);
    let id_killer = population.spawn(killer);
    grid.set(Coord::new(5, 5), id_killer);
    population.get_mut(id_killer).unwrap().last_move_dir = Dir(Compass::E);

    // Victim at (6,5) — directly east
    let victim = make_test_agent(population.next_id(), Coord::new(6, 5), &cfg, &mut rng);
    let id_victim = population.spawn(victim);
    grid.set(Coord::new(6, 5), id_victim);

    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let kill_idx = (0..reg.count()).find(|&i| reg.id(i) == "kill_forward").unwrap();

    // with_action_ctx has config_kill_enable = false
    let mut arng = Rng::seeded(0);
    let death_q = with_action_ctx(
        &cfg, &grid, &mut signals, &mut population, id_killer, &mut arng,
        |ctx| {
            for _ in 0..30 { reg.execute(kill_idx as u16, 10.0, ctx); }
            ctx.death_queue.clone()
        },
    );
    assert!(death_q.is_empty(),
        "kill_forward with kill_enable=false must not queue any deaths");
    assert!(population.get(id_victim).unwrap().alive,
        "victim must still be alive when kill is disabled");
}

/// KillForward with `config_kill_enable = true` and a deterministically high
/// activation level must queue the adjacent victim's ID in the death queue.
#[test]
fn kill_forward_enabled_queues_victim_when_adjacent() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(2);
    let mut rng = Rng::seeded(31);

    // Killer at (7,7) facing east
    let killer = make_test_agent(population.next_id(), Coord::new(7, 7), &cfg, &mut rng);
    let id_killer = population.spawn(killer);
    grid.set(Coord::new(7, 7), id_killer);
    population.get_mut(id_killer).unwrap().last_move_dir = Dir(Compass::E);

    // Victim at (8,7) — one step east of killer
    let victim = make_test_agent(population.next_id(), Coord::new(8, 7), &cfg, &mut rng);
    let id_victim = population.spawn(victim);
    grid.set(Coord::new(8, 7), id_victim);

    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let kill_idx = (0..reg.count()).find(|&i| reg.id(i) == "kill_forward").unwrap();

    // Execute once with a very high level so prob2bool fires deterministically
    let mut arng = Rng::seeded(0);
    let death_q = with_kill_ctx(
        &cfg, &grid, &mut signals, &mut population, id_killer, &mut arng,
        |ctx| {
            reg.execute(kill_idx as u16, 10.0, ctx);
            ctx.death_queue.clone()
        },
    );

    assert!(
        death_q.contains(&id_victim),
        "kill_forward should queue victim (id={}) for death, got {:?}", id_victim, death_q
    );
    // The killer itself must not appear in the death queue
    assert!(
        !death_q.contains(&id_killer),
        "killer must not appear in its own death queue"
    );
}

/// After calling `drain_death_queue`, the victim agent is marked dead and its
/// grid cell is cleared to empty.
#[test]
fn kill_forward_drain_marks_victim_dead_and_clears_grid() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(2);
    let mut rng = Rng::seeded(32);

    let victim_loc = Coord::new(10, 8);
    let killer_loc = Coord::new(9, 8);

    let killer = make_test_agent(population.next_id(), killer_loc, &cfg, &mut rng);
    let id_killer = population.spawn(killer);
    grid.set(killer_loc, id_killer);
    population.get_mut(id_killer).unwrap().last_move_dir = Dir(Compass::E);

    let victim = make_test_agent(population.next_id(), victim_loc, &cfg, &mut rng);
    let id_victim = population.spawn(victim);
    grid.set(victim_loc, id_victim);

    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let kill_idx = (0..reg.count()).find(|&i| reg.id(i) == "kill_forward").unwrap();

    // Collect the queued deaths
    let mut arng = Rng::seeded(0);
    let death_q = with_kill_ctx(
        &cfg, &grid, &mut signals, &mut population, id_killer, &mut arng,
        |ctx| {
            reg.execute(kill_idx as u16, 10.0, ctx);
            ctx.death_queue.clone()
        },
    );
    assert!(death_q.contains(&id_victim), "pre-condition: victim must be queued");

    // Now wire the queue into population and drain
    population.death_queue.extend(death_q);
    population.drain_death_queue(&mut grid);

    // Post-drain invariants
    let victim_agent = population.get(id_victim).expect("slot must still exist");
    assert!(!victim_agent.alive, "victim.alive must be false after drain");
    assert!(grid.is_empty_at(victim_loc),
        "grid cell at victim location must be cleared after drain");
    // Killer must be unaffected
    assert!(population.get(id_killer).unwrap().alive, "killer should remain alive");
    assert!(grid.is_occupied_at(killer_loc), "killer cell should remain occupied");
}

/// KillForward must do nothing when the cell directly in front is empty
/// (no victim to kill), even with kill enabled and a high activation level.
#[test]
fn kill_forward_no_victim_in_front_does_not_crash() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut population = Population::new(1);
    let mut rng = Rng::seeded(33);

    // Lone agent at (5,5) facing east — nothing east of it
    let agent = make_test_agent(population.next_id(), Coord::new(5, 5), &cfg, &mut rng);
    let id = population.spawn(agent);
    grid.set(Coord::new(5, 5), id);
    population.get_mut(id).unwrap().last_move_dir = Dir(Compass::E);

    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let kill_idx = (0..reg.count()).find(|&i| reg.id(i) == "kill_forward").unwrap();

    let mut arng = Rng::seeded(0);
    let death_q = with_kill_ctx(
        &cfg, &grid, &mut signals, &mut population, id, &mut arng,
        |ctx| {
            for _ in 0..20 { reg.execute(kill_idx as u16, 10.0, ctx); }
            ctx.death_queue.clone()
        },
    );
    assert!(death_q.is_empty(),
        "kill_forward with empty forward cell must produce no deaths, got {:?}", death_q);
}
