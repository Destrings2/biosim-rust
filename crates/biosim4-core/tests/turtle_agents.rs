//! Turtle-inspired agent contract: persistent heading updates only on movement,
//! color is deterministic from genome, property bag round-trips. This is the
//! third architectural pillar from the plan.

use biosim4_core::{
    agent::{Agent, AgentId, PropValue},
    grid::Grid,
    genome::neural_net::{create_wiring, WiringConfig},
    genome::genome::make_random_genome,
    population::Population,
    rng::Rng,
    sim_config::SimConfig,
    types::{Coord, Compass, Dir},
};

fn make_agent(id: AgentId, loc: Coord, cfg: &SimConfig, rng: &mut Rng) -> Agent {
    let g = make_random_genome(cfg, rng);
    let w = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: cfg.max_number_neurons };
    let n = create_wiring(&g, w);
    Agent::new(id, loc, g, n)
}

#[test]
fn drain_move_queue_updates_both_heading_and_last_move_dir() {
    // The plan says heading is persistent (turtle-style) and updated on successful move,
    // while last_move_dir is the C++-compat field updated only on movement.
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(0);

    let agent = make_agent(pop.next_id(), Coord::new(5, 5), &cfg, &mut rng);
    let id = pop.spawn(agent);
    grid.set(Coord::new(5, 5), id);

    // Queue a move east
    pop.queue_for_move(id, Coord::new(6, 5));
    pop.drain_move_queue(&mut grid);

    let a = pop.get(id).unwrap();
    assert_eq!(a.loc, Coord::new(6, 5), "agent should have moved");
    assert_eq!(a.last_move_dir.0, Compass::E, "last_move_dir should be E");
    assert_eq!(a.heading.0, Compass::E, "heading should also be E (persistent)");
}

#[test]
fn move_blocked_by_occupied_cell_does_not_update_state() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut pop = Population::new(2);
    let mut rng = Rng::seeded(0);

    let a = make_agent(pop.next_id(), Coord::new(5, 5), &cfg, &mut rng);
    let id_a = pop.spawn(a);
    grid.set(Coord::new(5, 5), id_a);

    // Block the east neighbor
    let b = make_agent(pop.next_id(), Coord::new(6, 5), &cfg, &mut rng);
    let id_b = pop.spawn(b);
    grid.set(Coord::new(6, 5), id_b);

    let original_heading = pop.get(id_a).unwrap().heading;

    pop.queue_for_move(id_a, Coord::new(6, 5));
    pop.drain_move_queue(&mut grid);

    let a_after = pop.get(id_a).unwrap();
    assert_eq!(a_after.loc, Coord::new(5, 5), "blocked move must not change loc");
    assert_eq!(a_after.heading, original_heading, "blocked move must not change heading");
}

#[test]
fn drain_death_queue_marks_agent_dead_and_clears_grid() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(0);

    let agent = make_agent(pop.next_id(), Coord::new(7, 7), &cfg, &mut rng);
    let id = pop.spawn(agent);
    grid.set(Coord::new(7, 7), id);
    assert_eq!(pop.alive_count(), 1);

    pop.queue_for_death(id);
    pop.drain_death_queue(&mut grid);

    let a = pop.get(id).unwrap();
    assert!(!a.alive, "agent should be marked dead");
    assert_eq!(pop.alive_count(), 0, "alive_count should drop to 0");
    assert!(grid.is_empty_at(Coord::new(7, 7)), "grid cell should be cleared");
}

#[test]
fn double_queue_for_death_does_not_double_kill() {
    let cfg = SimConfig::default();
    let mut grid = Grid::new(cfg.size_x, cfg.size_y);
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(0);
    let a = make_agent(pop.next_id(), Coord::new(3, 3), &cfg, &mut rng);
    let id = pop.spawn(a);
    grid.set(Coord::new(3, 3), id);

    pop.queue_for_death(id);
    pop.queue_for_death(id);  // duplicate
    pop.drain_death_queue(&mut grid);

    assert_eq!(pop.alive_count(), 0);
    // Should not panic from double-removal
}

#[test]
fn genome_color_is_deterministic() {
    // Same genome → same color, every time. Required for stable visualization.
    let cfg = SimConfig::default();
    let mut rng = Rng::seeded(123);
    let g = make_random_genome(&cfg, &mut rng);

    let w = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: cfg.max_number_neurons };
    let n1 = create_wiring(&g, w);
    let n2 = create_wiring(&g, w);

    let a1 = Agent::new(1, Coord::new(0, 0), g.clone(), n1);
    let a2 = Agent::new(2, Coord::new(5, 5), g, n2);
    assert_eq!(a1.color, a2.color, "same genome must produce same color");
}

#[test]
fn genome_color_is_not_too_dark() {
    // Plan says minimum brightness is enforced to keep agents visible on dark backgrounds.
    let cfg = SimConfig::default();
    let mut rng = Rng::seeded(0);
    for _ in 0..50 {
        let g = make_random_genome(&cfg, &mut rng);
        let w = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: cfg.max_number_neurons };
        let n = create_wiring(&g, w);
        let a = Agent::new(1, Coord::new(0, 0), g, n);
        let lum = (a.color[0] as u16 + a.color[1] as u16 + a.color[2] as u16) / 3;
        assert!(lum >= 60, "agent color too dim: {:?} (lum={})", a.color, lum);
    }
}

#[test]
fn agent_property_bag_roundtrips() {
    let cfg = SimConfig::default();
    let mut rng = Rng::seeded(0);
    let g = make_random_genome(&cfg, &mut rng);
    let w = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: cfg.max_number_neurons };
    let n = create_wiring(&g, w);
    let mut a = Agent::new(1, Coord::new(0, 0), g, n);

    a.set_prop("hunger", PropValue::F32(0.7));
    a.set_prop("alive_steps", PropValue::I32(42));
    a.set_prop("is_leader", PropValue::Bool(true));
    a.set_prop("breed_name", PropValue::Str("alpha".to_string()));

    assert_eq!(a.get_prop("hunger"), Some(&PropValue::F32(0.7)));
    assert_eq!(a.get_prop("alive_steps"), Some(&PropValue::I32(42)));
    assert_eq!(a.get_prop("is_leader"), Some(&PropValue::Bool(true)));
    assert_eq!(a.get_prop("breed_name"), Some(&PropValue::Str("alpha".to_string())));
    assert_eq!(a.get_prop("nonexistent"), None);
}

#[test]
fn agent_starts_with_default_state() {
    let cfg = SimConfig::default();
    let mut rng = Rng::seeded(0);
    let g = make_random_genome(&cfg, &mut rng);
    let w = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: cfg.max_number_neurons };
    let n = create_wiring(&g, w);
    let a = Agent::new(42, Coord::new(3, 4), g, n);

    assert_eq!(a.id, 42);
    assert!(a.alive);
    assert_eq!(a.age, 0);
    assert_eq!(a.loc, Coord::new(3, 4));
    assert_eq!(a.birth_loc, Coord::new(3, 4), "birth_loc starts equal to loc");
    assert_eq!(a.challenge_bits, 0);
    assert!(a.props.is_empty());
}
