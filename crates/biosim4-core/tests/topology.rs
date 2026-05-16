//! Integration tests for the world-topology setting.
//!
//! Topology unit tests live next to the type in
//! `biosim4-core/src/topology.rs`. This file covers the *consumers*:
//! Grid query helpers, movement on a torus, sensor probes that walk
//! through the wrap seam, and the distance-aware challenges.

use biosim4_challenges::register_builtin_challenges;
use biosim4_core::{
    agent::{Agent, AgentId},
    food_layer::FoodLayer,
    genome::neural_net::{create_wiring, WiringConfig},
    genome::ops::make_random_genome,
    grid::{Grid, BARRIER},
    population::Population,
    programmable::ProgrammablePool,
    registry::{ActionContext, ActionRegistry, ChallengeRegistry, SensorRegistry},
    rng::Rng,
    signals_layer::Signals,
    sim_config::SimConfig,
    topology::Topology,
    types::{Compass, Coord, Dir},
    world::World,
};
use biosim4_sensors::register_builtin_sensors;

fn make_agent(id: AgentId, loc: Coord, cfg: &SimConfig, rng: &mut Rng) -> Agent {
    let g = make_random_genome(cfg, rng);
    let w =
        WiringConfig { sensor_count: 40, action_count: 23, max_neurons: cfg.max_number_neurons };
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

// ── Grid helpers exposed through the topology ───────────────────────────

#[test]
fn grid_wrap_is_identity_on_plane_and_wraps_on_torus() {
    let plane = Grid::new(10, 10);
    assert_eq!(plane.wrap(Coord::new(5, 5)), Some(Coord::new(5, 5)));
    assert_eq!(plane.wrap(Coord::new(-1, 5)), None);
    assert_eq!(plane.wrap(Coord::new(5, 10)), None);

    let torus_x = Grid::with_topology(10, 10, Topology::TorusX);
    assert_eq!(torus_x.wrap(Coord::new(-1, 5)), Some(Coord::new(9, 5)));
    assert_eq!(torus_x.wrap(Coord::new(10, 5)), Some(Coord::new(0, 5)));
    // Y axis still bounded.
    assert_eq!(torus_x.wrap(Coord::new(5, -1)), None);

    let sphere = Grid::with_topology(10, 10, Topology::Sphere);
    assert_eq!(sphere.wrap(Coord::new(15, 27)), Some(Coord::new(5, 7)));
}

#[test]
fn grid_is_border_respects_topology() {
    let plane = Grid::new(10, 10);
    assert!(plane.is_border(Coord::new(0, 5)));
    assert!(plane.is_border(Coord::new(5, 9)));
    assert!(!plane.is_border(Coord::new(5, 5)));

    // TorusX wraps E/W → only N/S edges remain borders.
    let torus_x = Grid::with_topology(10, 10, Topology::TorusX);
    assert!(!torus_x.is_border(Coord::new(0, 5)));
    assert!(!torus_x.is_border(Coord::new(9, 5)));
    assert!(torus_x.is_border(Coord::new(5, 0)));

    // Sphere — no cell is a border.
    let sphere = Grid::with_topology(10, 10, Topology::Sphere);
    for x in 0..10 {
        for y in 0..10 {
            assert!(!sphere.is_border(Coord::new(x, y)));
        }
    }
}

#[test]
fn grid_distance_takes_the_short_wrap_path() {
    let torus_x = Grid::with_topology(10, 10, Topology::TorusX);
    // (1,5) <-> (9,5): direct = 8, wrap = 2. Wrap wins.
    assert_eq!(torus_x.dist_sq(Coord::new(1, 5), Coord::new(9, 5)), 4);
    assert_eq!(torus_x.chebyshev_dist(Coord::new(1, 5), Coord::new(9, 5)), 2);

    // On the plane the raw distance stands.
    let plane = Grid::new(10, 10);
    assert_eq!(plane.dist_sq(Coord::new(1, 5), Coord::new(9, 5)), 64);
}

#[test]
fn grid_norm_dist_helper_wraps() {
    let sphere = Grid::with_topology(10, 10, Topology::Sphere);
    // Agent at (0,0), target at normalized (0.95, 0.95). Naive distance
    // would be ~0.95*sqrt(2); wrap distance is short (~0.07*sqrt(2)).
    let d_torus = sphere.norm_dist_to_norm_point(Coord::new(0, 0), 0.95, 0.95);
    let plane = Grid::new(10, 10);
    let d_plane = plane.norm_dist_to_norm_point(Coord::new(0, 0), 0.95, 0.95);
    assert!(d_torus < d_plane, "torus distance ({d_torus}) must be smaller than plane ({d_plane})");
}

// ── Movement ────────────────────────────────────────────────────────────

fn run_one_step_east(cfg: &SimConfig, grid: &Grid, pop: &mut Population, id: AgentId) -> Coord {
    use biosim4_actions::register_builtin_actions;
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let programmable = ProgrammablePool::new();
    let mut reg = ActionRegistry::new();
    register_builtin_actions(&mut reg);
    let east_idx = (0..reg.count()).find(|&i| reg.id(i) == "move_east").unwrap();

    let mut move_q: Vec<(AgentId, Coord)> = Vec::new();
    let mut death_q: Vec<AgentId> = Vec::new();
    let grid_ptr: *const Grid = grid;
    let signals_const_ptr: *const Signals = &signals;
    let signals_mut_ptr: *mut Signals = &mut signals;
    let pop_ptr: *const Population = pop;
    let agent_ptr: *mut Agent = pop.get_mut(id).expect("agent exists");
    {
        let w = World {
            grid: unsafe { &*grid_ptr },
            signals: unsafe { &*signals_const_ptr },
            food: &food,
            population: unsafe { &*pop_ptr },
            programmable: &programmable,
            size_x: cfg.size_x,
            size_y: cfg.size_y,
            steps_per_generation: cfg.steps_per_generation,
            generation: 0,
            step: 0,
        };
        let mut arng = Rng::seeded(0);
        let agent_ref = unsafe { &mut *agent_ptr };
        agent_ref.responsiveness = 1.0;
        let mut ctx = ActionContext {
            agent: agent_ref,
            world: &w,
            move_queue: &mut move_q,
            death_queue: &mut death_q,
            signals: unsafe { &mut *signals_mut_ptr },
            rng: &mut arng,
            config_kill_enable: false,
            responsiveness_adjusted: biosim4_core::registry::action::response_curve(
                unsafe { &*agent_ptr }.responsiveness,
                cfg.responsiveness_curve_k_factor,
            ),
            move_x_urge: 0.0,
            move_y_urge: 0.0,
        };
        reg.execute(east_idx, 10.0, &mut ctx);
        biosim4_core::registry::action::resolve_movement(&mut ctx);
    }
    let moves = move_q.clone();
    // Use drain_move_queue_from with an owned grid.
    let mut grid_owned = Grid::with_topology(cfg.size_x, cfg.size_y, grid.topology);
    // Copy cells (not exposed) by reading every cell — for these tests the
    // grid is empty save for the agent, so just set the agent's cell.
    grid_owned.set(pop.get(id).unwrap().loc, id);
    pop.drain_move_queue_from(&mut grid_owned, moves);
    pop.get(id).unwrap().loc
}

#[test]
fn move_east_off_east_edge_wraps_on_torus_x() {
    let cfg =
        SimConfig { size_x: 16, size_y: 16, topology: Topology::TorusX, ..SimConfig::default() };
    let grid = Grid::with_topology(cfg.size_x, cfg.size_y, cfg.topology);
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(11);
    let id = pop.spawn(make_agent(pop.next_id(), Coord::new(15, 5), &cfg, &mut rng));

    let new_loc = run_one_step_east(&cfg, &grid, &mut pop, id);
    assert_eq!(new_loc, Coord::new(0, 5), "east step from x=15 must wrap to x=0 on TorusX");
}

#[test]
fn move_east_off_east_edge_is_blocked_on_plane() {
    let cfg =
        SimConfig { size_x: 16, size_y: 16, topology: Topology::Plane, ..SimConfig::default() };
    let grid = Grid::with_topology(cfg.size_x, cfg.size_y, cfg.topology);
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(11);
    let id = pop.spawn(make_agent(pop.next_id(), Coord::new(15, 5), &cfg, &mut rng));

    let new_loc = run_one_step_east(&cfg, &grid, &mut pop, id);
    assert_eq!(new_loc, Coord::new(15, 5), "east step from x=15 must be blocked on Plane");
}

// ── Sensors ─────────────────────────────────────────────────────────────

#[test]
fn longprobe_population_fwd_sees_through_wrap() {
    use biosim4_core::registry::SensorContext;

    let cfg =
        SimConfig { size_x: 16, size_y: 16, topology: Topology::TorusX, ..SimConfig::default() };
    let mut grid = Grid::with_topology(cfg.size_x, cfg.size_y, cfg.topology);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let programmable = ProgrammablePool::new();
    let mut pop = Population::new(2);
    let mut rng = Rng::seeded(0xCAFE);

    // Probe agent at the east edge (x=15) facing east; target two cells
    // west of x=0 across the wrap — i.e. x=1.
    let probe_id = pop.spawn(make_agent(pop.next_id(), Coord::new(15, 5), &cfg, &mut rng));
    pop.get_mut(probe_id).unwrap().last_move_dir = Dir(Compass::E);
    pop.get_mut(probe_id).unwrap().long_probe_dist = 8;
    let target_id = pop.spawn(make_agent(pop.next_id(), Coord::new(1, 5), &cfg, &mut rng));
    grid.set(Coord::new(15, 5), probe_id);
    grid.set(Coord::new(1, 5), target_id);

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let lp_idx = (0..reg.count()).find(|&i| reg.id(i) == "longprobe_pop_fwd").unwrap();

    let w = world(&grid, &signals, &food, &pop, &programmable, &cfg);
    let mut srng = Rng::seeded(0);
    let agent_ref = pop.get(probe_id).unwrap();
    let reading = reg.evaluate(
        lp_idx,
        &mut SensorContext { agent: agent_ref, world: &w, sim_step: 0, rng: &mut srng },
    );

    // Target is 2 cells away through wrap (x=15 → x=0 → x=1 is 2 steps east).
    // longprobe_pop_fwd returns `count / probe_dist` where count is the
    // number of empty cells walked before hitting the agent. We walked
    // (16,5) which wraps to (0,5) — that's empty — then (1,5) which is
    // the target. So count=1, probe_dist=8, reading = 1/8 = 0.125.
    assert!(
        reading < 0.5,
        "probe should see target through wrap (reading={reading}, expected ~0.125)"
    );
}

#[test]
fn longprobe_pop_fwd_blocked_at_plane_edge() {
    use biosim4_core::registry::SensorContext;

    // Same setup on Plane: probe walks east, immediately hits the edge,
    // returns 1.0 = "nothing in range".
    let cfg =
        SimConfig { size_x: 16, size_y: 16, topology: Topology::Plane, ..SimConfig::default() };
    let mut grid = Grid::with_topology(cfg.size_x, cfg.size_y, cfg.topology);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let programmable = ProgrammablePool::new();
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(0xBEEF);

    let id = pop.spawn(make_agent(pop.next_id(), Coord::new(15, 5), &cfg, &mut rng));
    pop.get_mut(id).unwrap().last_move_dir = Dir(Compass::E);
    pop.get_mut(id).unwrap().long_probe_dist = 8;
    grid.set(Coord::new(15, 5), id);

    let mut reg = SensorRegistry::new();
    register_builtin_sensors(&mut reg);
    let lp_idx = (0..reg.count()).find(|&i| reg.id(i) == "longprobe_pop_fwd").unwrap();

    let w = world(&grid, &signals, &food, &pop, &programmable, &cfg);
    let mut srng = Rng::seeded(0);
    let reading = reg.evaluate(
        lp_idx,
        &mut SensorContext { agent: pop.get(id).unwrap(), world: &w, sim_step: 0, rng: &mut srng },
    );
    assert!((reading - 1.0).abs() < 1e-6, "probe should saturate at edge on Plane (got {reading})");
}

// ── Challenges ──────────────────────────────────────────────────────────

#[test]
fn against_any_wall_loses_outer_edge_on_torus_x() {
    let cfg =
        SimConfig { size_x: 32, size_y: 32, topology: Topology::TorusX, ..SimConfig::default() };
    let grid = Grid::with_topology(cfg.size_x, cfg.size_y, cfg.topology);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let programmable = ProgrammablePool::new();
    let mut pop = Population::new(2);
    let mut rng = Rng::seeded(7);
    // East edge: no longer a border under TorusX.
    let east = pop.spawn(make_agent(pop.next_id(), Coord::new(31, 10), &cfg, &mut rng));
    // North edge: still a border (TorusX wraps X only).
    let north = pop.spawn(make_agent(pop.next_id(), Coord::new(10, 31), &cfg, &mut rng));

    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    reg.set_single("against_any_wall", None).unwrap();
    let w = world(&grid, &signals, &food, &pop, &programmable, &cfg);

    let (east_pass, _) = reg.evaluate(pop.get(east).unwrap(), &w);
    let (north_pass, _) = reg.evaluate(pop.get(north).unwrap(), &w);
    assert!(!east_pass, "east-edge agent should NOT pass on TorusX (axis wraps)");
    assert!(north_pass, "north-edge agent should still pass on TorusX (Y axis bounded)");
}

#[test]
fn lethal_borders_spares_wrapped_edges() {
    use biosim4_core::registry::challenge::WorldMut;
    let cfg =
        SimConfig { size_x: 16, size_y: 16, topology: Topology::Sphere, ..SimConfig::default() };
    let mut grid = Grid::with_topology(cfg.size_x, cfg.size_y, cfg.topology);
    let mut signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let mut programmable = ProgrammablePool::new();
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(0xACE);
    // On Sphere every "edge" cell still exists, but `is_border` returns
    // false for all of them — lethal_borders shouldn't fire.
    let id = pop.spawn(make_agent(pop.next_id(), Coord::new(0, 0), &cfg, &mut rng));

    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    reg.set_single("lethal_borders", Some(serde_json::json!({ "grace_steps": 0 }))).unwrap();

    let mut ctx_rng = Rng::seeded(0xB0DE);
    let mut ctx = WorldMut {
        grid: &mut grid,
        signals: &mut signals,
        population: &mut pop,
        programmable: &mut programmable,
        rng: &mut ctx_rng,
        step: 0,
        generation: 0,
        config: &cfg,
    };
    reg.on_sim_step(&mut ctx);
    assert!(
        !pop.death_queue.contains(&id),
        "Sphere has no borders — lethal_borders must not kill the corner peep"
    );
}

#[test]
fn near_barrier_sees_barrier_across_wrap_seam() {
    let cfg =
        SimConfig { size_x: 32, size_y: 32, topology: Topology::TorusX, ..SimConfig::default() };
    let mut grid = Grid::with_topology(cfg.size_x, cfg.size_y, cfg.topology);
    grid.set(Coord::new(0, 16), BARRIER);
    grid.barrier_centers.push(Coord::new(0, 16));

    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let programmable = ProgrammablePool::new();
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(0xBA1);
    // Agent at the east edge, two cells away through the wrap from the
    // barrier at x=0.
    pop.spawn(make_agent(pop.next_id(), Coord::new(31, 16), &cfg, &mut rng));

    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    // A tight radius (1.5 cells in normalized units = 1.5/32 ≈ 0.047) so
    // only the wrap-path distance (~1 cell) puts the agent inside.
    reg.set_single("near_barrier", Some(serde_json::json!({ "radius": 0.05 }))).unwrap();

    let w = world(&grid, &signals, &food, &pop, &programmable, &cfg);
    let (pass, _) = reg.evaluate(pop.get(1).unwrap(), &w);
    assert!(pass, "agent should read barrier across the wrap seam");
}
