//! Visual overlay contract for challenges. Every challenge with a known
//! geometric "safe zone" must produce overlays whose pixel coordinates fit
//! inside the world bounds; aggregate `get_overlays()` for the active set
//! must collect contributions from every active challenge.
//!
//! Catches regressions where a challenge silently returns `Vec::new()` or
//! emits geometry off-screen.

use biosim4_challenges::register_builtin_challenges;
use biosim4_core::{
    agent::{Agent, AgentId},
    food_layer::FoodLayer,
    genome::neural_net::{create_wiring, WiringConfig},
    genome::ops::make_random_genome,
    grid::Grid,
    population::Population,
    programmable::ProgrammablePool,
    registry::challenge::ChallengeOverlay,
    registry::{ChallengeConfig, ChallengeRegistry},
    rng::Rng,
    signals_layer::Signals,
    sim_config::SimConfig,
    types::Coord,
    world::World,
};

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

fn make_world(cfg: &SimConfig) -> (Grid, Signals, FoodLayer, Population) {
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(0xBEEF);
    let g = make_random_genome(cfg, &mut rng);
    let w =
        WiringConfig { sensor_count: 40, action_count: 17, max_neurons: cfg.max_number_neurons };
    let n = create_wiring(&g, w);
    pop.spawn(Agent::new(pop.next_id() as AgentId, Coord::new(0, 0), g, n));
    (grid, signals, food, pop)
}

/// Each spatial challenge with a region-based safe zone must emit at least
/// one overlay element. Challenges that depend on world state (barriers,
/// radioactive zones) are tested separately. Challenges that have no
/// geometric region by design (sequential, social, altruism) MUST emit zero
/// overlays — this guards against accidental Default-impl drift.
#[test]
fn region_challenges_emit_overlays_and_regionless_do_not() {
    let cfg = SimConfig { size_x: 128, size_y: 128, ..SimConfig::default() };
    let (grid, signals, food, pop) = make_world(&cfg);
    let programmable = ProgrammablePool::new();
    let w = world(&grid, &signals, &food, &pop, &programmable, &cfg);

    // Region-based challenges with non-empty default overlays.
    let with_overlay = [
        "circle",
        "right_half",
        "right_quarter",
        "left_eighth",
        "east_west_eighths",
        "center_weighted",
        "center_unweighted",
        "corner",
        "corner_weighted",
        "radioactive_walls",
        "altruism",
        "altruism_sacrifice",
        "location_sequence",
        "quarantine",
    ];
    for id in with_overlay {
        let mut reg = ChallengeRegistry::new();
        register_builtin_challenges(&mut reg);
        reg.set_single(id, None).expect("set_single");
        let overlays = reg.get_overlays(&w);
        assert!(
            !overlays.is_empty(),
            "challenge `{id}` should emit at least one overlay (none returned)",
        );
    }

    // Behavioral / non-geometric challenges must NOT emit overlays.
    let expected_no_overlay = [
        "against_any_wall",
        "pairs",
        "center_sparse",
        "string",
        "touch_any_wall",
        "migrate_distance",
        "diaspora",
        // `tag` overlays the location of every "it" agent — empty in the
        // test setup since no agent has been flagged yet.
        "tag",
    ];
    for id in expected_no_overlay {
        let mut reg = ChallengeRegistry::new();
        register_builtin_challenges(&mut reg);
        reg.set_single(id, None).expect("set_single");
        let overlays = reg.get_overlays(&w);
        assert!(
            overlays.is_empty(),
            "challenge `{id}` is expected to have no overlay but returned {} elements",
            overlays.len(),
        );
    }
}

#[test]
fn near_barrier_overlay_appears_only_when_barriers_exist() {
    let cfg = SimConfig { size_x: 64, size_y: 64, ..SimConfig::default() };
    let (grid, signals, food, pop) = make_world(&cfg);
    let programmable = ProgrammablePool::new();
    let w_empty = world(&grid, &signals, &food, &pop, &programmable, &cfg);

    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    reg.set_single("near_barrier", None).expect("set_single");
    assert!(
        reg.get_overlays(&w_empty).is_empty(),
        "near_barrier overlay should be empty when no barriers exist"
    );

    // Place a barrier center and verify a circle appears.
    let mut grid2 = Grid::new(cfg.size_x, cfg.size_y);
    grid2.barrier_centers.push(Coord::new(32, 32));
    let w_with = world(&grid2, &signals, &food, &pop, &programmable, &cfg);
    let overlays = reg.get_overlays(&w_with);
    assert!(
        !overlays.is_empty(),
        "near_barrier overlay should appear once barrier_centers is populated"
    );
    assert!(
        matches!(overlays[0], ChallengeOverlay::Circle { .. }),
        "near_barrier overlay should be a Circle"
    );
}

#[test]
fn sun_tracker_overlay_present_and_inside_world() {
    let cfg = SimConfig { size_x: 64, size_y: 64, ..SimConfig::default() };
    let (grid, signals, food, pop) = make_world(&cfg);
    let programmable = ProgrammablePool::new();
    let w = world(&grid, &signals, &food, &pop, &programmable, &cfg);

    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    reg.set_single("sun_tracker", None).expect("set_single");
    let overlays = reg.get_overlays(&w);
    assert!(!overlays.is_empty(), "sun_tracker should emit at least one overlay");
    if let ChallengeOverlay::Circle { cx, cy, radius, .. } = overlays[0] {
        assert!(radius > 0.0, "sun_tracker radius positive (got {radius})");
        let inside = cx + radius > 0.0
            && cy + radius > 0.0
            && cx - radius < cfg.size_x as f32
            && cy - radius < cfg.size_y as f32;
        assert!(inside, "sun_tracker circle ({cx}, {cy}, r={radius}) must overlap world");
    } else {
        panic!("sun_tracker overlay should be a Circle");
    }
}

#[test]
fn overlay_coordinates_fit_world_bounds() {
    let cfg = SimConfig { size_x: 64, size_y: 64, ..SimConfig::default() };
    let (grid, signals, food, pop) = make_world(&cfg);
    let programmable = ProgrammablePool::new();
    let w = world(&grid, &signals, &food, &pop, &programmable, &cfg);
    let sx = cfg.size_x as f32;
    let sy = cfg.size_y as f32;

    // Each spatial challenge's overlay should at minimum overlap the world.
    for id in [
        "circle",
        "right_half",
        "right_quarter",
        "left_eighth",
        "east_west_eighths",
        "center_weighted",
        "center_unweighted",
        "corner",
        "corner_weighted",
    ] {
        let mut reg = ChallengeRegistry::new();
        register_builtin_challenges(&mut reg);
        reg.set_single(id, None).expect("set_single");
        for ov in reg.get_overlays(&w) {
            match ov {
                ChallengeOverlay::Rectangle { x, y, w: rw, h, .. } => {
                    assert!(
                        rw > 0.0 && h > 0.0,
                        "{id}: rect must have positive size (got {rw}x{h})"
                    );
                    let inside = x + rw > 0.0 && y + h > 0.0 && x < sx && y < sy;
                    assert!(
                        inside,
                        "{id}: rect [{x},{y} {rw}x{h}] does not overlap [0,0,{sx},{sy}]"
                    );
                }
                ChallengeOverlay::Circle { cx, cy, radius, .. } => {
                    assert!(radius > 0.0, "{id}: circle radius must be positive (got {radius})");
                    let inside = cx + radius > 0.0
                        && cy + radius > 0.0
                        && cx - radius < sx
                        && cy - radius < sy;
                    assert!(
                        inside,
                        "{id}: circle (cx={cx}, cy={cy}, r={radius}) does not overlap world"
                    );
                }
                ChallengeOverlay::Points { points, .. } => {
                    assert!(!points.is_empty(), "{id}: empty points overlay");
                }
            }
        }
    }
}

#[test]
fn aggregate_overlays_includes_every_active_contribution() {
    let cfg = SimConfig { size_x: 32, size_y: 32, ..SimConfig::default() };
    let (grid, signals, food, pop) = make_world(&cfg);
    let programmable = ProgrammablePool::new();
    let w = world(&grid, &signals, &food, &pop, &programmable, &cfg);

    // Active = three spatial challenges; each is known to emit ≥1 overlay.
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    let cfg_active = ChallengeConfig {
        active: vec!["circle".into(), "right_half".into(), "corner".into()],
        composition: Default::default(),
        params: Default::default(),
    };
    reg.apply_config(cfg_active).expect("apply_config");

    let overlays = reg.get_overlays(&w);
    assert!(
        overlays.len() >= 3,
        "expected ≥3 overlays for 3 spatial challenges (got {})",
        overlays.len(),
    );
}

#[test]
fn circle_overlay_respects_configured_params() {
    // The circle challenge should produce one circle whose center / radius
    // match the configured normalized values multiplied by world dims.
    let cfg = SimConfig { size_x: 100, size_y: 100, ..SimConfig::default() };
    let (grid, signals, food, pop) = make_world(&cfg);
    let programmable = ProgrammablePool::new();
    let w = world(&grid, &signals, &food, &pop, &programmable, &cfg);

    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    reg.set_single(
        "circle",
        Some(serde_json::json!({
            "cx": 0.5, "cy": 0.5, "radius": 0.10, "weighted": false
        })),
    )
    .expect("set_single circle");

    let overlays = reg.get_overlays(&w);
    assert_eq!(overlays.len(), 1, "circle should produce exactly one overlay");
    match overlays[0] {
        ChallengeOverlay::Circle { cx, cy, radius, .. } => {
            assert!((cx - 50.0).abs() < 1e-3, "cx scaled wrong: {cx}");
            assert!((cy - 50.0).abs() < 1e-3, "cy scaled wrong: {cy}");
            assert!((radius - 10.0).abs() < 1e-3, "radius scaled wrong: {radius}");
        }
        _ => panic!("circle overlay should be ChallengeOverlay::Circle"),
    }
}

#[test]
fn no_active_challenges_produces_no_overlays() {
    let cfg = SimConfig { size_x: 32, size_y: 32, ..SimConfig::default() };
    let (grid, signals, food, pop) = make_world(&cfg);
    let programmable = ProgrammablePool::new();
    let w = world(&grid, &signals, &food, &pop, &programmable, &cfg);

    let reg = ChallengeRegistry::new();
    let overlays = reg.get_overlays(&w);
    assert!(overlays.is_empty(), "no-active registry should emit zero overlays");
}
