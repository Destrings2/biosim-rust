//! Challenge framework contract: this is the most user-facing extensibility point.
//! Tests cover JSON schema, configure() round-trip, evaluate() at boundary cases,
//! and ChallengeRegistry composition modes.

use biosim4_challenges::register_builtin_challenges;
use biosim4_core::{
    agent::{Agent, AgentId},
    food_layer::FoodLayer,
    genome::neural_net::{create_wiring, WiringConfig},
    genome::ops::make_random_genome,
    grid::Grid,
    population::Population,
    registry::{ChallengeComposition, ChallengeConfig, ChallengeRegistry},
    rng::Rng,
    signals_layer::Signals,
    sim_config::SimConfig,
    types::Coord,
    world::World,
};
use serde_json::json;

fn make_agent(id: AgentId, loc: Coord, cfg: &SimConfig, rng: &mut Rng) -> Agent {
    let g = make_random_genome(cfg, rng);
    let w =
        WiringConfig { sensor_count: 21, action_count: 17, max_neurons: cfg.max_number_neurons };
    let n = create_wiring(&g, w);
    Agent::new(id, loc, g, n)
}

fn world_with_agent(agent_loc: Coord, cfg: &SimConfig) -> (Grid, Signals, FoodLayer, Population) {
    let grid = Grid::new(cfg.size_x, cfg.size_y);
    let signals = Signals::new(1, cfg.size_x, cfg.size_y);
    let food = FoodLayer::new(cfg.size_x, cfg.size_y);
    let mut pop = Population::new(1);
    let mut rng = Rng::seeded(0xCAFE);
    let a = make_agent(pop.next_id(), agent_loc, cfg, &mut rng);
    pop.spawn(a);
    (grid, signals, food, pop)
}

#[test]
fn registry_lists_all_known_built_in_challenges() {
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    let schema = reg.schema_list();
    let arr = schema.as_array().expect("schema_list must be JSON array");

    let ids: Vec<&str> = arr.iter().map(|c| c["id"].as_str().unwrap()).collect();

    for required in [
        "circle",
        "right_half",
        "right_quarter",
        "left_eighth",
        "east_west_eighths",
        "center_weighted",
        "center_unweighted",
        "corner",
        "corner_weighted",
        "against_any_wall",
        "near_barrier",
        "pairs",
        "center_sparse",
        "string",
        "migrate_distance",
        "touch_any_wall",
        "location_sequence",
        "radioactive_walls",
        "altruism",
        "altruism_sacrifice",
    ] {
        assert!(ids.contains(&required), "missing built-in challenge: {required}");
    }
}

#[test]
fn schema_list_entries_have_required_fields() {
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    let schema = reg.schema_list();
    for entry in schema.as_array().unwrap() {
        assert!(entry["id"].is_string(), "entry missing id: {entry}");
        assert!(entry["name"].is_string(), "entry missing name: {entry}");
        assert!(entry["description"].is_string(), "entry missing description: {entry}");
        assert!(entry["schema"].is_object(), "entry missing schema object: {entry}");
        assert_eq!(entry["schema"]["type"], "object", "schema must declare type: object");
    }
}

#[test]
fn empty_active_set_passes_everyone() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let (grid, signals, food, pop) = world_with_agent(Coord::new(0, 0), &cfg);
    let world = World {
        grid: &grid,
        signals: &signals,
        food: &food,
        population: &pop,
        size_x: cfg.size_x,
        size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0,
        step: 0,
    };
    let reg = ChallengeRegistry::new();
    let agent = pop.get(1).unwrap();
    let (pass, score) = reg.evaluate(agent, &world);
    assert!(pass, "empty challenge set should pass");
    assert!((score - 1.0).abs() < 1e-6, "empty challenge set should score 1.0, got {}", score);
}

#[test]
fn right_half_pass_left_fails() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };

    // Agent on the right
    let (grid, signals, food, pop) = world_with_agent(Coord::new(12, 8), &cfg);
    let world = World {
        grid: &grid,
        signals: &signals,
        food: &food,
        population: &pop,
        size_x: cfg.size_x,
        size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0,
        step: 0,
    };
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    reg.set_single("right_half", None).unwrap();
    let (pass, score) = reg.evaluate(pop.get(1).unwrap(), &world);
    assert!(pass);
    assert!((score - 1.0).abs() < 1e-6);

    // Agent on the left should fail
    let (grid, signals, food, pop) = world_with_agent(Coord::new(2, 8), &cfg);
    let world = World {
        grid: &grid,
        signals: &signals,
        food: &food,
        population: &pop,
        size_x: cfg.size_x,
        size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0,
        step: 0,
    };
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    reg.set_single("right_half", None).unwrap();
    let (pass, score) = reg.evaluate(pop.get(1).unwrap(), &world);
    assert!(!pass);
    assert!(score.abs() < 1e-6);
}

#[test]
fn circle_challenge_configure_changes_evaluation() {
    let cfg = SimConfig { size_x: 17, size_y: 17, ..SimConfig::default() }; // mid = 8
    let (grid, signals, food, pop) = world_with_agent(Coord::new(8, 8), &cfg);
    let world = World {
        grid: &grid,
        signals: &signals,
        food: &food,
        population: &pop,
        size_x: cfg.size_x,
        size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0,
        step: 0,
    };

    // Default circle is centered at (0.25, 0.75) — agent at center (0.5, 0.5) should fail
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    reg.set_single("circle", None).unwrap();
    let (pass1, _) = reg.evaluate(pop.get(1).unwrap(), &world);
    assert!(!pass1, "default circle (NW quadrant) should not contain center");

    // Re-configure to center the circle ON the agent
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    reg.set_single("circle", Some(json!({ "cx": 0.5, "cy": 0.5, "radius": 0.1 }))).unwrap();
    let (pass2, score2) = reg.evaluate(pop.get(1).unwrap(), &world);
    assert!(pass2, "reconfigured circle on agent should pass");
    assert!(score2 > 0.99, "agent at center → score ≈ 1.0, got {}", score2);
}

#[test]
fn configure_with_bad_challenge_id_returns_err() {
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    let r = reg.set_single("does_not_exist", None);
    assert!(r.is_err(), "unknown challenge id should return Err");
}

#[test]
fn configure_with_bad_param_value_returns_err() {
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    // cx must be a number; pass a string
    let r = reg.set_single("circle", Some(json!({ "cx": "not a number" })));
    assert!(r.is_err(), "non-numeric cx should be rejected, got Ok");
}

#[test]
fn apply_config_with_any_composition_passes_if_any_active_passes() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    // Agent on left half (so right_half fails) but in left_eighth → at least one passes
    let (grid, signals, food, pop) = world_with_agent(Coord::new(1, 8), &cfg);
    let world = World {
        grid: &grid,
        signals: &signals,
        food: &food,
        population: &pop,
        size_x: cfg.size_x,
        size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0,
        step: 0,
    };
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    let cfg_json = ChallengeConfig {
        active: vec!["right_half".into(), "left_eighth".into()],
        composition: ChallengeComposition::Any,
        params: Default::default(),
    };
    reg.apply_config(cfg_json).unwrap();
    let (pass, _) = reg.evaluate(pop.get(1).unwrap(), &world);
    assert!(pass, "Any composition with one passing challenge should pass");
}

#[test]
fn apply_config_with_all_composition_fails_if_any_active_fails() {
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };
    let (grid, signals, food, pop) = world_with_agent(Coord::new(1, 8), &cfg);
    let world = World {
        grid: &grid,
        signals: &signals,
        food: &food,
        population: &pop,
        size_x: cfg.size_x,
        size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0,
        step: 0,
    };
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    let cfg_json = ChallengeConfig {
        active: vec!["right_half".into(), "left_eighth".into()],
        composition: ChallengeComposition::All,
        params: Default::default(),
    };
    reg.apply_config(cfg_json).unwrap();
    let (pass, _) = reg.evaluate(pop.get(1).unwrap(), &world);
    assert!(!pass, "All composition with one failing challenge should fail");
}

#[test]
fn challenge_config_serde_roundtrip() {
    // Frontend sends JSON like this — must deserialize cleanly:
    let json_str = r#"{
        "active": ["circle", "right_half"],
        "composition": "Any",
        "params": { "circle": { "cx": 0.5, "cy": 0.5, "radius": 0.2 } }
    }"#;
    let cfg: ChallengeConfig = serde_json::from_str(json_str).expect("valid JSON should parse");
    assert_eq!(cfg.active, vec!["circle", "right_half"]);
    assert!(matches!(cfg.composition, ChallengeComposition::Any));
    assert!(cfg.params.contains_key("circle"));
}

#[test]
fn evaluate_score_always_in_unit_interval() {
    // For every built-in challenge, evaluate returns a score in [0, 1].
    let cfg =
        SimConfig { size_x: 16, size_y: 16, steps_per_generation: 100, ..SimConfig::default() };
    let mut reg = ChallengeRegistry::new();
    register_builtin_challenges(&mut reg);
    let schema = reg.schema_list();
    let ids: Vec<String> =
        schema.as_array().unwrap().iter().map(|c| c["id"].as_str().unwrap().to_owned()).collect();

    // Test each at three positions: corner, edge, center
    let positions = [Coord::new(0, 0), Coord::new(8, 0), Coord::new(8, 8)];
    for id in &ids {
        for &p in &positions {
            let mut reg = ChallengeRegistry::new();
            register_builtin_challenges(&mut reg);
            if reg.set_single(id, None).is_err() {
                continue;
            }

            let (grid, signals, food, pop) = world_with_agent(p, &cfg);
            let world = World {
                grid: &grid,
                signals: &signals,
                food: &food,
                population: &pop,
                size_x: cfg.size_x,
                size_y: cfg.size_y,
                steps_per_generation: cfg.steps_per_generation,
                generation: 0,
                step: 0,
            };
            let (_pass, score) = reg.evaluate(pop.get(1).unwrap(), &world);
            assert!(
                score.is_finite() && (-1e-6..=1.0 + 1e-6).contains(&score),
                "challenge {} at {:?} returned out-of-range score: {}",
                id,
                p,
                score,
            );
        }
    }
}

#[test]
fn weighted_sum_composition_evaluates_correctly() {
    // WeightedSum: score = sum(score_i * weight_i) / sum(weights).
    // Use right_half (agent on right → score 1.0) and left_eighth (agent on
    // right at x=12 → fails → score 0.0) with equal weights → combined 0.5.
    // threshold=0.4 → passes; threshold=0.6 → fails.
    let cfg = SimConfig { size_x: 16, size_y: 16, ..SimConfig::default() };

    // Agent at x=12 is in the right half (passes right_half, score=1.0)
    // but NOT in the left eighth (fails left_eighth, score=0.0).
    let (grid, signals, food, pop) = world_with_agent(Coord::new(12, 8), &cfg);
    let world = World {
        grid: &grid,
        signals: &signals,
        food: &food,
        population: &pop,
        size_x: cfg.size_x,
        size_y: cfg.size_y,
        steps_per_generation: cfg.steps_per_generation,
        generation: 0,
        step: 0,
    };

    let make_reg = |threshold: f32| {
        let mut reg = ChallengeRegistry::new();
        register_builtin_challenges(&mut reg);
        let cfg_json = ChallengeConfig {
            active: vec!["right_half".into(), "left_eighth".into()],
            composition: ChallengeComposition::WeightedSum { weights: vec![1.0, 1.0], threshold },
            params: Default::default(),
        };
        reg.apply_config(cfg_json).unwrap();
        reg
    };

    // score = (1.0 * 1.0 + 0.0 * 1.0) / 2.0 = 0.5
    let (pass_low, score_low) = make_reg(0.4).evaluate(pop.get(1).unwrap(), &world);
    assert!((score_low - 0.5).abs() < 1e-5, "weighted-sum score should be 0.5, got {}", score_low);
    assert!(pass_low, "score 0.5 >= threshold 0.4 should pass");

    let (pass_high, score_high) = make_reg(0.6).evaluate(pop.get(1).unwrap(), &world);
    assert!(
        (score_high - 0.5).abs() < 1e-5,
        "weighted-sum score should be 0.5, got {}",
        score_high
    );
    assert!(!pass_high, "score 0.5 < threshold 0.6 should fail");
}
