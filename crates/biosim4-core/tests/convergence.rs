//! Sanity tests that survival rates trend upward over generations on
//! challenges with real selection pressure. Not exact (the GA is stochastic),
//! but the gap between "first 5 gens" and "last 5 gens" should be large.
//!
//! These were added after fixing the radioactive-walls challenge (which
//! previously had no on-step damage logic) and the inverted fitness-biased
//! parent selection in `generate_child_genome`.

use biosim4_core::{
    challenges::register_builtin_challenges,
    sim_config::SimConfig,
    sim_state::SimulationState,
    sim_step::step_generation,
    spawn::{initialize_generation_0, spawn_new_generation},
};
use biosim4_core::registry::challenge::ChallengeConfig;

fn run_and_collect_survival(challenge_id: &str, generations: u32, seed: u64) -> Vec<f32> {
    let mut cfg = SimConfig::default();
    cfg.size_x = 96;
    cfg.size_y = 96;
    cfg.population = 400;
    cfg.steps_per_generation = 150;
    cfg.deterministic = true;
    cfg.rng_seed = seed;
    cfg.point_mutation_rate = 0.01;
    cfg.choose_parents_by_fitness = true;
    cfg.barrier_type = 0;

    let mut state = SimulationState::new(cfg);
    register_builtin_challenges(&mut state.challenges);
    state.challenges.apply_config(ChallengeConfig {
        active: vec![challenge_id.to_string()],
        composition: biosim4_core::registry::challenge::ChallengeComposition::Any,
        params: Default::default(),
    }).expect("set challenge");

    initialize_generation_0(&mut state);

    let pop = state.config.population as f32;
    let mut rates = Vec::with_capacity(generations as usize);
    for _ in 0..generations {
        step_generation(&mut state);
        let survivors = spawn_new_generation(&mut state) as f32;
        rates.push(survivors / pop);
    }
    rates
}

fn mean(xs: &[f32]) -> f32 { xs.iter().sum::<f32>() / xs.len() as f32 }

fn assert_improves(challenge: &str, rates: &[f32], min_gain: f32) {
    let n = rates.len();
    let head = mean(&rates[0..5]);
    let tail = mean(&rates[n - 5..n]);
    let gain = tail - head;
    eprintln!(
        "[{}] head_5={:.3} tail_5={:.3} gain={:.3} (need ≥ {:.2})\n  series: {:?}",
        challenge, head, tail, gain, min_gain,
        rates.iter().map(|r| (r * 100.0).round() / 100.0).collect::<Vec<_>>(),
    );
    assert!(
        gain >= min_gain,
        "{}: survival rate did not improve enough (gain {:.3} < {:.2})",
        challenge, gain, min_gain,
    );
}

#[test]
fn radioactive_walls_population_converges() {
    let rates = run_and_collect_survival("radioactive_walls", 50, 42);
    assert_improves("radioactive_walls", &rates, 0.05);
}

#[test]
fn migrate_distance_population_converges() {
    let rates = run_and_collect_survival("migrate_distance", 40, 7);
    assert_improves("migrate_distance", &rates, 0.10);
}

#[test]
fn sun_tracker_population_converges() {
    // Hard challenge with default-eased params; just want to see *some* lift
    // from the bootstrap-from-extinction code path.
    let rates = run_and_collect_survival("sun_tracker", 60, 99);
    assert_improves("sun_tracker", &rates, 0.02);
}

#[test]
fn food_foraging_population_converges() {
    // Food_foraging tends to saturate within a few gens (gen 0 ~0% from
    // random walks, gen 1+ ~10-13% with the elitism+bootstrap kick), then
    // plateau, so we just check the tail is meaningfully above gen 0.
    let rates = run_and_collect_survival("food_foraging", 40, 555);
    let tail = rates[rates.len() - 5..].iter().sum::<f32>() / 5.0;
    eprintln!("[food_foraging] tail_5={:.3} gen_0={:.3}", tail, rates[0]);
    assert!(tail > 0.05, "food_foraging never lifted off (tail={:.3})", tail);
}

// Diaspora deliberately omitted: at the population densities used in tests
// (~4% of grid cells), spreading out is geometrically near-impossible, so
// the GA can't push the survival rate above noise floor. The challenge
// itself works (the per-agent eval is correct); tune density per use-case.
