use biosim4_core::{
    sim_config::SimConfig, sim_state::SimulationState, sim_step::step_generation,
    spawn::{initialize_generation_0, spawn_new_generation},
    registry::challenge::{ChallengeConfig, ChallengeComposition},
};

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
    biosim4_core::challenges::register_builtin_challenges(&mut state.challenges);
    state.challenges.apply_config(ChallengeConfig {
        active: vec![challenge_id.to_string()],
        composition: ChallengeComposition::Any,
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
        "[{}] head_5={:.3} tail_5={:.3} gain={:.3} (need >= {:.2})\n  series: {:?}",
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
    let rates = run_and_collect_survival("sun_tracker", 80, 99);
    assert_improves("sun_tracker", &rates, 0.02);
}
