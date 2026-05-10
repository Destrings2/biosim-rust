use biosim4_core::{
    sim_config::SimConfig, sim_state::SimulationState, sim_step::step_generation,
    spawn::{initialize_generation_0, spawn_new_generation},
    registry::challenge::{ChallengeConfig, ChallengeComposition},
};

fn run_sim(threads: u32, challenge: &str) -> u64 {
    let mut cfg = SimConfig::default();
    cfg.size_x = 64;
    cfg.size_y = 64;
    cfg.population = 200;
    cfg.steps_per_generation = 100;
    cfg.deterministic = true;
    cfg.rng_seed = 12345;
    cfg.num_threads = threads;

    let mut state = SimulationState::new(cfg);
    biosim4_core::challenges::register_builtin_challenges(&mut state.challenges);
    
    if challenge != "none" {
        state.challenges.apply_config(ChallengeConfig {
            active: vec![challenge.to_string()],
            composition: ChallengeComposition::Any,
            params: Default::default(),
        }).unwrap();
    }

    initialize_generation_0(&mut state);

    for _ in 0..5 {
        step_generation(&mut state);
        spawn_new_generation(&mut state);
    }

    // Compute a simple structural fingerprint from the alive population
    let mut hash = 0u64;
    for agent in state.population.iter_alive() {
        hash = hash.wrapping_add(agent.id as u64);
        hash = hash.wrapping_add((agent.loc.x as u64) << 16);
        hash = hash.wrapping_add((agent.loc.y as u64) << 32);
        hash = hash.wrapping_add((agent.age as u64) << 48);
    }
    hash
}

#[test]
fn same_seed_two_runs_identical_state() {
    let h1 = run_sim(1, "none");
    let h2 = run_sim(1, "none");
    assert_eq!(h1, h2, "Two single-threaded runs with the same seed must be identical.");
}

#[test]
fn challenged_run_matches_across_features() {
    let h1 = run_sim(1, "radioactive_walls");
    let h4 = run_sim(4, "radioactive_walls");
    assert_eq!(
        h1, h4,
        "A run with num_threads=1 must produce the exact same fingerprint as num_threads=4"
    );
}
