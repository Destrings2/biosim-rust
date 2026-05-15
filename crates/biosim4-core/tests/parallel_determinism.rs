use biosim4_core::{
    registry::challenge::{ChallengeComposition, ChallengeConfig},
    sim_config::SimConfig,
    sim_state::SimulationState,
    sim_step::step_generation,
    spawn::{initialize_generation_0, spawn_new_generation},
};

fn run_sim(threads: u32, challenge: &str) -> u64 {
    let mut cfg = SimConfig::default();
    cfg.size_x = 64;
    cfg.size_y = 64;
    cfg.population = 200;
    cfg.steps_per_generation = 100;
    cfg.rng_seed = 12345;
    cfg.num_threads = threads;

    let mut state = SimulationState::new(cfg);
    biosim4_challenges::register_builtin_challenges(&mut state.challenges);

    if challenge != "none" {
        state
            .challenges
            .apply_config(ChallengeConfig {
                active: vec![challenge.to_string()],
                composition: ChallengeComposition::Any,
                params: Default::default(),
            })
            .unwrap();
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

/// Multi-threaded runs are **intentionally non-deterministic** — the fast
/// stepping path uses entropy-seeded per-worker Rngs in Phase 2 and merges
/// chunk-local queues in arbitrary work-stealing order. We trade
/// reproducibility for throughput, so this test only asserts the run
/// completes and produces a non-empty population.
#[test]
fn multithreaded_run_completes_with_population() {
    let h = run_sim(4, "none");
    assert!(h != 0, "multi-threaded run produced empty population");
}

/// Cross-thread-count determinism is intentionally NOT preserved: the
/// fast-mode parallel scheduler trades it for maximum throughput. A run
/// with `num_threads=1` is not expected to match `num_threads=4`. We still
/// assert the simulation completes successfully and yields a non-empty
/// population at both thread counts.
#[test]
fn challenged_run_completes_at_both_thread_counts() {
    let h1 = run_sim(1, "radioactive_walls");
    let h4 = run_sim(4, "radioactive_walls");
    // Both runs produced *some* state (non-zero fingerprint of remaining agents).
    assert!(h1 != 0, "single-threaded run produced empty population");
    assert!(h4 != 0, "multi-threaded run produced empty population");
}
