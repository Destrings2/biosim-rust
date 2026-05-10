//! Parallel-vs-serial determinism check.
//!
//! `sim_step::step_all_agents` splits each tick into a (potentially parallel)
//! decide phase and a serial apply phase. The decide phase uses
//! `par_iter_mut().filter_map().collect()` over an indexed slice, which rayon
//! guarantees preserves source order. Combined with serially pre-forked
//! per-agent RNGs, this means parallel runs must produce byte-identical state
//! to serial runs.
//!
//! Run this file with both `cargo test` and `cargo test --features parallel`.
//! Both invocations must produce the same fingerprint — that is what proves
//! the parallel path didn't drift.

use biosim4_core::{
    sim_config::SimConfig,
    sim_state::SimulationState,
    sim_step::step_generation,
    registry::{ChallengeComposition, ChallengeConfig},
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn fingerprint(state: &SimulationState) -> u64 {
    let mut h = DefaultHasher::new();
    state.generation.hash(&mut h);
    state.sim_step.hash(&mut h);
    // Iterate in deterministic id order, hashing the observable state.
    let mut entries: Vec<_> = state.population.iter_alive()
        .map(|a| (a.id, a.loc.x, a.loc.y, a.age, a.heading.0, a.last_move_dir.0))
        .collect();
    entries.sort_by_key(|e| e.0);
    entries.hash(&mut h);
    h.finish()
}

fn run_config() -> SimConfig {
    SimConfig {
        size_x: 64,
        size_y: 64,
        population: 200,
        steps_per_generation: 80,
        max_generations: 4,
        rng_seed: 9999,
        max_number_neurons: 5,
        kill_enable: true,
        ..SimConfig::default()
    }
}

#[test]
fn same_seed_two_runs_identical_state() {
    // Sanity: within a single binary, two runs with the same seed must match.
    let cfg = run_config();
    let mut a = SimulationState::new(cfg.clone());
    let mut b = SimulationState::new(cfg);

    for _ in 0..3 {
        step_generation(&mut a);
        step_generation(&mut b);
    }
    assert_eq!(fingerprint(&a), fingerprint(&b));
}

#[test]
fn challenged_run_matches_across_features() {
    // This fingerprint must be identical when the test is run with
    // `--features parallel` and without. If the parallel decide-phase ever
    // re-orders side effects, this assert fails.
    let cfg = run_config();
    let mut state = SimulationState::new(cfg);

    let cc = ChallengeConfig {
        active: vec!["right_half".into()],
        composition: ChallengeComposition::Any,
        params: Default::default(),
    };
    state.set_challenge(&serde_json::to_string(&cc).unwrap()).unwrap();

    for _ in 0..3 {
        step_generation(&mut state);
    }

    let fp = fingerprint(&state);
    // Lock the value the first time we see it — a divergence here means the
    // parallel implementation is no longer bit-equivalent to the serial path.
    insta_like_assert(fp);
}

fn insta_like_assert(actual: u64) {
    // The reference fingerprint is captured below. To regenerate after an
    // intentional simulation-semantics change: run this test, copy the
    // "actual" value from the failure message, paste it here, and verify the
    // test then passes under both `cargo test` and
    // `cargo test --features parallel`.
    const EXPECTED: u64 = 0x056758cdca624deb;
    if EXPECTED == 0 {
        // First-run mode: just print the value so the developer can paste it.
        // Force the test to be useful by failing if the const is still 0.
        panic!(
            "PARALLEL_DETERMINISM_FINGERPRINT not yet recorded. Set EXPECTED = 0x{:016x} in this test.",
            actual,
        );
    }
    assert_eq!(
        actual, EXPECTED,
        "fingerprint changed — either an intentional sim-semantics change \
         (regenerate EXPECTED) or the parallel path diverged from serial",
    );
}
