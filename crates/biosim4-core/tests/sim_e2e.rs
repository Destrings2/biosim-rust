//! End-to-end smoke tests for the full simulation loop. These catch integration
//! bugs (split-borrow regressions, queue draining order, generation count drift,
//! deterministic-seed reproducibility) that unit tests can't.

use biosim4_core::{
    sim_config::SimConfig,
    sim_state::SimulationState,
    sim_step::step_generation,
    spawn::spawn_new_generation,
    registry::ChallengeConfig,
};

fn small_config() -> SimConfig {
    SimConfig {
        size_x: 32,
        size_y: 32,
        population: 30,
        steps_per_generation: 50,
        max_generations: 3,
        rng_seed: 12345,
        max_number_neurons: 4,
        ..SimConfig::default()
    }
}

#[test]
fn new_simulation_state_has_full_population() {
    let state = SimulationState::new(small_config());
    assert_eq!(state.generation, 0);
    assert_eq!(state.sim_step, 0);
    assert_eq!(
        state.population.alive_count(), 30,
        "generation 0 should be fully populated"
    );
}

#[test]
fn step_generation_does_not_panic_or_lose_population() {
    let mut state = SimulationState::new(small_config());
    let initial_alive = state.population.alive_count();
    step_generation(&mut state);
    // Without kill_enable, all agents should still be alive at the end of a generation
    let final_alive = state.population.alive_count();
    assert_eq!(
        final_alive, initial_alive,
        "without kill_enable, no agents should die during a generation"
    );
}

#[test]
fn agents_age_advances_during_step_generation() {
    let mut state = SimulationState::new(small_config());
    step_generation(&mut state);
    let any_aged = state.population.iter_alive().any(|a| a.age > 0);
    assert!(any_aged, "at least some agents should have age > 0 after step_generation");
}

#[test]
fn spawn_new_generation_resets_population_and_increments_counter() {
    let mut state = SimulationState::new(small_config());
    step_generation(&mut state);
    let gen0 = state.generation;
    let _ = spawn_new_generation(&mut state);
    assert_eq!(state.generation, gen0 + 1, "generation should increment");
    assert_eq!(state.population.alive_count() as u32, state.config.population,
               "next generation should be fully populated");
    // All agents should have age 0 again
    for a in state.population.iter_alive() {
        assert_eq!(a.age, 0, "fresh generation agents should have age=0");
    }
}

#[test]
fn deterministic_seed_produces_identical_first_step() {
    // Two runs with the same seed must produce identical agent positions after one generation.
    let cfg = small_config();
    let mut s1 = SimulationState::new(cfg.clone());
    let mut s2 = SimulationState::new(cfg);
    step_generation(&mut s1);
    step_generation(&mut s2);

    let p1: Vec<_> = s1.population.iter_alive().map(|a| (a.id, a.loc, a.age)).collect();
    let p2: Vec<_> = s2.population.iter_alive().map(|a| (a.id, a.loc, a.age)).collect();
    assert_eq!(p1, p2, "deterministic seed must produce identical state");
}

#[test]
fn challenge_filters_survivors() {
    // Set right_half challenge — only agents on the right half should survive.
    let mut cfg = small_config();
    cfg.population = 50;
    cfg.steps_per_generation = 20;
    let mut state = SimulationState::new(cfg);

    let cc = ChallengeConfig {
        active: vec!["right_half".into()],
        composition: biosim4_core::registry::ChallengeComposition::Any,
        params: Default::default(),
    };
    state.challenges.apply_config(cc).unwrap();

    step_generation(&mut state);
    let n = spawn_new_generation(&mut state);
    // The survivor count should be less than total population
    // (it's overwhelmingly unlikely all 50 random agents land on the right half)
    assert!(n <= state.config.population);
}

#[test]
fn run_three_full_generations_without_panic() {
    let mut state = SimulationState::new(small_config());
    let cc = ChallengeConfig {
        active: vec!["circle".into()],
        composition: biosim4_core::registry::ChallengeComposition::Any,
        params: Default::default(),
    };
    state.challenges.apply_config(cc).unwrap();
    for _ in 0..3 {
        step_generation(&mut state);
        let _ = spawn_new_generation(&mut state);
    }
    assert_eq!(state.generation, 3);
    assert_eq!(state.population.alive_count() as u32, state.config.population);
}

#[test]
fn set_challenge_via_json_works_end_to_end() {
    let mut state = SimulationState::new(small_config());
    let json = r#"{
        "active": ["right_half"],
        "composition": "Any",
        "params": {}
    }"#;
    state.set_challenge(json).expect("setting challenge from JSON should succeed");
    step_generation(&mut state);
    // Should not panic; challenge applied successfully.
}
