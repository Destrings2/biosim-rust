//! Generation initialization and reproduction.
//!
//! # `initialize_generation_0`
//!
//! Clears the world, places barriers, commits pending registry changes, then
//! fills the population with agents carrying random genomes. Neural nets are
//! compiled against the committed sensor/action set (`wiring_config()`).
//!
//! # `spawn_new_generation`
//!
//! Implements the full selection-and-reproduction cycle:
//!
//! 1. **Evaluate** all alive agents against active challenges; each returns
//!    `(pass: bool, fitness: f32)`.
//! 2. **Build survivor pool** from agents where `pass == true`.
//! 3. **Bootstrap fallback** — if no agents pass (common on hard challenges in
//!    early generations), take the top 10% (minimum 2) by raw fitness score
//!    regardless of `pass`. Without this, a generation with zero survivors
//!    would re-randomize the entire population, destroying any gradient the
//!    GA had accumulated.
//! 4. **Sort** the pool ascending by fitness so `generate_child_genome`'s
//!    fitness-biased parent selection works correctly (higher index = fitter).
//! 5. **Elitism** — copy the two fittest survivors unchanged into the next
//!    generation before reproduction. Protects hard-won genomes from
//!    being overwritten by mutation, especially valuable when few agents pass.
//! 6. **Reproduce** — fill the rest of the population via
//!    `generate_child_genome`, applying crossover and mutation.
//! 7. **Commit** pending sensor/action changes and compile new neural nets
//!    against the updated `wiring_config()`.
//! 8. **Run** `on_generation_start` hooks.

use crate::agent::Agent;
use crate::genome::ops::{make_random_genome, generate_child_genome, Genome, ReproductionParams};
use crate::genome::neural_net::create_wiring;
use crate::sim_state::SimulationState;
use crate::registry::challenge::WorldMut;

/// Apply sensor/action enabled state derived from the current config.
/// Must be called before `commit_enabled()` at each generation boundary.
fn apply_feature_enables(state: &mut SimulationState) {
    let cfg = &state.config;

    // Energy/food sensors — meaningful only when energy system is on
    let energy = cfg.enable_energy;
    for id in &["energy_level", "food_here", "food_fwd", "food_lr"] {
        state.sensors.set_enabled(id, energy);
    }

    // Signal layer 1 sensors/actions — meaningful only when signal_layers >= 2
    let s1 = cfg.signal_layers >= 2;
    for id in &["signal1", "signal1_fwd", "signal1_lr"] {
        state.sensors.set_enabled(id, s1);
    }
    state.actions.set_enabled("emit_signal1", s1);

    // Signal layer 2 sensors/actions — meaningful only when signal_layers >= 3
    let s2 = cfg.signal_layers >= 3;
    for id in &["signal2", "signal2_fwd", "signal2_lr"] {
        state.sensors.set_enabled(id, s2);
    }
    state.actions.set_enabled("emit_signal2", s2);
}

/// Populate generation 0 with agents carrying random genomes, placed randomly.
pub fn initialize_generation_0(state: &mut SimulationState) {
    state.population.clear();
    state.grid.zero_fill();
    crate::barriers::create_barrier(&mut state.grid, state.config.barrier_type);
    state.reapply_user_barriers();
    state.signals.zero_fill();
    if state.config.enable_energy {
        state.food.randomize(state.config.food_initial_density, &state.grid, &mut state.rng);
    } else {
        state.food.zero_fill();
    }

    apply_feature_enables(state);
    state.sensors.commit_enabled();
    state.actions.commit_enabled();
    let wiring_cfg = state.wiring_config();

    for _ in 0..state.config.population {
        let genome = make_random_genome(&state.config, &mut state.rng);
        let nnet = create_wiring(&genome, wiring_cfg);
        let loc = state.grid.find_empty_location(&mut state.rng);
        let id = state.population.next_id();
        let agent = Agent::new(id, loc, genome, nnet);
        let assigned_id = state.population.spawn(agent);
        debug_assert_eq!(id, assigned_id);
        state.grid.set(loc, assigned_id);
    }
}

/// Total order over fitness scores. Uses `f32::total_cmp` so NaN — which
/// can leak in from buggy challenge `evaluate` impls — sorts to a defined
/// position instead of being silently treated as "equal" by
/// `partial_cmp().unwrap_or(Equal)`. Higher score = greater.
#[inline]
fn fitness_cmp(a: f32, b: f32) -> std::cmp::Ordering { a.total_cmp(&b) }

/// Pick parent genomes from this generation's evaluation results.
///
/// Pool composition:
/// - Normally: every agent that passed the challenge.
/// - Extinction-recovery: if **no** agent passed but the population isn't
///   empty, take the top 10% (minimum 2) by raw fitness regardless of
///   pass/fail. Without this, hard challenges where no random agent passes
///   gen 0 would just re-randomize the population every gen and never
///   accumulate any selection pressure.
///
/// Returned vec is sorted ascending by fitness so `generate_child_genome`'s
/// fitness-biased parent selection works correctly (higher index = fitter).
/// Returns `(parents, survivor_count)` where `survivor_count` is the number
/// of agents that *actually passed* (not counting the extinction fallback).
fn select_parent_genomes(evaluated: Vec<(Genome, f32, bool)>) -> (Vec<Genome>, u32) {
    let survivor_count = evaluated.iter().filter(|(_, _, p)| *p).count() as u32;

    // Filter pass=true into the pool, moving genomes rather than cloning.
    let mut pool: Vec<(Genome, f32)> = Vec::with_capacity(survivor_count as usize);
    let mut rejected: Vec<(Genome, f32)> = Vec::new();
    for (g, f, pass) in evaluated {
        if pass {
            pool.push((g, f));
        } else {
            rejected.push((g, f));
        }
    }

    if pool.is_empty() && !rejected.is_empty() {
        // Extinction-recovery: sort rejected by fitness desc, take top 10%.
        rejected.sort_by(|a, b| fitness_cmp(b.1, a.1));
        let take = (rejected.len() / 10).max(2);
        rejected.truncate(take);
        pool = rejected;
    }

    pool.sort_by(|a, b| fitness_cmp(a.1, b.1));
    let parents = pool.into_iter().map(|(g, _)| g).collect();
    (parents, survivor_count)
}

/// Generate the next generation's genomes from a sorted parent pool.
///
/// - Empty parent pool → all random genomes.
/// - Non-empty → elitism preserves the top 2 unchanged, the rest are
///   produced via crossover/mutation against the parent pool.
fn generate_new_genomes(
    parent_genomes: &[Genome],
    cfg: &crate::sim_config::SimConfig,
    rng: &mut crate::rng::Rng,
    new_pop: usize,
) -> Vec<Genome> {
    if parent_genomes.is_empty() {
        return (0..new_pop).map(|_| make_random_genome(cfg, rng)).collect();
    }

    let elite_count = 2.min(parent_genomes.len());
    let repro = ReproductionParams {
        sexual: cfg.sexual_reproduction,
        choose_by_fitness: cfg.choose_parents_by_fitness,
        mutation_rate: cfg.point_mutation_rate,
        insertion_deletion_rate: cfg.gene_insertion_deletion_rate,
        deletion_ratio: cfg.deletion_ratio,
        max_len: cfg.genome_max_length,
    };

    let mut out = Vec::with_capacity(new_pop);
    // Elitism: copy the top-N survivors unchanged. Cheap insurance against
    // losing the best genome to mutation, especially on hard challenges.
    out.extend(parent_genomes.iter().rev().take(elite_count).cloned());
    while out.len() < new_pop {
        out.push(generate_child_genome(parent_genomes, &repro, rng));
    }
    out
}

/// Clear the world for a new generation: empty the grid, restamp barriers,
/// reset signals, regenerate food (if enabled).
fn reset_world(state: &mut SimulationState) {
    state.population.clear();
    state.grid.zero_fill();
    crate::barriers::create_barrier(&mut state.grid, state.config.barrier_type);
    state.reapply_user_barriers();
    state.signals.zero_fill();
    if state.config.enable_energy {
        state.food.randomize(state.config.food_initial_density, &state.grid, &mut state.rng);
    } else {
        state.food.zero_fill();
    }
}

/// Select survivors, reproduce, and populate the next generation. Returns
/// the number of agents that passed the challenge (excluding any extinction-
/// recovery fallback parents).
pub fn spawn_new_generation(state: &mut SimulationState) -> u32 {
    let world = state.world();

    let evaluated: Vec<(Genome, f32, bool)> = state
        .population
        .iter_alive()
        .map(|a| {
            let (pass, fitness) = state.challenges.evaluate(a, &world);
            (a.genome.clone(), fitness, pass)
        })
        .collect();

    let (parent_genomes, survivor_count) = select_parent_genomes(evaluated);

    let new_pop = state.config.population as usize;
    // Commit pending enable/disable changes: from this generation on, new
    // nnets are wired against the updated active sensor/action set.
    apply_feature_enables(state);
    state.sensors.commit_enabled();
    state.actions.commit_enabled();
    let wiring_cfg = state.wiring_config();

    let new_genomes = generate_new_genomes(
        parent_genomes.as_slice(),
        &state.config,
        &mut state.rng,
        new_pop,
    );

    reset_world(state);
    state.generation += 1;

    for genome in new_genomes {
        let nnet = create_wiring(&genome, wiring_cfg);
        let loc = state.grid.find_empty_location(&mut state.rng);
        let id = state.population.next_id();
        let agent = Agent::new(id, loc, genome, nnet);
        let assigned_id = state.population.spawn(agent);
        debug_assert_eq!(id, assigned_id);
        state.grid.set(loc, assigned_id);
    }

    // Run on_generation_start hooks.
    let mut world_mut = WorldMut {
        grid: &mut state.grid,
        signals: &mut state.signals,
        population: &mut state.population,
        rng: &mut state.rng,
        config: &state.config,
        step: 0,
        generation: state.generation,
    };
    state.challenges.on_generation_start(&mut world_mut);

    survivor_count
}
