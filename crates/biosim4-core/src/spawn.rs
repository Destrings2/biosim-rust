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
use crate::genome::neural_net::create_wiring;
use crate::genome::ops::{generate_child_genome, make_random_genome, Genome, ReproductionParams};
use crate::registry::challenge::WorldMut;
use crate::sim_state::SimulationState;

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
    state.programmable.clear(&mut state.grid);
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

    let starting_rate = state.config.point_mutation_rate;
    for _ in 0..state.config.population {
        let genome = make_random_genome(&state.config, &mut state.rng);
        let nnet = create_wiring(&genome, wiring_cfg);
        let dead = genome.len().saturating_sub(nnet.connection_count()) as u16;
        let loc = state.grid.find_empty_location(&mut state.rng);
        let id = state.population.next_id();
        let mut agent = Agent::new(id, loc, genome, nnet);
        // Seed the per-individual rate so adaptive runs have an anchor.
        agent.mutation_rate = starting_rate;
        agent.dead_gene_count = dead;
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
fn fitness_cmp(a: f32, b: f32) -> std::cmp::Ordering {
    a.total_cmp(&b)
}

/// Pick parents from this generation's evaluation results.
///
/// Pool composition:
/// - Normal: every agent that passed.
/// - Extinction-recovery: when nobody passes, take the top 10% by raw
///   fitness (minimum 2). Stops hard challenges from re-randomising
///   the population every generation and losing accumulated pressure.
///
/// Returns parents sorted ascending by fitness (higher index = fitter)
/// to match the tournament selector's index ordering. `survivor_count`
/// reports actual passes, excluding extinction-fallback fillers.
fn select_parent_genomes(evaluated: Vec<(Genome, f32, bool, f32)>) -> (Vec<(Genome, f32)>, u32) {
    let survivor_count = evaluated.iter().filter(|(_, _, p, _)| *p).count() as u32;

    let mut pool: Vec<(Genome, f32, f32)> = Vec::with_capacity(survivor_count as usize);
    let mut rejected: Vec<(Genome, f32, f32)> = Vec::new();
    for (g, f, pass, rate) in evaluated {
        if pass {
            pool.push((g, f, rate));
        } else {
            rejected.push((g, f, rate));
        }
    }

    if pool.is_empty() && !rejected.is_empty() {
        rejected.sort_by(|a, b| fitness_cmp(b.1, a.1));
        let take = (rejected.len() / 10).max(2);
        rejected.truncate(take);
        pool = rejected;
    }

    pool.sort_by(|a, b| fitness_cmp(a.1, b.1));
    let parents = pool.into_iter().map(|(g, _f, r)| (g, r)).collect();
    (parents, survivor_count)
}

/// Generate the next generation's `(genome, mutation_rate)` pairs from a
/// sorted parent pool. Empty pool → all random genomes seeded with
/// `cfg.point_mutation_rate`. Non-empty → `cfg.elitism_count` survivors
/// pass through unchanged, the rest are produced via crossover and
/// mutation.
fn generate_new_genomes(
    parent_genomes: &[(Genome, f32)],
    cfg: &crate::sim_config::SimConfig,
    rng: &mut crate::rng::Rng,
    new_pop: usize,
) -> Vec<(Genome, f32)> {
    if parent_genomes.is_empty() {
        return (0..new_pop)
            .map(|_| (make_random_genome(cfg, rng), cfg.point_mutation_rate))
            .collect();
    }

    // Clamp elitism so a hard challenge with few survivors doesn't ask
    // for more elites than exist.
    let elite_count = (cfg.elitism_count as usize).min(parent_genomes.len());
    let repro = ReproductionParams {
        sexual: cfg.sexual_reproduction,
        tournament_size: cfg.tournament_size,
        mutation_rate: cfg.point_mutation_rate,
        insertion_deletion_rate: cfg.gene_insertion_deletion_rate,
        deletion_ratio: cfg.deletion_ratio,
        max_len: cfg.genome_max_length,
        adaptive_mutation: cfg.adaptive_mutation,
        mutation_rate_jitter: cfg.mutation_rate_jitter,
    };

    let mut out = Vec::with_capacity(new_pop);
    // Elitism: copy the top-N survivors unchanged (genome AND rate, so a
    // well-tuned adaptive lineage isn't reset on promotion). Cheap
    // insurance against losing the best genome to mutation, especially
    // on hard challenges where survivors are scarce.
    out.extend(parent_genomes.iter().rev().take(elite_count).cloned());
    while out.len() < new_pop {
        out.push(generate_child_genome(parent_genomes, &repro, rng));
    }
    out
}

/// Clear the world for a new generation: empty the grid, restamp barriers,
/// reset signals, regenerate food (if enabled), wipe the programmable pool.
fn reset_world(state: &mut SimulationState) {
    state.population.clear();
    // Clear programmables before zero-filling so their grid cells are
    // released in one consistent step. The pool's clear walks alive_ids
    // and sets EMPTY at each loc; since zero_fill follows, the two are
    // equivalent here — but doing the pool clear up front keeps the
    // pool's internal state in sync regardless of grid contents.
    state.programmable.clear(&mut state.grid);
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
    // Parsimony pressure: agents carrying dead-end gene chains get a
    // small fitness deduction proportional to the dead fraction. The
    // `pass` boolean is preserved — challenge admission must not depend
    // on bloat — but the adjusted score re-orders the parent pool so
    // lean genomes outrank equally-fit bloated ones. Multiplying by 0.0
    // is a no-op when the feature is disabled (default).
    let bloat_weight = state.config.bloat_penalty_weight;

    let evaluated: Vec<(Genome, f32, bool, f32)> = state
        .population
        .iter_alive()
        .map(|a| {
            let (pass, fitness) = state.challenges.evaluate(a, &world);
            // `dead_norm` is in [0, 1]; the subtraction is bounded by
            // `bloat_weight`. With the default weight = 0 this is exactly
            // zero — no behavioural change and no float-rounding drift.
            let dead_norm = a.dead_gene_count as f32 / a.genome.len().max(1) as f32;
            let adjusted = fitness - bloat_weight * dead_norm;
            // Carry the agent's mutation_rate through selection so
            // adaptive lineages preserve their inherited rate.
            (a.genome.clone(), adjusted, pass, a.mutation_rate)
        })
        .collect();

    let (parent_pool, survivor_count) = select_parent_genomes(evaluated);

    let new_pop = state.config.population as usize;
    // Commit pending enable/disable changes: from this generation on, new
    // nnets are wired against the updated active sensor/action set.
    apply_feature_enables(state);
    state.sensors.commit_enabled();
    state.actions.commit_enabled();
    let wiring_cfg = state.wiring_config();

    let new_genomes =
        generate_new_genomes(parent_pool.as_slice(), &state.config, &mut state.rng, new_pop);

    reset_world(state);
    state.generation += 1;

    for (genome, mutation_rate) in new_genomes {
        let nnet = create_wiring(&genome, wiring_cfg);
        let dead = genome.len().saturating_sub(nnet.connection_count()) as u16;
        let loc = state.grid.find_empty_location(&mut state.rng);
        let id = state.population.next_id();
        let mut agent = Agent::new(id, loc, genome, nnet);
        agent.mutation_rate = mutation_rate;
        agent.dead_gene_count = dead;
        let assigned_id = state.population.spawn(agent);
        debug_assert_eq!(id, assigned_id);
        state.grid.set(loc, assigned_id);
    }

    // Run on_generation_start hooks. The programmable pool was already
    // wiped inside `reset_world`; challenges that own programmables
    // re-spawn them here.
    let mut world_mut = WorldMut {
        grid: &mut state.grid,
        signals: &mut state.signals,
        population: &mut state.population,
        programmable: &mut state.programmable,
        rng: &mut state.rng,
        config: &state.config,
        step: 0,
        generation: state.generation,
    };
    state.challenges.on_generation_start(&mut world_mut);

    survivor_count
}
