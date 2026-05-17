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
use crate::genome::ops::{
    generate_child_genome_interspecies_with_positions, generate_child_genome_with_positions,
    make_random_genome, Genome, ReproductionParams,
};
use crate::registry::challenge::WorldMut;
use crate::sim_config::OffspringPlacementMode;
use crate::sim_state::SimulationState;
use crate::types::Coord;

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
/// to match the tournament selector's index ordering, plus parallel
/// arrays of fitnesses and grid positions.
///
/// Fitness is propagated separately because the per-parent `(Genome,
/// f32)` tuple stores **mutation rate** (carried for adaptive lineages),
/// not fitness. The speciation pipeline needs the true fitness for
/// fitness sharing and stagnation tracking; conflating the two breaks
/// silently whenever `adaptive_mutation` is on. `parent_positions` is
/// used by spatial offspring inheritance ([`OffspringPlacementMode`]).
/// `survivor_count` reports actual passes, excluding extinction-fallback
/// fillers.
fn select_parent_genomes(
    evaluated: Vec<(Genome, f32, bool, f32, Coord)>,
) -> (Vec<(Genome, f32)>, Vec<f32>, Vec<Coord>, u32) {
    let survivor_count = evaluated.iter().filter(|(_, _, p, _, _)| *p).count() as u32;

    let mut pool: Vec<(Genome, f32, f32, Coord)> = Vec::with_capacity(survivor_count as usize);
    let mut rejected: Vec<(Genome, f32, f32, Coord)> = Vec::new();
    for (g, f, pass, rate, loc) in evaluated {
        if pass {
            pool.push((g, f, rate, loc));
        } else {
            rejected.push((g, f, rate, loc));
        }
    }

    if pool.is_empty() && !rejected.is_empty() {
        rejected.sort_by(|a, b| fitness_cmp(b.1, a.1));
        let take = (rejected.len() / 10).max(2);
        rejected.truncate(take);
        pool = rejected;
    }

    pool.sort_by(|a, b| fitness_cmp(a.1, b.1));
    let mut parents = Vec::with_capacity(pool.len());
    let mut fitnesses = Vec::with_capacity(pool.len());
    let mut positions = Vec::with_capacity(pool.len());
    for (g, f, r, loc) in pool {
        parents.push((g, r));
        fitnesses.push(f);
        positions.push(loc);
    }
    (parents, fitnesses, positions, survivor_count)
}

/// One child produced by reproduction. `species_id` is `None` when the
/// speciation pipeline didn't attribute the child (speciation disabled,
/// extinction-fallback random fill, or defensive top-up). `parent_a_pos`
/// and `parent_b_pos` are populated when reproduction tracked the
/// parents' grid positions, driving spatial offspring inheritance in the
/// placement loop (see [`OffspringPlacementMode`]). For elites,
/// `parent_a_pos` is the elite's own previous location and `parent_b_pos`
/// is `None`; for the bootstrap / extinction-fallback random fill both
/// are `None`.
struct NewChild {
    genome: Genome,
    mutation_rate: f32,
    species_id: Option<u32>,
    parent_a_pos: Option<Coord>,
    parent_b_pos: Option<Coord>,
}

/// Generate the next generation's children from a sorted parent pool
/// without speciation. Empty pool → all random genomes seeded with
/// `cfg.point_mutation_rate`. Non-empty → `cfg.elitism_count` survivors
/// pass through unchanged, the rest are produced via crossover and
/// mutation. `species_id` is `None` on every child since the speciation
/// pipeline never ran.
///
/// `parent_positions` is parallel to `parent_genomes` and propagates each
/// chosen parent's grid location into the returned `NewChild` so the
/// placement loop can apply [`OffspringPlacementMode`].
fn generate_new_genomes(
    parent_genomes: &[(Genome, f32)],
    parent_positions: &[Coord],
    cfg: &crate::sim_config::SimConfig,
    rng: &mut crate::rng::Rng,
    new_pop: usize,
) -> Vec<NewChild> {
    if parent_genomes.is_empty() {
        return (0..new_pop)
            .map(|_| NewChild {
                genome: make_random_genome(cfg, rng),
                mutation_rate: cfg.point_mutation_rate,
                species_id: None,
                parent_a_pos: None,
                parent_b_pos: None,
            })
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

    let mut out: Vec<NewChild> = Vec::with_capacity(new_pop);
    // Elitism: copy the top-N survivors unchanged (genome AND rate, so a
    // well-tuned adaptive lineage isn't reset on promotion). Cheap
    // insurance against losing the best genome to mutation, especially
    // on hard challenges where survivors are scarce. Each elite keeps
    // its own grid location so spatial placement modes preserve
    // territorial succession.
    let n = parent_genomes.len();
    for i in 0..elite_count {
        let idx = n - 1 - i;
        let (g, r) = &parent_genomes[idx];
        out.push(NewChild {
            genome: g.clone(),
            mutation_rate: *r,
            species_id: None,
            parent_a_pos: parent_positions.get(idx).copied(),
            parent_b_pos: None,
        });
    }
    while out.len() < new_pop {
        let r = generate_child_genome_with_positions(parent_genomes, parent_positions, &repro, rng);
        out.push(NewChild {
            genome: r.genome,
            mutation_rate: r.mutation_rate,
            species_id: None,
            parent_a_pos: r.parent_a_pos,
            parent_b_pos: r.parent_b_pos,
        });
    }
    out
}

/// Speciated genome generation. Buckets parents by genome distance,
/// allocates offspring slots based on species fitness, and breeds
/// within species. Each child carries the producing species' id so the
/// inspector and downstream analysis can attribute lineage.
///
/// `parent_fitnesses` and `parent_positions` are parallel to
/// `parent_genomes`. Fitness is passed in explicitly because
/// `parent_genomes[i].1` is the inherited mutation rate, not fitness —
/// see [`select_parent_genomes`] for the split. Positions are used by
/// spatial offspring inheritance.
fn generate_new_genomes_speciated(
    state: &mut SimulationState,
    parent_genomes: &[(Genome, f32)],
    parent_fitnesses: &[f32],
    parent_positions: &[Coord],
    new_pop: usize,
    wiring_cfg: crate::genome::neural_net::WiringConfig,
) -> Vec<NewChild> {
    if parent_genomes.is_empty() {
        return (0..new_pop)
            .map(|_| NewChild {
                genome: make_random_genome(&state.config, &mut state.rng),
                mutation_rate: state.config.point_mutation_rate,
                species_id: None,
                parent_a_pos: None,
                parent_b_pos: None,
            })
            .collect();
    }

    state.speciation.speciate(parent_genomes, &state.config, wiring_cfg);
    state.speciation.assign_offspring_slots(parent_fitnesses, new_pop as u32);
    state.speciation.prune_stagnant(parent_fitnesses, state.config.stagnation_limit);

    let repro = ReproductionParams {
        sexual: state.config.sexual_reproduction,
        tournament_size: state.config.tournament_size,
        mutation_rate: state.config.point_mutation_rate,
        insertion_deletion_rate: state.config.gene_insertion_deletion_rate,
        deletion_ratio: state.config.deletion_ratio,
        max_len: state.config.genome_max_length,
        adaptive_mutation: state.config.adaptive_mutation,
        mutation_rate_jitter: state.config.mutation_rate_jitter,
    };

    let mut out: Vec<NewChild> = Vec::with_capacity(new_pop);

    // Iterate a clone so we can mutably borrow `state.rng` inside without
    // clashing with the species list borrow.
    let species_snapshot = state.speciation.species.clone();
    for species in &species_snapshot {
        if species.allocated_offspring == 0 || species.members.is_empty() {
            continue;
        }

        // Build species sub-pool with fitness so we can co-sort by
        // fitness ascending (tournament selector expects highest index =
        // fittest). Sorting by the rate slot — which is what `(Genome,
        // f32)` carries — would silently break selection whenever
        // adaptive_mutation is on, since rates vary per individual and
        // would reorder parents by mutation rate instead of fitness.
        let mut species_indexed: Vec<(Genome, f32 /* rate */, f32 /* fitness */, Coord)> = species
            .members
            .iter()
            .map(|&idx| {
                let (g, r) = &parent_genomes[idx];
                (g.clone(), *r, parent_fitnesses[idx], parent_positions[idx])
            })
            .collect();
        species_indexed.sort_by(|a, b| a.2.total_cmp(&b.2));
        let species_parents: Vec<(Genome, f32)> =
            species_indexed.iter().map(|(g, r, _, _)| (g.clone(), *r)).collect();
        let species_positions: Vec<Coord> = species_indexed.iter().map(|(_, _, _, p)| *p).collect();

        let mut spawned = 0usize;

        // Within-species elitism: copy the species' best genome unchanged
        // when there are enough members for the rank to mean something.
        // Always counts against `allocated_offspring` so the population
        // total stays exact. The elite keeps its own location so spatial
        // placement preserves the species' territorial anchor.
        if species.members.len() >= state.config.species_elitism_min as usize
            && spawned < species.allocated_offspring
        {
            let (elite_genome, elite_rate, _elite_fit, elite_pos) =
                species_indexed.last().unwrap().clone();
            out.push(NewChild {
                genome: elite_genome,
                mutation_rate: elite_rate,
                species_id: Some(species.id),
                parent_a_pos: Some(elite_pos),
                parent_b_pos: None,
            });
            spawned += 1;
        }

        while spawned < species.allocated_offspring {
            // Interspecies mating: rare (default 0.001) cross-species crossover
            // to inject diversity. Build the "other species" pool lazily —
            // only when the dice say so — so the typical birth pays nothing.
            // Falls back to within-species crossover when this is the only
            // species or no other species has members.
            let try_interspecies = state.config.sexual_reproduction
                && species_snapshot.len() > 1
                && state.rng.gen_bool(state.config.interspecies_mating_rate);
            let r = if try_interspecies {
                let mut other_parents: Vec<(Genome, f32)> = Vec::new();
                let mut other_positions: Vec<Coord> = Vec::new();
                for other in &species_snapshot {
                    if other.id == species.id || other.members.is_empty() {
                        continue;
                    }
                    for &idx in &other.members {
                        other_parents.push(parent_genomes[idx].clone());
                        other_positions.push(parent_positions[idx]);
                    }
                }
                if other_parents.is_empty() {
                    generate_child_genome_with_positions(
                        &species_parents,
                        &species_positions,
                        &repro,
                        &mut state.rng,
                    )
                } else {
                    generate_child_genome_interspecies_with_positions(
                        &species_parents,
                        &species_positions,
                        &other_parents,
                        &other_positions,
                        &repro,
                        &mut state.rng,
                    )
                }
            } else {
                generate_child_genome_with_positions(
                    &species_parents,
                    &species_positions,
                    &repro,
                    &mut state.rng,
                )
            };
            out.push(NewChild {
                genome: r.genome,
                mutation_rate: r.mutation_rate,
                species_id: Some(species.id),
                parent_a_pos: r.parent_a_pos,
                parent_b_pos: r.parent_b_pos,
            });
            spawned += 1;
        }
    }

    // Adaptive τ adjustment based on this gen's species count.
    state.speciation.update_compatibility_threshold(
        state.config.species_count_target,
        state.config.species_count_target_tolerance,
        state.config.compatibility_threshold_step,
    );

    // Housekeeping for next gen: resample representatives and drop species
    // that received zero offspring this gen (stagnant or empty).
    state.speciation.end_of_generation(parent_genomes, &state.config, &mut state.rng, wiring_cfg);

    // Defensive fill: if any rounding edge case left us short, top up with
    // global tournament crossover. `species_id = None` flags them as
    // unattributed so the inspector doesn't pretend they were speciated.
    while out.len() < new_pop {
        let r = generate_child_genome_with_positions(
            parent_genomes,
            parent_positions,
            &repro,
            &mut state.rng,
        );
        out.push(NewChild {
            genome: r.genome,
            mutation_rate: r.mutation_rate,
            species_id: None,
            parent_a_pos: r.parent_a_pos,
            parent_b_pos: r.parent_b_pos,
        });
    }
    out.truncate(new_pop);
    out
}

/// Resolve a child's grid location given the configured placement mode,
/// the radius, and the parents' previous positions.
///
/// `Random` and any path with no `parent_a_pos` (extinction-fallback /
/// random fill / gen-0) go through [`crate::grid::Grid::find_empty_location`]
/// — the original code path, so the RNG trace matches the pre-feature
/// baseline byte-for-byte at default settings.
///
/// `NearPrimaryParent` samples within `radius` of parent A.
/// `MidpointOfParents` samples within `radius` of the wrap-aware midpoint
/// of A and B; falls through to `NearPrimaryParent` whenever B is
/// missing (asexual, interspecies, elite, defensive top-up).
fn placement_for(
    mode: OffspringPlacementMode,
    radius: u32,
    parent_a_pos: Option<Coord>,
    parent_b_pos: Option<Coord>,
    grid: &crate::grid::Grid,
    rng: &mut crate::rng::Rng,
) -> Coord {
    match (mode, parent_a_pos) {
        (OffspringPlacementMode::Random, _) | (_, None) => grid.find_empty_location(rng),
        (OffspringPlacementMode::NearPrimaryParent, Some(a)) => {
            grid.find_empty_location_near(a, radius, rng)
        }
        (OffspringPlacementMode::MidpointOfParents, Some(a)) => {
            let seed = match parent_b_pos {
                Some(b) => {
                    // Wrap-aware midpoint: `delta` picks the shorter
                    // path on wrapping axes, so parents straddling a
                    // seam don't get their child dropped in the
                    // unrelated grid centre.
                    let (dx, dy) = grid.delta(a, b);
                    Coord::new(a.x + (dx / 2) as i16, a.y + (dy / 2) as i16)
                }
                None => a,
            };
            grid.find_empty_location_near(seed, radius, rng)
        }
    }
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
    // small fitness deduction. `pass` is preserved — challenge admission
    // must not depend on bloat — but the adjusted score re-orders the
    // parent pool so lean genomes outrank equally-fit bloated ones.
    // Multiplying by 0.0 is a no-op when the feature is disabled (default).
    //
    // The curve is **quadratic** in `dead_norm`: a 30% dead fraction (the
    // normal operating range for healthy lineages, since most mutations
    // that wire a new connection also break an existing chain) pays only
    // 9% of `bloat_weight`, while 80% dead pays 64% and 100% pays the full
    // weight. The quadratic shape lets moderate bloat slide through as a
    // near-tie-breaker and only bites hard at extreme bloat. A linear
    // curve here punished exploration too aggressively: at weight=0.02
    // and dead_norm=0.3 the 0.006 deduction was enough to flip rankings
    // between (high-fitness, moderately-bloated) and (low-fitness, lean)
    // agents on tight-fitness challenges.
    let bloat_weight = state.config.bloat_penalty_weight;

    let evaluated: Vec<(Genome, f32, bool, f32, Coord)> = state
        .population
        .iter_alive()
        .map(|a| {
            let (pass, fitness) = state.challenges.evaluate(a, &world);
            // `dead_norm² ∈ [0, 1]`; the subtraction is bounded by
            // `bloat_weight`. With the default weight = 0 this is exactly
            // zero — no behavioural change and no float-rounding drift.
            let dead_norm = a.dead_gene_count as f32 / a.genome.len().max(1) as f32;
            let adjusted = fitness - bloat_weight * dead_norm * dead_norm;
            // Carry the agent's mutation_rate through selection so
            // adaptive lineages preserve their inherited rate, and its
            // grid location so spatial placement modes can inherit it.
            (a.genome.clone(), adjusted, pass, a.mutation_rate, a.loc)
        })
        .collect();

    let (parent_pool, parent_fitnesses, parent_positions, survivor_count) =
        select_parent_genomes(evaluated);

    let new_pop = state.config.population as usize;
    // Commit pending enable/disable changes: from this generation on, new
    // nnets are wired against the updated active sensor/action set.
    apply_feature_enables(state);
    state.sensors.commit_enabled();
    state.actions.commit_enabled();
    let wiring_cfg = state.wiring_config();

    let new_genomes = if state.config.enable_speciation {
        generate_new_genomes_speciated(
            state,
            parent_pool.as_slice(),
            parent_fitnesses.as_slice(),
            parent_positions.as_slice(),
            new_pop,
            wiring_cfg,
        )
    } else {
        generate_new_genomes(
            parent_pool.as_slice(),
            parent_positions.as_slice(),
            &state.config,
            &mut state.rng,
            new_pop,
        )
    };

    let placement_mode = state.config.offspring_placement_mode;
    let placement_radius = state.config.offspring_placement_radius;

    reset_world(state);
    state.generation += 1;

    for child in new_genomes {
        let NewChild { genome, mutation_rate, species_id, parent_a_pos, parent_b_pos } = child;
        let nnet = create_wiring(&genome, wiring_cfg);
        let dead = genome.len().saturating_sub(nnet.connection_count()) as u16;
        let loc = placement_for(
            placement_mode,
            placement_radius,
            parent_a_pos,
            parent_b_pos,
            &state.grid,
            &mut state.rng,
        );
        let id = state.population.next_id();
        let mut agent = Agent::new(id, loc, genome, nnet);
        agent.mutation_rate = mutation_rate;
        agent.species_id = species_id;
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
