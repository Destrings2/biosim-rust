use crate::agent::Agent;
use crate::genome::genome::{make_random_genome, generate_child_genome, Genome};
use crate::genome::neural_net::create_wiring;
use crate::sim_state::SimulationState;
use crate::registry::challenge::WorldMut;

/// Populate generation 0 with agents carrying random genomes, placed randomly.
pub fn initialize_generation_0(state: &mut SimulationState) {
    state.population.clear();
    state.grid.zero_fill();
    crate::barriers::create_barrier(&mut state.grid, state.config.barrier_type);
    state.reapply_user_barriers();
    state.signals.zero_fill();

    // Commit any pending sensor/action enable-disable changes before wiring.
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

/// Select survivors, reproduce, and populate the next generation. Returns survivor count.
///
/// Bootstrap fallback: if **no** agents pass the challenge, take the top 10%
/// (minimum 2) by fitness score regardless of pass/fail. Without this, hard
/// challenges (sun_tracker, location_sequence, ...) where no random agent
/// passes generation 0 would just re-randomize the population every gen and
/// never accumulate any selection pressure.
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

    let mut survivor_pool: Vec<(Genome, f32)> = evaluated.iter()
        .filter(|(_, _, p)| *p)
        .map(|(g, f, _)| (g.clone(), *f))
        .collect();

    let survivor_count = survivor_pool.len() as u32;

    if survivor_pool.is_empty() && !evaluated.is_empty() {
        // Extinction-recovery: take top 10% by fitness as "soft" parents so
        // the GA still has a gradient to climb on hard challenges.
        let mut all = evaluated.clone();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let take = (all.len() / 10).max(2);
        survivor_pool = all.into_iter().take(take).map(|(g, f, _)| (g, f)).collect();
    }

    // Sort ascending so generate_child_genome bias (higher index = fitter) works correctly
    survivor_pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let parent_genomes: Vec<Genome> = survivor_pool.into_iter().map(|(g, _)| g).collect();

    let new_pop = state.config.population as usize;
    // Commit pending enable/disable changes: from this generation on, new nnets
    // are wired against the updated active sensor/action set.
    state.sensors.commit_enabled();
    state.actions.commit_enabled();
    let wiring_cfg = state.wiring_config();
    let cfg = state.config.clone();

    let new_genomes: Vec<Genome> = if parent_genomes.is_empty() {
        (0..new_pop)
            .map(|_| make_random_genome(&cfg, &mut state.rng))
            .collect()
    } else {
        // Elitism: preserve the top 2 survivors unchanged. Cheap insurance
        // against losing the best genome to mutation, especially valuable
        // on hard challenges where good genomes are rare.
        let elite_count = 2.min(parent_genomes.len());
        let elites: Vec<Genome> = parent_genomes.iter()
            .rev() // ascending sort → fittest at end
            .take(elite_count)
            .cloned()
            .collect();

        let mut out = Vec::with_capacity(new_pop);
        out.extend(elites);
        while out.len() < new_pop {
            out.push(generate_child_genome(
                &parent_genomes,
                cfg.sexual_reproduction,
                cfg.choose_parents_by_fitness,
                cfg.point_mutation_rate,
                cfg.gene_insertion_deletion_rate,
                cfg.deletion_ratio,
                cfg.genome_max_length,
                &mut state.rng,
            ));
        }
        out
    };

    state.population.clear();
    state.grid.zero_fill();
    crate::barriers::create_barrier(&mut state.grid, cfg.barrier_type);
    state.reapply_user_barriers();
    state.signals.zero_fill();
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

    // Run on_generation_start hooks
    {
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
    }

    survivor_count
}
