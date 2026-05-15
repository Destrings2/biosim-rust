//! Per-generation statistics.
//!
//! [`EpochStats`] collects generation number, survivor count, population size,
//! and genetic diversity. `survival_rate()` is `survivors / population`.
//!
//! `genetic_diversity` samples up to 1000 random pairs from the genome pool
//! and averages their pairwise similarity (Jaro-Winkler, Hamming-bits, or
//! Hamming-bytes per `SimConfig.genome_comparison_method`). Sampling avoids
//! the O(N²) cost of exhaustive pair enumeration at large population sizes.

use crate::genome::ops::{genetic_diversity, Genome};
use crate::sim_state::SimulationState;

/// Per-generation statistics collected by [`collect_epoch_stats`].
pub struct EpochStats {
    /// Generation index (0-based).
    pub generation: u32,
    /// Number of agents that passed the active challenge(s).
    pub survivors: u32,
    /// Total population size (from `config.population`).
    pub population: u32,
    /// Genetic diversity in [0.0, 1.0], where 1.0 means all genomes are
    /// maximally different. Computed from up to 1000 random pairwise comparisons.
    pub diversity: f32,
}

impl EpochStats {
    /// Fraction of the population that survived: `survivors / population`.
    pub fn survival_rate(&self) -> f32 {
        if self.population == 0 {
            0.0
        } else {
            self.survivors as f32 / self.population as f32
        }
    }
}

/// Collect statistics for the just-completed generation.
///
/// `survivors` is the count returned by [`spawn_new_generation`](crate::spawn_new_generation).
/// Diversity sampling uses the comparison method in `state.config.genome_comparison_method`.
pub fn collect_epoch_stats(state: &mut SimulationState, survivors: u32) -> EpochStats {
    let genomes: Vec<&Genome> = state.population.iter_alive().map(|a| &a.genome).collect();

    let diversity = if genomes.len() >= 2 {
        let method = state.config.genome_comparison_method;
        genetic_diversity(&genomes, method, &mut state.rng)
    } else {
        0.0
    };

    EpochStats {
        generation: state.generation,
        survivors,
        population: state.config.population,
        diversity,
    }
}

/// Print a one-line summary of `stats` to stdout.
pub fn print_epoch_stats(stats: &EpochStats) {
    println!(
        "Gen {:>4}  survivors: {:>5}/{:<5}  ({:.1}%)  diversity: {:.4}",
        stats.generation,
        stats.survivors,
        stats.population,
        stats.survival_rate() * 100.0,
        stats.diversity
    );
}

/// Print the hexadecimal genome of the first `count` alive agents to stdout.
pub fn display_sample_genomes(state: &SimulationState, count: usize) {
    for agent in state.population.iter_alive().take(count) {
        let hex: String =
            agent.genome.iter().map(|g| format!("{:08x}", g.0)).collect::<Vec<_>>().join(" ");
        println!("  agent {:>4}: {}", agent.id, hex);
    }
}
