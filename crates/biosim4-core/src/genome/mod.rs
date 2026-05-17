//! Genome representation and operations.
//!
//! Re-exports the three submodules: [`gene`] (bit encoding), [`ops`]
//! (mutation, crossover, diversity), and [`neural_net`] (compilation and
//! feed-forward). Most callers import from this module directly.

pub mod gene;
pub mod neural_net;
pub mod ops;
pub mod speciation;

pub use gene::Gene;
pub use neural_net::{feed_forward, NeuralNet, Neuron, WiringConfig};
pub use ops::{
    apply_point_mutations, generate_child_genome, genetic_diversity, genome_similarity,
    make_random_gene, make_random_genome, random_insert_deletion, Genome, ReproductionParams,
};
pub use speciation::{Species, SpeciesId, SpeciationState};
