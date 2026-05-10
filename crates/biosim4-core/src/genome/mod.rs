//! Genome representation and operations.
//!
//! Re-exports the three submodules: [`gene`] (bit encoding), [`ops`]
//! (mutation, crossover, diversity), and [`neural_net`] (compilation and
//! feed-forward). Most callers import from this module directly.

pub mod gene;
pub mod ops;
pub mod neural_net;

pub use gene::Gene;
pub use ops::{Genome, ReproductionParams, make_random_genome, make_random_gene,
              apply_point_mutations, random_insert_deletion, generate_child_genome,
              genome_similarity, genetic_diversity};
pub use neural_net::{NeuralNet, Neuron, WiringConfig, feed_forward};
