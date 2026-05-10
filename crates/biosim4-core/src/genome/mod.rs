pub mod gene;
pub mod genome;
pub mod neural_net;

pub use gene::Gene;
pub use genome::{Genome, make_random_genome, make_random_gene, apply_point_mutations,
                  random_insert_deletion, generate_child_genome, genome_similarity,
                  genetic_diversity};
pub use neural_net::{NeuralNet, Neuron, WiringConfig, feed_forward};
