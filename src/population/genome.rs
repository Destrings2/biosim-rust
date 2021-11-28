pub mod gene;

use std::collections::HashMap;
use gene::Gene;
use crate::{Parameters};
use crate::population::brain::NeuralNet;
use crate::population::brain::sensor_actions::{ENABLED_ACTIONS, ENABLED_SENSORS};
use crate::population::genome::gene::NEURON;

// An individual's genome is a set of Genes, see [`Gene`]. Each
// gene is equivalent to one connection in a neural net. An individual's
// neural net is derived from its set of genes
pub type Genome = Vec<Gene>;

// Returns by value a single genome with random genes.
pub fn make_random_genome(num_genes: usize) -> Genome {
    let mut genome = Vec::with_capacity(num_genes);
    for _ in 0..num_genes {
        genome.push(Gene::make_random_gene());
    }
    return genome;
}

// Renumbers the genome to the range 0..p.max_number_neurons so that the wiring can be made
// to create the neural net.
pub fn renumber_genome(genome: &Genome, p: &Parameters) -> Genome {
    let mut new_genome = Vec::with_capacity(genome.len());
    for gene in genome.iter() {
        let mut conn: Gene = gene.clone();

        let new_source = if conn.get_source_type() == NEURON {
            conn.get_source_num() % p.max_number_neurons as u8
        } else {
            conn.get_source_num() % ENABLED_SENSORS.len() as u8
        };
        conn.set_source_num(new_source);

        let new_sink = if conn.get_sink_type() == NEURON {
            conn.get_sink_num() % p.max_number_neurons as u8
        } else {
            conn.get_sink_num() % ENABLED_ACTIONS.len() as u8
        };
        conn.set_sink_num(new_sink);
        new_genome.push(conn);
    }

    return new_genome;
}