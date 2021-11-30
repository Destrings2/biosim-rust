pub mod gene;
pub mod mutations;
pub mod similarity;

use std::collections::HashMap;
use gene::Gene;
use crate::population::brain::sensor_actions::{ENABLED_ACTIONS, ENABLED_SENSORS};
use crate::population::genome::gene::NEURON;

// An individual's genome is a set of Genes, see [`Gene`]. Each
// gene is equivalent to one connection in a neural net. An individual's
// neural net is derived from its set of genes
pub type Genome = Vec<Gene>;

pub fn genome_to_string(genome: &Genome) -> String {
    let mut string = String::new();
    for gene in genome {
        string.push_str(&gene.to_string());
        string.push_str(" ");
    }
    return string;
}

pub fn genome_to_hex(genome: &Genome) -> String {
    let mut string = String::new();
    for gene in genome {
        string.push_str(&gene.hex_string());
        string.push_str(":");
    }
    return string;
}

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
pub fn renumber_genome(genome: &Genome, max_number_neurons: u16) -> Genome {
    let mut new_genome = Vec::with_capacity(genome.len());
    for gene in genome.iter() {
        let mut conn: Gene = gene.clone();

        let new_source = if conn.get_source_type() == NEURON {
            conn.get_source_num() % max_number_neurons as u8
        } else {
            conn.get_source_num() % ENABLED_SENSORS.len() as u8
        };
        conn.set_source_num(new_source);

        let new_sink = if conn.get_sink_type() == NEURON {
            conn.get_sink_num() % max_number_neurons as u8
        } else {
            conn.get_sink_num() % ENABLED_ACTIONS.len() as u8
        };
        conn.set_sink_num(new_sink);
        new_genome.push(conn);
    }

    return new_genome;
}

pub struct Node {
    pub remapped_number: u8,
    pub outputs: u8,
    pub self_inputs: u8,
    pub other_inputs: u8,
}


pub fn get_connection_map_from_genome(genome: &Genome) -> HashMap<u8, Node> {
    let mut connection_map: HashMap<u8, Node> = HashMap::new();

    for gene in genome.iter() {
        // If we dont find the key, then we create the node
        // Otherwise we increment the outputs, inputs and self_inputs as appropriate
        if gene.get_sink_type() == NEURON {
            if !connection_map.contains_key(&gene.get_sink_num()) {
                connection_map.insert(gene.get_sink_num(), Node {
                    remapped_number: 0,
                    outputs: 0,
                    self_inputs: 0,
                    other_inputs: 0,
                });
            }
            let mut sink_connection = connection_map.get_mut(&gene.get_sink_num()).unwrap();

            // Increase the number of inputs
            if gene.get_source_type() == NEURON && gene.get_source_num() == gene.get_sink_num() {
                sink_connection.self_inputs += 1;
            } else {
                sink_connection.other_inputs += 1;
            }
        }
        if gene.get_source_type() == NEURON {
            if !connection_map.contains_key(&gene.get_source_num()) {
                connection_map.insert(gene.get_source_num(),Node {
                    remapped_number: 0,
                    outputs: 0,
                    self_inputs: 0,
                    other_inputs: 0,
                });
            }

            let mut source_connection = connection_map.get_mut(&gene.get_source_num()).unwrap();
            // Increase the number of outputs
            source_connection.outputs += 1;
        }
    }

    return connection_map;
}

fn remove_connections_to_neuron(genome: &mut Genome, connections: &mut HashMap<u8, Node>, neuron_num: u8) {
    genome.retain(|gene| {
        if gene.get_sink_type() == NEURON && gene.get_sink_num() == neuron_num {
            if gene.get_source_type() == NEURON {
                match connections.get_mut(&gene.get_source_num()) {
                    Some(node) => node.outputs -= 1,
                    None => panic!("Could not find node in connections map"),
                }
            }
            return false;
        }
        true
    });
}

pub fn remove_useless_neurons_from_genome(genome: &mut Genome, connection_map: &mut HashMap<u8, Node>) {
    let mut has_useless_neurons = true;

    while has_useless_neurons {
        has_useless_neurons = false;
        let mut keys_to_remove = Vec::new();
        for (key, value) in connection_map.iter() {
            // We're looking for neurons with zero outputs, or neurons that feed itself
            // and nobody else
            if value.outputs == value.self_inputs {
                has_useless_neurons = true;
                keys_to_remove.push(*key);
            }
        }

        for key in keys_to_remove {
            remove_connections_to_neuron(genome, connection_map, key);
            connection_map.remove(&key);
        }
    }
}