pub mod sensor_actions;

use std::cell::RefCell;
use std::collections::HashMap;
use crate::population::brain::sensor_actions::{ENABLED_ACTIONS, ENABLED_SENSORS};
use crate::population::genome::{Genome, get_connection_map_from_genome, Node, remove_useless_neurons_from_genome, renumber_genome};
use crate::population::genome::gene::{ACTION, NEURON, SENSOR};

pub struct Neuron {
    pub output: f32,
    pub driven: bool
}

impl Neuron {
    pub const fn initial_neuron_output() -> f32 {
        return 0.5;
    }
}

/// An individual's "brain" is a neural net specified by a set
/// of Genes where each Gene specifies one connection in the neural net (see
/// Genome comments above). Each neuron has a single output which is
/// connected to a set of sinks where each sink is either an action output
/// or another neuron. Each neuron has a set of input sources where each
/// source is either a sensor or another neuron. There is no concept of
/// layers in the net: it's a free-for-all topology with forward, backwards,
/// and sideways connection allowed. Weighted connections are allowed
/// directly from any source to any action.
///
/// Currently, the genome does not specify the activation function used in
/// the neurons. (May be hardcoded to std::tanh() !!!)
///
/// When the input is a sensor, the input value to the sink is the raw
/// sensor value of type float and depends on the sensor. If the output
/// is an action, the source's output value is interpreted by the action
/// node and whether the action occurs or not depends on the action's
/// implementation.
///
/// In the genome, neurons are identified by 15-bit unsigned indices,
/// which are reinterpreted as values in the range 0..p.genomeMaxLength-1
/// by taking the 15-bit index modulo the max number of allowed neurons.
/// In the neural net, the neurons that end up connected get new indices
/// assigned sequentially starting at 0.
pub struct NeuralNet {
    pub connections: Genome,
    pub neurons: Vec<RefCell<Neuron>>,
}

impl NeuralNet {
    pub fn new(genome: &Genome, max_number_neurons: u16) -> NeuralNet {
        let mut renumbered_genome = renumber_genome(&genome, max_number_neurons);
        let mut connection_map: HashMap<u8, Node> = get_connection_map_from_genome(&renumbered_genome);

        let mut neural_connections: Genome = vec![];
        let mut neural_neurons: Vec<RefCell<Neuron>> = vec![];

        remove_useless_neurons_from_genome(&mut renumbered_genome, &mut connection_map);

        // The neurons map now has all the referenced neurons, their neuron numbers, and
        // the number of outputs for each neuron. Now we'll renumber the connections
        // starting at zero.
        assert!(connection_map.len() <= max_number_neurons as usize);
        let mut counter: u8 = 0;
        for (_, value) in connection_map.iter_mut() {
            assert_ne!(value.outputs, 0);
            value.remapped_number = counter;
            counter += 1;
        }

        // First, the connections from sensor or neuron to a neuron
        for gene in renumbered_genome.iter() {
            if gene.get_sink_type() == NEURON {
                neural_connections.push(*gene);
                let connection = neural_connections.last_mut().unwrap();
                connection.set_sink_num(connection_map[&connection.get_sink_num()].remapped_number);

                if connection.get_source_type() == NEURON {
                    connection.set_source_num(connection_map[&connection.get_source_num()].remapped_number);
                }
            }
        }

        // Last, the connections from sensor or neuron to an action
        for gene in renumbered_genome.iter() {
            if gene.get_sink_type() == ACTION {
                neural_connections.push(*gene);
                let connection = neural_connections.last_mut().unwrap();

                if connection.get_source_type() == NEURON {
                    connection.set_source_num(connection_map[&connection.get_source_num()].remapped_number);
                }
            }
        }

        for key in connection_map.keys() {
            neural_neurons.push(RefCell::new(Neuron {
                output: Neuron::initial_neuron_output(),
                driven: (connection_map.get(key).unwrap().other_inputs != 0)
            }));
        }

        return NeuralNet {
            connections: neural_connections,
            neurons: neural_neurons
        }
    }

    pub fn to_graph_string(&self) -> String {
        let mut graph_string = String::new();
        for connection in &self.connections {
            if connection.get_source_type() == SENSOR {
                graph_string.push_str(&ENABLED_SENSORS[connection.get_source_num() as usize].to_string());
            } else {
                graph_string.push_str(&format!("N{}", connection.get_source_num()));
            }

            graph_string.push_str(" ");

            if connection.get_sink_type() == ACTION {
                graph_string.push_str(&ENABLED_ACTIONS[connection.get_sink_num() as usize].to_string());
            } else {
                graph_string.push_str(&format!("N{}", connection.get_sink_num()));
            }
            graph_string.push_str("\n");
        }

        return graph_string;
    }

    pub fn to_mathematica_string(&self) -> String {
        let mut graph_string = String::new();
        graph_string.push_str("{\"");
        for connection in &self.connections {
            if graph_string.len() > 2 {graph_string.push_str(",");
            graph_string.push_str("\"");}
            if connection.get_source_type() == SENSOR {
                graph_string.push_str(&ENABLED_SENSORS[connection.get_source_num() as usize].to_string());
            } else {
                graph_string.push_str(&format!("N{}", connection.get_source_num()));
            }

            graph_string.push_str("\"\\[DirectedEdge]\"");

            if connection.get_sink_type() == ACTION {
                graph_string.push_str(&ENABLED_ACTIONS[connection.get_sink_num() as usize].to_string());
            } else {
                graph_string.push_str(&format!("N{}", connection.get_sink_num()));
            }
            graph_string.push_str("\"");
        }
        graph_string.push_str("}");
        return graph_string;
    }
}
