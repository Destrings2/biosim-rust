use crate::Parameters;
use crate::population::brain::NeuralNet;
use crate::population::brain::sensor_actions::{ENABLED_ACTIONS, ENABLED_SENSORS};
use crate::population::brain::sensor_actions::sensor_implementation::get_sensor_dispatch;
use crate::population::genome::gene::{ACTION, SENSOR};
use crate::population::genome::Genome;
use crate::simulation::grid::Grid;
use crate::simulation::simulation::Simulation;
use crate::simulation::types::{Coord, Dir};

pub struct Individual {
    pub alive: bool,
    pub index: usize, //
    pub location: Coord,
    pub birth_location: Coord,
    pub age: u32,
    pub responsiveness: f32,
    pub oscillation_period: u32,
    pub long_probe_distance: u32,
    pub last_move_direction: Dir,
    pub challenge_bits: u32,
    pub neural_net: NeuralNet,
    pub genome: Genome,
    pub num_neurons: u16
}

impl Individual {
    pub fn new(index: usize, location: Coord, genome: Genome, p: &Parameters) {
        Individual {
            alive: true,
            index,
            location,
            birth_location: location,
            age: 0,
            num_neurons: p.max_number_neurons,
            responsiveness: 0.5,
            oscillation_period: 34,
            long_probe_distance: p.long_probe_distance,
            last_move_direction: Dir::random(),
            challenge_bits: 0,
            neural_net: NeuralNet::new(&genome, p.max_number_neurons),
            genome
        };
    }

    pub fn get_sensor_value(&self, source_num: u8, simulation: &Simulation) -> f32 {
        let sensor = &ENABLED_SENSORS[source_num as usize];
        let sensor_function = get_sensor_dispatch(sensor);
        return sensor_function(&self, &simulation.grid, simulation.simulation_step);
    }

    pub fn feed_forward(&mut self, simulation: &Simulation) -> [f32; ENABLED_ACTIONS.len()] {
        // This container is used to return values for all the action outputs. This array
        // contains one value per action neuron, which is the sum of all its weighted
        // input connections. The sum has an arbitrary range.
        let mut output = [0.0; ENABLED_ACTIONS.len()];

        // Weighted inputs to each neuron are summed in neuronAccumulators
        let mut neuron_accumulators = Vec::with_capacity(self.neural_net.neurons.len());

        // Connections were ordered at birth so that all connections to neurons get
        // processed here before any connections to actions. As soon as we encounter the
        // first connection to an action, we'll pass all the neuron input accumulators
        // through a transfer function and update the neuron outputs in the individual,
        // except for undriven neurons which act as bias feeds and don't change. The
        // transfer function will leave each neuron's output in the range -1.0..1.0.
        let mut neuron_outputs_computed = false;
        for gene in self.neural_net.connections.iter() {
            if gene.get_sink_type() == ACTION && !neuron_outputs_computed {
                // We've handled all the connections from sensors and now we are about to
                // start on the connections to the action outputs, so now it's time to
                // update and latch all the neuron outputs to their proper range (-1.0..1.0)
                for (neuron_index, neuron) in self.neural_net.neurons.iter_mut().enumerate() {
                    if neuron.driven {
                        let neuron_output = neuron_accumulators[neuron_index];
                        let neuron_output = f32::tanh(neuron_output);
                        neuron.output = neuron_output;
                    }
                }
                neuron_outputs_computed = true;
            }

            // Obtain the connection's input value from a sensor neuron or other neuron
            // The values are summed for now, later passed through a transfer function
            let mut input_value = 0.0f32;
            if gene.get_source_type() == SENSOR {
                input_value = self.get_sensor_value(gene.get_source_num(), simulation);
            } else {
                let source_neuron = &self.neural_net.neurons[gene.get_source_num() as usize];
                input_value = source_neuron.output;
            }

            // Weight the connection's value and add to neuron accumulator or action accumulator.
            // The action and neuron accumulators will therefore contain +- float values in
            // an arbitrary range.
            if gene.get_sink_type() == ACTION {
                output[gene.get_sink_num() as usize] += input_value * gene.weight_as_float();
            } else {
                *neuron_accumulators.get_mut(gene.get_sink_num() as usize).unwrap() +=
                    input_value * gene.weight_as_float();
            }
        }

        return output;
    }
}