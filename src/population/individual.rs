use crate::Parameters;
use crate::population::brain::NeuralNet;
use crate::population::brain::sensor_actions::ENABLED_SENSORS;
use crate::population::brain::sensor_actions::sensor_implementation::get_sensor_dispatch;
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

    // pub fn feed_forward(&self, simulation: &Simulation) -> f32 {
    //
    // }
}