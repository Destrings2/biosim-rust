use crate::Parameters;
use crate::population::brain::NeuralNet;
use crate::population::genome::Genome;
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
}

impl Individual {
    pub fn new(index: usize, location: Coord, genome: Genome, p: &Parameters) {
        let mut individual = Individual {
            alive: true,
            index,
            location,
            birth_location: location,
            age: 0,
            responsiveness: 0.5,
            oscillation_period: 34,
            long_probe_distance: p.long_probe_distance,
            last_move_direction: Dir::random(),
            challenge_bits: 0,
            neural_net: NeuralNet::new(&genome),
            genome
        };
    }
}