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
}