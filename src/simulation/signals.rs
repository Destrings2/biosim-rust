use crate::simulation::grid::Grid;
use crate::simulation::types::Coord;

pub struct Signals {
    layers: Vec<Grid>
}
//TODO: Implement Signals
impl Signals {
    pub fn get(&self, layer: usize, location: Coord) -> u16 {
        self.layers[layer].at_coord(location)
    }

    pub fn set(&mut self, layer: usize, location: Coord, value: u16) {
        self.layers[layer].set_at_coord(location, value);
    }
}