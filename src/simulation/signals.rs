use crate::simulation::grid::Grid;
use crate::simulation::types::Coord;

pub struct Signals {
    layers: Vec<Grid>
}
//TODO: Implement Signals
impl Signals {
    pub fn new(num_layers: u16, width: u16, height: u16) -> Signals {
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(Grid::new(width, height));
        }
        return Signals { layers };
    }

    pub fn get(&self, layer: usize, location: Coord) -> u16 {
        self.layers[layer].at_coord(location)
    }

    pub fn set(&mut self, layer: usize, location: Coord, value: u16) {
        self.layers[layer].set_at_coord(location, value);
    }
}