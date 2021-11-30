use crate::population::individual::Individual;
use crate::simulation::grid::Grid;
use crate::simulation::signals::Signals;
use crate::simulation::world::World;

pub struct Peeps {
    pub world: World,
    pub signals: Signals,
    pub population: Vec<Individual>,
    pub death_queue: Vec<u16>,
    pub move_queue: Vec<u16>
}

impl Peeps {
    
}