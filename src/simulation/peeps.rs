use crate::population::individual::Individual;
use crate::simulation::grid::{EMPTY_CELL, Grid};
use crate::simulation::signals::Signals;
use crate::simulation::types::{Coord, Dir};
use crate::simulation::world::World;

pub struct Peeps {
    pub world: World,
    pub signals: Signals,
    pub population: Vec<Individual>,
    pub death_queue: Vec<u16>,
    pub move_queue: Vec<(u16, Coord)>
}

impl Peeps {
    pub fn queue_for_death(&mut self, id: u16) {
        self.death_queue.push(id);
    }

    pub fn drain_death_queue(&mut self) {
        for id in self.death_queue.drain(..) {
            let individual: &mut Individual = self.population.get_mut(id as usize).unwrap();
            individual.alive = false;
        }
    }

    pub fn queue_for_move(&mut self, move_data: (u16, Coord)) {
        self.move_queue.push(move_data);
    }

    pub fn drain_move_queue(&mut self) {
        for (id, location) in self.move_queue.drain(..) {
            let individual: &mut Individual = self.population.get_mut(id as usize).unwrap();
            let move_dir: Dir = (location - individual.location).into();
            if self.world.is_empty_at(location) {
                self.world.set_at_coord(individual.location, EMPTY_CELL);
                self.world.set_at_coord(location, individual.index as u16);
                individual.location = location;
                individual.last_move_direction = move_dir;
            }
        }
    }

}