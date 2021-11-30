use rand::Rng;
use crate::simulation::grid::Grid;
use crate::simulation::types::Coord;

pub struct World {
    grid: Grid,
    barrier_locations: Vec<Coord>,
    barrier_centers: Vec<Coord>,
}

const BARRIER_CELL: u16 = 0xffff;

impl World {
    pub fn find_random_empty_location(&self) -> Coord {
        let mut rng = rand::thread_rng();
        let mut location = Coord(rng.gen_range(0..=self.width as i16), rng.gen_range(0..=self.height as i16));
        while !self.is_empty_at(location) {
            location = Coord(rng.gen_range(0..=self.width as i16), rng.gen_range(0..=self.height as i16));
        }
        return location;
    }

    #[inline]
    pub fn is_barrier_at(&self, location: Coord) -> bool {
        return self.grid.at_coord(location) == BARRIER_CELL;
    }

    #[inline]
    pub fn is_occupied_at(&self, location: Coord) -> bool {
        return !(self.grid.is_empty_at(location) || self.is_barrier_at(location));
    }


    //TODO: Implement the createBarrier in a better way
}

impl std::ops::Deref for World {
    type Target = Grid;

    #[inline]
    fn deref(&self) -> &Grid {
        return &self.grid;
    }
}