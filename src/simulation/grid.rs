use std::ops::Index;
use rand::Rng;
use crate::simulation::types::Coord;

//<editor-fold desc="Column implementation">
struct Column {
    data: Vec<u16>,
}

impl Column {
    fn new(size: usize) -> Column {
        Column {
            data: vec![0; size],
        }
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

impl std::ops::Index<usize> for Column {
    type Output = u16;

    fn index(&self, index: usize) -> &u16 {
        &self.data[index]
    }
}

impl std::ops::IndexMut<usize> for Column {
    fn index_mut(&mut self, index: usize) -> &mut u16 {
        &mut self.data[index]
    }
}

//</editor-fold>

const EMPTY_CELL: u16 = 0;

pub struct Grid {
    pub width: u16,
    pub height: u16,
    data: Vec<Column>,
}

impl Grid {
    pub fn new(width: u16, height: u16) -> Grid {
        let mut data = Vec::with_capacity(width as usize);
        for _ in 0..width {
            data.push(Column::new(height as usize));
        }
        return Grid { width, height, data };
    }

    pub fn at(&self, x: u16, y: u16) -> u16 {
        self.data[x as usize][y as usize]
    }

    pub fn at_coord(&self, location: Coord) -> u16 {
        self.data[location.0 as usize][location.1 as usize]
    }

    pub fn set_at(&mut self, x: u16, y: u16, value: u16) {
        self.data[x as usize][y as usize] = value;
    }

    pub fn set_at_coord(&mut self, location: Coord, value: u16) {
        self.data[location.0 as usize][location.1 as usize] = value;
    }

    #[inline]
    pub fn is_in_bounds(&self, location: Coord) -> bool {
        return location.0 < self.width as i16 && location.1 < self.height as i16;
    }

    #[inline]
    pub fn is_empty_at(&self, location: Coord) -> bool {
        return self.at_coord(location) == EMPTY_CELL;
    }

    #[inline]
    pub fn is_border_at(&self, location: Coord) -> bool {
        return location.0 == 0 || location.0 == self.width as i16 - 1
            || location.1 == 0 || location.1 == self.height as i16 - 1;
    }

    pub fn apply_neighborhood_to_f(&self, location: Coord, radius: i16, f: &mut dyn FnMut(&Grid, Coord)) {
        // Visits the Von Neumann neighborhood of the given location.
        // Then calls the given function on each of the visited locations.
        let mut x = location.0 - radius;
        while x <= location.0 + radius {
            let mut y = location.1 - radius;
            while y <= location.1 + radius {
                let neighbor = Coord(x, y);
                if self.is_in_bounds(neighbor) {
                    f(&self, neighbor);
                }
                y += 1;
            }
            x += 1;
        }
    }
}