//! First-class food layer for the energy system.
//!
//! Each grid cell holds a food value in [0.0, 1.0]. Agents absorb food
//! automatically by standing on a cell. Food regenerates at a configurable
//! rate each step and is re-scattered at each generation boundary.

use crate::grid::{visit_neighborhood, Grid};
use crate::rng::Rng;
use crate::types::{Coord, Dir};

/// Per-cell food values in [0.0, 1.0]. Column-major layout: `cells[x][y]`.
pub struct FoodLayer {
    cells: Vec<Vec<f32>>,
    pub size_x: u16,
    pub size_y: u16,
}

impl FoodLayer {
    /// Create a new food layer. Cells are initialized to 0; call `randomize`
    /// or `zero_fill` after construction to set initial state.
    pub fn new(size_x: u16, size_y: u16) -> Self {
        let cells = vec![vec![0.0f32; size_y as usize]; size_x as usize];
        Self { cells, size_x, size_y }
    }

    pub fn get(&self, loc: Coord) -> f32 {
        self.cells[loc.x as usize][loc.y as usize]
    }

    pub fn set(&mut self, loc: Coord, v: f32) {
        self.cells[loc.x as usize][loc.y as usize] = v.clamp(0.0, 1.0);
    }

    pub fn zero_fill(&mut self) {
        for col in self.cells.iter_mut() {
            col.fill(0.0);
        }
    }

    /// Scatter food randomly. Each non-barrier cell gets food=1.0 with
    /// probability `density`. Barrier cells are skipped.
    pub fn randomize(&mut self, density: f32, grid: &Grid, rng: &mut Rng) {
        for x in 0..self.size_x as usize {
            for y in 0..self.size_y as usize {
                let loc = Coord::new(x as i16, y as i16);
                if grid.is_barrier_at(loc) {
                    self.cells[x][y] = 0.0;
                } else {
                    self.cells[x][y] = if rng.gen_f32() < density { 1.0 } else { 0.0 };
                }
            }
        }
    }

    /// Add `rate` to every non-barrier cell, saturating at 1.0.
    pub fn regenerate(&mut self, rate: f32, grid: &Grid) {
        for x in 0..self.size_x as usize {
            for y in 0..self.size_y as usize {
                let loc = Coord::new(x as i16, y as i16);
                if !grid.is_barrier_at(loc) {
                    self.cells[x][y] = (self.cells[x][y] + rate).min(1.0);
                }
            }
        }
    }

    /// Mean food density in a neighborhood, normalized [0.0, 1.0].
    pub fn get_density(&self, center: Coord, radius: f32, grid: &Grid) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0u32;
        visit_neighborhood(grid, center, radius, |loc| {
            sum += self.get(loc);
            count += 1;
        });
        if count == 0 { return 0.0; }
        (sum / count as f32).clamp(0.0, 1.0)
    }

    /// Weighted food density in a half-plane defined by `dir` — higher values
    /// mean more food in that direction. Returns [0.0, 1.0].
    pub fn get_density_fwd(&self, center: Coord, dir: Dir, radius: f32, grid: &Grid) -> f32 {
        let mut fwd_sum = 0.0f32;
        let mut bwd_sum = 0.0f32;
        visit_neighborhood(grid, center, radius, |loc| {
            if loc == center { return; }
            let offset = loc - center;
            let mag = self.get(loc);
            let sameness = offset.ray_sameness_dir(dir);
            if sameness >= 0.0 { fwd_sum += mag * sameness; }
            else               { bwd_sum += mag * (-sameness); }
        });
        let total = fwd_sum + bwd_sum;
        if total == 0.0 { return 0.5; }
        (fwd_sum / total).clamp(0.0, 1.0)
    }
}
