//! 2D arena grid.
//!
//! # Cell encoding
//!
//! Each cell stores one of four values:
//! - `EMPTY = 0` — unoccupied.
//! - `BARRIER = 0xFFFF_FFFF` — static, blocking obstacle.
//! - `KILL_BARRIER = 0xFFFF_FFFE` — hazard cell. Movement into it kills
//!   the moving agent instead of blocking the move. See
//!   `Population::drain_move_queue`.
//! - Agent ID (1..population) — occupied by a live agent.
//!
//! Zero maps to `EMPTY` (not an agent), which is why `INVALID_AGENT = 0` in
//! `population`. The sentinel `0xFFFF_FFFF` is safely distinct from any
//! plausible population size.
//!
//! # Layout
//!
//! Cells are stored as a single contiguous `Vec<u32>` in row-major order
//! (`cells[y * size_x + x]`). Sensor neighborhood scans walk in row-major
//! order, so this layout keeps adjacent cells on the same cache line.
//!
//! # `visit_neighborhood`
//!
//! Iterates all in-bounds cells within a circular radius of a center cell.
//! Uses `dx² + dy² ≤ radius²` with an explicit per-cell check to guard
//! against rounding artifacts from the `floor` on `dy_max`.
//!
//! `find_empty_location` spin-loops until a random empty cell is found.
//! The caller must ensure population < grid area to avoid an infinite loop.

use crate::types::Coord;

/// Grid cell value indicating an unoccupied cell.
pub const EMPTY: u32 = 0;
/// Grid cell value indicating a static impassable wall.
pub const BARRIER: u32 = 0xFFFF_FFFF;
/// User-painted hazard cell. Agents attempting to move into it die rather
/// than being blocked.
pub const KILL_BARRIER: u32 = 0xFFFF_FFFE;

/// 2D arena grid. Each cell stores EMPTY, BARRIER, or an agent ID (1..population).
/// Flat row-major storage: `cells[y * size_x + x]`.
pub struct Grid {
    pub size_x: u16,
    pub size_y: u16,
    cells: Vec<u32>,
    pub barrier_locations: Vec<Coord>,
    pub barrier_centers: Vec<Coord>,
}

impl Grid {
    pub fn new(size_x: u16, size_y: u16) -> Self {
        let cells = vec![EMPTY; size_x as usize * size_y as usize];
        Self { size_x, size_y, cells, barrier_locations: vec![], barrier_centers: vec![] }
    }

    #[inline]
    fn idx(&self, loc: Coord) -> usize {
        (loc.y as usize) * (self.size_x as usize) + (loc.x as usize)
    }

    pub fn zero_fill(&mut self) {
        self.cells.fill(EMPTY);
    }

    #[inline]
    pub fn at(&self, loc: Coord) -> u32 {
        self.cells[self.idx(loc)]
    }

    #[inline]
    pub fn set(&mut self, loc: Coord, val: u32) {
        let i = self.idx(loc);
        self.cells[i] = val;
    }

    pub fn is_in_bounds(&self, loc: Coord) -> bool {
        loc.x >= 0 && loc.y >= 0 && (loc.x as u16) < self.size_x && (loc.y as u16) < self.size_y
    }

    pub fn is_border(&self, loc: Coord) -> bool {
        loc.x == 0
            || loc.y == 0
            || loc.x as u16 == self.size_x - 1
            || loc.y as u16 == self.size_y - 1
    }

    pub fn is_empty_at(&self, loc: Coord) -> bool {
        self.is_in_bounds(loc) && self.at(loc) == EMPTY
    }
    pub fn is_barrier_at(&self, loc: Coord) -> bool {
        self.is_in_bounds(loc) && self.at(loc) == BARRIER
    }
    pub fn is_kill_barrier_at(&self, loc: Coord) -> bool {
        self.is_in_bounds(loc) && self.at(loc) == KILL_BARRIER
    }
    /// True if the cell blocks movement (regular wall or kill barrier).
    /// Use this to test "can an agent move here?" — kill barriers are
    /// drained specially in `drain_move_queue` so the agent doesn't end
    /// up on the cell.
    pub fn is_blocking_at(&self, loc: Coord) -> bool {
        let v = if self.is_in_bounds(loc) {
            self.at(loc)
        } else {
            return false;
        };
        v == BARRIER || v == KILL_BARRIER
    }
    pub fn is_occupied_at(&self, loc: Coord) -> bool {
        let v = if self.is_in_bounds(loc) {
            self.at(loc)
        } else {
            return false;
        };
        v != EMPTY && v != BARRIER && v != KILL_BARRIER
    }

    /// Find a random empty location. Spin-loops — caller must ensure population < grid area.
    pub fn find_empty_location(&self, rng: &mut crate::rng::Rng) -> Coord {
        loop {
            let x = rng.gen_range_u32(0, self.size_x as u32) as i16;
            let y = rng.gen_range_u32(0, self.size_y as u32) as i16;
            let loc = Coord::new(x, y);
            if self.is_empty_at(loc) {
                return loc;
            }
        }
    }
}

/// Visit all valid grid locations within a circular radius of `center`.
/// Calls `f` for each coordinate within the circle that is in bounds.
pub fn visit_neighborhood(grid: &Grid, center: Coord, radius: f32, mut f: impl FnMut(Coord)) {
    let r = radius.ceil() as i16;
    let r2 = radius * radius;
    for dx in -r..=r {
        let dx_sq = (dx as f32).powi(2);
        if dx_sq > r2 {
            continue;
        }
        let dy_max = ((r2 - dx_sq).max(0.0).sqrt()).floor() as i16;
        for dy in -dy_max..=dy_max {
            // Explicit distance check — guards against rounding edge cases
            if dx_sq + (dy as f32).powi(2) > r2 {
                continue;
            }
            let loc = Coord::new(center.x + dx, center.y + dy);
            if grid.is_in_bounds(loc) {
                f(loc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_fill() {
        let mut g = Grid::new(4, 4);
        g.set(Coord::new(1, 1), 5);
        g.zero_fill();
        assert_eq!(g.at(Coord::new(1, 1)), EMPTY);
    }

    #[test]
    fn visit_neighborhood_center_only() {
        let g = Grid::new(10, 10);
        let mut count = 0;
        visit_neighborhood(&g, Coord::new(5, 5), 0.5, |_| count += 1);
        assert_eq!(count, 1);
    }

    #[test]
    fn visit_neighborhood_radius1() {
        let g = Grid::new(10, 10);
        let mut count = 0;
        visit_neighborhood(&g, Coord::new(5, 5), 1.0, |_| count += 1);
        // 3x3 = 9, but only cells where dx²+dy² <= 1 → center + 4 cardinal = 5
        assert_eq!(count, 5);
    }
}
