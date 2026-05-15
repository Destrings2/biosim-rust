use crate::agent::AgentId;
use crate::grid::{self, Grid};
use crate::programmable::ProgramContext;
use crate::types::Coord;

pub mod sensors {
    use super::*;

    /// Validates path clearance against static barriers.
    ///
    /// Allows line of sight through other agents and programmable entities.
    #[inline]
    pub fn has_line_of_sight(grid: &Grid, start: Coord, end: Coord) -> bool {
        let mut x = start.x;
        let mut y = start.y;
        let x2 = end.x;
        let y2 = end.y;

        let dx = (x2 - x).abs();
        let dy = -(y2 - y).abs();
        let sx = if x < x2 { 1 } else { -1 };
        let sy = if y < y2 { 1 } else { -1 };
        let mut err = dx + dy;

        loop {
            // Ignore endpoints to avoid self-collision or blocking on the target.
            if x != start.x || y != start.y {
                if x == x2 && y == y2 {
                    break;
                }

                let current_loc = Coord::new(x, y);

                if grid.is_blocking_at(current_loc) {
                    return false;
                }
            } else if x == x2 && y == y2 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }

        true
    }

    /// Locates the nearest agent visible within `max_range`.
    ///
    /// Prioritizes closer agents to minimize unnecessary line-of-sight checks.
    pub fn nearest_peep_in_los(
        ctx: &ProgramContext,
        center: Coord,
        max_range: u16,
    ) -> Option<(Coord, AgentId)> {
        let grid = ctx.world.grid;
        let max_r = max_range as i16;
        let max_r_sq = (max_range as u32) * (max_range as u32);

        for ring in 1..=max_r {
            let mut best_target: Option<(Coord, AgentId)> = None;
            let mut best_dist_sq = u32::MAX;

            for dy in -ring..=ring {
                for dx in -ring..=ring {
                    if dx.abs() != ring && dy.abs() != ring {
                        continue;
                    }

                    let d_sq = ((dx as i32) * (dx as i32) + (dy as i32) * (dy as i32)) as u32;
                    if d_sq > max_r_sq {
                        continue;
                    }

                    let loc = Coord::new(center.x + dx, center.y + dy);
                    if !grid.is_in_bounds(loc) {
                        continue;
                    }

                    let cell = grid.at(loc);
                    if let grid::CellKind::Agent(agent_id) = grid::cell_kind(cell) {
                        if has_line_of_sight(grid, center, loc) {
                            if d_sq < best_dist_sq {
                                best_dist_sq = d_sq;
                                best_target = Some((loc, agent_id));
                            }
                        }
                    }
                }
            }

            // Short-circuit when the next ring is guaranteed to be further than the current best.
            if let Some(target) = best_target {
                let next_ring_min_dist_sq = ((ring + 1) as u32) * ((ring + 1) as u32);
                if best_dist_sq <= next_ring_min_dist_sq {
                    return Some(target);
                } else if ring == max_r {
                    return Some(target);
                }
            }
        }

        None
    }
}

pub mod actions {
    use super::*;

    /// Determines the next step coordinate for moving from the current location towards the target.
    ///
    /// Returns `None` when the current location matches the target.
    #[inline]
    pub fn move_towards(from: Coord, to: Coord) -> Option<Coord> {
        let dx = (to.x - from.x).signum();
        let dy = (to.y - from.y).signum();

        if dx == 0 && dy == 0 {
            None
        } else {
            Some(Coord::new(from.x + dx, from.y + dy))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid, BARRIER, EMPTY};

    #[test]
    fn test_line_of_sight_clear() {
        let grid = Grid::new(10, 10);
        assert!(sensors::has_line_of_sight(&grid, Coord::new(1, 1), Coord::new(8, 8)));
    }

    #[test]
    fn test_line_of_sight_blocked() {
        let mut grid = Grid::new(10, 10);
        grid.set(Coord::new(5, 5), BARRIER);
        assert!(!sensors::has_line_of_sight(&grid, Coord::new(1, 1), Coord::new(9, 9)));
    }

    #[test]
    fn test_line_of_sight_adjacent() {
        let grid = Grid::new(10, 10);
        assert!(sensors::has_line_of_sight(&grid, Coord::new(1, 1), Coord::new(1, 2)));
    }
}
