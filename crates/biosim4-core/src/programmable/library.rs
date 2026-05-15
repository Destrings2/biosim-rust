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

        // Best across *all* rings scanned so far. Carrying it across the
        // outer loop matters: ring `r` can find a peep at L2² > (r+1)² —
        // not close enough to early-terminate, but still the closest peep
        // we'll see if later rings come up empty. Resetting per ring (the
        // earlier version) silently dropped that peep.
        let mut best_target: Option<(Coord, AgentId)> = None;
        let mut best_dist_sq = u32::MAX;

        for ring in 1..=max_r {
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
                        if d_sq < best_dist_sq && has_line_of_sight(grid, center, loc) {
                            best_dist_sq = d_sq;
                            best_target = Some((loc, agent_id));
                        }
                    }
                }
            }

            // No ring r' > r can hold a peep with L2² < (r+1)² — the
            // nearest cell on ring r+1 sits at exactly (r+1)². So if our
            // current best is already at or below that floor, it's the
            // global best and we can stop.
            if let Some(target) = best_target {
                let next_ring_min_dist_sq = ((ring + 1) as u32) * ((ring + 1) as u32);
                if best_dist_sq <= next_ring_min_dist_sq {
                    return Some(target);
                }
            }
        }

        best_target
    }
}

pub mod actions {
    use super::*;
    use crate::rng::Rng;

    /// Next step coordinate when moving from `from` toward `to`. Returns
    /// `None` when the two are already equal.
    ///
    /// The step is a single Chebyshev move (one of 8 directions). Doesn't
    /// check whether the destination is empty — the caller resolves the
    /// move through the usual `out.move_to` channel, which blocks on
    /// occupied cells.
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

    /// Pick a random 8-directional step from `from`, with a 1-in-9 chance
    /// of staying put. Returns `None` when the roll lands on "stay", so
    /// the caller can skip writing `out.move_to` entirely.
    ///
    /// Shared by `Wanderer` (the smoke-test program) and `Predator`'s
    /// full / no-target branches so the random-walk behaviour stays
    /// consistent across programs.
    #[inline]
    pub fn random_walk_step(from: Coord, rng: &mut Rng) -> Option<Coord> {
        let roll = rng.gen_range_u32(0, 9) as i16;
        let (dx, dy) = match roll {
            0 => return None, // stay put
            1 => (-1, 0),
            2 => (1, 0),
            3 => (0, -1),
            4 => (0, 1),
            5 => (-1, -1),
            6 => (1, -1),
            7 => (-1, 1),
            _ => (1, 1),
        };
        Some(Coord::new(from.x + dx, from.y + dy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::food_layer::FoodLayer;
    use crate::grid::{Grid, BARRIER};
    use crate::population::Population;
    use crate::programmable::{ProgramContext, ProgrammablePool};
    use crate::rng::Rng;
    use crate::signals_layer::Signals;
    use crate::world::World;

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

    /// Run `nearest_peep_in_los` against a hand-built minimal world. The
    /// agent id values here are placeholders — the sensor only decodes the
    /// cell value and never dereferences the `Population`.
    fn run_nearest(
        peep_cells: &[(Coord, u32)],
        center: Coord,
        max_range: u16,
    ) -> Option<(Coord, crate::agent::AgentId)> {
        let mut grid = Grid::new(64, 64);
        for (loc, id) in peep_cells {
            grid.set(*loc, *id);
        }
        let signals = Signals::new(1, 64, 64);
        let food = FoodLayer::new(64, 64);
        let population = Population::new(0);
        let pool = ProgrammablePool::new();
        let world = World::new(&grid, &signals, &food, &population, &pool, 1, 0, 0);
        let mut rng = Rng::seeded(1);
        let mut ctx = ProgramContext { world: &world, sim_step: 0, generation: 0, rng: &mut rng };
        sensors::nearest_peep_in_los(&mut ctx, center, max_range)
    }

    #[test]
    fn nearest_peep_finds_target_across_rings() {
        // Regression: previously `best_target` was reset every ring, so a
        // peep on ring 5 at L2² = 41 (no early-terminate because 41 > 36)
        // was forgotten when rings 6..10 were empty and the function
        // returned None. With best carried across rings, the lone peep is
        // returned.
        let peep = Coord::new(15, 14);
        let center = Coord::new(10, 10);
        let result = run_nearest(&[(peep, 7)], center, 10);
        assert_eq!(result.map(|(loc, _)| loc), Some(peep));
    }

    #[test]
    fn nearest_peep_picks_closest_when_two_visible() {
        // (15, 14) lives on Chebyshev ring 5 at L2² = 41; (14, 14) on ring 4
        // at L2² = 32. With both visible, the ring-4 cell wins.
        let center = Coord::new(10, 10);
        let result = run_nearest(&[(Coord::new(15, 14), 7), (Coord::new(14, 14), 8)], center, 10);
        assert_eq!(result.map(|(loc, _)| loc), Some(Coord::new(14, 14)));
    }

    #[test]
    fn nearest_peep_respects_max_range() {
        // L2 distance ~7.07 — outside max_range 5.
        let result = run_nearest(&[(Coord::new(15, 15), 7)], Coord::new(10, 10), 5);
        assert_eq!(result, None);
    }

    #[test]
    fn nearest_peep_blocked_by_barrier() {
        let mut grid = Grid::new(32, 32);
        grid.set(Coord::new(15, 14), 7);
        // Place a barrier directly between center and the peep.
        for x in 12..=14 {
            grid.set(Coord::new(x, 12), BARRIER);
        }
        let signals = Signals::new(1, 32, 32);
        let food = FoodLayer::new(32, 32);
        let population = Population::new(0);
        let pool = ProgrammablePool::new();
        let world = World::new(&grid, &signals, &food, &population, &pool, 1, 0, 0);
        let mut rng = Rng::seeded(1);
        let mut ctx = ProgramContext { world: &world, sim_step: 0, generation: 0, rng: &mut rng };
        let result = sensors::nearest_peep_in_los(&mut ctx, Coord::new(10, 10), 10);
        assert_eq!(result, None);
    }

    #[test]
    fn nearest_peep_empty_world_returns_none() {
        assert_eq!(run_nearest(&[], Coord::new(10, 10), 10), None);
    }
}
