//! 2D arena grid.
//!
//! # Cell encoding
//!
//! Each cell stores one of five values, distinguishable by inspecting bit 31
//! and the two sentinel exact values:
//!
//! - `EMPTY = 0` — unoccupied.
//! - `BARRIER = 0xFFFF_FFFF` — static, blocking obstacle.
//! - `KILL_BARRIER = 0xFFFF_FFFE` — hazard cell. Movement into it kills
//!   the moving agent instead of blocking the move. See
//!   `Population::drain_move_queue`.
//! - **Agent ID** (1..0x7FFF_FFFF, bit 31 clear) — occupied by a live agent.
//! - **Programmable ID** (bit 31 set, *except* the BARRIER/KILL_BARRIER
//!   sentinels) — occupied by a challenge-owned programmable entity.
//!   See [`crate::programmable`]. Encoded as `id | PROGRAMMABLE_FLAG`;
//!   decoded with `cell & !PROGRAMMABLE_FLAG`.
//!
//! Zero maps to `EMPTY` (not an agent), which is why `INVALID_AGENT = 0` in
//! `population`. The sentinels `0xFFFF_FFFF` / `0xFFFF_FFFE` are safely
//! distinct from any plausible agent or programmable id.
//!
//! Use [`CellKind`] + [`cell_kind`] when code needs to discriminate the
//! kinds. `is_empty_at` / `is_occupied_at` / `is_barrier_at` continue to do
//! the right thing for both agents and programmables (programmables count
//! as "occupied" for movement-blocking purposes).
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

/// Bit set on the encoded cell when the occupant is a programmable entity
/// (a [`crate::programmable::Programmable`] owned by a challenge), not a peep.
///
/// Encoding: `cell = programmable_id | PROGRAMMABLE_FLAG`.
/// Decoding: `programmable_id = cell & !PROGRAMMABLE_FLAG`.
///
/// The BARRIER / KILL_BARRIER sentinels also have bit 31 set; they are
/// disambiguated by their exact values before testing the flag.
pub const PROGRAMMABLE_FLAG: u32 = 0x8000_0000;

/// Tagged decoding of a raw cell value. Cheaper than wrapping `at` in
/// `match` chains at every call site; the helpers below also exist for
/// the common single-question case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellKind {
    Empty,
    Barrier,
    KillBarrier,
    /// A live peep — value is the `AgentId`.
    Agent(u32),
    /// A challenge-owned programmable entity — value is the raw
    /// `ProgrammableId` (with the flag bit stripped).
    Programmable(u32),
}

/// Decode a raw cell value into its kind. Order of tests matters:
/// the two sentinels are exact and short-circuit before the flag check.
#[inline]
pub fn cell_kind(cell: u32) -> CellKind {
    match cell {
        EMPTY => CellKind::Empty,
        BARRIER => CellKind::Barrier,
        KILL_BARRIER => CellKind::KillBarrier,
        v if v & PROGRAMMABLE_FLAG != 0 => CellKind::Programmable(v & !PROGRAMMABLE_FLAG),
        v => CellKind::Agent(v),
    }
}

/// Encode a programmable id as a grid cell value.
///
/// # Panics
///
/// Debug-asserts that `id` fits within the 31 non-flag bits and is not zero.
/// Programmable ids start at 1 (slot 0 reserved), same convention as agents.
#[inline]
pub fn encode_programmable(id: u32) -> u32 {
    debug_assert!(id != 0, "programmable id 0 is reserved");
    debug_assert!(id & PROGRAMMABLE_FLAG == 0, "programmable id overflowed 31 bits");
    id | PROGRAMMABLE_FLAG
}

/// True if a cell value represents a programmable entity (and not a barrier).
#[inline]
pub fn is_programmable_cell(cell: u32) -> bool {
    cell != BARRIER && cell != KILL_BARRIER && (cell & PROGRAMMABLE_FLAG) != 0
}

/// Extract the programmable id from a programmable cell value. Caller must
/// have already verified the cell is programmable (e.g. via
/// [`is_programmable_cell`] or [`cell_kind`]).
#[inline]
pub fn programmable_id_of(cell: u32) -> u32 {
    cell & !PROGRAMMABLE_FLAG
}

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

    #[test]
    fn cell_kind_dispatch() {
        assert_eq!(cell_kind(EMPTY), CellKind::Empty);
        assert_eq!(cell_kind(BARRIER), CellKind::Barrier);
        assert_eq!(cell_kind(KILL_BARRIER), CellKind::KillBarrier);
        assert_eq!(cell_kind(1), CellKind::Agent(1));
        assert_eq!(cell_kind(0x7FFF_FFFE), CellKind::Agent(0x7FFF_FFFE));
        assert_eq!(cell_kind(encode_programmable(1)), CellKind::Programmable(1));
        assert_eq!(cell_kind(encode_programmable(42)), CellKind::Programmable(42));
    }

    #[test]
    fn programmable_encoding_round_trip() {
        for id in [1, 2, 100, 0x1234, 0x7FFE_FFFD] {
            let cell = encode_programmable(id);
            assert!(is_programmable_cell(cell), "expected programmable for id {id}");
            assert_eq!(programmable_id_of(cell), id, "round-trip failed for id {id}");
            assert!(cell != BARRIER && cell != KILL_BARRIER);
            // Programmable cell must not collide with the agent-id range.
            assert!(cell & PROGRAMMABLE_FLAG != 0);
        }
    }

    #[test]
    fn programmable_is_distinct_from_barriers() {
        assert!(!is_programmable_cell(BARRIER));
        assert!(!is_programmable_cell(KILL_BARRIER));
        assert!(!is_programmable_cell(EMPTY));
        assert!(!is_programmable_cell(1));
    }

    #[test]
    fn programmable_cell_is_not_empty_or_barrier() {
        // Place a programmable cell on a real grid and verify the existing
        // helpers behave correctly: not empty, not a barrier, is occupied.
        let mut g = Grid::new(8, 8);
        let loc = Coord::new(3, 4);
        g.set(loc, encode_programmable(7));
        assert!(!g.is_empty_at(loc));
        assert!(!g.is_barrier_at(loc));
        assert!(!g.is_kill_barrier_at(loc));
        assert!(g.is_occupied_at(loc));
        assert!(
            !g.is_blocking_at(loc),
            "programmables don't block like barriers (they're moveable occupants)"
        );
    }
}
