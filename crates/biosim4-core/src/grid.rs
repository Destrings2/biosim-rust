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

use crate::topology::Topology;
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
///
/// All geometry queries — wrap, displacement, distance, border, in-bounds
/// — are routed through the embedded [`Topology`]. Callers don't match on
/// the topology directly; they call [`Grid::wrap`], [`Grid::delta`],
/// [`Grid::dist_sq`], etc. and the right behaviour falls out of whatever
/// variant the world was configured with. See the [`crate::topology`]
/// module for the contract.
pub struct Grid {
    pub size_x: u16,
    pub size_y: u16,
    /// Which axes wrap. Set at construction (from `SimConfig`); never mutated.
    pub topology: Topology,
    cells: Vec<u32>,
    pub barrier_locations: Vec<Coord>,
    pub barrier_centers: Vec<Coord>,
}

impl Grid {
    /// Build a bounded-plane grid. Equivalent to `with_topology(size_x,
    /// size_y, Topology::Plane)`; preserved as the historical default
    /// constructor so existing tests and tools keep working.
    pub fn new(size_x: u16, size_y: u16) -> Self {
        Self::with_topology(size_x, size_y, Topology::default())
    }

    /// Build a grid with an explicit topology.
    pub fn with_topology(size_x: u16, size_y: u16, topology: Topology) -> Self {
        let cells = vec![EMPTY; size_x as usize * size_y as usize];
        Self { size_x, size_y, topology, cells, barrier_locations: vec![], barrier_centers: vec![] }
    }

    #[inline]
    fn idx(&self, loc: Coord) -> usize {
        (loc.y as usize) * (self.size_x as usize) + (loc.x as usize)
    }

    pub fn zero_fill(&mut self) {
        self.cells.fill(EMPTY);
    }

    /// Map `loc` to its canonical in-bounds coordinate. On wrapping axes,
    /// out-of-range coords are wrapped; on non-wrapping axes, out-of-range
    /// returns `None`. This is the one call sites should use whenever a
    /// coordinate is constructed from `loc + (dx, dy)` and may have run off
    /// an edge — let the Grid decide whether the edge is real.
    #[inline]
    pub fn wrap(&self, loc: Coord) -> Option<Coord> {
        self.topology.wrap(loc, self.size_x, self.size_y)
    }

    /// Topology-aware signed displacement vector from `from` to `to`. On
    /// wrapping axes picks the shorter of the two possible paths around
    /// the cylinder; on non-wrapping axes returns the raw subtraction.
    /// Building block for [`Grid::dist_sq`] / [`Grid::chebyshev_dist`].
    #[inline]
    pub fn delta(&self, from: Coord, to: Coord) -> (i32, i32) {
        self.topology.delta(from, to, self.size_x, self.size_y)
    }

    /// Topology-aware squared Euclidean distance. Cheaper than `dist` —
    /// callers comparing distances should prefer this and only `sqrt` at
    /// the end if they need the metric value.
    #[inline]
    pub fn dist_sq(&self, from: Coord, to: Coord) -> i32 {
        let (dx, dy) = self.delta(from, to);
        dx * dx + dy * dy
    }

    /// Topology-aware Euclidean distance.
    #[inline]
    pub fn dist(&self, from: Coord, to: Coord) -> f32 {
        (self.dist_sq(from, to) as f32).sqrt()
    }

    /// Topology-aware Chebyshev (L∞) distance: `max(|dx|, |dy|)` over
    /// the wrap-aware displacement. Used by 8-neighbourhood logic.
    #[inline]
    pub fn chebyshev_dist(&self, from: Coord, to: Coord) -> i32 {
        let (dx, dy) = self.delta(from, to);
        dx.abs().max(dy.abs())
    }

    /// Wrap-aware signed displacement from `from` to a fractional point
    /// (`to_x`, `to_y`). For challenges whose targets aren't integer
    /// cells — orbiting suns, jittered safe-zones, normalised coords.
    #[inline]
    pub fn delta_to_point(&self, from: Coord, to_x: f32, to_y: f32) -> (f32, f32) {
        self.topology.delta_f(from, to_x, to_y, self.size_x, self.size_y)
    }

    /// Wrap-aware squared L2 distance from `from` to a fractional point.
    #[inline]
    pub fn dist_sq_to_point(&self, from: Coord, to_x: f32, to_y: f32) -> f32 {
        let (dx, dy) = self.delta_to_point(from, to_x, to_y);
        dx * dx + dy * dy
    }

    /// Wrap-aware L2 distance from `from` to a fractional point.
    #[inline]
    pub fn dist_to_point(&self, from: Coord, to_x: f32, to_y: f32) -> f32 {
        self.dist_sq_to_point(from, to_x, to_y).sqrt()
    }

    /// Wrap-aware Euclidean distance to a target expressed in normalised
    /// `[0, 1]` coordinates (the convention several spatial challenges
    /// use). Each axis is normalised by its own `size - 1`, matching the
    /// pre-topology math. Returned distance is also in normalised units.
    pub fn norm_dist_to_norm_point(&self, from: Coord, ncx: f32, ncy: f32) -> f32 {
        let sx = (self.size_x.saturating_sub(1)).max(1) as f32;
        let sy = (self.size_y.saturating_sub(1)).max(1) as f32;
        let cx_px = ncx * sx;
        let cy_px = ncy * sy;
        let (dx_px, dy_px) = self.delta_to_point(from, cx_px, cy_px);
        let dx = dx_px / sx;
        let dy = dy_px / sy;
        (dx * dx + dy * dy).sqrt()
    }

    /// Internal: read a raw cell value. Wraps before indexing so any
    /// caller that constructed an over-edge coord on a wrapping axis
    /// still resolves to a valid index. Panics on out-of-bounds on a
    /// non-wrapping axis — caller should have checked with [`wrap`] first.
    #[inline]
    pub fn at(&self, loc: Coord) -> u32 {
        let c = self.wrap(loc).expect("Grid::at called with OOB coord on non-wrapping axis");
        self.cells[self.idx(c)]
    }

    /// Internal: write a raw cell value. Wraps before indexing. Same
    /// out-of-bounds semantics as [`at`].
    #[inline]
    pub fn set(&mut self, loc: Coord, val: u32) {
        let c = self.wrap(loc).expect("Grid::set called with OOB coord on non-wrapping axis");
        let i = self.idx(c);
        self.cells[i] = val;
    }

    /// True if `loc` (after wrapping where applicable) lands on a valid
    /// cell. Equivalent to `self.wrap(loc).is_some()`. On a torus, all
    /// finite coords are in bounds (they wrap); on a plane this is the
    /// classical 0 ≤ x < size_x check.
    #[inline]
    pub fn is_in_bounds(&self, loc: Coord) -> bool {
        self.wrap(loc).is_some()
    }

    /// True if `loc` sits on an outer edge the agent can't cross. On the
    /// `Plane` topology that's the full rectangular boundary; wrapping
    /// axes don't contribute borders since the agent can step through.
    /// On `Sphere` no cell is a border.
    pub fn is_border(&self, loc: Coord) -> bool {
        self.topology.is_border(loc, self.size_x, self.size_y)
    }

    #[inline]
    pub fn is_empty_at(&self, loc: Coord) -> bool {
        match self.wrap(loc) {
            Some(c) => self.cells[self.idx(c)] == EMPTY,
            None => false,
        }
    }
    #[inline]
    pub fn is_barrier_at(&self, loc: Coord) -> bool {
        match self.wrap(loc) {
            Some(c) => self.cells[self.idx(c)] == BARRIER,
            None => false,
        }
    }
    #[inline]
    pub fn is_kill_barrier_at(&self, loc: Coord) -> bool {
        match self.wrap(loc) {
            Some(c) => self.cells[self.idx(c)] == KILL_BARRIER,
            None => false,
        }
    }
    /// True if the cell blocks movement (regular wall or kill barrier).
    /// Use this to test "can an agent move here?" — kill barriers are
    /// drained specially in `drain_move_queue` so the agent doesn't end
    /// up on the cell.
    #[inline]
    pub fn is_blocking_at(&self, loc: Coord) -> bool {
        match self.wrap(loc) {
            Some(c) => {
                let v = self.cells[self.idx(c)];
                v == BARRIER || v == KILL_BARRIER
            }
            None => false,
        }
    }
    #[inline]
    pub fn is_occupied_at(&self, loc: Coord) -> bool {
        match self.wrap(loc) {
            Some(c) => {
                let v = self.cells[self.idx(c)];
                v != EMPTY && v != BARRIER && v != KILL_BARRIER
            }
            None => false,
        }
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
            // Wrap via Grid so wrapping axes deliver the canonical coord;
            // bounded axes drop OOB neighbours. Same call shape works on
            // every topology — sensors reading densities don't need to
            // know whether they're sweeping a torus or a plane.
            if let Some(loc) = grid.wrap(Coord::new(center.x + dx, center.y + dy)) {
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
