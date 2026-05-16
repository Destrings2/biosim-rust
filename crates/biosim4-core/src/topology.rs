//! World topology: choose which axes wrap around at the edge.
//!
//! The default [`Topology::Plane`] preserves the historical behaviour — the
//! grid is a bounded rectangle, edges are impassable. The other variants
//! turn one or both axes into cylinders so peeps stepping off one side
//! reappear on the other.
//!
//! # Why an enum, not a trait
//!
//! Every cell lookup, sensor probe, and challenge distance check consults
//! the topology. A trait would either force a vtable indirection in the
//! hot path or thread a generic parameter through `Grid`, `World`, every
//! registry, and every consumer — invasive for four variants. The enum
//! compiles to a small jump table the optimizer can collapse.
//!
//! # Programming-to-the-abstraction
//!
//! Outside this module, code never matches on `Topology` directly. Use
//! the [`crate::grid::Grid`] helpers — `wrap`, `delta`, `dist_sq`, `dist`,
//! `chebyshev_dist`, `is_border`, `is_in_bounds` — and topology comes
//! along for free. The variants below exist for the [`SimConfig`] surface,
//! `Default`, and the unit-tested geometry primitives in this file.
//!
//! [`SimConfig`]: crate::sim_config::SimConfig

use serde::{Deserialize, Serialize};

use crate::types::Coord;

/// Which axes of the world wrap around at the edge.
///
/// See the module-level docs for the design contract; see
/// [`crate::grid::Grid`] for the helpers that consume this type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Topology {
    /// Bounded rectangle. Edges block. The historical default.
    #[default]
    Plane,
    /// Cylinder around the N-S axis: east and west edges are connected,
    /// north and south remain borders.
    TorusX,
    /// Cylinder around the E-W axis: north and south edges are connected,
    /// east and west remain borders.
    TorusY,
    /// Both axes wrap. Topologically a torus, called "sphere" in the UI
    /// because the user-facing semantic is "no edges, you can go forever
    /// in any direction".
    Sphere,
}

impl Topology {
    /// True when stepping off the east/west edge teleports you to the
    /// opposite side instead of being blocked.
    #[inline]
    pub fn wraps_x(self) -> bool {
        matches!(self, Self::TorusX | Self::Sphere)
    }

    /// True when stepping off the north/south edge teleports you to the
    /// opposite side instead of being blocked.
    #[inline]
    pub fn wraps_y(self) -> bool {
        matches!(self, Self::TorusY | Self::Sphere)
    }

    /// Map a (possibly out-of-bounds) coordinate to its canonical
    /// in-bounds form. Wraps any axis the topology declares wrapping for;
    /// returns `None` if the coord is out of bounds on a non-wrapping axis.
    ///
    /// `size_x` / `size_y` are taken as parameters (not stored) because
    /// the topology is independent of grid dimensions — the same value
    /// describes a 32×32 and a 1024×1024 world.
    ///
    /// `#[inline(always)]` because this sits on every cell lookup in the
    /// hot path — sensors, neighbourhood scans, movement. The `Plane`
    /// fast-path below collapses to the same bounds check the
    /// pre-topology grid did when the caller's `topology` is `Plane`.
    #[inline(always)]
    pub fn wrap(self, loc: Coord, size_x: u16, size_y: u16) -> Option<Coord> {
        // Hot-path: most worlds run on `Plane`. Spelling that case out
        // explicitly lets the compiler skip the per-axis `wrap_axis`
        // dispatch when the caller's topology is bounded.
        if matches!(self, Topology::Plane) {
            if loc.x >= 0 && loc.y >= 0 && (loc.x as u16) < size_x && (loc.y as u16) < size_y {
                return Some(loc);
            }
            return None;
        }
        let sx = size_x as i32;
        let sy = size_y as i32;
        let x = wrap_axis(loc.x as i32, sx, self.wraps_x())?;
        let y = wrap_axis(loc.y as i32, sy, self.wraps_y())?;
        Some(Coord::new(x as i16, y as i16))
    }

    /// True if `loc` sits on a border the agent cannot escape through.
    /// On `Plane` every outer row/column counts; on `TorusX` only the
    /// north and south edges remain borders (E-W wrap removes the others);
    /// on `Sphere` no cells are borders.
    #[inline]
    pub fn is_border(self, loc: Coord, size_x: u16, size_y: u16) -> bool {
        let on_x_edge = loc.x == 0 || loc.x as u16 == size_x.saturating_sub(1);
        let on_y_edge = loc.y == 0 || loc.y as u16 == size_y.saturating_sub(1);
        // An edge only counts as a border on an axis that *doesn't* wrap.
        let x_border = !self.wraps_x() && on_x_edge;
        let y_border = !self.wraps_y() && on_y_edge;
        x_border || y_border
    }

    /// Shortest signed displacement from `from` to `to`, picking the
    /// minimum-wrap path on each wrapping axis. On a non-wrapping axis
    /// this is just `to - from`. On a wrapping axis it's the same value
    /// folded into `(-size/2, size/2]` so the closer of the two paths
    /// around the cylinder wins.
    #[inline(always)]
    pub fn delta(self, from: Coord, to: Coord, size_x: u16, size_y: u16) -> (i32, i32) {
        // Bare subtraction on `Plane` matches the pre-topology code path;
        // keep it explicit so the optimizer doesn't pay for the
        // wrap-axis dispatch on every distance check.
        if matches!(self, Topology::Plane) {
            return (to.x as i32 - from.x as i32, to.y as i32 - from.y as i32);
        }
        let dx = delta_axis(to.x as i32 - from.x as i32, size_x as i32, self.wraps_x());
        let dy = delta_axis(to.y as i32 - from.y as i32, size_y as i32, self.wraps_y());
        (dx, dy)
    }

    /// Float-precision variant of [`delta`] for challenges whose targets
    /// live at fractional positions (orbiting suns, hazard centres). Same
    /// shortest-wrap semantics — the magnitude is `min(|raw|, size − |raw|)`
    /// on each wrapping axis.
    #[inline]
    pub fn delta_f(
        self,
        from: Coord,
        to_x: f32,
        to_y: f32,
        size_x: u16,
        size_y: u16,
    ) -> (f32, f32) {
        if matches!(self, Topology::Plane) {
            return (to_x - from.x as f32, to_y - from.y as f32);
        }
        let dx = delta_axis_f(to_x - from.x as f32, size_x as f32, self.wraps_x());
        let dy = delta_axis_f(to_y - from.y as f32, size_y as f32, self.wraps_y());
        (dx, dy)
    }
}

/// Normalise an axis coordinate. Returns `Some(canonical)` if either in
/// the range `[0, size)` already, or after wrapping if the axis allows it;
/// `None` if out of range on a non-wrapping axis.
#[inline]
fn wrap_axis(v: i32, size: i32, wrap: bool) -> Option<i32> {
    if v >= 0 && v < size {
        return Some(v);
    }
    if !wrap {
        return None;
    }
    // `rem_euclid` always returns a non-negative result in `[0, size)`
    // for size > 0 — no second branch needed for negatives.
    Some(v.rem_euclid(size))
}

/// Fold a raw axis displacement into the shortest-wrap version for an
/// axis of the given size. `wrap = false` returns the input unchanged.
#[inline]
fn delta_axis(d: i32, size: i32, wrap: bool) -> i32 {
    if !wrap || size == 0 {
        return d;
    }
    // Map to `[0, size)` then split: anything past `size/2` is shorter
    // going the other way.
    let m = d.rem_euclid(size);
    let half = size / 2;
    if m > half {
        m - size
    } else {
        m
    }
}

/// Float-precision variant of [`delta_axis`].
#[inline]
fn delta_axis_f(d: f32, size: f32, wrap: bool) -> f32 {
    if !wrap || size == 0.0 {
        return d;
    }
    // Manual rem_euclid for f32: subtract the floor of d/size scaled back.
    let m = d - (d / size).floor() * size;
    let half = size * 0.5;
    if m > half {
        m - size
    } else {
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(x: i16, y: i16) -> Coord {
        Coord::new(x, y)
    }

    // ── wrap ─────────────────────────────────────────────────────────────

    #[test]
    fn plane_wrap_passes_inside_and_rejects_outside() {
        let t = Topology::Plane;
        assert_eq!(t.wrap(c(0, 0), 10, 10), Some(c(0, 0)));
        assert_eq!(t.wrap(c(9, 9), 10, 10), Some(c(9, 9)));
        assert_eq!(t.wrap(c(-1, 0), 10, 10), None);
        assert_eq!(t.wrap(c(10, 0), 10, 10), None);
        assert_eq!(t.wrap(c(0, -1), 10, 10), None);
        assert_eq!(t.wrap(c(0, 10), 10, 10), None);
    }

    #[test]
    fn torus_x_wraps_east_west_only() {
        let t = Topology::TorusX;
        assert_eq!(t.wrap(c(-1, 5), 10, 10), Some(c(9, 5)));
        assert_eq!(t.wrap(c(10, 5), 10, 10), Some(c(0, 5)));
        assert_eq!(t.wrap(c(15, 5), 10, 10), Some(c(5, 5)));
        assert_eq!(t.wrap(c(5, -1), 10, 10), None, "y still bounded on TorusX");
        assert_eq!(t.wrap(c(5, 10), 10, 10), None);
    }

    #[test]
    fn torus_y_wraps_north_south_only() {
        let t = Topology::TorusY;
        assert_eq!(t.wrap(c(5, -1), 10, 10), Some(c(5, 9)));
        assert_eq!(t.wrap(c(5, 10), 10, 10), Some(c(5, 0)));
        assert_eq!(t.wrap(c(-1, 5), 10, 10), None);
    }

    #[test]
    fn sphere_wraps_both_axes() {
        let t = Topology::Sphere;
        assert_eq!(t.wrap(c(-1, -1), 10, 10), Some(c(9, 9)));
        assert_eq!(t.wrap(c(10, 10), 10, 10), Some(c(0, 0)));
        assert_eq!(t.wrap(c(15, 27), 10, 10), Some(c(5, 7)));
    }

    // ── is_border ────────────────────────────────────────────────────────

    #[test]
    fn plane_has_full_border() {
        let t = Topology::Plane;
        assert!(t.is_border(c(0, 5), 10, 10));
        assert!(t.is_border(c(9, 5), 10, 10));
        assert!(t.is_border(c(5, 0), 10, 10));
        assert!(t.is_border(c(5, 9), 10, 10));
        assert!(!t.is_border(c(5, 5), 10, 10));
    }

    #[test]
    fn torus_x_keeps_only_horizontal_border() {
        let t = Topology::TorusX;
        // E/W edges no longer borders (they wrap).
        assert!(!t.is_border(c(0, 5), 10, 10));
        assert!(!t.is_border(c(9, 5), 10, 10));
        // N/S edges still borders.
        assert!(t.is_border(c(5, 0), 10, 10));
        assert!(t.is_border(c(5, 9), 10, 10));
    }

    #[test]
    fn sphere_has_no_borders() {
        let t = Topology::Sphere;
        for x in 0..10 {
            for y in 0..10 {
                assert!(!t.is_border(c(x, y), 10, 10), "({x},{y}) should not be a border");
            }
        }
    }

    // ── delta ────────────────────────────────────────────────────────────

    #[test]
    fn plane_delta_is_raw_subtraction() {
        let t = Topology::Plane;
        assert_eq!(t.delta(c(2, 3), c(7, 1), 10, 10), (5, -2));
        assert_eq!(t.delta(c(0, 0), c(9, 9), 10, 10), (9, 9));
    }

    #[test]
    fn torus_x_picks_shorter_path() {
        let t = Topology::TorusX;
        // (1,5) -> (9,5): direct +8, wrap −2 → wrap wins.
        assert_eq!(t.delta(c(1, 5), c(9, 5), 10, 10), (-2, 0));
        // Mirror direction.
        assert_eq!(t.delta(c(9, 5), c(1, 5), 10, 10), (2, 0));
        // Y still raw (non-wrapping on TorusX).
        assert_eq!(t.delta(c(0, 1), c(0, 8), 10, 10), (0, 7));
    }

    #[test]
    fn sphere_wraps_both_directions() {
        let t = Topology::Sphere;
        assert_eq!(t.delta(c(1, 1), c(9, 9), 10, 10), (-2, -2));
    }

    #[test]
    fn delta_for_distance_squared_on_torus() {
        // Two cells diagonally across the wrap boundary; the squared L2
        // should match the wrapped path, not the raw subtraction.
        let t = Topology::Sphere;
        let (dx, dy) = t.delta(c(1, 1), c(9, 9), 10, 10);
        assert_eq!(dx * dx + dy * dy, 8, "wrap path should give (-2)^2 + (-2)^2 = 8");
    }
}
