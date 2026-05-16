//! Shared helper functions for sensor implementations.
//!
//! These are free functions used by multiple sensors in `mod.rs`.
//! `population_density_along_axis` computes the weighted density difference
//! between the forward and backward (or left and right) half of a neighborhood,
//! enabling directional population and signal sensors.

use biosim4_core::grid::{visit_neighborhood, Grid};
use biosim4_core::population::Population;
use biosim4_core::signals_layer::Signals;
use biosim4_core::types::{Coord, Dir};

/// Average of a probe taken in the two directions perpendicular to `dir`
/// (left + right), clamped to `[0.0, 1.0]`. Used by every `*_lr` sensor.
///
/// Generic + `#[inline]` so the probe closure monomorphises and inlines
/// into the caller — no extra cost vs. spelling the average out by hand.
#[inline]
pub fn lr_average<F: FnMut(Dir) -> f32>(dir: Dir, mut probe: F) -> f32 {
    let left = probe(dir.rotate90ccw());
    let right = probe(dir.rotate90cw());
    ((left + right) / 2.0).clamp(0.0, 1.0)
}

/// Weighted population density along a direction, normalized to `[0, 1]`.
///
/// Each occupied neighbor contributes `proj / (dx² + dy²)`, where `proj`
/// is the signed dot product of the offset with the unit `dir` vector
/// (positive ahead, negative behind). The sum is normalized by
/// `6 · radius` (the empirical maximum magnitude for a full disc of
/// neighbors) and mapped from `[−1, 1]` to `[0, 1]`, so:
///
/// - `0.5` means population is symmetric around the axis (or empty).
/// - `>0.5` means more occupied cells in the forward direction.
/// - `<0.5` means more in the reverse direction.
pub fn population_density_along_axis(
    loc: Coord,
    dir: Dir,
    radius: f32,
    grid: &Grid,
    _population: &Population,
) -> f32 {
    let dir_unit = unit_dir(dir);
    let mut sum = 0.0f64;
    visit_neighborhood(grid, loc, radius, |nloc| {
        if nloc == loc || !grid.is_occupied_at(nloc) {
            return;
        }
        let dx = (nloc.x - loc.x) as f64;
        let dy = (nloc.y - loc.y) as f64;
        let proj = dir_unit.0 * dx + dir_unit.1 * dy;
        sum += proj / (dx * dx + dy * dy);
    });
    let max_mag = 6.0 * radius as f64;
    (((sum / max_mag).clamp(-1.0, 1.0) + 1.0) / 2.0) as f32
}

/// Unit vector for a `Dir`, returned as an `(x, y)` pair of f64s.
/// Normalizes by Euclidean length so diagonal directions become
/// `(±1/√2, ±1/√2)` — matching the C++ `dirVec.asNormalizedCoord()`
/// followed by `len = √(x² + y²)` then division.
#[inline]
fn unit_dir(dir: Dir) -> (f64, f64) {
    let c = dir.as_normalized_coord();
    let x = c.x as f64;
    let y = c.y as f64;
    let len = (x * x + y * y).sqrt();
    if len == 0.0 {
        (0.0, 0.0)
    } else {
        (x / len, y / len)
    }
}

/// Bidirectional short-probe barrier reading along the given axis,
/// normalized to `[0, 1]`. Scans up to `probe_dist` cells in the forward
/// direction (`+dir`) and the same in the reverse direction (`−dir`),
/// counting non-barrier cells in each. Locations that run off the grid
/// without finding a barrier saturate that side to `probe_dist` (treated
/// as "no barrier within range").
///
/// Returns `((count_fwd − count_rev) + probe_dist) / (2 · probe_dist)`,
/// so values near `0.5` mean symmetric barrier proximity on both sides,
/// `>0.5` means barriers are farther in the forward direction (more
/// non-barrier space ahead), and `<0.5` means barriers are farther in the
/// reverse direction.
///
/// Both `BARRIER` and `KILL_BARRIER` cells stop the probe — for an agent
/// "is there a barrier ahead?" includes hazards. Sensors that need to
/// distinguish hazards from walls use `kill_barrier_fwd` instead.
pub fn short_probe_barrier_distance(loc: Coord, dir: Dir, probe_dist: u32, grid: &Grid) -> f32 {
    let step = dir.as_normalized_coord();
    let pd = probe_dist as i16;

    let scan = |sign: i16| -> u32 {
        let mut count: u32 = 0;
        for i in 1..=pd {
            let target = Coord::new(loc.x + step.x * sign * i, loc.y + step.y * sign * i);
            if !grid.is_in_bounds(target) {
                // Off the grid without finding a barrier — saturate this
                // side to the maximum reading.
                return probe_dist;
            }
            if grid.is_blocking_at(target) {
                return count;
            }
            count += 1;
        }
        count
    };

    let count_fwd = scan(1) as f32;
    let count_rev = scan(-1) as f32;
    let pd_f = probe_dist as f32;
    ((count_fwd - count_rev) + pd_f) / (2.0 * pd_f)
}

/// Distance in steps to the nearest agent ahead, normalized to `[0, 1]`.
///
/// Walks forward up to `probe_dist` empty cells. If an agent is hit at
/// step `i`, returns `(i − 1) / probe_dist` — i.e. higher = farther away,
/// `0` if an agent is in the cell immediately ahead. If the probe runs
/// into a grid boundary or barrier before any agent, returns `1.0`
/// (treated as "no agent within range"). Same shape if no agent is found
/// within `probe_dist` empty cells.
pub fn long_probe_population_fwd(loc: Coord, dir: Dir, probe_dist: u32, grid: &Grid) -> f32 {
    let step = dir.as_normalized_coord();
    let mut count: u32 = 0;
    for _ in 0..probe_dist {
        let target =
            Coord::new(loc.x + step.x * (count as i16 + 1), loc.y + step.y * (count as i16 + 1));
        if !grid.is_in_bounds(target) || grid.is_blocking_at(target) {
            return 1.0;
        }
        if grid.is_occupied_at(target) {
            return count as f32 / probe_dist as f32;
        }
        count += 1;
    }
    1.0
}

/// Distance in steps to the nearest barrier ahead, normalized to `[0, 1]`.
///
/// Walks forward up to `probe_dist` non-barrier cells, ignoring agents. If
/// a barrier is hit at step `i`, returns `(i − 1) / probe_dist` — higher
/// = farther. Running off the grid before finding a barrier returns
/// `1.0`, matching the "no barrier within range" reading.
///
/// Both walls and kill barriers count as barriers here; use
/// `kill_barrier_fwd` to single out hazards.
pub fn long_probe_barrier_fwd(loc: Coord, dir: Dir, probe_dist: u32, grid: &Grid) -> f32 {
    let step = dir.as_normalized_coord();
    let mut count: u32 = 0;
    for _ in 0..probe_dist {
        let target =
            Coord::new(loc.x + step.x * (count as i16 + 1), loc.y + step.y * (count as i16 + 1));
        if !grid.is_in_bounds(target) {
            return 1.0;
        }
        if grid.is_blocking_at(target) {
            return count as f32 / probe_dist as f32;
        }
        count += 1;
    }
    1.0
}

/// Weighted signal density along a direction, normalized to `[0, 1]`.
///
/// Each neighbor cell contributes `mag · proj / (dx² + dy²)`, where `mag`
/// is the cell's raw signal magnitude (0..=SIGNAL_MAX) and `proj` is the
/// signed dot product of the offset with the unit `dir` vector. The
/// summed magnitudes are normalized by `6 · radius · SIGNAL_MAX` and
/// mapped from `[−1, 1]` to `[0, 1]`, with `0.5` meaning symmetric
/// density and `>0.5` meaning stronger pheromone trail in the forward
/// direction.
pub fn signal_density_along_axis(
    layer: u8,
    loc: Coord,
    dir: Dir,
    radius: f32,
    grid: &Grid,
    signals: &Signals,
) -> f32 {
    let dir_unit = unit_dir(dir);
    let mut sum = 0.0f64;
    visit_neighborhood(grid, loc, radius, |nloc| {
        if nloc == loc {
            return;
        }
        let dx = (nloc.x - loc.x) as f64;
        let dy = (nloc.y - loc.y) as f64;
        let proj = dir_unit.0 * dx + dir_unit.1 * dy;
        let mag = signals.get(layer, nloc) as f64;
        sum += (mag * proj) / (dx * dx + dy * dy);
    });
    let max_mag = 6.0 * radius as f64 * biosim4_core::signals_layer::SIGNAL_MAX as f64;
    (((sum / max_mag).clamp(-1.0, 1.0) + 1.0) / 2.0) as f32
}
