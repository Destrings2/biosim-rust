use crate::grid::{Grid, visit_neighborhood};
use crate::population::Population;
use crate::signals_layer::Signals;
use crate::types::{Coord, Dir};

/// Weighted population density along a direction, normalized 0..1.
pub fn population_density_along_axis(
    loc: Coord, dir: Dir, radius: f32, grid: &Grid, population: &Population,
) -> f32 {
    let dir_coord = dir.as_normalized_coord();
    let mut sum = 0.0f32;
    let mut count = 0u32;
    visit_neighborhood(grid, loc, radius, |nloc| {
        if nloc == loc { return; }
        if grid.is_occupied_at(nloc) {
            let offset = nloc - loc;
            let sameness = offset.ray_sameness_dir(dir);
            sum += (sameness + 1.0) / 2.0; // map [-1,1] → [0,1]
        }
        count += 1;
    });
    // Normalize: max possible sum ≈ count (all neighbors in-direction)
    if count == 0 { return 0.0; }
    (sum / count as f32).clamp(0.0, 1.0)
}

/// Distance to nearest barrier in forward direction (1..probe_dist), normalized.
/// Returns 1.0 if no barrier found (far away).
pub fn short_probe_barrier_fwd(
    loc: Coord, dir: Dir, probe_dist: u32, grid: &Grid,
) -> f32 {
    let step = dir.as_normalized_coord();
    for i in 1..=(probe_dist as i16) {
        let target = Coord::new(loc.x + step.x * i, loc.y + step.y * i);
        if !grid.is_in_bounds(target) || grid.is_barrier_at(target) {
            return i as f32 / probe_dist as f32;
        }
    }
    1.0
}

/// Distance to nearest barrier in left-right axis, normalized.
pub fn short_probe_barrier_lr(
    loc: Coord, dir: Dir, probe_dist: u32, grid: &Grid,
) -> f32 {
    let left  = short_probe_barrier_fwd(loc, dir.rotate90ccw(), probe_dist, grid);
    let right = short_probe_barrier_fwd(loc, dir.rotate90cw(),  probe_dist, grid);
    ((left + right) / 2.0).clamp(0.0, 1.0)
}

/// Distance in steps to nearest agent ahead, normalized 0..1. 0 = right next to one.
pub fn long_probe_population_fwd(
    loc: Coord, dir: Dir, probe_dist: u32, grid: &Grid,
) -> f32 {
    let step = dir.as_normalized_coord();
    for i in 1..=(probe_dist as i16) {
        let target = Coord::new(loc.x + step.x * i, loc.y + step.y * i);
        if !grid.is_in_bounds(target) { return 1.0; }
        if grid.is_occupied_at(target) {
            return 1.0 - i as f32 / probe_dist as f32;
        }
    }
    0.0
}

/// Distance in steps to nearest barrier ahead, normalized 0..1.
pub fn long_probe_barrier_fwd(
    loc: Coord, dir: Dir, probe_dist: u32, grid: &Grid,
) -> f32 {
    let step = dir.as_normalized_coord();
    for i in 1..=(probe_dist as i16) {
        let target = Coord::new(loc.x + step.x * i, loc.y + step.y * i);
        if !grid.is_in_bounds(target) || grid.is_barrier_at(target) {
            return 1.0 - i as f32 / probe_dist as f32;
        }
    }
    0.0
}

/// Signal density along a direction (forward half vs backward half), normalized 0..1.
pub fn signal_density_along_axis(
    layer: u8, loc: Coord, dir: Dir, radius: f32, grid: &Grid, signals: &Signals,
) -> f32 {
    let mut fwd_sum = 0.0f32;
    let mut bwd_sum = 0.0f32;
    let dir_coord = dir.as_normalized_coord();
    visit_neighborhood(grid, loc, radius, |nloc| {
        if nloc == loc { return; }
        let offset = nloc - loc;
        let mag = signals.get(layer, nloc) as f32 / 255.0;
        let sameness = offset.ray_sameness_dir(dir);
        if sameness >= 0.0 { fwd_sum += mag * sameness; }
        else              { bwd_sum += mag * (-sameness); }
    });
    // Return normalized directional difference
    ((fwd_sum - bwd_sum) / (radius * radius)).clamp(-1.0, 1.0) * 0.5 + 0.5
}

/// Genetic similarity to nearest forward neighbor, normalized 0..1.
pub fn genetic_sim_fwd(
    loc: Coord, dir: Dir, grid: &Grid, population: &Population, method: u8,
) -> f32 {
    let step = dir.as_normalized_coord();
    for i in 1..=4i16 {
        let target = Coord::new(loc.x + step.x * i, loc.y + step.y * i);
        if !grid.is_in_bounds(target) { break; }
        if let Some(neighbor) = population.get_at(grid, target) {
            // Get the current agent's genome for comparison — passed as arg
            return 0.5; // placeholder; caller fills in proper genome ref
        }
    }
    0.0
}
