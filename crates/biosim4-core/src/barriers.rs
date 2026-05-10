//! Procedural barrier placement.
//!
//! [`create_barrier`] stamps a preset obstacle layout onto the grid based on
//! `SimConfig.barrier_type` (0..7; values ≥ 8 are no-ops). Type 0 places no
//! barriers. Types 1–7 place rectangles, bars, staggered blocks, and strips.
//!
//! All placed cells are recorded in `grid.barrier_locations` and cluster
//! centers in `grid.barrier_centers` for use by sensors and challenges.
//!
//! # `user_barriers` override
//!
//! After every `create_barrier` call, `SimulationState::reapply_user_barriers`
//! re-stamps the user's manual overrides (painted via the frontend). This is
//! necessary because `initialize_generation_0` and `spawn_new_generation` both
//! call `grid.zero_fill()` followed by `create_barrier`, which would erase
//! any previously painted cells.

use crate::grid::{Grid, BARRIER};
use crate::types::Coord;

/// Place barriers on the grid based on the barrier type id.
pub fn create_barrier(grid: &mut Grid, barrier_type: u8) {
    grid.barrier_locations.clear();
    grid.barrier_centers.clear();

    match barrier_type {
        0 => {} // No barriers
        1 => barrier_three_floaters(grid),
        2 => barrier_vertical_bar(grid),
        3 => barrier_horizontal_bar(grid),
        4 => barrier_staggered_blocks(grid),
        5 => barrier_left_right_walls(grid),
        6 => barrier_five_blocks(grid),
        7 => barrier_horizontal_strips(grid),
        _ => {}
    }
}

fn place_barrier(grid: &mut Grid, loc: Coord) {
    if grid.is_in_bounds(loc) {
        grid.set(loc, BARRIER);
        grid.barrier_locations.push(loc);
    }
}

/// Three small floating rectangular islands near grid center regions.
fn barrier_three_floaters(grid: &mut Grid) {
    let sx = grid.size_x as i16;
    let sy = grid.size_y as i16;
    let centers = [
        Coord::new(sx / 4,     sy / 2),
        Coord::new(sx / 2,     sy / 4),
        Coord::new(3 * sx / 4, 3 * sy / 4),
    ];
    for center in centers {
        grid.barrier_centers.push(center);
        for dx in -sx / 8..=sx / 8 {
            for dy in -sy / 8..=sy / 8 {
                place_barrier(grid, Coord::new(center.x + dx, center.y + dy));
            }
        }
    }
}

/// Thin vertical bar near center.
fn barrier_vertical_bar(grid: &mut Grid) {
    let sx = grid.size_x as i16;
    let sy = grid.size_y as i16;
    let x = sx / 2;
    let center = Coord::new(x, sy / 2);
    grid.barrier_centers.push(center);
    for y in sy / 4..=(3 * sy / 4) {
        place_barrier(grid, Coord::new(x, y));
    }
}

/// Thin horizontal bar near center.
fn barrier_horizontal_bar(grid: &mut Grid) {
    let sx = grid.size_x as i16;
    let sy = grid.size_y as i16;
    let y = sy / 2;
    let center = Coord::new(sx / 2, y);
    grid.barrier_centers.push(center);
    for x in sx / 4..=(3 * sx / 4) {
        place_barrier(grid, Coord::new(x, y));
    }
}

/// Checkerboard of small square blocks.
fn barrier_staggered_blocks(grid: &mut Grid) {
    let sx = grid.size_x as i16;
    let sy = grid.size_y as i16;
    let bw = (sx / 16).max(1);
    let bh = (sy / 16).max(1);
    let mut row = 0i16;
    while row < sy {
        let offset = if (row / bh) % 2 == 0 { 0 } else { bw };
        let mut col = offset;
        while col < sx {
            let center = Coord::new(col + bw / 2, row + bh / 2);
            grid.barrier_centers.push(center);
            for dx in 0..bw {
                for dy in 0..bh {
                    place_barrier(grid, Coord::new(col + dx, row + dy));
                }
            }
            col += bw * 3;
        }
        row += bh * 3;
    }
}

/// Two full-height vertical walls dividing the grid into thirds, each with a gap.
fn barrier_left_right_walls(grid: &mut Grid) {
    let sx = grid.size_x as i16;
    let sy = grid.size_y as i16;
    let gap_y = sy / 3;
    let gap_size = sy / 6;
    for &x in &[sx / 3, 2 * sx / 3] {
        let center = Coord::new(x, sy / 2);
        grid.barrier_centers.push(center);
        for y in 0..sy {
            if y < gap_y || y >= gap_y + gap_size {
                place_barrier(grid, Coord::new(x, y));
            }
        }
    }
}

/// Five square blocks arranged in a plus/quincunx pattern.
fn barrier_five_blocks(grid: &mut Grid) {
    let sx = grid.size_x as i16;
    let sy = grid.size_y as i16;
    let bw = sx / 10;
    let bh = sy / 10;
    let positions = [
        Coord::new(sx / 2, sy / 2),
        Coord::new(sx / 4, sy / 4),
        Coord::new(3 * sx / 4, sy / 4),
        Coord::new(sx / 4, 3 * sy / 4),
        Coord::new(3 * sx / 4, 3 * sy / 4),
    ];
    for center in positions {
        grid.barrier_centers.push(center);
        for dx in -bw / 2..=bw / 2 {
            for dy in -bh / 2..=bh / 2 {
                place_barrier(grid, Coord::new(center.x + dx, center.y + dy));
            }
        }
    }
}

/// Three horizontal strips, one near each third of the grid height.
fn barrier_horizontal_strips(grid: &mut Grid) {
    let sx = grid.size_x as i16;
    let sy = grid.size_y as i16;
    let thickness = (sy / 20).max(1);
    for &y_center in &[sy / 4, sy / 2, 3 * sy / 4] {
        let center = Coord::new(sx / 2, y_center);
        grid.barrier_centers.push(center);
        for x in sx / 8..=(7 * sx / 8) {
            for dy in 0..thickness {
                place_barrier(grid, Coord::new(x, y_center + dy - thickness / 2));
            }
        }
    }
}
