//! Multi-layer pheromone grid.
//!
//! Each cell in each layer stores a `u8` magnitude (0..255). Currently the
//! simulation initializes one layer.
//!
//! # Increment pattern
//!
//! `increment(layer, center)` deposits: **+3 at `center`** (one from the
//! radius-1.5 neighborhood pass plus an explicit +2 bump) and **+1** at
//! each of the 8 surrounding cells (the 4 cardinals; diagonals sit at
//! distance √2 ≈ 1.41 ≤ 1.5 and also receive +1). Values saturate at 255.
//!
//! # Fade
//!
//! `fade(layer)` decrements every cell by 1 (saturating at 0), called once
//! per step. This simulates pheromone evaporation with a fixed decay rate.
//!
//! # Concurrent increments
//!
//! Cells are `AtomicU8`, so `increment(&self, ...)` is safe to call from
//! multiple threads concurrently during Phase 2 of `step_all_agents`. Reads
//! happen during Phase 1 (sensors); the `fade` and `zero_fill` writes happen
//! sequentially between steps. There is no data race.

use std::sync::atomic::{AtomicU8, Ordering};

use crate::grid::{visit_neighborhood, Grid};
use crate::types::Coord;

/// Maximum pheromone value per cell (saturating ceiling for atomic increments).
pub const SIGNAL_MAX: u8 = 255;

/// Multi-layer pheromone grid. Each cell is an atomic `u8` magnitude
/// (0..SIGNAL_MAX). Atomic so concurrent agent actions can increment cells
/// without locking. Each layer is a flat row-major `Vec<AtomicU8>` of length
/// `size_x * size_y` (indexed `y * size_x + x`).
pub struct Signals {
    layers: Vec<Vec<AtomicU8>>,
    pub size_x: u16,
    pub size_y: u16,
}

impl Signals {
    pub fn new(num_layers: u8, size_x: u16, size_y: u16) -> Self {
        let cells_per_layer = size_x as usize * size_y as usize;
        let layers = (0..num_layers)
            .map(|_| (0..cells_per_layer).map(|_| AtomicU8::new(0)).collect())
            .collect();
        Self { layers, size_x, size_y }
    }

    #[inline]
    fn idx(&self, loc: Coord) -> usize {
        (loc.y as usize) * (self.size_x as usize) + (loc.x as usize)
    }

    pub fn zero_fill(&mut self) {
        for layer in self.layers.iter_mut() {
            for cell in layer.iter_mut() {
                *cell.get_mut() = 0;
            }
        }
    }

    pub fn layer_count(&self) -> u8 {
        self.layers.len() as u8
    }

    /// Read a single cell's signal level. Returns `0` if `layer` is beyond
    /// the configured layer count — sensors/actions wired to a layer that
    /// doesn't exist (e.g. `signal2` while `signal_layers = 1`) should
    /// degrade silently rather than crash the simulation. The disabled_mask
    /// path normally prevents this, but breeds and registry toggles can
    /// re-enable a feature-gated sensor mid-generation; the bounds check
    /// here is a backstop against that.
    pub fn get(&self, layer: u8, loc: Coord) -> u8 {
        let Some(l) = self.layers.get(layer as usize) else { return 0 };
        l[self.idx(loc)].load(Ordering::Relaxed)
    }

    /// Deposit a pheromone burst centered on `center`. The center cell gains
    /// **+3** (one from the radius-1.5 neighborhood pass plus an explicit
    /// +2 bump) and the 8 surrounding cells gain **+1** each. All updates
    /// saturate at `SIGNAL_MAX`.
    ///
    /// Takes `&self` (not `&mut`): cells are `AtomicU8` so concurrent calls
    /// from different agents on different threads are safe.
    ///
    /// No-op if `layer` is beyond the configured layer count — same
    /// rationale as [`Signals::get`].
    pub fn increment(&self, layer: u8, center: Coord, grid: &Grid) {
        let Some(l) = self.layers.get(layer as usize) else { return };
        let size_x = self.size_x;
        let add = |loc: Coord, v: u8| {
            if loc.x >= 0
                && loc.y >= 0
                && (loc.x as u16) < grid.size_x
                && (loc.y as u16) < grid.size_y
            {
                let i = (loc.y as usize) * (size_x as usize) + (loc.x as usize);
                let cell = &l[i];
                let _ = cell.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                    let new = old.saturating_add(v);
                    if new == old {
                        None
                    } else {
                        Some(new)
                    }
                });
            }
        };

        // Radius-1.5 neighborhood pass: +1 to the center and every cell
        // within √2 of it (the 4 cardinals and 4 diagonals).
        for dx in -1i16..=1 {
            for dy in -1i16..=1 {
                let d2 = (dx * dx + dy * dy) as f32;
                if d2 <= 1.5 * 1.5 {
                    add(Coord::new(center.x + dx, center.y + dy), 1);
                }
            }
        }
        // Extra `+2` at the deposit site, on top of the `+1` already
        // contributed by the neighborhood pass — center cell ends at `+3`.
        add(center, 2);
    }

    /// Decrement all values in a layer by 1 (floor 0) to simulate pheromone decay.
    pub fn fade(&mut self, layer: u8) {
        for cell in self.layers[layer as usize].iter_mut() {
            let v = cell.get_mut();
            *v = v.saturating_sub(1);
        }
    }

    /// Parallel fade across every layer. Uses `&mut self` so the per-cell
    /// decrements skip the atomic fence (each chunk is owned by exactly one
    /// worker). Chunks within a layer (not just whole layers) so a typical
    /// 1-layer config still scales across cores. Called from
    /// `sim_step::fade_signals` when the parallel feature + multi-threaded
    /// config is on; falls back to the sequential `fade` otherwise.
    #[cfg(feature = "parallel")]
    pub fn fade_all_parallel(&mut self) {
        use rayon::prelude::*;
        // Tune the chunk size so each worker fades on the order of a few
        // KiB at a time — small enough to keep cores busy when there's a
        // single layer, large enough that fork/join overhead stays below
        // the actual work.
        const CHUNK: usize = 4096;
        for layer in self.layers.iter_mut() {
            layer.par_chunks_mut(CHUNK).for_each(|chunk| {
                for cell in chunk.iter_mut() {
                    let v = cell.get_mut();
                    *v = v.saturating_sub(1);
                }
            });
        }
    }

    /// Get the total signal density in a neighborhood (sum / count, normalized 0..1).
    pub fn get_density(&self, layer: u8, center: Coord, radius: f32, grid: &Grid) -> f32 {
        let mut sum = 0u32;
        let mut count = 0u32;
        visit_neighborhood(grid, center, radius, |loc| {
            sum += self.get(layer, loc) as u32;
            count += 1;
        });
        if count == 0 {
            return 0.0;
        }
        (sum as f32 / count as f32) / SIGNAL_MAX as f32
    }
}
