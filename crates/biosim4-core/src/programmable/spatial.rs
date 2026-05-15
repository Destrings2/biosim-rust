//! Uniform-grid spatial index for [`ProgrammablePool`].
//!
//! Sensors that ask "what's the nearest programmable to this peep?" used to
//! walk the entire `alive_ids` slice every call. With 1500 peeps and tens
//! of programmables the linear scan is cheap (~45 K ops/step); push to
//! hundreds of programmables and the constant factor starts to matter.
//!
//! This index buckets entities into a coarse grid (`BUCKET_SIZE × BUCKET_SIZE`
//! cells per bucket) and answers nearest-neighbour queries via an L∞
//! expanding-ring scan. Each query short-circuits as soon as the next ring's
//! minimum reachable distance exceeds the current best — so most queries
//! touch only the agent's own bucket plus the surrounding ring, regardless
//! of total entity count.
//!
//! # Build vs query cost
//!
//! Build: O(N) — one pass over `alive_ids`, one `Vec::push` per entity.
//! Query: O(K + visited) where K is entities checked and `visited` is the
//! count of buckets walked before the early-termination condition fires.
//!
//! The index is rebuilt once per step in [`crate::sim_step::step_one`]
//! before the parallel peep loop, then queried read-only by sensors during
//! the parallel section. Mutations (spawn / despawn / clear / step_all
//! merge) mark the pool dirty; the next refresh picks them up.

use crate::types::Coord;

use super::{Programmable, ProgrammableId};

/// Bucket side length, in grid cells. 8 gives 16×16 = 256 buckets on a
/// 128×128 grid — small enough that a typical scan visits two or three
/// rings, large enough that the bucket bookkeeping isn't dominated by
/// per-cell overhead.
const BUCKET_SIZE: u16 = 8;

/// Coarse spatial bucketing of programmable entities. Owned by
/// [`super::ProgrammablePool`]; consumers query through the pool rather
/// than touching this directly.
pub struct SpatialIndex {
    bucket_size: u16,
    buckets_x: u16,
    buckets_y: u16,
    /// Flat `buckets_x * buckets_y` bucket list. Each bucket holds the ids
    /// of entities whose `loc` falls into its region. Vec capacity is
    /// retained across rebuilds so steady-state queries don't allocate.
    buckets: Vec<Vec<ProgrammableId>>,
}

impl SpatialIndex {
    pub fn new() -> Self {
        Self { bucket_size: BUCKET_SIZE, buckets_x: 0, buckets_y: 0, buckets: Vec::new() }
    }

    /// Resize the bucket grid for a world of `size_x × size_y`. Idempotent
    /// when called with the same dimensions, so callers can invoke it on
    /// every rebuild without paying for unchanged sizes.
    pub fn resize_if_needed(&mut self, size_x: u16, size_y: u16) {
        let bsize = self.bucket_size.max(1);
        // Ceiling division so the last partial row/column still gets a
        // bucket — entities at the world's right/bottom edge would otherwise
        // index out of bounds.
        let new_bx = size_x.div_ceil(bsize);
        let new_by = size_y.div_ceil(bsize);
        if new_bx == self.buckets_x && new_by == self.buckets_y {
            return;
        }
        self.buckets_x = new_bx;
        self.buckets_y = new_by;
        let n = new_bx as usize * new_by as usize;
        self.buckets.resize_with(n, Vec::new);
    }

    /// Clear every bucket and re-insert each alive programmable's id.
    /// O(`alive_ids.len()`) plus one `Vec::clear` per bucket.
    pub fn rebuild(
        &mut self,
        size_x: u16,
        size_y: u16,
        agents: &[Option<Programmable>],
        alive_ids: &[ProgrammableId],
    ) {
        self.resize_if_needed(size_x, size_y);
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        for &id in alive_ids {
            let Some(e) = agents.get(id as usize).and_then(|s| s.as_ref()) else { continue };
            if !e.alive {
                continue;
            }
            let Some(idx) = self.bucket_idx(e.loc) else { continue };
            self.buckets[idx].push(id);
        }
    }

    /// Squared L2 distance to the nearest entry, or `None` if the index
    /// holds no entries. The squared form lets callers compare distances
    /// without a `sqrt` until the final normalize.
    ///
    /// The scan walks buckets in L∞ rings around `loc`'s bucket and
    /// terminates as soon as the next ring's minimum reachable cell
    /// distance exceeds the current best. In dense pools this typically
    /// inspects two rings; for sparse pools it can fall back to scanning
    /// the whole grid, but that case is still O(buckets) not O(entities).
    pub fn nearest_dist_sq(&self, loc: Coord, agents: &[Option<Programmable>]) -> Option<u32> {
        if self.buckets.is_empty() || self.buckets_x == 0 || self.buckets_y == 0 {
            return None;
        }
        let bsize = self.bucket_size as i32;
        let bx0 = (loc.x as i32) / bsize;
        let by0 = (loc.y as i32) / bsize;

        let mut best_sq = u32::MAX;
        let max_ring = self.buckets_x.max(self.buckets_y) as i32;

        for ring in 0..=max_ring {
            self.scan_ring(ring, bx0, by0, loc, agents, &mut best_sq);

            // The next ring's closest reachable cell is at least
            // `ring * bsize + 1` cells away (see module-level math note).
            // If our current best is already inside that bound we can
            // bail — no closer entity exists in an unvisited bucket.
            if best_sq != u32::MAX {
                let next_ring_min = ring * bsize + 1;
                if (next_ring_min as i64) * (next_ring_min as i64) > best_sq as i64 {
                    break;
                }
            }
        }

        if best_sq == u32::MAX {
            None
        } else {
            Some(best_sq)
        }
    }

    /// Scan only the boundary buckets of the L∞ ring at radius `ring`. Ring
    /// 0 is the single bucket containing `(bx0, by0)`; higher rings form a
    /// hollow square.
    #[inline]
    fn scan_ring(
        &self,
        ring: i32,
        bx0: i32,
        by0: i32,
        loc: Coord,
        agents: &[Option<Programmable>],
        best_sq: &mut u32,
    ) {
        let bxw = self.buckets_x as i32;
        let byw = self.buckets_y as i32;
        for dy in -ring..=ring {
            for dx in -ring..=ring {
                if dx.abs() != ring && dy.abs() != ring {
                    continue; // interior — already scanned in a smaller ring
                }
                let bx = bx0 + dx;
                let by = by0 + dy;
                if bx < 0 || by < 0 || bx >= bxw || by >= byw {
                    continue;
                }
                let bucket = &self.buckets[(by * bxw + bx) as usize];
                for &id in bucket {
                    let Some(e) = agents.get(id as usize).and_then(|s| s.as_ref()) else {
                        continue;
                    };
                    let cdx = (e.loc.x as i32) - (loc.x as i32);
                    let cdy = (e.loc.y as i32) - (loc.y as i32);
                    let d_sq = (cdx * cdx + cdy * cdy) as u32;
                    if d_sq < *best_sq {
                        *best_sq = d_sq;
                    }
                }
            }
        }
    }

    #[inline]
    fn bucket_idx(&self, loc: Coord) -> Option<usize> {
        if loc.x < 0 || loc.y < 0 {
            return None;
        }
        let bx = (loc.x as u16) / self.bucket_size;
        let by = (loc.y as u16) / self.bucket_size;
        if bx >= self.buckets_x || by >= self.buckets_y {
            return None;
        }
        Some(by as usize * self.buckets_x as usize + bx as usize)
    }
}

impl Default for SpatialIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::programmable::Programmable;
    use crate::types::Dir;

    fn entity(id: ProgrammableId, x: i16, y: i16) -> Option<Programmable> {
        Some(Programmable {
            id,
            loc: Coord::new(x, y),
            heading: Dir::center(),
            alive: true,
            program: 0,
            owner: 0,
            state: [0.0; 8],
            color: [0, 0, 0],
        })
    }

    /// Build a small `agents` Vec wrapping the supplied entities, padded so
    /// `agents.get(id as usize)` resolves for every id.
    fn pool_of(
        entities: Vec<Option<Programmable>>,
    ) -> (Vec<Option<Programmable>>, Vec<ProgrammableId>) {
        let mut agents = vec![None]; // slot 0 reserved
        let mut alive_ids = Vec::new();
        for e in entities {
            if let Some(p) = e {
                let id = agents.len() as ProgrammableId;
                let mut p = p;
                p.id = id;
                agents.push(Some(p));
                alive_ids.push(id);
            }
        }
        (agents, alive_ids)
    }

    #[test]
    fn nearest_returns_none_for_empty() {
        let mut ix = SpatialIndex::new();
        let (agents, alive_ids) = pool_of(vec![]);
        ix.rebuild(64, 64, &agents, &alive_ids);
        assert!(ix.nearest_dist_sq(Coord::new(10, 10), &agents).is_none());
    }

    #[test]
    fn nearest_finds_only_entity() {
        let mut ix = SpatialIndex::new();
        let (agents, alive_ids) = pool_of(vec![entity(0, 20, 30)]);
        ix.rebuild(64, 64, &agents, &alive_ids);
        let d_sq = ix.nearest_dist_sq(Coord::new(20, 30), &agents).unwrap();
        assert_eq!(d_sq, 0);
        // 3-4-5 triangle: (23, 34) → entity at (20, 30) → dist 5, dist² 25.
        let d_sq = ix.nearest_dist_sq(Coord::new(23, 34), &agents).unwrap();
        assert_eq!(d_sq, 9 + 16);
    }

    #[test]
    fn nearest_picks_closer_of_two() {
        let mut ix = SpatialIndex::new();
        // Entity A nearby, B far. Agent should find A.
        let (agents, alive_ids) = pool_of(vec![entity(0, 12, 12), entity(0, 60, 60)]);
        ix.rebuild(64, 64, &agents, &alive_ids);
        let d_sq = ix.nearest_dist_sq(Coord::new(10, 10), &agents).unwrap();
        // (10,10) → (12,12): 2² + 2² = 8
        assert_eq!(d_sq, 8);
    }

    #[test]
    fn matches_linear_scan_under_stress() {
        // Random-ish positions; verify against a brute-force linear scan
        // for many query points so the ring-search early termination
        // cannot accidentally skip a closer bucket.
        let entities: Vec<Option<Programmable>> =
            (0..50).map(|i| entity(0, ((i * 7) % 128) as i16, ((i * 13) % 128) as i16)).collect();
        let (agents, alive_ids) = pool_of(entities);
        let mut ix = SpatialIndex::new();
        ix.rebuild(128, 128, &agents, &alive_ids);

        for qx in (0..128).step_by(13) {
            for qy in (0..128).step_by(11) {
                let q = Coord::new(qx, qy);
                let indexed = ix.nearest_dist_sq(q, &agents);
                // Brute force baseline.
                let brute = alive_ids
                    .iter()
                    .filter_map(|&id| agents.get(id as usize).and_then(|s| s.as_ref()))
                    .map(|e| {
                        let dx = (e.loc.x as i32) - (q.x as i32);
                        let dy = (e.loc.y as i32) - (q.y as i32);
                        (dx * dx + dy * dy) as u32
                    })
                    .min();
                assert_eq!(indexed, brute, "mismatch at ({qx}, {qy})");
            }
        }
    }

    #[test]
    fn resize_grows_bucket_grid() {
        let mut ix = SpatialIndex::new();
        ix.resize_if_needed(64, 64);
        let small = ix.buckets.len();
        ix.resize_if_needed(256, 256);
        assert!(ix.buckets.len() > small);
    }

    #[test]
    fn rebuild_clears_stale_entries() {
        let mut ix = SpatialIndex::new();
        let (agents_a, alive_a) = pool_of(vec![entity(0, 5, 5)]);
        ix.rebuild(64, 64, &agents_a, &alive_a);
        assert!(ix.nearest_dist_sq(Coord::new(5, 5), &agents_a).is_some());

        // Rebuild against an empty pool — the previous entity must vanish.
        let (agents_b, alive_b) = pool_of(vec![]);
        ix.rebuild(64, 64, &agents_b, &alive_b);
        assert!(ix.nearest_dist_sq(Coord::new(5, 5), &agents_b).is_none());
    }
}
