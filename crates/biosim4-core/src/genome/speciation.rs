use crate::genome::neural_net::{create_wiring, WiringConfig};
use crate::genome::ops::{edge_fingerprint, genome_similarity, jaccard_sorted, Genome};
use crate::sim_config::SimConfig;

pub type SpeciesId = u32;

#[derive(Clone, Debug)]
pub struct Species {
    pub id: SpeciesId,
    /// Representative genome used for placement this gen.
    /// Resampled from current members at end-of-gen.
    pub representative: Genome,
    /// Cached topology fingerprint of `representative` (sorted, deduped
    /// `Vec<u32>` packed by [`crate::genome::ops::edge_key`]). Empty when
    /// the cache is invalidated (wiring config changed) or when speciation
    /// is using a bit-string method that doesn't need it. Populated
    /// lazily inside `speciate` on first access.
    pub representative_edges: Vec<u32>,
    /// Indices into the current generation's evaluated pool.
    pub members: Vec<usize>,
    /// Sum of (raw fitness / species size) across members. Drives allocation.
    pub adjusted_fitness_sum: f32,
    /// Best raw fitness seen by any member, ever.
    pub all_time_best_fitness: f32,
    /// Generations since `all_time_best_fitness` last improved.
    pub gens_since_improvement: u32,
    /// Offspring slots allotted this gen.
    pub allocated_offspring: usize,
}

#[derive(Debug, Clone)]
pub struct SpeciationState {
    pub species: Vec<Species>,
    pub next_id: SpeciesId,
    pub compatibility_threshold: f32,
    /// Last `WiringConfig` seen by `speciate`. When the current call's
    /// wiring differs (sensor or action toggled, max_neurons changed) the
    /// cached `representative_edges` on each species become stale and are
    /// invalidated. Tracked here so callers don't have to plumb a "wiring
    /// changed" signal.
    pub last_wiring_cfg: Option<WiringConfig>,
}

impl Default for SpeciationState {
    fn default() -> Self {
        Self::new(0.30)
    }
}

impl SpeciationState {
    pub fn new(initial_threshold: f32) -> Self {
        Self {
            species: Vec::new(),
            next_id: 1,
            compatibility_threshold: initial_threshold,
            last_wiring_cfg: None,
        }
    }

    /// Bucket the parent pool into species. Reads
    /// `cfg.speciation_similarity_method` to pick the distance metric:
    /// methods `0/1/2` are bitstring similarity on raw genomes (existing
    /// `genome_similarity` path); method `3` is Jaccard over each agent's
    /// post-cull connection edge set (the topology metric).
    ///
    /// For method 3 the placement loop computes one fingerprint per
    /// parent up front (`create_wiring` + `edge_fingerprint`), then a
    /// sorted `Vec<u32>` merge intersection drives each per-species
    /// comparison. Cached `representative_edges` survive across
    /// generations until the rep is resampled (in `end_of_generation`)
    /// or `wiring_cfg` changes.
    pub fn speciate(
        &mut self,
        parent_pool: &[(Genome, f32)],
        cfg: &SimConfig,
        wiring_cfg: WiringConfig,
    ) {
        // Invalidate the rep-fingerprint cache when wiring shape changes
        // (the user toggled a sensor; the old packed indices no longer
        // map to the same connections).
        if self.last_wiring_cfg != Some(wiring_cfg) {
            for species in &mut self.species {
                species.representative_edges.clear();
            }
            self.last_wiring_cfg = Some(wiring_cfg);
        }

        for species in &mut self.species {
            species.members.clear();
        }

        let method = cfg.speciation_similarity_method;
        if method == 3 {
            self.speciate_topology(parent_pool, wiring_cfg);
        } else {
            self.speciate_bitstring(parent_pool, method);
        }
    }

    /// Topology-distance placement. One `create_wiring` + `edge_fingerprint`
    /// per parent, then a `jaccard_sorted` merge per (parent × species)
    /// comparison. No allocations inside the species loop — the inner
    /// `mem::take` reuses the parent's pre-computed fingerprint when a
    /// new species is created.
    fn speciate_topology(&mut self, parent_pool: &[(Genome, f32)], wiring_cfg: WiringConfig) {
        // Pre-build parent fingerprints once. `mem::take` later moves
        // selected entries into newly-created species, so this owns
        // mutable slots that may be drained as we go.
        let mut parent_fps: Vec<Vec<u32>> = parent_pool
            .iter()
            .map(|(g, _)| edge_fingerprint(&create_wiring(g, wiring_cfg)))
            .collect();

        // Lazy-fill rep fingerprints for any species that lost theirs
        // via cache invalidation above (or are brand new from last gen).
        for species in &mut self.species {
            if species.representative_edges.is_empty() {
                species.representative_edges =
                    edge_fingerprint(&create_wiring(&species.representative, wiring_cfg));
            }
        }

        for i in 0..parent_pool.len() {
            let mut placed = false;
            for species in &mut self.species {
                let sim = jaccard_sorted(&parent_fps[i], &species.representative_edges);
                let dist = 1.0 - sim;
                if dist <= self.compatibility_threshold {
                    species.members.push(i);
                    placed = true;
                    break;
                }
            }
            if !placed {
                let id = self.next_id;
                self.next_id += 1;
                // Move the fingerprint into the new species — we already
                // computed it and no other parent needs it.
                let rep_edges = std::mem::take(&mut parent_fps[i]);
                self.species.push(Species {
                    id,
                    representative: parent_pool[i].0.clone(),
                    representative_edges: rep_edges,
                    members: vec![i],
                    adjusted_fitness_sum: 0.0,
                    all_time_best_fitness: f32::NEG_INFINITY,
                    gens_since_improvement: 0,
                    allocated_offspring: 0,
                });
            }
        }
    }

    /// Bitstring placement — unchanged from the pre-topology behaviour,
    /// kept available for users who want bit-distance buckets or for A/B
    /// comparisons against the topology metric.
    fn speciate_bitstring(&mut self, parent_pool: &[(Genome, f32)], method: u8) {
        for (i, (genome, _fitness)) in parent_pool.iter().enumerate() {
            let mut placed = false;
            for species in &mut self.species {
                let sim = genome_similarity(&species.representative, genome, method);
                let dist = 1.0 - sim;
                if dist <= self.compatibility_threshold {
                    species.members.push(i);
                    placed = true;
                    break;
                }
            }
            if !placed {
                let id = self.next_id;
                self.next_id += 1;
                self.species.push(Species {
                    id,
                    representative: genome.clone(),
                    representative_edges: Vec::new(),
                    members: vec![i],
                    adjusted_fitness_sum: 0.0,
                    all_time_best_fitness: f32::NEG_INFINITY,
                    gens_since_improvement: 0,
                    allocated_offspring: 0,
                });
            }
        }
    }

    /// Allocate `total_population` offspring slots across species.
    ///
    /// Each non-empty species is guaranteed **one** slot before any
    /// proportional distribution. Without this floor, a newly-split
    /// species with low fitness routinely rounds to zero in the
    /// largest-remainder pass and is deleted by `end_of_generation`
    /// before its lineage can mutate into something competitive —
    /// defeating the entire point of speciation.
    ///
    /// The remaining slots are distributed by largest-remainder over
    /// each species' fitness-shared sum. When total adjusted fitness
    /// is zero or negative (e.g. an entire generation that scored 0),
    /// the remainder is split equally instead.
    pub fn assign_offspring_slots(&mut self, parent_pool: &[(Genome, f32)], total_population: u32) {
        // Reset per-generation accumulators on every species, including
        // ones that went empty this gen (so they get dropped by
        // `end_of_generation`'s retain rather than carrying a stale slot
        // count forward).
        for species in &mut self.species {
            species.adjusted_fitness_sum = 0.0;
            species.allocated_offspring = 0;
        }

        // Fitness sharing: each member contributes `raw / species_size`,
        // so a 50-member species summing to 25.0 ranks the same as a
        // 5-member species summing to 2.5. Keeps small niches viable.
        let mut total_adjusted: f32 = 0.0;
        for species in &mut self.species {
            if species.members.is_empty() {
                continue;
            }
            let size = species.members.len() as f32;
            for &idx in &species.members {
                species.adjusted_fitness_sum += parent_pool[idx].1 / size;
            }
            total_adjusted += species.adjusted_fitness_sum;
        }

        // Indices of species that placed a member this gen — both the
        // reserved-slot pass and the proportional pass iterate these
        // by-index so we avoid quadratic id lookups.
        let active: Vec<usize> =
            (0..self.species.len()).filter(|&i| !self.species[i].members.is_empty()).collect();
        if active.is_empty() {
            return;
        }

        // Reserved floor: 1 slot per active species. When the requested
        // population is smaller than the species count (pathological),
        // only the first N species get the reservation — the rest fall
        // through to `end_of_generation`'s retain and are removed.
        let reserved = (active.len() as u32).min(total_population);
        let mut remaining = total_population - reserved;
        for &si in active.iter().take(reserved as usize) {
            self.species[si].allocated_offspring = 1;
        }
        if remaining == 0 {
            return;
        }

        if total_adjusted <= 0.0 {
            // No fitness signal — split the remainder equally and give
            // the first `leftover` species an extra slot to absorb
            // integer-division loss.
            let n = active.len() as u32;
            let extra = remaining / n;
            let leftover = (remaining - extra * n) as usize;
            for (slot, &si) in active.iter().enumerate() {
                self.species[si].allocated_offspring += extra as usize;
                if slot < leftover {
                    self.species[si].allocated_offspring += 1;
                }
            }
            return;
        }

        // Largest-remainder proportional allocation on the surplus.
        let mut frac: Vec<(usize, u32, f32)> = active
            .iter()
            .map(|&si| {
                let s = &self.species[si];
                let exact = (s.adjusted_fitness_sum / total_adjusted) * remaining as f32;
                let base = exact.floor() as u32;
                (si, base, exact - base as f32)
            })
            .collect();
        for &(si, base, _) in &frac {
            self.species[si].allocated_offspring += base as usize;
            remaining -= base;
        }
        frac.sort_by(|a, b| b.2.total_cmp(&a.2));
        for &(si, _, _) in &frac {
            if remaining == 0 {
                break;
            }
            self.species[si].allocated_offspring += 1;
            remaining -= 1;
        }
    }

    pub fn prune_stagnant(&mut self, parent_pool: &[(Genome, f32)], stagnation_limit: u32) {
        for species in &mut self.species {
            if species.members.is_empty() {
                continue;
            }
            let mut current_best = f32::NEG_INFINITY;
            for &idx in &species.members {
                let fitness = parent_pool[idx].1;
                if fitness > current_best {
                    current_best = fitness;
                }
            }

            if current_best > species.all_time_best_fitness {
                species.all_time_best_fitness = current_best;
                species.gens_since_improvement = 0;
            } else {
                species.gens_since_improvement += 1;
            }
        }

        let mut ordered_ids: Vec<_> = self
            .species
            .iter()
            .filter(|s| !s.members.is_empty())
            .map(|s| (s.id, s.all_time_best_fitness))
            .collect();
        ordered_ids.sort_by(|a, b| b.1.total_cmp(&a.1));
        let protected_ids: Vec<_> = ordered_ids.iter().take(2).map(|&(id, _)| id).collect();

        for species in &mut self.species {
            if !protected_ids.contains(&species.id)
                && species.gens_since_improvement >= stagnation_limit
            {
                species.allocated_offspring = 0;
            }
        }
    }

    pub fn update_compatibility_threshold(&mut self, target: u32, tolerance: u32, step: f32) {
        let count = self.species.iter().filter(|s| !s.members.is_empty()).count() as u32;
        if count > target + tolerance {
            self.compatibility_threshold += step;
        } else if count < target.saturating_sub(tolerance) {
            self.compatibility_threshold -= step;
        }
        self.compatibility_threshold = self.compatibility_threshold.clamp(0.01, 1.0);
    }

    pub fn end_of_generation(
        &mut self,
        parent_pool: &[(Genome, f32)],
        cfg: &SimConfig,
        rng: &mut crate::rng::Rng,
        wiring_cfg: WiringConfig,
    ) {
        let need_topology_cache = cfg.speciation_similarity_method == 3;
        // Pick new representatives from current members
        for species in &mut self.species {
            if !species.members.is_empty() {
                let rep_idx = species.members[rng.gen_range_usize(0, species.members.len())];
                species.representative = parent_pool[rep_idx].0.clone();
                if need_topology_cache {
                    // Refresh the cached fingerprint so next gen's
                    // placement loop hits the cache. ~5µs per species.
                    species.representative_edges =
                        edge_fingerprint(&create_wiring(&species.representative, wiring_cfg));
                } else {
                    // Bit-string mode: keep the cache empty so a later
                    // switch back to method 3 rebuilds against the new
                    // representative, not a stale fingerprint from an
                    // earlier rep.
                    species.representative_edges.clear();
                }
            }
        }

        // Retain only species that received offspring allocation this generation
        self.species.retain(|s| s.allocated_offspring > 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::gene::Gene;

    fn gene(raw: u32) -> Gene {
        Gene::from_raw(raw)
    }

    fn genome_of(vals: &[u32]) -> Genome {
        vals.iter().map(|&v| gene(v)).collect()
    }

    fn cfg_with_method(method: u8) -> SimConfig {
        let mut c = SimConfig::default();
        c.speciation_similarity_method = method;
        c
    }

    /// Wiring config used by speciate calls in these tests. The genome
    /// values used by the fixtures don't actually compile to interesting
    /// nnets — the bit-string methods (0, 1, 2) don't read it — but the
    /// signature still requires one. Big enough for any genome the
    /// fixtures throw at it.
    fn dummy_wiring_cfg() -> WiringConfig {
        WiringConfig { sensor_count: 8, action_count: 8, max_neurons: 8 }
    }

    /// Two identical genomes → they should share one species.
    #[test]
    fn identical_genomes_share_one_species() {
        let g = genome_of(&[0x1234ABCD, 0xDEADBEEF]);
        let pool: Vec<(Genome, f32)> = vec![(g.clone(), 1.0), (g.clone(), 1.0)];
        let mut state = SpeciationState::new(0.5);
        state.speciate(&pool, &cfg_with_method(0), dummy_wiring_cfg());
        assert_eq!(state.species.len(), 1, "should be 1 species");
        assert_eq!(state.species[0].members.len(), 2);
    }

    /// Two maximally-different genomes → two species when threshold is small.
    #[test]
    fn distant_genomes_form_two_species() {
        let a = genome_of(&[0x00000000, 0x00000000]);
        let b = genome_of(&[0xFFFFFFFF, 0xFFFFFFFF]);
        let pool: Vec<(Genome, f32)> = vec![(a, 1.0), (b, 1.0)];
        let mut state = SpeciationState::new(0.05);
        state.speciate(&pool, &cfg_with_method(1), dummy_wiring_cfg());
        assert_eq!(state.species.len(), 2, "should be 2 species");
    }

    /// Offspring allocation sums to `total_population`.
    #[test]
    fn allocation_sums_to_population() {
        let g1 = genome_of(&[0x00000000]);
        let g2 = genome_of(&[0xFFFFFFFF]);
        let pool: Vec<(Genome, f32)> = vec![(g1, 0.8), (g2, 0.2)];
        let mut state = SpeciationState::new(0.05);
        state.speciate(&pool, &cfg_with_method(1), dummy_wiring_cfg());
        let total_pop: u32 = 100;
        state.assign_offspring_slots(&pool, total_pop);
        let sum: usize = state.species.iter().map(|s| s.allocated_offspring).sum();
        assert_eq!(sum as u32, total_pop, "allocations must sum to population");
    }

    /// A stagnating species loses its allocation (unless it's in the top 2).
    #[test]
    fn stagnant_species_loses_allocation() {
        // Need 3 species: only the bottom-ranked stagnant one should be pruned.
        // With only 2 species, both are always in the top-2-immune set.
        let g1 = genome_of(&[0x00000000, 0x11111111]);
        let g2 = genome_of(&[0xFFFFFFFF, 0xEEEEEEEE]);
        let g3 = genome_of(&[0xAAAAAAAA, 0x55555555]);
        let pool: Vec<(Genome, f32)> = vec![(g1, 1.0), (g2, 0.8), (g3, 0.3)];
        let mut state = SpeciationState::new(0.05);
        state.speciate(&pool, &cfg_with_method(1), dummy_wiring_cfg()); // hamming → 3 very distinct species
        state.assign_offspring_slots(&pool, 100);

        // Pre-set species 3 (id=3, lowest fitness) as stagnant.
        // Set all_time_best low (0.2) so it ranks below species 1 (1.0) and 2 (0.8).
        // current pool best for species 3 is 0.3 > 0.2, which would normally reset the
        // counter — so also set it > 0.3 to block improvement detection.
        if let Some(s) = state.species.iter_mut().find(|s| s.id == 3) {
            s.all_time_best_fitness = 0.35; // just above current pool best 0.3 → no improvement
            s.gens_since_improvement = 100;
        }

        state.prune_stagnant(&pool, 15);

        // Top-2 by all_time_best_fitness: species 1 (1.0) and species 2 (0.8) → protected.
        // Species 3 has all_time_best=0.9 but is third and stagnant → pruned.
        let s1 = state.species.iter().find(|s| s.id == 1).unwrap();
        let s3 = state.species.iter().find(|s| s.id == 3);
        assert!(s1.allocated_offspring > 0, "top species should survive");
        if let Some(s3) = s3 {
            assert_eq!(s3.allocated_offspring, 0, "stagnant non-top species should be pruned");
        }
    }

    /// Adaptive threshold increases when too many species exist.
    #[test]
    fn adaptive_threshold_increases_when_over_target() {
        let g1 = genome_of(&[0x00000000]);
        let g2 = genome_of(&[0xFFFFFFFF]);
        let g3 = genome_of(&[0xAAAAAAAA]);
        let pool: Vec<(Genome, f32)> = vec![(g1, 1.0), (g2, 1.0), (g3, 1.0)];
        let mut state = SpeciationState::new(0.05);
        state.speciate(&pool, &cfg_with_method(1), dummy_wiring_cfg());

        let initial_threshold = state.compatibility_threshold;
        // Target 1 species but we have 3; threshold should increase.
        state.update_compatibility_threshold(1, 0, 0.02);
        assert!(
            state.compatibility_threshold > initial_threshold,
            "threshold should increase when species count > target"
        );
    }

    /// Adaptive threshold decreases when too few species exist.
    #[test]
    fn adaptive_threshold_decreases_when_under_target() {
        let g = genome_of(&[0xABCDEF01]);
        let pool: Vec<(Genome, f32)> = vec![(g, 1.0)];
        let mut state = SpeciationState::new(0.50);
        state.speciate(&pool, &cfg_with_method(1), dummy_wiring_cfg());

        let initial_threshold = state.compatibility_threshold;
        // Target 10 species but we have 1; threshold should decrease.
        state.update_compatibility_threshold(10, 2, 0.02);
        assert!(
            state.compatibility_threshold < initial_threshold,
            "threshold should decrease when species count < target"
        );
    }
}
