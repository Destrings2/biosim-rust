use crate::genome::ops::{genome_similarity, Genome};
use crate::sim_config::SimConfig;

pub type SpeciesId = u32;

#[derive(Clone, Debug)]
pub struct Species {
    pub id: SpeciesId,
    /// Representative genome used for placement this gen.
    /// Resampled from current members at end-of-gen.
    pub representative: Genome,
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
        }
    }

    pub fn speciate(&mut self, parent_pool: &[(Genome, f32)], method: u8, _cfg: &SimConfig) {
        for species in &mut self.species {
            species.members.clear();
        }

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
                    members: vec![i],
                    adjusted_fitness_sum: 0.0,
                    all_time_best_fitness: f32::NEG_INFINITY,
                    gens_since_improvement: 0,
                    allocated_offspring: 0,
                });
            }
        }
    }

    pub fn assign_offspring_slots(&mut self, parent_pool: &[(Genome, f32)], total_population: u32) {
        let mut total_adjusted_fitness = 0.0;
        
        for species in &mut self.species {
            species.adjusted_fitness_sum = 0.0;
            if species.members.is_empty() {
                continue;
            }
            
            let size = species.members.len() as f32;
            for &idx in &species.members {
                let fitness = parent_pool[idx].1;
                species.adjusted_fitness_sum += fitness / size;
            }
            total_adjusted_fitness += species.adjusted_fitness_sum;
        }

        let mut allocations = Vec::with_capacity(self.species.len());
        let mut remaining_slots = total_population;

        if total_adjusted_fitness <= 0.0 {
            let count = self.species.iter().filter(|s| !s.members.is_empty()).count() as u32;
            if count > 0 {
                let per_species = total_population / count;
                for species in &mut self.species {
                    if !species.members.is_empty() {
                        species.allocated_offspring = per_species as usize;
                        remaining_slots -= per_species;
                    } else {
                        species.allocated_offspring = 0;
                    }
                }
            }
        } else {
            for species in &mut self.species {
                if species.adjusted_fitness_sum > 0.0 {
                    let exact = (species.adjusted_fitness_sum / total_adjusted_fitness) * (total_population as f32);
                    let base = exact.floor() as u32;
                    let remainder = exact - exact.floor();
                    allocations.push((species.id, base, remainder));
                } else {
                    allocations.push((species.id, 0, 0.0));
                }
            }
            
            for &(_, base, _) in &allocations {
                remaining_slots -= base;
            }
            
            allocations.sort_by(|a, b| b.2.total_cmp(&a.2));
            
            for (id, base, _) in &mut allocations {
                if remaining_slots > 0 {
                    *base += 1;
                    remaining_slots -= 1;
                }
                
                if let Some(s) = self.species.iter_mut().find(|s| s.id == *id) {
                    s.allocated_offspring = *base as usize;
                }
            }
        }

        if remaining_slots > 0 {
             if let Some(best) = self.species.iter_mut().filter(|s| !s.members.is_empty()).max_by_key(|s| s.allocated_offspring) {
                 best.allocated_offspring += remaining_slots as usize;
             }
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

        let mut ordered_ids: Vec<_> = self.species.iter().filter(|s| !s.members.is_empty()).map(|s| (s.id, s.all_time_best_fitness)).collect();
        ordered_ids.sort_by(|a, b| b.1.total_cmp(&a.1));
        let protected_ids: Vec<_> = ordered_ids.iter().take(2).map(|&(id, _)| id).collect();

        for species in &mut self.species {
            if !protected_ids.contains(&species.id) && species.gens_since_improvement >= stagnation_limit {
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

    pub fn end_of_generation(&mut self, parent_pool: &[(Genome, f32)], rng: &mut crate::rng::Rng) {
        // Pick new representatives from current members
        for species in &mut self.species {
            if !species.members.is_empty() {
                let rep_idx = species.members[rng.gen_range_usize(0, species.members.len())];
                species.representative = parent_pool[rep_idx].0.clone();
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

    fn dummy_cfg() -> SimConfig {
        SimConfig::default()
    }

    /// Two identical genomes → they should share one species.
    #[test]
    fn identical_genomes_share_one_species() {
        let g = genome_of(&[0x1234ABCD, 0xDEADBEEF]);
        let pool: Vec<(Genome, f32)> = vec![(g.clone(), 1.0), (g.clone(), 1.0)];
        let mut state = SpeciationState::new(0.5);
        state.speciate(&pool, 0, &dummy_cfg());
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
        state.speciate(&pool, 1, &dummy_cfg());
        assert_eq!(state.species.len(), 2, "should be 2 species");
    }

    /// Offspring allocation sums to `total_population`.
    #[test]
    fn allocation_sums_to_population() {
        let g1 = genome_of(&[0x00000000]);
        let g2 = genome_of(&[0xFFFFFFFF]);
        let pool: Vec<(Genome, f32)> = vec![(g1, 0.8), (g2, 0.2)];
        let mut state = SpeciationState::new(0.05);
        state.speciate(&pool, 1, &dummy_cfg());
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
        state.speciate(&pool, 1, &dummy_cfg()); // hamming → 3 very distinct species
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
        state.speciate(&pool, 1, &dummy_cfg());

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
        state.speciate(&pool, 1, &dummy_cfg());

        let initial_threshold = state.compatibility_threshold;
        // Target 10 species but we have 1; threshold should decrease.
        state.update_compatibility_threshold(10, 2, 0.02);
        assert!(
            state.compatibility_threshold < initial_threshold,
            "threshold should decrease when species count < target"
        );
    }
}

