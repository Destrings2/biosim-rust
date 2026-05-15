//! Genome type and mutation/reproduction operators.
//!
//! `Genome = Vec<Gene>`. A genome is a flat sequence of connection genes with
//! no inherent ordering; `create_wiring` determines which connections survive.
//!
//! # Mutation operators
//!
//! - `apply_point_mutations` — with probability `rate`, flips one random bit
//!   in each gene independently.
//! - `random_insert_deletion` — with probability `rate`, either inserts a new
//!   random gene at a random position or deletes an existing gene. The split
//!   is controlled by `deletion_ratio`.
//! - `random_bit_flip` — flips exactly one bit in one random gene (used in
//!   tests and one-off contexts; the hot path uses `apply_point_mutations`).
//!
//! # Reproduction
//!
//! `generate_child_genome(parents, params, rng)` assumes `parents` is sorted
//! ascending by fitness (higher index = fitter). With `choose_by_fitness`,
//! parent selection uses the transform `idx = (1 - r²) × N` — squaring `r`
//! concentrates draws near index N-1 (the fittest parent).
//!
//! Sexual crossover overlays a contiguous slice of parent B onto a clone of
//! parent A; the result length is the average of A and B. Both mutation
//! operators are applied to the child after crossover.

use crate::genome::gene::Gene;
use crate::rng::Rng;
use crate::sim_config::SimConfig;

pub type Genome = Vec<Gene>;

// ── Random genome generation ──────────────────────────────────────────────

pub fn make_random_gene(rng: &mut Rng) -> Gene {
    Gene::from_raw(rng.gen_u32())
}

pub fn make_random_genome(cfg: &SimConfig, rng: &mut Rng) -> Genome {
    // gen_range_u32 is half-open; +1 makes the user-facing range inclusive.
    let lo = cfg.genome_initial_length_min as u32;
    let hi = (cfg.genome_initial_length_max as u32).max(lo) + 1;
    let len = rng.gen_range_u32(lo, hi) as usize;
    (0..len).map(|_| make_random_gene(rng)).collect()
}

// ── Mutation ──────────────────────────────────────────────────────────────

/// Flip a random bit in a random gene.
pub fn random_bit_flip(genome: &mut Genome, rng: &mut Rng) {
    if genome.is_empty() {
        return;
    }
    let idx = rng.gen_range_usize(0, genome.len());
    let bit = rng.gen_range_u32(0, 32);
    genome[idx] = Gene(genome[idx].0 ^ (1u32 << bit));
}

/// With probability `rate`, apply a point mutation to each gene.
pub fn apply_point_mutations(genome: &mut Genome, rate: f32, rng: &mut Rng) {
    for gene in genome.iter_mut() {
        if rng.gen_bool(rate) {
            let bit = rng.gen_range_u32(0, 32);
            *gene = Gene(gene.0 ^ (1u32 << bit));
        }
    }
}

/// With probability `rate`, insert or delete a gene. `deletion_ratio` controls the split.
pub fn random_insert_deletion(
    genome: &mut Genome,
    rate: f32,
    deletion_ratio: f32,
    max_len: u16,
    rng: &mut Rng,
) {
    if !rng.gen_bool(rate) {
        return;
    }
    if rng.gen_bool(deletion_ratio) {
        // deletion
        if !genome.is_empty() {
            let idx = rng.gen_range_usize(0, genome.len());
            genome.remove(idx);
        }
    } else {
        // insertion (if under max)
        if genome.len() < max_len as usize {
            let idx = rng.gen_range_usize(0, genome.len() + 1);
            genome.insert(idx, make_random_gene(rng));
        }
    }
}

// ── Reproduction ─────────────────────────────────────────────────────────

/// Parameters for [`generate_child_genome`].
#[derive(Clone, Debug)]
pub struct ReproductionParams {
    pub sexual: bool,
    pub choose_by_fitness: bool,
    pub mutation_rate: f32,
    pub insertion_deletion_rate: f32,
    pub deletion_ratio: f32,
    pub max_len: u16,
}

/// Generate a child genome from a pool of parent genomes.
/// Assumes parents are sorted ascending by fitness (higher index = fitter).
pub fn generate_child_genome(
    parents: &[Genome],
    params: &ReproductionParams,
    rng: &mut Rng,
) -> Genome {
    let ReproductionParams {
        sexual,
        choose_by_fitness,
        mutation_rate,
        insertion_deletion_rate,
        deletion_ratio,
        max_len,
    } = *params;
    if parents.is_empty() {
        return vec![];
    }

    let pick = |rng: &mut Rng| -> usize {
        if choose_by_fitness && parents.len() > 1 {
            // Bias toward higher indices (fitter parents): take (1 - r²)*N so
            // r near 0 (the high-density region of r²) maps to N-1 (fittest).
            let r = rng.gen_f32();
            let n = parents.len() as f32;
            let idx = ((1.0 - r * r) * n) as usize;
            idx.min(parents.len() - 1)
        } else {
            rng.gen_range_usize(0, parents.len())
        }
    };

    let mut child = if sexual && parents.len() > 1 {
        let a = pick(rng);
        let mut b = pick(rng);
        while b == a && parents.len() > 1 {
            b = pick(rng);
        }
        sexual_crossover(&parents[a], &parents[b], rng)
    } else {
        parents[pick(rng)].clone()
    };

    random_insert_deletion(&mut child, insertion_deletion_rate, deletion_ratio, max_len, rng);
    apply_point_mutations(&mut child, mutation_rate, rng);
    // Trim to max length
    if child.len() > max_len as usize {
        let trim = rng.gen_range_usize(0, child.len() - max_len as usize + 1);
        child.drain(..trim);
        child.truncate(max_len as usize);
    }
    child
}

fn sexual_crossover(a: &Genome, b: &Genome, rng: &mut Rng) -> Genome {
    // Overlay a random slice of `b` onto `a`; result length = average of a and b.
    let target_len = (a.len() + b.len()) / 2;
    let mut child = a.clone();
    if !b.is_empty() && !child.is_empty() {
        let start = rng.gen_range_usize(0, child.len().min(b.len()));
        let len = rng.gen_range_usize(1, (child.len().min(b.len()) - start + 1).max(2));
        for i in 0..len {
            if start + i < child.len() && start + i < b.len() {
                child[start + i] = b[start + i];
            }
        }
    }
    child.truncate(target_len.max(1));
    child
}

// ── Genome comparison ─────────────────────────────────────────────────────

/// Return similarity in [0, 1]. Method: 0=jaro-winkler, 1=hamming-bits, 2=hamming-bytes.
pub fn genome_similarity(a: &Genome, b: &Genome, method: u8) -> f32 {
    match method {
        1 => hamming_distance_bits(a, b),
        2 => hamming_distance_bytes(a, b),
        _ => jaro_winkler(a, b),
    }
}

fn jaro_winkler(a: &Genome, b: &Genome) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    // Sample at most 20 genes for long genomes (C++ behavior)
    let limit = 20;
    let a: Vec<u32> = a.iter().take(limit).map(|g| g.0).collect();
    let b: Vec<u32> = b.iter().take(limit).map(|g| g.0).collect();
    let n = a.len();
    let m = b.len();
    let window = (n.max(m) / 2).saturating_sub(1).max(1);
    let mut a_match = vec![false; n];
    let mut b_match = vec![false; m];
    let mut matches = 0usize;
    for i in 0..n {
        let lo = i.saturating_sub(window);
        let hi = (i + window + 1).min(m);
        for j in lo..hi {
            if !b_match[j] && a[i] == b[j] {
                a_match[i] = true;
                b_match[j] = true;
                matches += 1;
                break;
            }
        }
    }
    if matches == 0 {
        return 0.0;
    }
    let mut transpositions = 0usize;
    let mut k = 0;
    for i in 0..n {
        if a_match[i] {
            while !b_match[k] {
                k += 1;
            }
            if a[i] != b[k] {
                transpositions += 1;
            }
            k += 1;
        }
    }
    let jaro = (matches as f32 / n as f32
        + matches as f32 / m as f32
        + (matches - transpositions / 2) as f32 / matches as f32)
        / 3.0;
    // Winkler prefix bonus (use first 4 common u8 bytes as proxy for prefix).
    // Clamp to 1.0: float rounding can push jaro slightly above 1.0 (e.g.
    // 1.0000001), making (1-jaro) negative and the bonus subtractive. The
    // clamp ensures the result is always in [0, 1].
    let prefix = a.iter().zip(b.iter()).take(4).filter(|(x, y)| x == y).count();
    (jaro + prefix as f32 * 0.1 * (1.0 - jaro)).min(1.0)
}

fn hamming_distance_bits(a: &Genome, b: &Genome) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let len = a.len().min(b.len());
    let bits_total = len * 32;
    let diff_bits: u32 = a.iter().zip(b.iter()).map(|(x, y)| (x.0 ^ y.0).count_ones()).sum();
    1.0 - diff_bits as f32 / bits_total as f32
}

fn hamming_distance_bytes(a: &Genome, b: &Genome) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let len = a.len().min(b.len());
    let bytes_total = len * 4;
    let mut same_bytes = 0u32;
    for (x, y) in a.iter().zip(b.iter()) {
        for shift in [0u32, 8, 16, 24] {
            if ((x.0 >> shift) & 0xFF) == ((y.0 >> shift) & 0xFF) {
                same_bytes += 1;
            }
        }
    }
    same_bytes as f32 / bytes_total as f32
}

/// Sample up to 1000 random pairs and return 1.0 - average similarity.
pub fn genetic_diversity(genomes: &[&Genome], method: u8, rng: &mut Rng) -> f32 {
    if genomes.len() < 2 {
        return 0.0;
    }
    let samples = 1000.min(genomes.len() * (genomes.len() - 1) / 2);
    let mut total = 0.0f32;
    for _ in 0..samples {
        let i = rng.gen_range_usize(0, genomes.len());
        let mut j = rng.gen_range_usize(0, genomes.len());
        if j == i {
            j = (i + 1) % genomes.len();
        }
        total += genome_similarity(genomes[i], genomes[j], method);
    }
    1.0 - total / samples as f32
}
