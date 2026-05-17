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
//! `generate_child_genome(parents, params, rng)` requires `parents`
//! sorted ascending by fitness (higher index = fitter). Parent selection
//! is tournament-of-`tournament_size`: `k = 1` removes fitness pressure,
//! `k = 3` (default) gives a balanced gradient.
//!
//! Sexual crossover is uniform per-gene so individual genes survive with
//! probability ½, preserving useful structure across recombination.
//!
//! Under `adaptive_mutation` the child's rate is the parent's rate
//! perturbed by `exp(τ · (r − 0.5))`, an Evolution-Strategies
//! self-adaptation step. Off by default.

use crate::genome::gene::Gene;
use crate::rng::Rng;
use crate::sim_config::SimConfig;
use crate::types::Coord;

/// An ordered sequence of [`Gene`] values representing one agent's genome.
///
/// The ordering has no inherent biological meaning — `create_wiring` uses
/// the genes as an unordered set of synaptic-connection specifications.
/// Duplicate genes produce proportionally stronger connections.
pub type Genome = Vec<Gene>;

// ── Random genome generation ──────────────────────────────────────────────

/// Generate a single random gene by drawing a random `u32`.
pub fn make_random_gene(rng: &mut Rng) -> Gene {
    Gene::from_raw(rng.gen_u32())
}

/// Generate a random genome with a length sampled uniformly in
/// [`genome_initial_length_min`, `genome_initial_length_max`].
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
///
/// At low rates the naive "draw `gen_bool(rate)` per gene" wastes 24+
/// RNG draws per child to land roughly one flip. The geometric path
/// samples the gap to the next flip in a single draw, so a default
/// `rate = 0.05` child uses ~3 RNG calls instead of ~26. The cliff at
/// `rate > 0.25` falls back to the per-gene loop because the geometric
/// expected gap shrinks below 4 and the overhead of `ln` per gap
/// stops paying off.
pub fn apply_point_mutations(genome: &mut Genome, rate: f32, rng: &mut Rng) {
    if genome.is_empty() || rate <= 0.0 {
        return;
    }
    if rate >= 0.25 {
        for gene in genome.iter_mut() {
            if rng.gen_bool(rate) {
                let bit = rng.gen_range_u32(0, 32);
                *gene = Gene(gene.0 ^ (1u32 << bit));
            }
        }
        return;
    }
    // Geometric gap sampling: skip = floor(ln(u) / ln(1 - rate)).
    // `ln_q` is negative for rate ∈ (0, 1); the quotient is positive.
    let ln_q = (1.0_f32 - rate).ln();
    let len = genome.len();
    let mut idx = 0usize;
    while idx < len {
        let u = rng.gen_f32().max(f32::MIN_POSITIVE);
        let skip = (u.ln() / ln_q).floor() as usize;
        idx += skip;
        if idx >= len {
            break;
        }
        let bit = rng.gen_range_u32(0, 32);
        genome[idx] = Gene(genome[idx].0 ^ (1u32 << bit));
        idx += 1;
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

/// Clamp bounds for the per-individual mutation rate. The floor keeps
/// adaptive runs from collapsing to a zero-search-pressure state; the
/// ceiling prevents per-bit randomisation of every gene every generation.
pub const MIN_MUTATION_RATE: f32 = 1e-4;
pub const MAX_MUTATION_RATE: f32 = 0.5;

/// Reproduction parameters passed to [`generate_child_genome`].
///
/// Parents are `(Genome, mutation_rate)` so the rate flows through
/// selection alongside the genome. Under `adaptive_mutation` the rate
/// evolves with the lineage; otherwise it is just `cfg.point_mutation_rate`
/// forwarded.
#[derive(Clone, Debug)]
pub struct ReproductionParams {
    /// Use two-parent uniform crossover; clone a single parent when `false`.
    pub sexual: bool,
    /// Tournament size `k` for parent selection. `k = 1` removes fitness
    /// pressure; `k = 3` is the recommended default; higher values
    /// concentrate draws on top parents at the cost of diversity.
    pub tournament_size: u32,
    /// Per-gene bit-flip probability applied to the child. Used directly
    /// when `adaptive_mutation` is `false`; otherwise it seeds gen-0
    /// agents and is then superseded by the inherited per-individual rate.
    pub mutation_rate: f32,
    /// Per-child insertion-or-deletion probability.
    pub insertion_deletion_rate: f32,
    /// Fraction of indel events that delete (vs. insert).
    pub deletion_ratio: f32,
    /// Maximum genome length after mutations.
    pub max_len: u16,
    /// Evolve the per-individual mutation rate alongside the genome.
    pub adaptive_mutation: bool,
    /// Jitter scale `τ` for adaptive inheritance. Consulted only when
    /// [`adaptive_mutation`] is `true`.
    ///
    /// [`adaptive_mutation`]: Self::adaptive_mutation
    pub mutation_rate_jitter: f32,
}

/// Tournament-of-`k` selection. Returns the highest sampled index, which
/// is the fittest among `k` since `parents` is sorted ascending by
/// fitness. Selection pressure is smooth in `k`:
/// `P(top is chosen) = 1 − (1 − 1/N)^k`.
#[inline]
fn tournament_pick(parents_len: usize, k: u32, rng: &mut Rng) -> usize {
    debug_assert!(parents_len > 0);
    let k = k.max(1);
    let mut best = rng.gen_range_usize(0, parents_len);
    for _ in 1..k {
        let cand = rng.gen_range_usize(0, parents_len);
        if cand > best {
            best = cand;
        }
    }
    best
}

/// Log-uniform multiplicative jitter on a parent's mutation rate. Keeps
/// the rate strictly positive and symmetric in log space, which suits
/// ES-style self-adaptation. Cheaper than a Gaussian (no Box-Muller)
/// and accurate enough at this population scale.
#[inline]
fn jitter_rate(parent_rate: f32, tau: f32, rng: &mut Rng) -> f32 {
    let r = rng.gen_f32() - 0.5;
    (parent_rate * (tau * r).exp()).clamp(MIN_MUTATION_RATE, MAX_MUTATION_RATE)
}

/// One reproduction outcome with optional parent-position telemetry.
///
/// `parent_a_pos` and `parent_b_pos` are populated only when the caller
/// passed parallel `parent_positions` / `global_positions` slices to the
/// `_with_positions` variant. They drive spatial offspring inheritance in
/// [`crate::spawn::spawn_new_generation`] (cellular GA / parapatric
/// speciation) — see [`crate::sim_config::OffspringPlacementMode`].
///
/// `parent_b_pos` is `None` for asexual / single-parent paths and for
/// the extinction-recovery fallback (empty pool). Spatial placement
/// modes that depend on B (e.g. `MidpointOfParents`) should fall back to
/// `parent_a_pos` when B is missing.
#[derive(Clone, Debug)]
pub struct ChildResult {
    pub genome: Genome,
    pub mutation_rate: f32,
    pub parent_a_pos: Option<Coord>,
    pub parent_b_pos: Option<Coord>,
}

/// Generate one child `(genome, mutation_rate)` from a fitness-sorted
/// parent pool of `(genome, rate)` pairs. Higher pool index = fitter.
pub fn generate_child_genome(
    parents: &[(Genome, f32)],
    params: &ReproductionParams,
    rng: &mut Rng,
) -> (Genome, f32) {
    let r = generate_child_genome_impl(parents, None, params, rng, None, None);
    (r.genome, r.mutation_rate)
}

pub fn generate_child_genome_interspecies(
    parents: &[(Genome, f32)],
    global_parents: &[(Genome, f32)],
    params: &ReproductionParams,
    rng: &mut Rng,
) -> (Genome, f32) {
    let r = generate_child_genome_impl(parents, None, params, rng, Some(global_parents), None);
    (r.genome, r.mutation_rate)
}

/// Position-aware variant that returns the chosen parents' grid
/// positions alongside the child. `parent_positions` must be parallel
/// to `parents` (same length, same order); the returned
/// `parent_a_pos` / `parent_b_pos` are `None` when the slice is empty
/// or the spawning path was single-parent.
pub fn generate_child_genome_with_positions(
    parents: &[(Genome, f32)],
    parent_positions: &[Coord],
    params: &ReproductionParams,
    rng: &mut Rng,
) -> ChildResult {
    generate_child_genome_impl(parents, Some(parent_positions), params, rng, None, None)
}

pub fn generate_child_genome_interspecies_with_positions(
    parents: &[(Genome, f32)],
    parent_positions: &[Coord],
    global_parents: &[(Genome, f32)],
    global_positions: &[Coord],
    params: &ReproductionParams,
    rng: &mut Rng,
) -> ChildResult {
    generate_child_genome_impl(
        parents,
        Some(parent_positions),
        params,
        rng,
        Some(global_parents),
        Some(global_positions),
    )
}

fn generate_child_genome_impl(
    parents: &[(Genome, f32)],
    parent_positions: Option<&[Coord]>,
    params: &ReproductionParams,
    rng: &mut Rng,
    global_parents: Option<&[(Genome, f32)]>,
    global_positions: Option<&[Coord]>,
) -> ChildResult {
    let ReproductionParams {
        sexual,
        tournament_size,
        mutation_rate,
        insertion_deletion_rate,
        deletion_ratio,
        max_len,
        adaptive_mutation,
        mutation_rate_jitter,
    } = *params;
    if parents.is_empty() {
        return ChildResult {
            genome: vec![],
            mutation_rate,
            parent_a_pos: None,
            parent_b_pos: None,
        };
    }

    let pick = |rng: &mut Rng| -> usize { tournament_pick(parents.len(), tournament_size, rng) };

    let (mut child, parent_rate, parent_a_pos, parent_b_pos) = if sexual && parents.len() > 1 {
        let a = pick(rng);
        let mut b = pick(rng);

        // Interspecies mating: when callers pass `global_parents` they've
        // already filtered it to "members of OTHER species", so we can
        // pick B from that pool unconditionally. Caller-side filtering
        // keeps this draw symmetrical with the within-species path.
        let (b_genome, b_pos) = match global_parents {
            Some(global) if global.len() > 1 => {
                b = tournament_pick(global.len(), tournament_size, rng);
                let pos = global_positions.and_then(|p| p.get(b).copied());
                (&global[b].0, pos)
            }
            _ => {
                // Within-species mating with bounded retry: a low-diversity
                // pool must not stall the GA. Up to 4 tries to avoid self-
                // crossover, then accept the duplicate.
                for _ in 0..4 {
                    if b != a {
                        break;
                    }
                    b = pick(rng);
                }
                let pos = parent_positions.and_then(|p| p.get(b).copied());
                (&parents[b].0, pos)
            }
        };

        // Inherit from parent A; uniform crossover uses A's slot order
        // as the structural primary.
        let a_pos = parent_positions.and_then(|p| p.get(a).copied());
        (uniform_crossover(&parents[a].0, b_genome, rng), parents[a].1, a_pos, b_pos)
    } else {
        let p = pick(rng);
        let p_pos = parent_positions.and_then(|pp| pp.get(p).copied());
        (parents[p].0.clone(), parents[p].1, p_pos, None)
    };

    let child_rate = if adaptive_mutation {
        jitter_rate(parent_rate, mutation_rate_jitter, rng)
    } else {
        parent_rate
    };
    let effective_rate = if adaptive_mutation { child_rate } else { mutation_rate };

    random_insert_deletion(&mut child, insertion_deletion_rate, deletion_ratio, max_len, rng);
    apply_point_mutations(&mut child, effective_rate, rng);
    if child.len() > max_len as usize {
        let trim = rng.gen_range_usize(0, child.len() - max_len as usize + 1);
        child.drain(..trim);
        child.truncate(max_len as usize);
    }
    ChildResult { genome: child, mutation_rate: child_rate, parent_a_pos, parent_b_pos }
}

/// Uniform per-gene crossover. Each child position takes A[i] or B[i]
/// with 50/50 probability, falling back to the longer parent when the
/// chosen one runs short. Non-destructive: any individual gene survives
/// with probability ½.
///
/// Child length uses **random rounding** of `(a + b) / 2` so the
/// expected length is unbiased. Integer-division floor-bias otherwise
/// drains length by ~0.25 genes per child per generation, collapsing
/// genomes to zero within ~200 gens at default settings.
fn uniform_crossover(a: &Genome, b: &Genome, rng: &mut Rng) -> Genome {
    // One RNG draw funds the rounding bit plus 31 per-gene picks; refill
    // every 32 picks. Saves ~24 RNG calls per crossover compared to a
    // `gen_bool(0.5)` per gene — the dominant per-spawn cost on default
    // genome lengths.
    let mut bits = rng.gen_u32();
    let mut bits_left = 31u8;
    let round_up = (bits & 1) as usize;
    bits >>= 1;
    let sum = a.len() + b.len();
    let target_len = (sum + round_up) / 2;
    if target_len == 0 {
        return Vec::new();
    }
    let mut child = Vec::with_capacity(target_len);
    for i in 0..target_len {
        if bits_left == 0 {
            bits = rng.gen_u32();
            bits_left = 32;
        }
        let pick_a = (bits & 1) != 0;
        bits >>= 1;
        bits_left -= 1;
        // Fall back when the chosen parent is too short at `i`;
        // `target_len ≤ max(len_a, len_b)` guarantees one side has it.
        let gene = match (pick_a, i < a.len(), i < b.len()) {
            (true, true, _) => a[i],
            (false, _, true) => b[i],
            (true, false, true) => b[i],
            (false, true, false) => a[i],
            (_, false, false) => break,
        };
        child.push(gene);
    }
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

// ── Network-topology similarity ────────────────────────────────────────────
//
// `genome_similarity` compares raw bit patterns, which clusters agents by
// gene byte-packing rather than behaviour. The topology metric below clusters
// by the *culled* connection graph (`NeuralNet.all_connections()`), which is
// what `feed_forward` actually executes. Two agents that wire up the same
// edges — same sources, same sinks, weights in the same coarse bin — sort
// into the same species; agents whose surviving graphs diverge sort apart.
//
// Each edge is packed into a `u32` key; the fingerprint of a network is a
// sorted, deduped `Vec<u32>`. Jaccard between two fingerprints is a single
// linear merge — no hashing, no allocations inside the comparison loop.

/// Bucket size for the weight component of an edge key. Two weights land in
/// the same bucket when `round(w / WEIGHT_BUCKET_SIZE)` matches, so values
/// drifting by less than half a bucket don't change the key.
///
/// `0.5` puts a weight of `0.49` and `−0.49` in the same bucket (0) — small
/// mutations stay in-niche — while `+0.6` and `−0.6` land in different
/// buckets (1 and −1), splitting a sign-flipped lineage into its own species.
pub const WEIGHT_BUCKET_SIZE: f32 = 0.5;

/// Discretise a float weight into a signed 5-bit bucket. Clamped to fit the
/// packing layout in [`edge_key`] even if a weight escapes the typical
/// `[-8, +8]` range produced by the i16 → f32 mapping.
#[inline]
pub fn weight_bucket(w: f32) -> i8 {
    (w / WEIGHT_BUCKET_SIZE).round().clamp(-16.0, 15.0) as i8
}

/// Pack a single gene into a 23-bit `u32` topology key.
///
/// Bit layout:
/// - bit 0:        `source_type` (sensor=1, neuron=0)
/// - bits 1..=8:   `source_num`  (0..=255)
/// - bit 9:        `sink_type`   (action=1, neuron=0)
/// - bits 10..=17: `sink_num`    (0..=255)
/// - bits 18..=22: `weight_bucket` (two's-complement i5, range −16..=15)
///
/// 23 bits used; the upper 9 stay zero. Two genes with the same wiring and
/// weight bucket produce identical keys, so the fingerprint dedupes them.
#[inline]
pub fn edge_key(g: &Gene) -> u32 {
    let st = (g.source_type() as u32) & 0x1;
    let sn = (g.source_num() as u32) & 0xFF;
    let kt = (g.sink_type() as u32) & 0x1;
    let kn = (g.sink_num() as u32) & 0xFF;
    let wb = weight_bucket(g.weight_as_float()) as i32;
    let wb_packed = (wb & 0x1F) as u32; // 5 bits, two's-complement
    st | (sn << 1) | (kt << 9) | (kn << 10) | (wb_packed << 18)
}

/// Build the sorted, deduped topology fingerprint of a compiled network.
/// Walks `nnet.all_connections()` once; the resulting `Vec<u32>` is the
/// canonical edge set that [`jaccard_sorted`] consumes.
pub fn edge_fingerprint(nnet: &crate::genome::neural_net::NeuralNet) -> Vec<u32> {
    let mut keys: Vec<u32> = nnet.all_connections().map(edge_key).collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// Jaccard similarity of two sorted, deduped `u32` sets, computed via a
/// single merge pass: `|A ∩ B| / |A ∪ B|`. Empty-vs-empty returns `1.0`
/// to match `genome_similarity`'s convention; otherwise empty-vs-non-empty
/// returns `0.0`.
#[inline]
pub fn jaccard_sorted(a: &[u32], b: &[u32]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let (mut i, mut j, mut inter) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
        }
    }
    let union = a.len() + b.len() - inter;
    if union == 0 {
        1.0
    } else {
        inter as f32 / union as f32
    }
}

/// Convenience wrapper for tests and one-off use: build both fingerprints
/// and compute their Jaccard. The hot path in `speciate` caches fingerprints
/// on `Species` and calls [`jaccard_sorted`] directly.
pub fn nnet_topology_similarity(
    a: &crate::genome::neural_net::NeuralNet,
    b: &crate::genome::neural_net::NeuralNet,
) -> f32 {
    jaccard_sorted(&edge_fingerprint(a), &edge_fingerprint(b))
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
