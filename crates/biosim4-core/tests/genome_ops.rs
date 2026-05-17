//! Genome operations: mutation, similarity, child generation. The evolution
//! loop is built on these — a bug here breaks the entire simulation silently.

use biosim4_core::{
    genome::ops::{
        apply_point_mutations, generate_child_genome, genetic_diversity, genome_similarity,
        make_random_genome, random_bit_flip, random_insert_deletion, Genome, ReproductionParams,
    },
    rng::Rng,
    sim_config::SimConfig,
};

#[test]
fn make_random_genome_respects_length_range() {
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 10;
    cfg.genome_initial_length_max = 20;
    let mut rng = Rng::seeded(7);
    for _ in 0..50 {
        let g = make_random_genome(&cfg, &mut rng);
        assert!(g.len() >= 10 && g.len() <= 20, "genome length {} not in [10, 20]", g.len());
    }
}

#[test]
fn random_bit_flip_changes_exactly_one_bit() {
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 8;
    cfg.genome_initial_length_max = 8;
    let mut rng = Rng::seeded(99);
    let original = make_random_genome(&cfg, &mut rng);
    let mut mutated = original.clone();
    random_bit_flip(&mut mutated, &mut rng);

    let diff_bits: u32 =
        original.iter().zip(mutated.iter()).map(|(a, b)| (a.0 ^ b.0).count_ones()).sum();
    assert_eq!(diff_bits, 1, "bit flip should change exactly one bit, got {}", diff_bits);
}

#[test]
fn point_mutation_at_zero_rate_is_noop() {
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 24;
    cfg.genome_initial_length_max = 24;
    let mut rng = Rng::seeded(0);
    let original = make_random_genome(&cfg, &mut rng);
    let mut mutated = original.clone();
    apply_point_mutations(&mut mutated, 0.0, &mut rng);
    assert_eq!(mutated, original, "rate=0 should leave genome unchanged");
}

#[test]
fn random_insert_deletion_respects_max_len() {
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 30;
    cfg.genome_initial_length_max = 30;
    let mut rng = Rng::seeded(42);
    let mut genome = make_random_genome(&cfg, &mut rng);
    // Force many insertions, deletion_ratio=0
    for _ in 0..1000 {
        random_insert_deletion(&mut genome, 1.0, 0.0, 50, &mut rng);
    }
    assert!(genome.len() <= 50, "genome length {} exceeded max 50", genome.len());
}

#[test]
fn genome_similarity_self_is_one() {
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 20;
    cfg.genome_initial_length_max = 20;
    let mut rng = Rng::seeded(1);
    let g = make_random_genome(&cfg, &mut rng);
    // Try all comparison methods (0=jw, 1=hamming-bits, 2=hamming-bytes)
    for method in 0..=2u8 {
        let s = genome_similarity(&g, &g, method);
        assert!(
            (s - 1.0).abs() < 1e-3,
            "method {} self-similarity should be 1.0, got {}",
            method,
            s
        );
    }
}

#[test]
fn genome_similarity_is_in_unit_interval() {
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 20;
    cfg.genome_initial_length_max = 20;
    let mut rng_a = Rng::seeded(1);
    let mut rng_b = Rng::seeded(2);
    let g1 = make_random_genome(&cfg, &mut rng_a);
    let g2 = make_random_genome(&cfg, &mut rng_b);
    for method in 0..=2u8 {
        let s = genome_similarity(&g1, &g2, method);
        assert!(
            s.is_finite() && (0.0..=1.0).contains(&s),
            "method {} similarity {} out of [0,1]",
            method,
            s
        );
    }
}

#[test]
fn generate_child_from_single_parent_is_close_to_parent() {
    // Asexual reproduction with zero mutation rate should give an identical genome.
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 24;
    cfg.genome_initial_length_max = 24;
    let mut rng = Rng::seeded(0);
    let parent = make_random_genome(&cfg, &mut rng);
    let pool = vec![(parent.clone(), 0.0)];
    let params = ReproductionParams {
        sexual: false,
        tournament_size: 1,
        mutation_rate: 0.0,
        insertion_deletion_rate: 0.0,
        deletion_ratio: 0.5,
        max_len: 100,
        adaptive_mutation: false,
        mutation_rate_jitter: 0.0,
    };
    let (child, _rate) = generate_child_genome(&pool, &params, &mut rng);
    assert_eq!(child, parent, "asexual zero-mutation reproduction must clone parent");
}

#[test]
fn generate_child_with_high_mutation_diverges_from_parent() {
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 24;
    cfg.genome_initial_length_max = 24;
    let mut rng = Rng::seeded(0);
    let parent = make_random_genome(&cfg, &mut rng);
    let pool = vec![(parent.clone(), 1.0)];
    let params = ReproductionParams {
        sexual: false,
        tournament_size: 1,
        mutation_rate: 1.0,
        insertion_deletion_rate: 0.0,
        deletion_ratio: 0.5,
        max_len: 100,
        adaptive_mutation: false,
        mutation_rate_jitter: 0.0,
    };
    let (child, _rate) = generate_child_genome(&pool, &params, &mut rng);
    assert_ne!(child, parent, "high mutation rate should produce a different child");
}

#[test]
fn generate_child_from_empty_pool_returns_empty() {
    let mut rng = Rng::seeded(0);
    let pool: Vec<(Genome, f32)> = Vec::new();
    let params = ReproductionParams {
        sexual: false,
        tournament_size: 1,
        mutation_rate: 0.001,
        insertion_deletion_rate: 0.0,
        deletion_ratio: 0.5,
        max_len: 100,
        adaptive_mutation: false,
        mutation_rate_jitter: 0.0,
    };
    let (child, _rate) = generate_child_genome(&pool, &params, &mut rng);
    assert!(child.is_empty(), "empty parent pool should give empty child");
}

#[test]
fn genetic_diversity_returns_unit_value() {
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 16;
    cfg.genome_initial_length_max = 16;
    let mut rng = Rng::seeded(0);
    let pool: Vec<Genome> = (0..10).map(|_| make_random_genome(&cfg, &mut rng)).collect();
    let refs: Vec<&Genome> = pool.iter().collect();
    let d = genetic_diversity(&refs, 0, &mut rng);
    assert!(d.is_finite() && (0.0..=1.0).contains(&d), "diversity {} out of [0,1]", d);
}

#[test]
fn sexual_crossover_child_length_bounded_by_average_of_parents() {
    // Uniform crossover walks 0..(a.len + b.len)/2 and copies A[i] or
    // B[i] per index; with both parents at length 12 the target is 12
    // and neither parent ever runs short, so the loop always completes
    // 12 iterations.
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 12;
    cfg.genome_initial_length_max = 12;
    let mut rng = Rng::seeded(42);
    let a = make_random_genome(&cfg, &mut rng);
    let b = make_random_genome(&cfg, &mut rng);
    // Both parents are length 12 → target_len = 12 → child always length 12.
    let expected_len = (a.len() + b.len()) / 2;
    let pool = vec![(a, 0.0), (b, 0.0)];
    let params = ReproductionParams {
        sexual: true,
        tournament_size: 1,
        mutation_rate: 0.0,
        insertion_deletion_rate: 0.0,
        deletion_ratio: 0.5,
        max_len: 100,
        adaptive_mutation: false,
        mutation_rate_jitter: 0.0,
    };
    for seed in 0..20u64 {
        let mut r = Rng::seeded(seed);
        let (child, _rate) = generate_child_genome(&pool, &params, &mut r);
        assert_eq!(
            child.len(),
            expected_len,
            "seed {}: equal-length parents crossover child length should be {}, got {}",
            seed,
            expected_len,
            child.len()
        );
    }
}

#[test]
fn jaro_winkler_stays_in_unit_interval_for_nearly_identical_genomes() {
    // The Winkler prefix bonus can push jaro slightly above 1.0 due to float
    // rounding when jaro ≈ 1.0 and prefix = 4. The clamp must prevent this.
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 20;
    cfg.genome_initial_length_max = 20;
    let mut rng = Rng::seeded(7);
    let base = make_random_genome(&cfg, &mut rng);
    // Construct a genome that shares the first 4 genes exactly (max prefix bonus)
    // and has very high overall similarity.
    let mut similar = base.clone();
    // Flip a single bit in gene 15 so they are not identical, giving jaro < 1.
    similar[15] = biosim4_core::genome::gene::Gene(similar[15].0 ^ 1);
    // method 0 = jaro-winkler
    let s = genome_similarity(&base, &similar, 0);
    assert!(
        s.is_finite() && (0.0..=1.0).contains(&s),
        "jaro-winkler similarity {} out of [0, 1]",
        s
    );
}

#[test]
fn generate_child_with_sexual_true_and_two_parents_returns_nonempty() {
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 8;
    cfg.genome_initial_length_max = 8;
    let mut rng = Rng::seeded(1);
    let p1 = make_random_genome(&cfg, &mut rng);
    let p2 = make_random_genome(&cfg, &mut rng);
    let pool = vec![(p1, 0.0), (p2, 0.0)];
    let params = ReproductionParams {
        sexual: true,
        tournament_size: 1,
        mutation_rate: 0.0,
        insertion_deletion_rate: 0.0,
        deletion_ratio: 0.5,
        max_len: 100,
        adaptive_mutation: false,
        mutation_rate_jitter: 0.0,
    };
    let (child, _rate) = generate_child_genome(&pool, &params, &mut rng);
    assert!(
        !child.is_empty(),
        "sexual reproduction from two non-empty parents must produce a non-empty child"
    );
}

// ── Topology similarity (method 3) ───────────────────────────────────────
//
// These tests pin the topology metric used by speciation: Jaccard on the
// canonical post-cull edge set with coarse weight bucketing. Any change to
// the bit-packing in `edge_key` or the bucket size in `weight_bucket` will
// shift these numbers and break the asserts — that's the point.

#[test]
fn topology_similarity_identical_networks_returns_one() {
    use biosim4_core::genome::gene::{Gene, SINK_ACTION, SOURCE_SENSOR};
    use biosim4_core::genome::neural_net::{create_wiring, WiringConfig};
    use biosim4_core::genome::ops::nnet_topology_similarity;

    let cfg = WiringConfig { sensor_count: 2, action_count: 2, max_neurons: 2 };
    let g = vec![
        Gene::new(SOURCE_SENSOR, 0, SINK_ACTION, 0, 4096),
        Gene::new(SOURCE_SENSOR, 1, SINK_ACTION, 1, 4096),
    ];
    let nnet_a = create_wiring(&g, cfg);
    let nnet_b = create_wiring(&g, cfg);
    assert_eq!(nnet_topology_similarity(&nnet_a, &nnet_b), 1.0);
}

#[test]
fn topology_similarity_disjoint_networks_returns_zero() {
    use biosim4_core::genome::gene::{Gene, SINK_ACTION, SOURCE_SENSOR};
    use biosim4_core::genome::neural_net::{create_wiring, WiringConfig};
    use biosim4_core::genome::ops::nnet_topology_similarity;

    let cfg = WiringConfig { sensor_count: 2, action_count: 2, max_neurons: 2 };
    let g_a = vec![Gene::new(SOURCE_SENSOR, 0, SINK_ACTION, 0, 4096)];
    let g_b = vec![Gene::new(SOURCE_SENSOR, 1, SINK_ACTION, 1, 4096)];
    let nnet_a = create_wiring(&g_a, cfg);
    let nnet_b = create_wiring(&g_b, cfg);
    assert_eq!(nnet_topology_similarity(&nnet_a, &nnet_b), 0.0);
}

#[test]
fn topology_similarity_partial_overlap_matches_jaccard() {
    use biosim4_core::genome::gene::{Gene, SINK_ACTION, SOURCE_SENSOR};
    use biosim4_core::genome::neural_net::{create_wiring, WiringConfig};
    use biosim4_core::genome::ops::nnet_topology_similarity;

    let cfg = WiringConfig { sensor_count: 2, action_count: 2, max_neurons: 2 };
    // A edges: {s0→a0, s0→a1, s1→a0}
    // B edges: {s0→a0, s0→a1, s1→a1}
    // intersection = {s0→a0, s0→a1} (2); union = 4; Jaccard = 0.5.
    let g_a = vec![
        Gene::new(SOURCE_SENSOR, 0, SINK_ACTION, 0, 4096),
        Gene::new(SOURCE_SENSOR, 0, SINK_ACTION, 1, 4096),
        Gene::new(SOURCE_SENSOR, 1, SINK_ACTION, 0, 4096),
    ];
    let g_b = vec![
        Gene::new(SOURCE_SENSOR, 0, SINK_ACTION, 0, 4096),
        Gene::new(SOURCE_SENSOR, 0, SINK_ACTION, 1, 4096),
        Gene::new(SOURCE_SENSOR, 1, SINK_ACTION, 1, 4096),
    ];
    let sim = nnet_topology_similarity(&create_wiring(&g_a, cfg), &create_wiring(&g_b, cfg));
    assert!((sim - 0.5).abs() < 1e-6, "expected 0.5 Jaccard, got {sim}");
}

#[test]
fn topology_similarity_dedupes_duplicate_connections() {
    use biosim4_core::genome::gene::{Gene, SINK_ACTION, SOURCE_SENSOR};
    use biosim4_core::genome::neural_net::{create_wiring, WiringConfig};
    use biosim4_core::genome::ops::nnet_topology_similarity;

    let cfg = WiringConfig { sensor_count: 1, action_count: 1, max_neurons: 2 };
    let g_single = vec![Gene::new(SOURCE_SENSOR, 0, SINK_ACTION, 0, 4096)];
    let g_double = vec![
        Gene::new(SOURCE_SENSOR, 0, SINK_ACTION, 0, 4096),
        Gene::new(SOURCE_SENSOR, 0, SINK_ACTION, 0, 4096),
    ];
    // Multiplicity collapses under set semantics → similarity is 1.0, not 0.5.
    let sim =
        nnet_topology_similarity(&create_wiring(&g_single, cfg), &create_wiring(&g_double, cfg));
    assert_eq!(sim, 1.0, "duplicate connections must dedupe to a single edge");
}

#[test]
fn topology_similarity_weight_buckets_split_strong_sign_flip() {
    use biosim4_core::genome::gene::{Gene, SINK_ACTION, SOURCE_SENSOR};
    use biosim4_core::genome::neural_net::{create_wiring, WiringConfig};
    use biosim4_core::genome::ops::nnet_topology_similarity;

    let cfg = WiringConfig { sensor_count: 1, action_count: 1, max_neurons: 2 };
    // Same topology (sensor 0 → action 0); weights flipped past one
    // bucket boundary. WEIGHT_BUCKET_SIZE = 0.5, so +1.0 → bucket 2 and
    // −1.0 → bucket −2. Different packed keys, no overlap.
    let g_pos = vec![Gene::new(SOURCE_SENSOR, 0, SINK_ACTION, 0, 8192)];
    let g_neg = vec![Gene::new(SOURCE_SENSOR, 0, SINK_ACTION, 0, -8192)];
    let sim = nnet_topology_similarity(&create_wiring(&g_pos, cfg), &create_wiring(&g_neg, cfg));
    assert!(sim < 1.0, "strongly-flipped weight must split species; got similarity {sim}");
    assert_eq!(sim, 0.0, "with only one edge each, sign-flip gives Jaccard 0");
}

/// Dead-end gene chains must show up as `genome.len() − connection_count()`.
///
/// `create_wiring` Step 3 iteratively culls neurons whose outputs are zero;
/// Step 5 then drops every gene that referenced a culled neuron. The plan
/// surfaces that count to the GA via the `bloat_penalty_weight` parsimony
/// pressure, so this test pins the arithmetic: if a future refactor stops
/// dropping dead genes (or starts dropping live ones), the penalty would
/// silently lose its signal.
#[test]
fn dead_gene_count_matches_genome_minus_connections() {
    use biosim4_core::genome::gene::{
        Gene, SINK_ACTION, SINK_NEURON, SOURCE_NEURON, SOURCE_SENSOR,
    };
    use biosim4_core::genome::neural_net::{create_wiring, WiringConfig};

    let cfg = WiringConfig { sensor_count: 1, action_count: 1, max_neurons: 4 };

    // Two parallel paths share the genome:
    //   live:  sensor 0 → neuron 0 → action 0
    //   dead:  neuron 1 → neuron 2 → neuron 3
    //
    // Neuron 3 has no outgoing connection, so Step 3 culls it; the cull
    // cascades back through neuron 2 → neuron 1. Step 5 drops the two
    // chain genes, leaving the live path intact. dead = 4 − 2 = 2.
    let genome = vec![
        Gene::new(SOURCE_SENSOR, 0, SINK_NEURON, 0, 1000),
        Gene::new(SOURCE_NEURON, 0, SINK_ACTION, 0, 1000),
        Gene::new(SOURCE_NEURON, 1, SINK_NEURON, 2, 1000),
        Gene::new(SOURCE_NEURON, 2, SINK_NEURON, 3, 1000),
    ];
    let nnet = create_wiring(&genome, cfg);
    let dead = genome.len() - nnet.connection_count();
    assert_eq!(nnet.connection_count(), 2, "live sensor→neuron→action path should survive");
    assert_eq!(dead, 2, "neuron→neuron→neuron chain should be culled, leaving 2 dead genes");
}
