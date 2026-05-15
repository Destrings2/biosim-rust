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
    let pool = vec![parent.clone()];
    let params = ReproductionParams {
        sexual: false,
        choose_by_fitness: false,
        mutation_rate: 0.0,
        insertion_deletion_rate: 0.0,
        deletion_ratio: 0.5,
        max_len: 100,
    };
    let child = generate_child_genome(&pool, &params, &mut rng);
    assert_eq!(child, parent, "asexual zero-mutation reproduction must clone parent");
}

#[test]
fn generate_child_with_high_mutation_diverges_from_parent() {
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 24;
    cfg.genome_initial_length_max = 24;
    let mut rng = Rng::seeded(0);
    let parent = make_random_genome(&cfg, &mut rng);
    let pool = vec![parent.clone()];
    let params = ReproductionParams {
        sexual: false,
        choose_by_fitness: false,
        mutation_rate: 1.0,
        insertion_deletion_rate: 0.0,
        deletion_ratio: 0.5,
        max_len: 100,
    };
    let child = generate_child_genome(&pool, &params, &mut rng);
    assert_ne!(child, parent, "high mutation rate should produce a different child");
}

#[test]
fn generate_child_from_empty_pool_returns_empty() {
    let mut rng = Rng::seeded(0);
    let pool: Vec<Genome> = Vec::new();
    let params = ReproductionParams {
        sexual: false,
        choose_by_fitness: false,
        mutation_rate: 0.001,
        insertion_deletion_rate: 0.0,
        deletion_ratio: 0.5,
        max_len: 100,
    };
    let child = generate_child_genome(&pool, &params, &mut rng);
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
    // sexual_crossover clones parent `a` then overlays a slice from `b`, and
    // truncates to target_len = (a.len + b.len) / 2. Because truncate only
    // shrinks, the result length is min(a.len, target_len). With equal-length
    // parents the child always has exactly target_len genes.
    let mut cfg = SimConfig::default();
    cfg.genome_initial_length_min = 12;
    cfg.genome_initial_length_max = 12;
    let mut rng = Rng::seeded(42);
    let a = make_random_genome(&cfg, &mut rng);
    let b = make_random_genome(&cfg, &mut rng);
    // Both parents are length 12 → target_len = 12 → child always length 12.
    let expected_len = (a.len() + b.len()) / 2;
    let pool = vec![a, b];
    let params = ReproductionParams {
        sexual: true,
        choose_by_fitness: false,
        mutation_rate: 0.0,
        insertion_deletion_rate: 0.0,
        deletion_ratio: 0.5,
        max_len: 100,
    };
    for seed in 0..20u64 {
        let mut r = Rng::seeded(seed);
        let child = generate_child_genome(&pool, &params, &mut r);
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
    let pool = vec![p1, p2];
    let params = ReproductionParams {
        sexual: true,
        choose_by_fitness: false,
        mutation_rate: 0.0,
        insertion_deletion_rate: 0.0,
        deletion_ratio: 0.5,
        max_len: 100,
    };
    let child = generate_child_genome(&pool, &params, &mut rng);
    assert!(
        !child.is_empty(),
        "sexual reproduction from two non-empty parents must produce a non-empty child"
    );
}
