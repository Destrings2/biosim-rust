use biosim4_core::{
    registry::challenge::{ChallengeComposition, ChallengeConfig},
    sim_config::SimConfig,
    sim_state::SimulationState,
    sim_step::step_generation,
    spawn::{initialize_generation_0, spawn_new_generation},
};

fn run_and_collect_survival(challenge_id: &str, generations: u32, seed: u64) -> Vec<f32> {
    run_with_speciation(challenge_id, generations, seed, false)
}

fn run_with_speciation(
    challenge_id: &str,
    generations: u32,
    seed: u64,
    enable_speciation: bool,
) -> Vec<f32> {
    // Default to the configured speciation method (currently 3 = Network
    // topology) when speciation is on; ignored when it's off.
    run_with_speciation_method(
        challenge_id,
        generations,
        seed,
        enable_speciation,
        SimConfig::default().speciation_similarity_method,
    )
}

fn run_with_speciation_method(
    challenge_id: &str,
    generations: u32,
    seed: u64,
    enable_speciation: bool,
    speciation_method: u8,
) -> Vec<f32> {
    let mut cfg = SimConfig::default();
    cfg.size_x = 96;
    cfg.size_y = 96;
    cfg.population = 400;
    cfg.steps_per_generation = 150;
    cfg.rng_seed = seed;
    cfg.point_mutation_rate = 0.01;
    cfg.barrier_type = 0;
    // Single-thread for stable assertions — the multi-threaded stepping
    // path is intentionally non-deterministic, so parallel runs would make
    // the per-generation survival series flaky against a fixed threshold.
    cfg.num_threads = 1;
    cfg.enable_speciation = enable_speciation;
    cfg.speciation_similarity_method = speciation_method;

    let mut state = SimulationState::new(cfg);
    biosim4_sensors::register_builtin_sensors(&mut state.sensors);
    biosim4_actions::register_builtin_actions(&mut state.actions);
    biosim4_challenges::register_builtin_challenges(&mut state.challenges);
    state
        .challenges
        .apply_config(ChallengeConfig {
            active: vec![challenge_id.to_string()],
            composition: ChallengeComposition::Any,
            params: Default::default(),
        })
        .expect("set challenge");

    initialize_generation_0(&mut state);

    let pop = state.config.population as f32;
    let mut rates = Vec::with_capacity(generations as usize);
    for _ in 0..generations {
        step_generation(&mut state);
        let survivors = spawn_new_generation(&mut state) as f32;
        rates.push(survivors / pop);
    }
    rates
}

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len() as f32
}

fn assert_improves(challenge: &str, rates: &[f32], min_gain: f32) {
    let n = rates.len();
    let head = mean(&rates[0..5]);
    let tail = mean(&rates[n - 5..n]);
    let gain = tail - head;
    eprintln!(
        "[{}] head_5={:.3} tail_5={:.3} gain={:.3} (need >= {:.2})\n  series: {:?}",
        challenge,
        head,
        tail,
        gain,
        min_gain,
        rates.iter().map(|r| (r * 100.0).round() / 100.0).collect::<Vec<_>>(),
    );
    assert!(
        gain >= min_gain,
        "{}: survival rate did not improve enough (gain {:.3} < {:.2})",
        challenge,
        gain,
        min_gain,
    );
}

#[test]
fn radioactive_walls_population_converges() {
    let rates = run_and_collect_survival("radioactive_walls", 50, 42);
    assert_improves("radioactive_walls", &rates, 0.05);
}

#[test]
fn migrate_distance_population_converges() {
    let rates = run_and_collect_survival("migrate_distance", 40, 7);
    assert_improves("migrate_distance", &rates, 0.10);
}

/// Run a challenge, snapshot mean genome length AND survival rate every
/// `stride` generations. Returns `(lengths, survival_rates)`
/// parallel-indexed. Indel rate is bumped (0.1) so length actually drifts
/// inside the test's short window — at the default 0.01 the population
/// barely sees one indel event per agent over 100 gens, making the
/// penalty signal unmeasurable.
fn run_and_snapshot_health(
    challenge_id: &str,
    generations: u32,
    stride: u32,
    seed: u64,
    bloat_weight: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut cfg = SimConfig::default();
    cfg.size_x = 96;
    cfg.size_y = 96;
    cfg.population = 400;
    cfg.steps_per_generation = 150;
    cfg.rng_seed = seed;
    cfg.point_mutation_rate = 0.01;
    cfg.gene_insertion_deletion_rate = 0.1;
    cfg.deletion_ratio = 0.5;
    cfg.barrier_type = 0;
    cfg.num_threads = 1;
    cfg.bloat_penalty_weight = bloat_weight;

    let pop_f = cfg.population as f32;

    let mut state = SimulationState::new(cfg);
    biosim4_sensors::register_builtin_sensors(&mut state.sensors);
    biosim4_actions::register_builtin_actions(&mut state.actions);
    biosim4_challenges::register_builtin_challenges(&mut state.challenges);
    state
        .challenges
        .apply_config(ChallengeConfig {
            active: vec![challenge_id.to_string()],
            composition: ChallengeComposition::Any,
            params: Default::default(),
        })
        .expect("set challenge");

    initialize_generation_0(&mut state);

    let cap = (generations / stride) as usize;
    let mut lengths = Vec::with_capacity(cap);
    let mut survivals = Vec::with_capacity(cap);
    for gen in 1..=generations {
        step_generation(&mut state);
        let survivors = spawn_new_generation(&mut state);
        if gen % stride == 0 {
            let alive: Vec<_> = state.population.iter_alive().collect();
            let mean_len = if alive.is_empty() {
                0.0
            } else {
                alive.iter().map(|a| a.genome.len() as f32).sum::<f32>() / alive.len() as f32
            };
            lengths.push(mean_len);
            survivals.push(survivors as f32 / pop_f);
        }
    }
    (lengths, survivals)
}

/// Bloat penalty must (a) not let mean genome length grow above the
/// no-penalty baseline and (b) not regress survival convergence
/// meaningfully. The survival guard is the more important of the two:
/// the first iteration of this penalty was a linear curve that quietly
/// hurt convergence on real runs even at small weights (0.02), because
/// it punished exploring lineages whose mutations transiently raised
/// dead_norm. The quadratic curve in `spawn.rs` is supposed to keep
/// moderate-bloat agents almost untouched — this test pins that
/// property so future tweaks can't silently break it again.
#[test]
fn migrate_distance_with_bloat_penalty_doesnt_grow_genome() {
    let gens = 100;
    let stride = 10;
    let seed = 0xB10A7;

    let (baseline_lens, baseline_surv) =
        run_and_snapshot_health("migrate_distance", gens, stride, seed, 0.0);
    let (penalized_lens, penalized_surv) =
        run_and_snapshot_health("migrate_distance", gens, stride, seed, 0.05);

    // Tail-5 means (the last half of the 10 snapshots) instead of tail-3:
    // length and especially survival are noisy from gen to gen, and a 3-sample
    // window aliases that noise straight into the assertion. 5 samples
    // averages out the per-gen swing while still capturing late-run dynamics.
    let baseline_len_tail = mean(&baseline_lens[baseline_lens.len() - 5..]);
    let penalized_len_tail = mean(&penalized_lens[penalized_lens.len() - 5..]);
    let baseline_surv_tail = mean(&baseline_surv[baseline_surv.len() - 5..]);
    let penalized_surv_tail = mean(&penalized_surv[penalized_surv.len() - 5..]);

    eprintln!(
        "[bloat_penalty migrate_distance seed={seed:#x}]\n  \
         baseline  (w=0.00) lengths:   {baseline_lens:?}\n  \
         penalized (w=0.05) lengths:   {penalized_lens:?}\n  \
         baseline  (w=0.00) survival:  {baseline_surv:?}\n  \
         penalized (w=0.05) survival:  {penalized_surv:?}\n  \
         len_tail   baseline={baseline_len_tail:.2}  penalized={penalized_len_tail:.2}\n  \
         surv_tail  baseline={baseline_surv_tail:.3} penalized={penalized_surv_tail:.3}"
    );

    // Length guard: penalty should keep mean genome length at or below
    // baseline. 1 gene of slack for stochastic noise.
    assert!(
        penalized_len_tail <= baseline_len_tail + 1.0,
        "bloat penalty did not curb genome growth: \
         baseline_len_tail={baseline_len_tail:.2}, penalized_len_tail={penalized_len_tail:.2}"
    );

    // Survival guard: penalty must not cost more than 2% absolute
    // convergence. The linear curve at weight=0.05 was empirically much
    // worse than this; the quadratic curve should land well inside it.
    let max_surv_regression = 0.02;
    assert!(
        penalized_surv_tail >= baseline_surv_tail - max_surv_regression,
        "bloat penalty regressed survival by more than {max_surv_regression:.2}: \
         baseline_surv_tail={baseline_surv_tail:.3}, penalized_surv_tail={penalized_surv_tail:.3}"
    );
}

/// Speciation A/B/C on `sun_tracker`.
///
/// Three conditions compared per seed:
///   - **baseline**: no speciation, plain tournament selection
///   - **bitstring**: speciation with `method = 0` (Jaro-Winkler on raw
///     gene bytes — the historically-broken approach)
///   - **topology**: speciation with `method = 3` (Jaccard on the
///     post-cull NN edge set with coarse weight bucketing — the new
///     default)
///
/// The topology metric was added precisely because bitstring buckets
/// were behaviourally meaningless and routinely *hurt* convergence on
/// this challenge (see git history of this test). The hard guard is
/// that topology must not regress vs the no-speciation baseline by
/// more than 1% absolute — if it does, the new metric or one of its
/// supporting pieces (cache, fingerprinting, weight bucketing) has
/// broken. The bitstring number is logged for context but not asserted
/// on; we expect it to be ≤ baseline and don't want to pin a sad
/// number into the test as truth.
#[test]
fn sun_tracker_speciation_parity() {
    let seeds = [0xBEEF42, 0xC0FFEE, 0xDEAD99];
    let gens = 200;

    let mut baseline_means = Vec::with_capacity(seeds.len());
    let mut bitstring_means = Vec::with_capacity(seeds.len());
    let mut topology_means = Vec::with_capacity(seeds.len());
    for &seed in &seeds {
        let base = run_with_speciation_method("sun_tracker", gens, seed, false, 0);
        let bits = run_with_speciation_method("sun_tracker", gens, seed, true, 0);
        let topo = run_with_speciation_method("sun_tracker", gens, seed, true, 3);
        let base_tail = mean(&base[base.len() - 10..]);
        let bits_tail = mean(&bits[bits.len() - 10..]);
        let topo_tail = mean(&topo[topo.len() - 10..]);
        eprintln!(
            "[sun_tracker seed={seed:#x}] baseline={base_tail:.3} \
             bitstring={bits_tail:.3} topology={topo_tail:.3}"
        );
        baseline_means.push(base_tail);
        bitstring_means.push(bits_tail);
        topology_means.push(topo_tail);
    }

    let baseline_mean = mean(&baseline_means);
    let bitstring_mean = mean(&bitstring_means);
    let topology_mean = mean(&topology_means);
    eprintln!(
        "[sun_tracker] baseline={baseline_mean:.3} bitstring={bitstring_mean:.3} \
         topology={topology_mean:.3}"
    );

    // Topology must not regress vs baseline by more than 1% absolute.
    // Tightened from the previous 5% parity guard now that the
    // bit-similarity-clusters-noise problem is solved.
    let max_regression = 0.01;
    assert!(
        topology_mean >= baseline_mean - max_regression,
        "sun_tracker topology speciation regressed by more than {max_regression:.2} vs baseline \
         (baseline={baseline_mean:.3}, topology={topology_mean:.3}, \
         bitstring={bitstring_mean:.3})"
    );
}
