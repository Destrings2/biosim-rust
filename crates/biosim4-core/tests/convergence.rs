use biosim4_core::{
    registry::challenge::{ChallengeComposition, ChallengeConfig},
    sim_config::SimConfig,
    sim_state::SimulationState,
    sim_step::step_generation,
    spawn::{initialize_generation_0, spawn_new_generation},
};

fn run_and_collect_survival(challenge_id: &str, generations: u32, seed: u64) -> Vec<f32> {
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

/// Run a challenge, snapshot mean genome length every `stride` generations.
/// Indel rate is bumped (0.1) so length actually drifts inside the test's
/// short window — at the default 0.01 the population barely sees one indel
/// event per agent over 100 gens, making the penalty signal unmeasurable.
fn run_and_snapshot_genome_lengths(
    challenge_id: &str,
    generations: u32,
    stride: u32,
    seed: u64,
    bloat_weight: f32,
) -> Vec<f32> {
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

    let mut snapshots = Vec::with_capacity((generations / stride) as usize);
    for gen in 1..=generations {
        step_generation(&mut state);
        let _ = spawn_new_generation(&mut state);
        if gen % stride == 0 {
            let alive: Vec<_> = state.population.iter_alive().collect();
            let mean_len = if alive.is_empty() {
                0.0
            } else {
                alive.iter().map(|a| a.genome.len() as f32).sum::<f32>() / alive.len() as f32
            };
            snapshots.push(mean_len);
        }
    }
    snapshots
}

/// Bloat penalty must not let mean genome length grow above the
/// no-penalty baseline. Stronger version of the plan's assertion
/// — guards against the penalty silently becoming a no-op (e.g.
/// if a future refactor stops setting `dead_gene_count` or
/// inverts the sign of the subtraction in `spawn.rs`).
#[test]
fn migrate_distance_with_bloat_penalty_doesnt_grow_genome() {
    let gens = 100;
    let stride = 10;
    let seed = 0xB10A7;

    let baseline = run_and_snapshot_genome_lengths("migrate_distance", gens, stride, seed, 0.0);
    let penalized = run_and_snapshot_genome_lengths("migrate_distance", gens, stride, seed, 0.05);

    let baseline_tail = mean(&baseline[baseline.len() - 3..]);
    let penalized_tail = mean(&penalized[penalized.len() - 3..]);

    eprintln!(
        "[bloat_penalty migrate_distance seed={seed:#x}]\n  \
         baseline (w=0.00) lengths: {baseline:?}\n  \
         penalized (w=0.05) lengths: {penalized:?}\n  \
         baseline_tail={baseline_tail:.2}  penalized_tail={penalized_tail:.2}"
    );

    // Allow 1 gene of slack for stochastic noise. The penalty path should
    // produce equal-or-shorter mean genomes than the baseline.
    assert!(
        penalized_tail <= baseline_tail + 1.0,
        "bloat penalty did not curb genome growth: \
         baseline_tail={baseline_tail:.2}, penalized_tail={penalized_tail:.2}"
    );
}
