//! Genome-health inspector — detect population collapse during long runs.
//!
//! Reports per-stride length percentiles, dead-net counts, survival,
//! diversity, and mean mutation rate so genome-shrinkage or
//! dead-cell pathologies surface immediately.
//!
//! Usage:
//! ```text
//! cargo run -p biosim4-core --example genome_health --release [challenge] [gens] [stride]
//! ```
//!
//! Defaults: `right_half`, 500 generations, 25-gen stride. The challenge
//! id must match a built-in from `biosim4_challenges::register_builtin_challenges`.

use biosim4_core::{
    genome::ops::{genetic_diversity, Genome},
    registry::challenge::{ChallengeComposition, ChallengeConfig},
    sim_config::SimConfig,
    sim_state::SimulationState,
    sim_step::step_generation,
    spawn::{initialize_generation_0, spawn_new_generation},
};

fn main() {
    let challenge = std::env::args().nth(1).unwrap_or_else(|| "right_half".to_string());
    let gens: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let stride: u32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(25);

    let mut cfg = SimConfig::default();
    // Single-thread for reproducible dashboard rows; the parallel
    // stepping path is non-deterministic.
    cfg.num_threads = 1;
    cfg.rng_seed = 0xC0FFEE;

    let mut state = SimulationState::new(cfg);
    biosim4_sensors::register_builtin_sensors(&mut state.sensors);
    biosim4_actions::register_builtin_actions(&mut state.actions);
    biosim4_challenges::register_builtin_challenges(&mut state.challenges);
    state
        .challenges
        .apply_config(ChallengeConfig {
            active: vec![challenge.clone()],
            composition: ChallengeComposition::Any,
            params: Default::default(),
        })
        .expect("challenge not registered");
    initialize_generation_0(&mut state);

    println!(
        "config: pop={pop} steps_per_gen={spg} grid={sx}x{sy} mut_rate={mut_rate:.4} \
         indel={indel:.4} sexual={sexual} tournament={tk} elitism={elite} adaptive={ad}",
        pop = state.config.population,
        spg = state.config.steps_per_generation,
        sx = state.config.size_x,
        sy = state.config.size_y,
        mut_rate = state.config.point_mutation_rate,
        indel = state.config.gene_insertion_deletion_rate,
        sexual = state.config.sexual_reproduction,
        tk = state.config.tournament_size,
        elite = state.config.elitism_count,
        ad = state.config.adaptive_mutation,
    );
    println!("challenge={challenge} gens={gens} stride={stride}");
    println!();
    println!(
        "{:>5} {:>6} {:>5} {:>5} {:>5} {:>5} {:>5} {:>6} {:>6} {:>6} {:>6} {:>8} {:>9}",
        "gen",
        "alive",
        "minL",
        "p25L",
        "medL",
        "p75L",
        "maxL",
        "deadN",
        "0conn",
        "noAct",
        "surv%",
        "diverz",
        "mut_rate"
    );

    // Per-stride snapshot. Three failure modes get their own columns:
    // `0conn` = empty genome, `deadN` = nnet has neither neurons nor
    // action connections, `noAct` = nnet has no action connections at
    // all (the agent is structurally inert even if neurons exist).
    fn dump(state: &SimulationState, g: u32, last_surv_pct: f32) {
        let alive: Vec<_> = state.population.iter_alive().collect();
        if alive.is_empty() {
            println!("{:>5} {:>6} (no alive agents)", g, 0);
            return;
        }
        let mut lengths: Vec<usize> = alive.iter().map(|a| a.genome.len()).collect();
        lengths.sort_unstable();
        let n = lengths.len();
        let p = |q: f32| {
            let idx = ((n as f32 * q) as usize).min(n - 1);
            lengths[idx]
        };

        let dead_nnet = alive
            .iter()
            .filter(|a| a.nnet.neurons.is_empty() && a.nnet.action_connections.is_empty())
            .count();
        let no_action = alive.iter().filter(|a| a.nnet.action_connections.is_empty()).count();
        let zero_genome = alive.iter().filter(|a| a.genome.is_empty()).count();

        let mean_rate: f32 =
            alive.iter().map(|a| a.mutation_rate).sum::<f32>() / alive.len() as f32;

        let genome_refs: Vec<&Genome> = alive.iter().map(|a| &a.genome).collect();
        let mut div_rng = biosim4_core::rng::Rng::seeded(0xD1B45 ^ g as u64);
        let div = genetic_diversity(&genome_refs, 0, &mut div_rng);

        println!(
            "{:>5} {:>6} {:>5} {:>5} {:>5} {:>5} {:>5} {:>6} {:>6} {:>6} {:>5.1}% {:>8.3} {:>9.4}",
            g,
            alive.len(),
            lengths[0],
            p(0.25),
            p(0.50),
            p(0.75),
            lengths[n - 1],
            zero_genome,
            dead_nnet,
            no_action,
            last_surv_pct,
            div,
            mean_rate
        );
    }

    let pop_total = state.config.population as f32;

    // Baseline row before any reproduction.
    dump(&state, 0, 0.0);

    for g in 1..=gens {
        step_generation(&mut state);
        let survivors = spawn_new_generation(&mut state);
        let last_surv_pct = (survivors as f32 / pop_total) * 100.0;
        if g % stride == 0 || g == gens {
            dump(&state, g, last_surv_pct);
        }
    }

    println!();
    println!("final 5 sample agents:");
    for (i, a) in state.population.iter_alive().take(5).enumerate() {
        let neurons = a.nnet.neurons.len();
        let n_conns = a.nnet.neuron_connections.len();
        let a_conns = a.nnet.action_connections.len();
        let driven = a.nnet.neurons.iter().filter(|n| n.driven).count();
        println!(
            "  [{i}] genome_len={glen:>3} neurons={neurons:>2} (driven={driven:>2}) \
             neuron_conns={nc:>2} action_conns={ac:>2} mut_rate={mr:.4} loc=({x:>3},{y:>3})",
            i = i,
            glen = a.genome.len(),
            neurons = neurons,
            driven = driven,
            nc = n_conns,
            ac = a_conns,
            mr = a.mutation_rate,
            x = a.loc.x,
            y = a.loc.y
        );
        if a.genome.len() < 8 {
            let raw: Vec<String> = a.genome.iter().map(|g| format!("{:08x}", g.0)).collect();
            println!("       genome bytes: [{}]", raw.join(", "));
        }
    }
}
