//! Native CLI binary.
//!
//! Runs the simulation headless at maximum speed for a given number of
//! generations, with a progress bar and optional multi-threading.
//!
//! ```text
//! biosim4-native [--config FILE] [--generations N] [--threads N]
//!                [--seed N] [--quiet] [--verbose]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use biosim4_core::{
    SimConfig, SimulationState,
    analysis::{collect_epoch_stats, display_sample_genomes},
    sim_step::step_generation,
    spawn::spawn_new_generation,
};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser, Debug)]
#[command(name = "biosim4-native", about = "Run biosim4 headless at maximum speed")]
struct Args {
    /// Path to a JSON config file (overrides defaults; CLI flags override the file).
    #[arg(long, short = 'c')]
    config: Option<PathBuf>,

    /// Number of generations to simulate. Overrides config.max_generations.
    #[arg(long, short = 'g')]
    generations: Option<u32>,

    /// Worker threads for parallel Phase 1 (sensor + neural net evaluation).
    /// `0` = use all available cores. `1` = sequential. Overrides config.num_threads.
    #[arg(long, short = 't')]
    threads: Option<u32>,

    /// RNG seed. `0` = non-deterministic (entropy). Overrides config.rng_seed.
    #[arg(long)]
    seed: Option<u64>,

    /// Suppress per-generation log lines (progress bar still shown).
    #[arg(long, short = 'q', conflicts_with = "verbose")]
    quiet: bool,

    /// Print per-generation stats above the progress bar (default: every 10th gen).
    #[arg(long, short = 'v')]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    // ── Load config ──────────────────────────────────────────────────────
    let mut config = match &args.config {
        Some(path) => {
            let json = std::fs::read_to_string(path).unwrap_or_else(|e| {
                eprintln!("Cannot read config {}: {e}", path.display());
                std::process::exit(1);
            });
            SimConfig::from_json(&json).unwrap_or_else(|e| {
                eprintln!("Invalid config JSON: {e}");
                std::process::exit(1);
            })
        }
        None => SimConfig::default(),
    };

    if let Some(g) = args.generations { config.max_generations = g; }
    if let Some(s) = args.seed        { config.rng_seed = s; }
    if let Some(t) = args.threads {
        config.num_threads = if t == 0 { num_cpus_available() } else { t };
    } else if config.num_threads == 0 {
        config.num_threads = num_cpus_available();
    }

    // ── Configure rayon thread pool ──────────────────────────────────────
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.num_threads as usize)
        .build_global()
        .ok(); // ignore "already initialized" — only happens in tests

    eprintln!(
        "biosim4-rs  {}×{}  pop={}  gens={}  threads={}  seed={}",
        config.size_x,
        config.size_y,
        config.population,
        config.max_generations,
        config.num_threads,
        config.rng_seed,
    );

    let max_generations = config.max_generations;
    let analysis_stride = config.genome_analysis_stride.max(1);
    let display_sample_count = config.display_sample_genomes as usize;
    let verbose = args.verbose;
    let quiet = args.quiet;

    let mut state = SimulationState::new(config);

    // ── Progress bar ─────────────────────────────────────────────────────
    let bar = ProgressBar::new(max_generations as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{bar:40.cyan/blue} {pos}/{len} gens  {percent}%  elapsed {elapsed}  eta {eta}  {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let start = Instant::now();

    for _ in 0..max_generations {
        step_generation(&mut state);
        let survivors = spawn_new_generation(&mut state);
        let stats = collect_epoch_stats(&mut state, survivors);

        let log_this_gen = if quiet {
            false
        } else if verbose {
            true
        } else {
            stats.generation % 10 == 0 || stats.generation == max_generations
        };

        if log_this_gen {
            let line = format!(
                "Gen {:>4}  survivors: {:>5}/{:<5}  ({:>5.1}%)  diversity: {:.4}",
                stats.generation,
                stats.survivors,
                stats.population,
                stats.survival_rate() * 100.0,
                stats.diversity,
            );
            // When the bar is hidden (non-TTY: piping, redirecting, CI), its
            // `println` is suppressed. Fall back to eprintln so stats reach
            // the user either way.
            if bar.is_hidden() {
                eprintln!("{line}");
            } else {
                bar.println(line);
            }
        }

        if display_sample_count > 0
            && verbose
            && stats.generation % analysis_stride == 0
        {
            // display_sample_genomes prints to stdout; capture-and-emit via bar
            // would require restructuring, so just route through bar.suspend.
            bar.suspend(|| display_sample_genomes(&state, display_sample_count));
        }

        bar.set_message(format!(
            "survivors {}/{}",
            stats.survivors, stats.population
        ));
        bar.inc(1);
    }

    bar.finish_and_clear();

    let elapsed = start.elapsed();
    let gens_per_sec = max_generations as f64 / elapsed.as_secs_f64();
    eprintln!(
        "Done. {} generations in {:.2}s  ({:.2} gen/s).",
        max_generations,
        elapsed.as_secs_f64(),
        gens_per_sec,
    );
}

fn num_cpus_available() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}
