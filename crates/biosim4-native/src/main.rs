use biosim4_core::{
    SimConfig, SimulationState,
    analysis::{collect_epoch_stats, print_epoch_stats, display_sample_genomes},
    sim_step::step_generation,
    spawn::spawn_new_generation,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let config = match args.get(1) {
        Some(path) => {
            let json = std::fs::read_to_string(path)
                .unwrap_or_else(|e| { eprintln!("Cannot read config {path}: {e}"); std::process::exit(1); });
            SimConfig::from_json(&json)
                .unwrap_or_else(|e| { eprintln!("Invalid config JSON: {e}"); std::process::exit(1); })
        }
        None => SimConfig::default(),
    };

    eprintln!("biosim4-rs  {}×{}  pop={}  gens={}",
        config.size_x, config.size_y, config.population, config.max_generations);

    let mut state = SimulationState::new(config);

    for _gen in 0..state.config.max_generations {
        step_generation(&mut state);
        let survivors = spawn_new_generation(&mut state);
        let stats = collect_epoch_stats(&mut state, survivors);
        print_epoch_stats(&stats);

        if state.config.display_sample_genomes > 0 && stats.generation % state.config.genome_analysis_stride == 0 {
            display_sample_genomes(&state, state.config.display_sample_genomes as usize);
        }
    }

    eprintln!("Done.");
}
