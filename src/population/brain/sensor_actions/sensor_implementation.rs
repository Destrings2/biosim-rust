use crate::population::brain::sensor_actions::Sensor;
use crate::population::individual::Individual;
use crate::simulation::grid::Grid;

pub fn get_sensor_dispatch(sensor: &Sensor) -> fn(&Individual, &Grid, u32) -> f32 {
    match sensor {
        Sensor::LocX => loc_x,
        Sensor::LocY => loc_y,
        Sensor::BoundaryDistX => boundary_distance_x,
        Sensor::BoundaryDist => boundary_distance,
        Sensor::BoundaryDistY => boundary_distance_y,
        Sensor::GeneticSimFwd => genetic_similitude_fwd,
        Sensor::LastMoveDirX => last_move_dir_x,
        Sensor::LastMoveDirY => last_move_dir_y,
        Sensor::LongProbePopFwd => long_probe_population_fwd,
        Sensor::LongProbeBarFwd => long_probe_barrier_fwd,
        Sensor::Population => population,
        Sensor::PopulationFwd => population_fwd,
        Sensor::PopulationLR => population_lr,
        Sensor::Osc1 => oscillation,
        Sensor::Age => age,
        Sensor::BarrierFwd => barrier_fwd,
        Sensor::BarrierLR => barrier_lr,
        Sensor::Rnd => random,
        Sensor::Signal0 => signal,
        Sensor::Signal0Fwd => signal_fwd,
        Sensor::Signal0LR => signal_lr,
    }
}

fn loc_x(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn loc_y(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn boundary_distance_x(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn boundary_distance(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn boundary_distance_y(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn genetic_similitude_fwd(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn last_move_dir_x(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn last_move_dir_y(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn long_probe_population_fwd(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn long_probe_barrier_fwd(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn population(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn population_fwd(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn population_lr(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn oscillation(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn age(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn barrier_fwd(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn barrier_lr(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn random(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn signal(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn signal_fwd(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}

fn signal_lr(individual: &Individual, grid: &Grid, simulation_step: u32) -> f32 {0.0}