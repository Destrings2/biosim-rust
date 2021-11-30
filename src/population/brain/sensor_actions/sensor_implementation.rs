#![allow(dead_code)]
#![allow(unused_variables)]
use crate::Parameters;
use crate::population::brain::sensor_actions::Sensor;
use crate::population::individual::Individual;
use crate::simulation::peeps::Peeps;

pub fn get_sensor_dispatch(sensor: &Sensor) -> fn(&Individual, &Peeps, &Parameters, u32) -> f32 {
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

fn loc_x(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {
    (individual.location.0 / (peeps.world.width as i16 - 1)) as f32
}

fn loc_y(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {
    (individual.location.1 / (peeps.world.height as i16 - 1)) as f32
}

fn boundary_distance_x(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {
    let distance_x = i16::min(individual.location.0, (peeps.world.width as i16 - individual.location.0 - 1) as i16);
    return distance_x as f32/(peeps.world.width as f32 /2.0)
}

fn boundary_distance(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {
    let distance_x = i16::min(individual.location.0, (peeps.world.width as i16 - individual.location.0 - 1) as i16);
    let distance_y = i16::min(individual.location.1, (peeps.world.height as i16 - individual.location.1 - 1) as i16);
    let closest_distance = i16::min(distance_x, distance_y);
    let max_possible = u16::max(peeps.world.width/2 - 1, peeps.world.height/2 - 1);
    return closest_distance as f32/max_possible as f32
}

fn boundary_distance_y(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {
    let distance_y = i16::min(individual.location.1, (peeps.world.height as i16 - individual.location.1 - 1) as i16);
    return distance_y as f32/(peeps.world.height as f32 /2.0)
}

fn genetic_similitude_fwd(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn last_move_dir_x(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn last_move_dir_y(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn long_probe_population_fwd(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn long_probe_barrier_fwd(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn population(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn population_fwd(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn population_lr(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn oscillation(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn age(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn barrier_fwd(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn barrier_lr(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn random(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn signal(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn signal_fwd(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}

fn signal_lr(individual: &Individual, peeps: &Peeps, p: &Parameters, simulation_step: u32) -> f32 {0.0}