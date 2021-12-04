#![allow(dead_code)]
#![allow(unused_variables)]

use std::f32::consts::PI;

use rand::{thread_rng, Rng};

use crate::Parameters;
use crate::population::genome::similarity::{genome_similarity, SimilarityMetric};
use crate::population::brain::sensor_actions::Sensor;
use crate::population::genome::Genome;
use crate::population::individual::Individual;
use crate::simulation::signals::Signals;
use crate::simulation::types::{Coord, Dir};
use crate::simulation::world::World;

fn long_probe_population_forward_sensor(start_location: Coord, direction: Dir, range: u32, world: &World) -> u32 {
    let mut location = start_location;
    let mut count = 0;
    while count < range && world.is_in_bounds(location) && world.is_empty_at(location) {
        location = location + direction;
        count += 1;
    }
    if !world.is_in_bounds(location) || world.is_barrier_at(location) {
        return range;
    } else {
        return count;
    }
}

fn long_probe_barrier_forward_sensor(start_location: Coord, direction: Dir, range: u32, world: &World) -> u32 {
    let mut location = start_location;
    let mut count = 0;
    while count < range && world.is_in_bounds(location) && !world.is_barrier_at(location) {
        location = location + direction;
        count += 1;
    }
    if !world.is_in_bounds(location) {
        return range;
    } else {
        return count;
    }
}

fn population_density(start_location: Coord, direction: Dir, range: u32, world: &World) -> f32 {
    let mut sum = 0.0;
    world.apply_neighborhood_to_f(start_location, range as i16, |coord: Coord| {
        if start_location != coord && world.is_occupied_at(coord) {
            let offset = coord - start_location;
            let angle = offset.ray_sameness_dir(direction);
            let distance = f32::sqrt((offset.0*offset.0 + offset.1*offset.1) as f32);
            let scaled = (1.0 / distance) * angle;
            sum += scaled as f32;
        }
    });
    let max_sum = 6.0 * range as f32;
    let sensor_val = sum/max_sum;
    return (sensor_val + 1.0) / 2.0;
}

fn short_probe_barrier_distance(location: Coord, dir: Dir, range: u32, world: &World) -> f32 {
    let mut count_forward = 0u32;
    let mut current_location = location + dir;
    while count_forward < range && world.is_in_bounds(current_location) && !world.is_barrier_at(current_location) {
        count_forward += 1;
        current_location = current_location + dir;
    }

    if !world.is_in_bounds(current_location) {
        count_forward = range;
    }

    let mut count_backward = 0u32;
    current_location = location - dir;
    while count_backward < range && world.is_in_bounds(current_location) && !world.is_barrier_at(current_location) {
        count_backward += 1;
        current_location = current_location - dir;
    }
    if !world.is_in_bounds(current_location) {
        count_backward = range;
    }

    let sensor_value = ((count_forward - count_backward) + range) as f32;
    return sensor_value / (2.0 * range as f32);
}

pub fn get_sensor_dispatch(sensor: &Sensor) -> fn(&Individual, &Vec<Genome>, &World, &Signals, &Parameters, u32) -> f32 {
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

fn loc_x(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    (individual.location.0 / (world.width as i16 - 1)) as f32
}

fn loc_y(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    (individual.location.1 / (world.height as i16 - 1)) as f32
}

fn boundary_distance_x(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    let distance_x = i16::min(individual.location.0, (world.width as i16 - individual.location.0 - 1) as i16);
    return distance_x as f32/(world.width as f32 /2.0)
}

fn boundary_distance(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    let distance_x = i16::min(individual.location.0, (world.width as i16 - individual.location.0 - 1) as i16);
    let distance_y = i16::min(individual.location.1, (world.height as i16 - individual.location.1 - 1) as i16);
    let closest_distance = i16::min(distance_x, distance_y);
    let max_possible = u16::max(world.width/2 - 1, world.height/2 - 1);
    return closest_distance as f32/max_possible as f32
}

fn boundary_distance_y(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    let distance_y = i16::min(individual.location.1, (world.height as i16 - individual.location.1 - 1) as i16);
    return distance_y as f32/(world.height as f32 /2.0)
}

fn genetic_similitude_fwd(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    let loc2 = individual.location + individual.last_move_direction;
    if world.is_in_bounds(loc2) && world.is_occupied_at(loc2) {
        let other_genome = population_genomes.get(world.at_coord(loc2) as usize);
        match other_genome {
            Some(other_genome) => {
                return genome_similarity(&individual.genome , &other_genome, SimilarityMetric::JaroWinkler);
            },
            None => {
                return 0.0;
            }
        }
    }
    return 0.0;
}

fn last_move_dir_x(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    let last_x: Coord = individual.last_move_direction.into();
    match last_x.0 {
        0 => 0.5,
        1 => 1.0,
        -1 => 0.0,
        _ => panic!("Invalid last move direction")
    }
}

fn last_move_dir_y(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    let last_y: Coord = individual.last_move_direction.into();
    match last_y.1 {
        0 => 0.5,
        1 => 1.0,
        -1 => 0.0,
        _ => panic!("Invalid last move direction")
    }
}

fn long_probe_population_fwd(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    let direction = individual.last_move_direction;
    return long_probe_population_forward_sensor(individual.location, direction, p.long_probe_distance, world) as f32;
}

fn long_probe_barrier_fwd(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    let direction = individual.last_move_direction;
    return long_probe_barrier_forward_sensor(individual.location, direction, p.long_probe_distance, world) as f32;
}

fn population(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    let location = individual.location;
    let mut occupied= 0;
    let mut checked = 0;
    world.apply_neighborhood_to_f(location, p.population_sensor_radius, |coord: Coord| {
        checked += 1;
        if world.is_occupied_at(coord) {
            occupied += 1;
        }
    });
    return occupied as f32/checked as f32;
}

fn population_fwd(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    return population_density(individual.location, individual.last_move_direction, p.long_probe_distance, world);
}

fn population_lr(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    return population_density(individual.location, individual.last_move_direction.rotate90deg_cw(), p.long_probe_distance, world);
}

fn oscillation(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    let phase = (simulation_step % individual.oscillation_period) as f32 / individual.oscillation_period as f32;
    let mut factor = -f32::cos(phase * 2.0 * PI);
    factor += 1.0;
    factor /= 2.0;
    //Clip any round off errors
    return factor.clamp(0.0, 1.0);
}

fn age(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    return (individual.age / p.steps_per_generation as u32) as f32;
}

fn barrier_fwd(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    return short_probe_barrier_distance(individual.location, individual.last_move_direction, p.long_probe_distance, world);
}

fn barrier_lr(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    return short_probe_barrier_distance(individual.location, individual.last_move_direction.rotate90deg_cw(), p.long_probe_distance, world);
}

fn random(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    return thread_rng().gen_range(0.0..=1.0);
}

fn signal(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    0.0
}

fn signal_fwd(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    0.0
}

fn signal_lr(individual: &Individual, population_genomes: &Vec<Genome>,  world: &World, signals: &Signals, p: &Parameters, simulation_step: u32) -> f32 {
    0.0
}