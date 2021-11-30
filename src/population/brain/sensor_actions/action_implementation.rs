#![allow(unused_variables)]
use std::cell::RefCell;
use crate::Parameters;
use crate::population::brain::sensor_actions::Action;
use crate::population::individual::Individual;
use crate::simulation::peeps::Peeps;
use crate::simulation::types::{Coord, Dir};

// Gets the function corresponding to the given action, which accepts za
// individual, a grid, and the input level.
pub fn get_action_dispatch(action: &Action) -> fn(&RefCell<Individual>, &RefCell<Peeps>, &Parameters, f32) {
    match action {
        Action::MoveX => move_x,
        Action::MoveY => move_y,
        Action::MoveForward => move_forward,
        Action::MoveRL => move_rl,
        Action::MoveRandom => move_random,
        Action::SetOscillatorPeriod => set_oscillator_period,
        Action::SetLongProbeDist => set_long_probe_distance,
        Action::SetResponsiveness => set_responsiveness,
        Action::EmitSignal0 => emit_signal0,
        Action::MoveEast => move_east,
        Action::MoveWest => move_west,
        Action::MoveNorth => move_north,
        Action::MoveSouth => move_forward,
        Action::MoveLeft => move_left,
        Action::MoveRight => move_right,
        Action::MoveReverse => move_reverse,
        Action::KillForward => kill_forward,
    }
}

fn move_x(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let individual = individual_ref.borrow();
    let mut peeps = peeps.borrow_mut();
    peeps.queue_for_move(individual.index, (level, 0.0));
}

fn move_y(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let individual = individual_ref.borrow();
    let mut peeps = peeps.borrow_mut();
    peeps.queue_for_move(individual.index, (0.0, level));
}

fn move_forward(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let individual = individual_ref.borrow();
    let mut peeps = peeps.borrow_mut();
    let last_move_offset: Coord = individual.last_move_direction.into();
    peeps.queue_for_move(individual.index, (last_move_offset.0 as f32 * level,
                                                   last_move_offset.1 as f32 *level));
}

fn move_rl(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let individual = individual_ref.borrow();
    let mut peeps = peeps.borrow_mut();
    let last_move_offset: Coord = individual.last_move_direction.rotate90deg_cw().into();
    peeps.queue_for_move(individual.index, (last_move_offset.0 as f32 * level,
                                                   last_move_offset.1 as f32 * -level));
}

fn move_random(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let offset: Coord = Dir::random().into();
    let individual = individual_ref.borrow();
    let mut peeps = peeps.borrow_mut();
    peeps.queue_for_move(individual.index, (offset.0 as f32 * level, offset.1 as f32 * level));
}

fn set_oscillator_period(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let mut individual = individual_ref.borrow_mut();
    let exponent = (f32::tanh(level) + 1.0)/2.0;
    let new_period = 1 + (1.5 + f32::exp(7.0 * exponent)) as u32;
    individual.oscillation_period = new_period;
}

fn set_long_probe_distance(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let mut individual = individual_ref.borrow_mut();
    let normalized_level = (f32::tanh(level) + 1.0)/2.0;
    individual.long_probe_distance += 1 + (normalized_level * p.long_probe_distance as f32) as u32;
}

fn set_responsiveness(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let mut individual = individual_ref.borrow_mut();
    let normalized_level = (f32::tanh(level) + 1.0)/2.0;
    individual.responsiveness += normalized_level;
}

//TODO
fn emit_signal0(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {}

fn move_east(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let individual = individual_ref.borrow();
    let mut peeps = peeps.borrow_mut();
    peeps.queue_for_move(individual.index, (level, 0.0));
}


fn move_west(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let individual = individual_ref.borrow();
    let mut peeps = peeps.borrow_mut();
    peeps.queue_for_move(individual.index, (-level, 0.0));
}

fn move_north(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let individual = individual_ref.borrow();
    let mut peeps = peeps.borrow_mut();
    peeps.queue_for_move(individual.index, (0.0, level));
}

fn move_south(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let individual = individual_ref.borrow();
    let mut peeps = peeps.borrow_mut();
    peeps.queue_for_move(individual.index, (0.0, -level));
}

fn move_left(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let individual = individual_ref.borrow();
    let mut peeps = peeps.borrow_mut();
    let last_move_offset: Coord = individual.last_move_direction.rotate90deg_ccw().into();
    peeps.queue_for_move(individual.index, (last_move_offset.0 as f32 * level,
                                                   last_move_offset.1 as f32 *level));
}

fn move_right(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let individual = individual_ref.borrow();
    let mut peeps = peeps.borrow_mut();
    let last_move_offset: Coord = individual.last_move_direction.rotate90deg_cw().into();
    peeps.queue_for_move(individual.index, (last_move_offset.0 as f32 * level,
                                                   last_move_offset.1 as f32 *level));
}

fn move_reverse(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {
    let individual = individual_ref.borrow();
    let mut peeps = peeps.borrow_mut();
    let last_move_offset: Coord = individual.last_move_direction.into();
    peeps.queue_for_move(individual.index, (-last_move_offset.0 as f32 * level,
                                                   -last_move_offset.1 as f32 *level));
}

//TODO
fn kill_forward(individual_ref: &RefCell<Individual>, peeps: &RefCell<Peeps>, p: &Parameters, level: f32) {}