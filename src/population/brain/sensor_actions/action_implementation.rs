use std::cell::Cell;
use crate::population::brain::sensor_actions::Action;
use crate::population::individual::Individual;
use crate::simulation::peeps::Peeps;

// Gets the function corresponding to the given action, which accepts za
// individual, a grid, and the input level.
pub fn get_action_dispatch(action: &Action) -> fn(&mut Individual, &Cell<Peeps>, f32) {
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

fn move_x(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn move_y(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn move_forward(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn move_rl(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn move_random(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn set_oscillator_period(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn set_long_probe_distance(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn set_responsiveness(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn emit_signal0(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn move_east(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn move_west(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn move_north(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn move_south(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn move_left(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn move_right(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn move_reverse(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}

fn kill_forward(individual: &mut Individual, peeps: &Cell<Peeps>, level: f32) {}