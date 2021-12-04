#![allow(dead_code)]

use crate::simulation::parameters::Parameters;
use crate::simulation::simulation::Simulation;

mod simulation;
mod population;
mod util;

use crate::util::event::{Config, Event, Events};
use std::sync::RwLock;
use std::{error::Error, io, time::Duration};
use termion::{event::Key, input::MouseTerminal, raw::IntoRawMode, screen::AlternateScreen};
use tui::widgets::Paragraph;
use tui::{
    backend::TermionBackend,
    layout::{Constraint, Direction, Layout},
    style::Color,
    widgets::{
        canvas::{Canvas, Points},
        Block, Borders,
    },
    Terminal,
};

struct App<'a> {
    pub simulation: RwLock<Simulation<'a>>,
}

impl<'a> App<'a> {
    fn new(parameters: &'a Parameters) -> App<'a> {
        App {
            simulation: RwLock::new(Simulation::initialize(parameters))
        }
    }

    fn update(&mut self) {
        
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Terminal initialization
    let stdout = io::stdout().into_raw_mode()?;
    let stdout = MouseTerminal::from(stdout);
    let stdout = AlternateScreen::from(stdout);
    let backend = TermionBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Setup event handlers
    let config = Config {
        tick_rate: Duration::from_millis(250),
        ..Default::default()
    };
    let events = Events::with_config(config);

    let parameters = Parameters::defaults();

    // App
    let mut app = App::new(&parameters);

    loop {
        app.simulation.write().unwrap().run_simulation_step();
        let locations = app.simulation.read().unwrap().peeps.get_population_locations();
        let points = Points {
            color: Color::Red,
            coords: locations.as_slice(),
        };

        terminal.draw(|f| {
            let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints(
                [
                    Constraint::Percentage(90),
                    Constraint::Percentage(10),
                ].as_ref()
            )
            .split(f.size());
        let block = Canvas::default()
            .paint(|ctx| {
                ctx.draw(&points)
            })
            .x_bounds([0.0, parameters.size_x as f64])
            .y_bounds([0.0, parameters.size_y as f64]);
        f.render_widget(block, chunks[0]);
        let block = Paragraph::new(format!("Step: {}, Generation: {}", app.simulation.read().unwrap().simulation_step % parameters.steps_per_generation as u32, app.simulation.read().unwrap().simulation_step / parameters.steps_per_generation as u32))
        .block(Block::default().borders(Borders::ALL).title("Statistics"));
        f.render_widget(block, chunks[1]);
        })?;

        match events.next()? {
            Event::Input(input) => match input {
                Key::Char('q') => {
                    break;
                }

                Key::Char('c') => {
                    let lock = app.simulation.write();
                    let mut sim = lock.unwrap();
                    sim.run_simulation(100, parameters.steps_per_generation as u32);
                }

                Key::Char('s') => {
                    let lock = app.simulation.write();
                    let mut sim = lock.unwrap();
                    sim.run_simulation(1, parameters.steps_per_generation as u32);
                }

                _ => {}
            },
            Event::Tick => {
                app.update();
            }
        }
    }

    Ok(())
}