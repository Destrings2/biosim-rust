//! Tool input dispatcher.
//!
//! Translates mouse clicks/drags on the grid into [`SimCommand`]s based on the
//! currently selected [`Tool`]. Hover state feeds the HUD (`HoveredCell`).
//!
//! egui's `wants_pointer_input` is consulted before each gesture so a click
//! inside a panel doesn't paint a barrier behind it.

use bevy::input::ButtonInput;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::EguiContexts;

use crate::camera::SimCamera;
use crate::grid_render::sprite_to_cell;
use crate::sim::{Sim, SimCommand, SimCommandQueue, SimControls, Tool};

#[derive(Resource, Default, Clone, Copy)]
pub struct HoveredCell {
    pub cell: Option<(u16, u16)>,
    /// What the cell currently is. Driven by the same query that updates `cell`.
    pub kind: CellKind,
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellKind {
    #[default]
    Empty,
    Barrier,
    KillBarrier,
    Agent(u32),
    /// Holds the raw grid value (`programmable_id | PROGRAMMABLE_FLAG`),
    /// not the decoded id — callers pass it straight back to
    /// `SimCommand::Kill`/`Inspect` etc., and decoding happens at the
    /// final use site.
    Programmable(u32),
    OutOfBounds,
}

pub struct ToolPlugin;

impl Plugin for ToolPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HoveredCell>().add_systems(
            Update,
            // Tools are user-input driven against the visible canvas; FF
            // hides the canvas, so input handling is wasted work there.
            (track_hover, handle_tool_input, tool_keyboard_shortcuts)
                .run_if(crate::sim::fast_forward_inactive),
        );
    }
}

fn track_hover(
    sim: Res<Sim>,
    controls: Res<SimControls>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cam_q: Query<(&Camera, &GlobalTransform), With<SimCamera>>,
    mut hovered: ResMut<HoveredCell>,
) {
    let Ok(window) = windows.single() else { return };
    let Ok((cam, cam_xform)) = cam_q.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        hovered.cell = None;
        hovered.kind = CellKind::OutOfBounds;
        return;
    };
    let Ok(world) = cam.viewport_to_world_2d(cam_xform, cursor) else {
        hovered.cell = None;
        hovered.kind = CellKind::OutOfBounds;
        return;
    };
    let sx = sim.state.config.size_x as u32;
    let sy = sim.state.config.size_y as u32;
    match sprite_to_cell(world, sx, sy, controls.pixel_scale) {
        Some((x, y)) => {
            hovered.cell = Some((x, y));
            let loc = biosim4_core::types::Coord::new(x as i16, y as i16);
            let raw = sim.state.grid.at(loc);
            hovered.kind = match biosim4_core::grid::cell_kind(raw) {
                biosim4_core::grid::CellKind::Empty => CellKind::Empty,
                biosim4_core::grid::CellKind::Barrier => CellKind::Barrier,
                biosim4_core::grid::CellKind::KillBarrier => CellKind::KillBarrier,
                // Decoded id is in `agent_id`; we re-encode-by-keeping
                // the raw cell value so downstream sites have a single
                // discriminator (the bit-31 flag).
                biosim4_core::grid::CellKind::Agent(_) => CellKind::Agent(raw),
                biosim4_core::grid::CellKind::Programmable(_) => CellKind::Programmable(raw),
            };
        }
        None => {
            hovered.cell = None;
            hovered.kind = CellKind::OutOfBounds;
        }
    }
}

fn handle_tool_input(
    mouse: Res<ButtonInput<MouseButton>>,
    mut controls: ResMut<SimControls>,
    hovered: Res<HoveredCell>,
    mut queue: ResMut<SimCommandQueue>,
    mut contexts: EguiContexts,
) {
    // Egui wins pointer events if it wants them. We also defer to the camera
    // pan/zoom on middle and right buttons.
    if let Ok(ctx) = contexts.ctx_mut() {
        if ctx.wants_pointer_input() {
            return;
        }
    }

    let Some((x, y)) = hovered.cell else { return };
    let left = mouse.pressed(MouseButton::Left);
    let left_pressed = mouse.just_pressed(MouseButton::Left);
    let right_pressed = mouse.just_pressed(MouseButton::Right);

    match controls.tool {
        Tool::Inspect => {
            if left_pressed {
                // `selected_agent` stores the raw grid value (peeps =
                // bit 31 clear, programmables = bit 31 set) so the
                // inspector can dispatch on the encoding rather than
                // carry a sibling enum through `SimControls`.
                controls.selected_agent = match hovered.kind {
                    CellKind::Agent(raw) | CellKind::Programmable(raw) => Some(raw),
                    _ => None,
                };
            }
        }
        Tool::Barrier => {
            use biosim4_core::sim_state::BarrierTile;
            if left {
                queue.items.push(SimCommand::SetBarrier { x, y, tile: Some(BarrierTile::Wall) });
            } else if right_pressed {
                queue.items.push(SimCommand::SetBarrier { x, y, tile: None });
            }
        }
        Tool::KillBarrier => {
            use biosim4_core::sim_state::BarrierTile;
            if left {
                queue.items.push(SimCommand::SetBarrier { x, y, tile: Some(BarrierTile::Kill) });
            } else if right_pressed {
                queue.items.push(SimCommand::SetBarrier { x, y, tile: None });
            }
        }
        Tool::Kill => {
            if left_pressed {
                queue.items.push(SimCommand::Kill { x, y });
            }
        }
        Tool::Reproduce => {
            if left_pressed {
                queue.items.push(SimCommand::Reproduce { x, y });
            }
        }
    }
}

/// Keyboard shortcuts mirror the React frontend: 1/2/3/4 for tools, space for
/// play/pause, C for challenge picker (not wired here — UI handles it).
fn tool_keyboard_shortcuts(
    keys: Res<ButtonInput<KeyCode>>,
    mut controls: ResMut<SimControls>,
    mut contexts: EguiContexts,
) {
    if let Ok(ctx) = contexts.ctx_mut() {
        // Don't fire shortcuts while typing in an egui text field.
        if ctx.wants_keyboard_input() {
            return;
        }
    }

    if keys.just_pressed(KeyCode::KeyI) || keys.just_pressed(KeyCode::Digit1) {
        controls.tool = Tool::Inspect;
    } else if keys.just_pressed(KeyCode::KeyB) || keys.just_pressed(KeyCode::Digit2) {
        controls.tool = Tool::Barrier;
    } else if keys.just_pressed(KeyCode::KeyZ) || keys.just_pressed(KeyCode::Digit3) {
        controls.tool = Tool::KillBarrier;
    } else if keys.just_pressed(KeyCode::KeyK) || keys.just_pressed(KeyCode::Digit4) {
        controls.tool = Tool::Kill;
    } else if keys.just_pressed(KeyCode::KeyR) || keys.just_pressed(KeyCode::Digit5) {
        controls.tool = Tool::Reproduce;
    } else if keys.just_pressed(KeyCode::Space) {
        controls.running = !controls.running;
    } else if keys.just_pressed(KeyCode::Escape) {
        controls.selected_agent = None;
    }
}
