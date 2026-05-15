//! 2D camera with pan + zoom.
//!
//! - Pan: middle-mouse drag, or right-mouse drag (matches the floating-toolbar
//!   keyboard shortcut philosophy of leaving left-click free for the tool).
//! - Zoom: scroll wheel, around the cursor position so the spot under the
//!   cursor stays put.
//! - Auto-fit: on `SimControls::refit_camera`, size the orthographic projection
//!   so the grid sprite fills ~90% of the window's smaller axis with the right
//!   panel accounted for.
//!
//! The cursor-anchored zoom is the difference between an OK 2D camera and a
//! great one. Without it, every scroll tick yanks the world out from under
//! the mouse.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::EguiContexts;

use crate::sim::{Sim, SimControls};
use crate::ui::RIGHT_PANEL_WIDTH;

const ZOOM_MIN: f32 = 0.05;
const ZOOM_MAX: f32 = 8.0;
const ZOOM_STEP: f32 = 1.15;
const FIT_MARGIN: f32 = 0.92; // fraction of the available axis to fill

#[derive(Component)]
pub struct SimCamera;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_camera)
            .add_systems(Update, (pan_camera, zoom_camera, fit_to_grid).chain());
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((
        SimCamera,
        Camera2d,
        Projection::from(OrthographicProjection {
            scale: 1.0,
            ..OrthographicProjection::default_2d()
        }),
        Transform::default(),
    ));
}

fn pan_camera(
    motion: Res<AccumulatedMouseMotion>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut cam_q: Query<(&mut Transform, &Projection), With<SimCamera>>,
    mut contexts: EguiContexts,
) {
    // egui takes pointer input first — don't pan when the user is dragging in
    // a panel.
    if let Ok(ctx) = contexts.ctx_mut() {
        if ctx.wants_pointer_input() {
            return;
        }
    }

    let panning = buttons.pressed(MouseButton::Middle) || buttons.pressed(MouseButton::Right);
    if !panning {
        return;
    }
    let Ok((mut t, proj)) = cam_q.single_mut() else { return };
    let scale = match proj {
        Projection::Orthographic(o) => o.scale,
        _ => 1.0,
    };
    let delta = motion.delta;
    if delta == Vec2::ZERO {
        return;
    }
    // Mouse motion is in screen pixels (y-down); world is y-up. Negate y and
    // multiply by scale so pan speed feels consistent across zoom levels.
    t.translation.x -= delta.x * scale;
    t.translation.y += delta.y * scale;
}

fn zoom_camera(
    scroll: Res<AccumulatedMouseScroll>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cam_q: Query<(&mut Transform, &mut Projection, &Camera, &GlobalTransform), With<SimCamera>>,
    mut contexts: EguiContexts,
) {
    if let Ok(ctx) = contexts.ctx_mut() {
        if ctx.wants_pointer_input() {
            return;
        }
    }

    let Ok(window) = windows.single() else {
        return;
    };
    let Ok((mut t, mut proj, cam, cam_xform)) = cam_q.single_mut() else {
        return;
    };

    let total: f32 = scroll.delta.y;
    if total == 0.0 {
        return;
    }

    let Projection::Orthographic(ortho) = &mut *proj else { return };
    let old_scale = ortho.scale;
    let factor = if total > 0.0 { 1.0 / ZOOM_STEP } else { ZOOM_STEP };
    let new_scale = (old_scale * factor).clamp(ZOOM_MIN, ZOOM_MAX);
    if (new_scale - old_scale).abs() < f32::EPSILON {
        return;
    }

    // Cursor-anchored zoom: keep the world point under the cursor fixed.
    if let Some(cursor) = window.cursor_position() {
        if let Ok(world_before) = cam.viewport_to_world_2d(cam_xform, cursor) {
            ortho.scale = new_scale;
            // After scale change we need a fresh camera/global xform read.
            // The cheap way: compute the offset in world space using the
            // ratio of new/old scales.
            let cam_pos = t.translation.truncate();
            let from_cam = world_before - cam_pos;
            let after = cam_pos + from_cam * (new_scale / old_scale);
            // Shift camera so the point that was under cursor stays under
            // cursor in the new projection.
            let delta = world_before - after;
            t.translation.x += delta.x;
            t.translation.y += delta.y;
            return;
        }
    }
    ortho.scale = new_scale;
}

fn fit_to_grid(
    mut controls: ResMut<SimControls>,
    sim: Res<Sim>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cam_q: Query<(&mut Transform, &mut Projection), With<SimCamera>>,
) {
    if !controls.refit_camera {
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Ok((mut t, mut proj)) = cam_q.single_mut() else { return };

    // Top bar (~44px) + bottom playback bar (~52px) + right panel slot.
    let usable_w = (window.width() - RIGHT_PANEL_WIDTH).max(200.0);
    let usable_h = (window.height() - 96.0).max(200.0);

    let world_w = sim.state.config.size_x as f32 * controls.pixel_scale;
    let world_h = sim.state.config.size_y as f32 * controls.pixel_scale;

    let scale_x = world_w / (usable_w * FIT_MARGIN);
    let scale_y = world_h / (usable_h * FIT_MARGIN);
    let target_scale = scale_x.max(scale_y).clamp(ZOOM_MIN, ZOOM_MAX);

    if let Projection::Orthographic(ortho) = &mut *proj {
        ortho.scale = target_scale;
    }
    // Shift the camera to the RIGHT in world coords so the world sprite lands
    // in the LEFT-of-screen-center canvas area (the right panel eats the
    // right RIGHT_PANEL_WIDTH px of the screen). +x camera → -x sprite onscreen.
    let panel_world_offset = (RIGHT_PANEL_WIDTH * 0.5) * target_scale;
    t.translation.x = panel_world_offset;
    t.translation.y = 0.0;

    controls.refit_camera = false;
}
