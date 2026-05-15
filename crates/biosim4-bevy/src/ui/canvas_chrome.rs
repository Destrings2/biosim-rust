//! Canvas frame chrome — corner crosshairs and four corner labels that frame
//! the rendered grid. Mirrors the `frame-label tl/tr/bl/br` + cross elements
//! from the React frontend (`frontend/src/styles.css`).
//!
//! The labels are rendered as a single `egui::Area` covering the canvas region
//! (left of the right panel, below the top bar). Because the area uses
//! `Order::Background` it sits **under** the floating toolbar and playback
//! controls but **above** the bevy sprite — keeps them legible while leaving
//! the simulation visible underneath.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::sim::{Sim, SimControls};
use crate::theme;
use crate::tool::{CellKind, HoveredCell};

// `content_rect` already excludes docked panels — no need to subtract them.

const CHROME_INSET: f32 = 12.0;
const FRAME_INSET: f32 = 18.0; // distance from canvas edge to the frame border

pub fn draw_canvas_chrome(
    mut contexts: EguiContexts,
    sim: Res<Sim>,
    controls: Res<SimControls>,
    hovered: Res<HoveredCell>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    // `content_rect` is the area left over after docked panels (top bar +
    // right panel) have claimed their space — exactly the canvas region.
    let canvas = ctx.content_rect();
    let frame = canvas.shrink(FRAME_INSET);
    if frame.width() < 100.0 || frame.height() < 100.0 {
        return;
    }

    egui::Area::new(egui::Id::new("canvas_chrome"))
        .order(egui::Order::Background)
        .fixed_pos(canvas.min)
        .interactable(false)
        .show(ctx, |ui| {
            let painter = ui.painter_at(canvas);

            // Corner labels
            let label_color = theme::MUTED;
            let label_color_dim = theme::TEXT_2;
            let label_size = 10.0;
            let font = egui::FontId::monospace(label_size);

            let sx = sim.state.config.size_x;
            let sy = sim.state.config.size_y;
            let painted = controls.painted_count;
            let running = controls.running;
            let speed = controls.speed;

            // ── Top-left: GRID size
            painter.text(
                frame.left_top() + egui::vec2(CHROME_INSET, -CHROME_INSET - 6.0),
                egui::Align2::LEFT_BOTTOM,
                format!("GRID  {sx}×{sy}"),
                font.clone(),
                label_color,
            );

            // ── Top-right: painted or procedural
            let tr_text =
                if painted > 0 { format!("{painted} PAINTED") } else { "PROCEDURAL".to_string() };
            painter.text(
                frame.right_top() + egui::vec2(-CHROME_INSET, -CHROME_INSET - 6.0),
                egui::Align2::RIGHT_BOTTOM,
                tr_text,
                font.clone(),
                label_color_dim,
            );

            // ── Bottom-left: hover cell info (replaces the separate hover badge)
            let bl_text = match (hovered.cell, hovered.kind) {
                (Some((x, y)), CellKind::Empty) => format!("({x:>3}, {y:>3}) · empty"),
                (Some((x, y)), CellKind::Barrier) => format!("({x:>3}, {y:>3}) · barrier"),
                (Some((x, y)), CellKind::KillBarrier) => format!("({x:>3}, {y:>3}) · kill zone"),
                (Some((x, y)), CellKind::Agent(id)) => format!("({x:>3}, {y:>3}) · agent #{id}"),
                _ => "—".into(),
            };
            painter.text(
                frame.left_bottom() + egui::vec2(CHROME_INSET, CHROME_INSET + 6.0),
                egui::Align2::LEFT_TOP,
                bl_text,
                font.clone(),
                label_color,
            );

            // ── Bottom-right: RUNNING / PAUSED · speed
            let br_text =
                format!("{}  ·  {}× SPF", if running { "RUNNING" } else { "PAUSED" }, speed,);
            painter.text(
                frame.right_bottom() + egui::vec2(-CHROME_INSET, CHROME_INSET + 6.0),
                egui::Align2::RIGHT_TOP,
                br_text,
                font,
                if running { theme::ACCENT } else { label_color_dim },
            );
        });
}
