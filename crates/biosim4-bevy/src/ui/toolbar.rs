//! Floating tool picker — top-center of the canvas area.
//!
//! Hover info ("KILL · (42, 68)") moved out of this module — it's now part of
//! the canvas frame chrome (bottom-left), so the toolbar stays compact.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::sim::{SimControls, Tool};
use crate::theme;
use crate::ui::RIGHT_PANEL_WIDTH;

pub fn draw_floating_toolbar(
    mut contexts: EguiContexts,
    mut controls: ResMut<SimControls>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    let dx = -(RIGHT_PANEL_WIDTH * 0.5);

    egui::Area::new(egui::Id::new("floating_toolbar"))
        .anchor(egui::Align2::CENTER_TOP, egui::vec2(dx, 56.0))
        .show(ctx, |ui| {
            egui::Frame::default()
                .fill(theme::PANEL)
                .stroke(egui::Stroke::new(1.0, theme::LINE))
                .corner_radius(egui::CornerRadius::same(10))
                .shadow(theme::float_shadow())
                .inner_margin(egui::Margin::same(5))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for t in [Tool::Inspect, Tool::Barrier, Tool::Kill, Tool::Reproduce] {
                            tool_button(ui, t, &mut controls);
                        }
                    });
                });
        });
}

fn tool_button(ui: &mut egui::Ui, tool: Tool, controls: &mut SimControls) {
    let active = controls.tool == tool;

    let (fill, stroke, fg) = if active {
        (theme::ACCENT_SOFT, theme::ACCENT, theme::ACCENT)
    } else {
        (egui::Color32::TRANSPARENT, theme::LINE, theme::TEXT_2)
    };

    let glyph = match tool {
        Tool::Inspect   => "◎",
        Tool::Barrier   => "▣",
        Tool::Kill      => "✕",
        Tool::Reproduce => "✦",
    };

    let btn = egui::Button::new(
        egui::RichText::new(format!("{glyph}  {}", tool.label()))
            .size(11.0)
            .color(fg)
            .strong(),
    )
    .fill(fill)
    .stroke(egui::Stroke::new(1.0, stroke))
    .corner_radius(egui::CornerRadius::same(5))
    .min_size(egui::vec2(96.0, 26.0));

    let r = ui.add(btn).on_hover_text(tool.description());
    if r.clicked() {
        controls.tool = tool;
    }
    ui.add_space(2.0);
    theme::kbd_hint(ui, tool.shortcut());
}
