//! Floating tool picker — top-center of the canvas area.
//!
//! Hover info ("KILL · (42, 68)") moved out of this module — it's now part of
//! the canvas frame chrome (bottom-left), so the toolbar stays compact.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::sim::{SimControls, Tool};
use crate::theme;
use crate::ui::RIGHT_PANEL_WIDTH;

pub fn draw_floating_toolbar(mut contexts: EguiContexts, mut controls: ResMut<SimControls>) {
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
                        for t in [
                            Tool::Inspect,
                            Tool::Barrier,
                            Tool::KillBarrier,
                            Tool::Kill,
                            Tool::Reproduce,
                        ] {
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
    let icon = match tool {
        Tool::Inspect => theme::Icon::Inspect,
        Tool::Barrier => theme::Icon::Barrier,
        Tool::KillBarrier => theme::Icon::KillBarrier,
        Tool::Kill => theme::Icon::Kill,
        Tool::Reproduce => theme::Icon::Reproduce,
    };

    // Painted icon + label inside one clickable frame so the icon glyphs
    // don't depend on the default font having `☠ ▣ ◎ ✕ ✦` coverage. Sized
    // to match the prior text-button row height (≈26px) — 2px margin top/
    // bottom + 20px content + the 1px stroke on each side.
    let resp = egui::Frame::default()
        .fill(fill)
        .stroke(egui::Stroke::new(1.0, stroke))
        .corner_radius(egui::CornerRadius::same(5))
        .inner_margin(egui::Margin::symmetric(8, 2))
        .show(ui, |ui| {
            ui.set_min_size(egui::vec2(80.0, 20.0));
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 5.0;
                let (icon_rect, _) =
                    ui.allocate_exact_size(egui::vec2(13.0, 13.0), egui::Sense::hover());
                theme::paint_icon(ui.painter(), icon_rect, icon, fg);
                ui.label(egui::RichText::new(tool.label()).size(11.0).color(fg).strong());
            });
        })
        .response
        .interact(egui::Sense::click());
    if resp.clone().on_hover_text(tool.description()).clicked() {
        controls.tool = tool;
    }
    ui.add_space(2.0);
    theme::kbd_hint(ui, tool.shortcut());
}
