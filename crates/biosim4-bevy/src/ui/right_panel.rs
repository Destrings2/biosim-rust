//! Right-docked side panel — tabbed sub-views with a polished underline
//! indicator and consistent section layout.
//!
//! Tabs: Stats / Challenge / Registry / Config. The tab strip uses a 1-px
//! bottom border for the active tab (vs. the dense pill style in the first
//! cut). Each tab body lives in a `ScrollArea` so long content can't escape
//! the panel.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use biosim4_core::sim_config::SimConfig;

use crate::sim::{Sim, SimCommand, SimCommandQueue, SimControls, SimHistory};
use crate::theme;
use crate::ui::{RightPanelTab, UiState, RIGHT_PANEL_WIDTH};

/// Horizontal padding between panel edge and body content. Same value left
/// and right so right-aligned values land symmetrically with section labels.
const BODY_INSET: i8 = 14;

pub fn draw_right_panel(
    mut contexts: EguiContexts,
    sim: Res<Sim>,
    controls: Res<SimControls>,
    history: Res<SimHistory>,
    mut queue: ResMut<SimCommandQueue>,
    mut ui_state: ResMut<UiState>,
    mut local_state: Local<RightPanelLocal>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    egui::SidePanel::right("right_panel")
        .exact_width(RIGHT_PANEL_WIDTH)
        .resizable(false)
        .frame(
            egui::Frame::default()
                .fill(theme::BG_2)
                .stroke(egui::Stroke::new(1.0, theme::LINE))
                .inner_margin(egui::Margin { left: 0, right: 0, top: 0, bottom: 0 }),
        )
        .show(ctx, |ui| {
            // Tab strip docks at the very top of the panel, flush edge-to-edge.
            tab_bar(ui, &mut ui_state.right_panel_tab);

            // Body content sits inside a frame that gives it consistent left+
            // right padding. The scrollbar lives OUTSIDE this frame (still
            // inside the panel) so the right margin stays even as content
            // grows past the viewport.
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    egui::Frame::default()
                        .inner_margin(egui::Margin {
                            left: BODY_INSET,
                            right: BODY_INSET,
                            top: 10,
                            bottom: 20,
                        })
                        .show(ui, |ui| {
                            ui.style_mut().spacing.item_spacing.y = 6.0;
                            match ui_state.right_panel_tab {
                                RightPanelTab::Stats     => stats_tab(ui, &sim, &controls, &history, &mut queue, &mut local_state),
                                RightPanelTab::Challenge => challenge_tab(ui, &sim, &mut queue, &mut local_state),
                                RightPanelTab::Registry  => registry_tab(ui, &sim, &mut queue, &mut local_state),
                                RightPanelTab::Config    => config_tab(ui, &sim, &controls, &mut queue, &mut local_state),
                            }
                        });
                });
        });
}

pub struct RightPanelLocal {
    selected_challenge: String,
    /// Live param values, keyed by challenge id. Persists across tab
    /// switches so half-filled forms don't reset.
    challenge_params: std::collections::HashMap<String, serde_json::Value>,
    sensor_filter: String,
    action_filter: String,
    edit_config: Option<SimConfig>,
    /// Pending fast-forward target — number of generations to simulate.
    ff_gens: u32,
}

impl Default for RightPanelLocal {
    fn default() -> Self {
        Self {
            selected_challenge: String::new(),
            challenge_params: Default::default(),
            sensor_filter: String::new(),
            action_filter: String::new(),
            edit_config: None,
            ff_gens: 100,
        }
    }
}

fn tab_bar(ui: &mut egui::Ui, current: &mut RightPanelTab) {
    egui::Frame::default()
        .fill(theme::BG_2)
        .stroke(egui::Stroke::new(1.0, theme::LINE))
        .inner_margin(egui::Margin {
            left: BODY_INSET - 4,
            right: BODY_INSET - 4,
            top: 6,
            bottom: 0,
        })
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                // Equal-width tabs that span the body inset boundary so the
                // underline indicator aligns visually with the section
                // labels below it.
                let avail = ui.available_width();
                let tab_w = (avail / 4.0).max(60.0);
                for tab in [
                    RightPanelTab::Stats,
                    RightPanelTab::Challenge,
                    RightPanelTab::Registry,
                    RightPanelTab::Config,
                ] {
                    let active = *current == tab;
                    let label = egui::RichText::new(tab.label())
                        .size(10.5)
                        .color(if active { theme::ACCENT } else { theme::TEXT_2 })
                        .strong();
                    let btn = egui::Button::new(label)
                        .fill(egui::Color32::TRANSPARENT)
                        .stroke(egui::Stroke::NONE)
                        .corner_radius(egui::CornerRadius::ZERO)
                        .min_size(egui::vec2(tab_w, 30.0));
                    let r = ui.add(btn);
                    if r.clicked() { *current = tab; }
                    if active {
                        let underline = egui::Rect::from_min_max(
                            egui::pos2(r.rect.left() + 8.0, r.rect.bottom() - 1.0),
                            egui::pos2(r.rect.right() - 8.0, r.rect.bottom()),
                        );
                        ui.painter().rect_filled(underline, 0.0, theme::ACCENT);
                    }
                }
            });
        });
}

// ── Stats ───────────────────────────────────────────────────────────────────

fn stats_tab(
    ui: &mut egui::Ui,
    sim: &Sim,
    controls: &SimControls,
    history: &SimHistory,
    queue: &mut SimCommandQueue,
    local: &mut RightPanelLocal,
) {
    // Hero numbers
    hero_number(
        ui,
        "GENERATION",
        format!("{}", sim.state.generation),
        Some(format!(
            "step {} / {}",
            sim.state.sim_step, sim.state.config.steps_per_generation
        )),
    );
    let alive = sim.alive();
    let pop = sim.state.config.population;
    let alive_pct = if pop == 0 { 0.0 } else { alive as f32 / pop as f32 };
    hero_number(
        ui,
        "POPULATION",
        format!("{alive}"),
        Some(format!("{:.1}% of {} cap", alive_pct * 100.0, pop)),
    );

    section(ui, "WORLD");
    kv_row(ui, "Grid", &format!("{}×{}", sim.state.config.size_x, sim.state.config.size_y));
    kv_row(ui, "Painted", &format!("{} cells", controls.painted_count));
    kv_row(ui, "Signal layers", &format!("{}", sim.state.config.signal_layers));
    kv_row(ui, "Barrier type", &format!("{}", sim.state.config.barrier_type));

    section(ui, "REGISTRY");
    kv_row(ui, "Sensors", &format!(
        "{} / {} active",
        sim.state.sensors.enabled_count(),
        sim.state.sensors.count()
    ));
    kv_row(ui, "Actions", &format!(
        "{} / {} active",
        sim.state.actions.enabled_count(),
        sim.state.actions.count()
    ));

    section(ui, "LAST EPOCH");
    if let Some(p) = history.latest() {
        kv_row(ui, "Generation", &format!("#{}", p.generation));
        kv_row(ui, "Survival", &format!("{:.1}%", p.survival_rate * 100.0));
        kv_row(ui, "Diversity", &format!("{:.4}", p.diversity));
        kv_row(ui, "Survivors", &format!("{}", p.alive));
    } else {
        ui.label(
            egui::RichText::new("No completed generations yet — press EPOCH or PLAY.")
                .size(11.0)
                .color(theme::MUTED)
                .italics(),
        );
    }

    section(ui, "PARALLELISM");
    ui.label(
        egui::RichText::new(
            "Phase 1 (sensors + neural net), Phase 2 (actions + aging), the energy phase, and signal-layer fade all run in parallel via rayon. Multi-thread runs are non-deterministic by design — set num_threads=1 if you need reproducible runs."
        )
        .size(11.0)
        .color(theme::TEXT_2),
    );
    ui.add_space(2.0);
    kv_row(ui, "Threads", &format!("{}", controls.num_threads));
    kv_row(ui, "FPS", &format!("{:.0}", controls.fps));
    kv_row(ui, "Steps/frame", &format!("{}", controls.speed));

    section(ui, "FAST FORWARD");
    ui.label(
        egui::RichText::new(
            "Run N generations at full CPU speed with rendering paused.",
        )
        .size(11.0)
        .color(theme::TEXT_2),
    );
    ui.add_space(4.0);

    // Preset row — sized so all five chips share the body width evenly.
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let presets: &[u32] = &[10, 50, 100, 500, 1000];
        let chip_w = ((ui.available_width() - 4.0 * 4.0) / presets.len() as f32).max(38.0);
        for &n in presets {
            let active = local.ff_gens == n;
            let btn = egui::Button::new(
                egui::RichText::new(format!("+{n}"))
                    .monospace()
                    .size(10.5)
                    .color(if active { theme::ACCENT } else { theme::TEXT_2 })
                    .strong(),
            )
            .fill(if active { theme::ACCENT_SOFT } else { egui::Color32::TRANSPARENT })
            .stroke(egui::Stroke::new(1.0, if active { theme::ACCENT } else { theme::LINE }))
            .corner_radius(egui::CornerRadius::same(4))
            .min_size(egui::vec2(chip_w, 22.0));
            if ui.add(btn).clicked() {
                local.ff_gens = n;
            }
        }
    });

    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("generations").size(11.0).color(theme::TEXT_2));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add(
                egui::DragValue::new(&mut local.ff_gens)
                    .range(1..=100_000)
                    .speed(1.0),
            );
        });
    });

    ui.add_space(4.0);
    let n = local.ff_gens;
    let r = full_width_primary(ui, &format!("▶  RUN {n} GENS"));
    if r.clicked() {
        queue.items.push(SimCommand::FastForward(n));
    }
}

// ── Challenge ───────────────────────────────────────────────────────────────

fn challenge_tab(
    ui: &mut egui::Ui,
    sim: &Sim,
    queue: &mut SimCommandQueue,
    local: &mut RightPanelLocal,
) {
    ui.add_space(2.0);
    ui.label(
        egui::RichText::new(
            "Survival challenges decide which agents seed the next generation. Pick one and APPLY — change takes effect immediately."
        )
        .size(11.0)
        .color(theme::TEXT_2),
    );
    ui.add_space(8.0);

    let schemas = sim.state.challenges.schema_list();
    let Some(arr) = schemas.as_array() else { return };

    if local.selected_challenge.is_empty() {
        if let Some(first) = arr.first().and_then(|s| s.get("id")).and_then(|v| v.as_str()) {
            local.selected_challenge = first.to_string();
        }
    }

    let current = arr
        .iter()
        .find(|e| e.get("id").and_then(|v| v.as_str()) == Some(&local.selected_challenge));
    let current_name = current
        .and_then(|e| e.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("—")
        .to_string();

    // ── Dropdown
    let body_w = ui.available_width();
    egui::ComboBox::from_id_salt("challenge_dd")
        .selected_text(
            egui::RichText::new(current_name)
                .size(12.0)
                .color(theme::TEXT)
                .strong(),
        )
        .width(body_w - 4.0)
        .height(360.0)
        .show_ui(ui, |ui| {
            ui.style_mut().spacing.item_spacing.y = 0.0;
            for entry in arr {
                let id = entry.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let name = entry.get("name").and_then(|v| v.as_str()).unwrap_or(id);
                let selected = id == local.selected_challenge;
                if ui
                    .selectable_label(
                        selected,
                        egui::RichText::new(name)
                            .size(11.5)
                            .color(if selected { theme::ACCENT } else { theme::TEXT }),
                    )
                    .clicked()
                {
                    local.selected_challenge = id.to_string();
                }
            }
        });

    if let Some(entry) = current {
        if let Some(desc) = entry.get("description").and_then(|v| v.as_str()) {
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(desc)
                    .size(11.0)
                    .color(theme::TEXT_2),
            );
        }

        // ── Params form generated from the JSON Schema's `properties` map.
        let properties = entry
            .get("schema")
            .and_then(|s| s.get("properties"))
            .and_then(|p| p.as_object());

        if let Some(props) = properties {
            if !props.is_empty() {
                ui.add_space(8.0);
                section(ui, "PARAMETERS");

                // Seed local param state from schema defaults on first visit.
                let entry_id = local.selected_challenge.clone();
                let params = local
                    .challenge_params
                    .entry(entry_id.clone())
                    .or_insert_with(|| {
                        let mut m = serde_json::Map::new();
                        for (k, v) in props.iter() {
                            if let Some(d) = v.get("default") {
                                m.insert(k.clone(), d.clone());
                            }
                        }
                        serde_json::Value::Object(m)
                    });

                egui::Frame::default()
                    .fill(theme::BG)
                    .stroke(egui::Stroke::new(1.0, theme::LINE))
                    .corner_radius(egui::CornerRadius::same(4))
                    .inner_margin(egui::Margin::symmetric(8, 6))
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 5.0;
                        let obj = params.as_object_mut().expect("seeded as Object above");
                        for (key, prop) in props.iter() {
                            render_param_field(ui, key, prop, obj);
                        }
                    });
            }
        }
    }

    ui.add_space(10.0);

    ui.horizontal(|ui| {
        let apply = primary_button(ui, "APPLY  ▸")
            .on_hover_text(format!(
                "Activate `{}` for upcoming generations",
                local.selected_challenge
            ))
            .clicked();
        if apply {
            let mut params_map = serde_json::Map::new();
            if let Some(p) = local.challenge_params.get(&local.selected_challenge) {
                params_map.insert(local.selected_challenge.clone(), p.clone());
            }
            let cfg = serde_json::json!({
                "active": [local.selected_challenge],
                "composition": "Any",
                "params": params_map,
            });
            queue.items.push(SimCommand::SetChallenge(cfg.to_string()));
        }
        if ghost_button(ui, "CLEAR")
            .on_hover_text("Run without any survival pressure")
            .clicked()
        {
            let cfg = serde_json::json!({
                "active": [],
                "composition": "Any",
                "params": {}
            });
            queue.items.push(SimCommand::SetChallenge(cfg.to_string()));
        }
    });

    ui.add_space(10.0);
    section(ui, "WORLD EDITS");
    ui.label(
        egui::RichText::new(
            "Painted barriers persist across generation rollovers; clear them to revert to the procedural pattern.",
        )
        .size(11.0)
        .color(theme::TEXT_2),
    );
    ui.add_space(4.0);
    if ghost_button(ui, "CLEAR PAINTED BARRIERS").clicked() {
        queue.items.push(SimCommand::ClearUserBarriers);
    }
}

/// Two-way-bound JSON-Schema property → egui widget.
fn render_param_field(
    ui: &mut egui::Ui,
    key: &str,
    prop: &serde_json::Value,
    obj: &mut serde_json::Map<String, serde_json::Value>,
) {
    let ty = prop.get("type").and_then(|v| v.as_str()).unwrap_or("string");
    let title = prop
        .get("title")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| key.to_string());

    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(title)
                .size(11.0)
                .color(theme::TEXT_2),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            match ty {
                "number" => {
                    let min = prop.get("minimum").and_then(|v| v.as_f64()).unwrap_or(-1e9);
                    let max = prop.get("maximum").and_then(|v| v.as_f64()).unwrap_or(1e9);
                    let mut v = obj
                        .get(key)
                        .and_then(|v| v.as_f64())
                        .unwrap_or_else(|| prop.get("default").and_then(|v| v.as_f64()).unwrap_or(0.0));
                    let speed = ((max - min).abs() / 200.0).max(0.001) as f64;
                    let r = ui.add(
                        egui::DragValue::new(&mut v)
                            .range(min..=max)
                            .speed(speed)
                            .fixed_decimals(3),
                    );
                    if r.changed() {
                        obj.insert(
                            key.to_string(),
                            serde_json::Number::from_f64(v)
                                .map(serde_json::Value::Number)
                                .unwrap_or(serde_json::Value::Null),
                        );
                    }
                }
                "integer" => {
                    let min = prop.get("minimum").and_then(|v| v.as_i64()).unwrap_or(i64::MIN);
                    let max = prop.get("maximum").and_then(|v| v.as_i64()).unwrap_or(i64::MAX);
                    let mut v = obj
                        .get(key)
                        .and_then(|v| v.as_i64())
                        .unwrap_or_else(|| prop.get("default").and_then(|v| v.as_i64()).unwrap_or(0));
                    let r = ui.add(egui::DragValue::new(&mut v).range(min..=max));
                    if r.changed() {
                        obj.insert(key.to_string(), serde_json::Value::Number(v.into()));
                    }
                }
                "boolean" => {
                    let mut v = obj
                        .get(key)
                        .and_then(|v| v.as_bool())
                        .unwrap_or_else(|| {
                            prop.get("default").and_then(|v| v.as_bool()).unwrap_or(false)
                        });
                    if ui.checkbox(&mut v, "").changed() {
                        obj.insert(key.to_string(), serde_json::Value::Bool(v));
                    }
                }
                "string" => {
                    let mut v = obj
                        .get(key)
                        .and_then(|v| v.as_str())
                        .unwrap_or_else(|| prop.get("default").and_then(|v| v.as_str()).unwrap_or(""))
                        .to_string();
                    if ui
                        .add(egui::TextEdit::singleline(&mut v).desired_width(120.0))
                        .changed()
                    {
                        obj.insert(key.to_string(), serde_json::Value::String(v));
                    }
                }
                _ => {
                    ui.label(
                        egui::RichText::new(format!("({ty})"))
                            .size(10.0)
                            .color(theme::MUTED)
                            .italics(),
                    );
                }
            }
        });
    });
}


// ── Registry ────────────────────────────────────────────────────────────────

fn registry_tab(
    ui: &mut egui::Ui,
    sim: &Sim,
    queue: &mut SimCommandQueue,
    local: &mut RightPanelLocal,
) {
    ui.label(
        egui::RichText::new(
            "Changes take effect on next generation — neural-net wiring is frozen mid-generation.",
        )
        .size(11.0)
        .color(theme::TEXT_2),
    );
    ui.add_space(6.0);

    egui::CollapsingHeader::new(
        egui::RichText::new(format!(
            "SENSORS  {} / {}",
            sim.state.sensors.enabled_count(),
            sim.state.sensors.count(),
        ))
        .size(11.0)
        .color(theme::TEXT)
        .strong(),
    )
    .default_open(true)
    .show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Filter").size(10.5).color(theme::MUTED));
            ui.add(
                egui::TextEdit::singleline(&mut local.sensor_filter)
                    .hint_text("name…")
                    .desired_width(180.0),
            );
        });
        ui.add_space(2.0);
        registry_list(ui, &local.sensor_filter, sim.state.sensors.iter().map(|(_, s, e)| (s.id().to_string(), s.name().to_string(), e)), |id, on| {
            queue.items.push(SimCommand::SetSensorEnabled(id, on));
        });
    });

    ui.add_space(4.0);

    egui::CollapsingHeader::new(
        egui::RichText::new(format!(
            "ACTIONS  {} / {}",
            sim.state.actions.enabled_count(),
            sim.state.actions.count(),
        ))
        .size(11.0)
        .color(theme::TEXT)
        .strong(),
    )
    .default_open(true)
    .show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Filter").size(10.5).color(theme::MUTED));
            ui.add(
                egui::TextEdit::singleline(&mut local.action_filter)
                    .hint_text("name…")
                    .desired_width(180.0),
            );
        });
        ui.add_space(2.0);
        registry_list(ui, &local.action_filter, sim.state.actions.iter().map(|(_, a, e)| (a.id().to_string(), a.name().to_string(), e)), |id, on| {
            queue.items.push(SimCommand::SetActionEnabled(id, on));
        });
    });
}

fn registry_list(
    ui: &mut egui::Ui,
    filter: &str,
    items: impl Iterator<Item = (String, String, bool)>,
    mut on_toggle: impl FnMut(String, bool),
) {
    let filter_lower = filter.to_lowercase();
    egui::Frame::default()
        .fill(theme::BG)
        .stroke(egui::Stroke::new(1.0, theme::LINE))
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 3.0;
            for (id, name, enabled) in items {
                if !filter_lower.is_empty() && !name.to_lowercase().contains(&filter_lower) {
                    continue;
                }
                ui.horizontal(|ui| {
                    let mut on = enabled;
                    if ui.checkbox(&mut on, "").changed() {
                        on_toggle(id.clone(), on);
                    }
                    ui.label(
                        egui::RichText::new(name)
                            .size(11.5)
                            .color(if enabled { theme::TEXT } else { theme::MUTED }),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(&id)
                                .monospace()
                                .size(10.0)
                                .color(theme::MUTED),
                        );
                    });
                });
            }
        });
}

// ── Config ──────────────────────────────────────────────────────────────────

fn config_tab(
    ui: &mut egui::Ui,
    sim: &Sim,
    controls: &SimControls,
    queue: &mut SimCommandQueue,
    local: &mut RightPanelLocal,
) {
    if local.edit_config.is_none() {
        local.edit_config = Some(sim.state.config.clone());
    }
    let Some(cfg) = local.edit_config.as_mut() else { return };

    // The full SimConfig has 35-ish fields. We surface every one, organized
    // into collapsible groups that mirror the order/comments in
    // `crates/biosim4-core/src/sim_config.rs`.

    egui::CollapsingHeader::new(strong_label("WORLD"))
        .default_open(true)
        .show(ui, |ui| {
            drag_u16(ui, "size_x",        &mut cfg.size_x,        32..=512);
            drag_u16(ui, "size_y",        &mut cfg.size_y,        32..=512);
            drag_u32(ui, "population",    &mut cfg.population,    50..=20_000);
            drag_u32(ui, "num_threads",   &mut cfg.num_threads,   1..=64);
            drag_u8 (ui, "signal_layers", &mut cfg.signal_layers, 1..=4);
            drag_u64(ui, "rng_seed",      &mut cfg.rng_seed,      0..=u64::MAX);
        });

    egui::CollapsingHeader::new(strong_label("EVOLUTION"))
        .default_open(true)
        .show(ui, |ui| {
            drag_u32(ui, "steps_per_generation",      &mut cfg.steps_per_generation,      30..=5_000);
            drag_u32(ui, "max_generations",           &mut cfg.max_generations,           1..=100_000);
            drag_u16(ui, "genome_initial_length_min", &mut cfg.genome_initial_length_min, 1..=512);
            drag_u16(ui, "genome_initial_length_max", &mut cfg.genome_initial_length_max, 1..=512);
            drag_u16(ui, "genome_max_length",         &mut cfg.genome_max_length,         8..=2_048);
            drag_u16(ui, "max_number_neurons",        &mut cfg.max_number_neurons,        1..=64);
            drag_f32(ui, "point_mutation_rate",       &mut cfg.point_mutation_rate,       0.0..=0.5);
            drag_f32(ui, "gene_insertion_deletion_rate", &mut cfg.gene_insertion_deletion_rate, 0.0..=0.5);
            drag_f32(ui, "deletion_ratio",            &mut cfg.deletion_ratio,            0.0..=1.0);
            ui.checkbox(&mut cfg.sexual_reproduction,        "sexual_reproduction");
            ui.checkbox(&mut cfg.choose_parents_by_fitness,  "choose_parents_by_fitness");
            ui.checkbox(&mut cfg.kill_enable,                "kill_enable");
        });

    egui::CollapsingHeader::new(strong_label("AGENT DEFAULTS"))
        .default_open(false)
        .show(ui, |ui| {
            drag_f32(ui, "responsiveness",                  &mut cfg.responsiveness,                  0.0..=2.0);
            drag_f32(ui, "responsiveness_curve_k_factor",   &mut cfg.responsiveness_curve_k_factor,   0.0..=10.0);
            drag_f32(ui, "population_sensor_radius",        &mut cfg.population_sensor_radius,        0.5..=16.0);
            drag_f32(ui, "signal_sensor_radius",            &mut cfg.signal_sensor_radius,            0.5..=16.0);
            drag_u32(ui, "long_probe_distance",             &mut cfg.long_probe_distance,             1..=64);
            drag_u32(ui, "short_probe_barrier_distance",    &mut cfg.short_probe_barrier_distance,    1..=32);
        });

    egui::CollapsingHeader::new(strong_label("ENERGY"))
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut cfg.enable_energy, "enable_energy");
            drag_f32(ui, "energy_per_step_cost", &mut cfg.energy_per_step_cost, 0.0..=0.05);
            drag_f32(ui, "food_regen_rate",      &mut cfg.food_regen_rate,      0.0..=0.05);
            drag_f32(ui, "food_initial_density", &mut cfg.food_initial_density, 0.0..=1.0);
        });

    egui::CollapsingHeader::new(strong_label("ENVIRONMENT"))
        .default_open(false)
        .show(ui, |ui| {
            drag_u8(ui, "barrier_type", &mut cfg.barrier_type, 0..=7);
        });

    egui::CollapsingHeader::new(strong_label("ANALYSIS / OUTPUT"))
        .default_open(false)
        .show(ui, |ui| {
            drag_u32(ui, "genome_analysis_stride",   &mut cfg.genome_analysis_stride,   1..=10_000);
            drag_u32(ui, "display_sample_genomes",   &mut cfg.display_sample_genomes,   0..=64);
            // 0 = Jaro-Winkler · 1 = Hamming bits · 2 = Hamming bytes
            drag_u8 (ui, "genome_comparison_method", &mut cfg.genome_comparison_method, 0..=2);
            ui.checkbox(&mut cfg.save_video, "save_video");
            drag_u32(ui, "video_stride",             &mut cfg.video_stride,             1..=1_000);
        });

    ui.add_space(8.0);

    // Capture intent before the closure so we don't hold `&mut local.edit_config`
    // (via `cfg`) and `&mut local` simultaneously.
    let cfg_snapshot = cfg.clone();
    let mut apply = false;
    let mut discard = false;
    ui.horizontal(|ui| {
        if primary_button(ui, "APPLY  ·  RESET").clicked() { apply = true; }
        if ghost_button(ui, "DISCARD").clicked() { discard = true; }
    });
    if apply   { queue.items.push(SimCommand::Recreate(cfg_snapshot)); }
    if discard { local.edit_config = Some(sim.state.config.clone()); }

    ui.add_space(6.0);
    ui.label(
        egui::RichText::new(format!(
            "Running on {} threads at {} step{}/frame.",
            controls.num_threads, controls.speed,
            if controls.speed == 1 { "" } else { "s" },
        ))
        .size(11.0)
        .color(theme::MUTED)
        .italics(),
    );
}

// ── Shared widgets used inside the right panel ──────────────────────────────

fn section(ui: &mut egui::Ui, text: &str) {
    ui.add_space(10.0);
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(text)
                .size(10.0)
                .color(theme::MUTED)
                .strong(),
        );
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width() - 4.0, 1.0),
            egui::Sense::hover(),
        );
        ui.painter().rect_filled(rect, 0.0, theme::LINE);
    });
    ui.add_space(2.0);
}

fn strong_label(text: &str) -> egui::RichText {
    egui::RichText::new(text)
        .size(11.0)
        .color(theme::TEXT)
        .strong()
}

fn hero_number(ui: &mut egui::Ui, label: &str, value: String, hint: Option<String>) {
    ui.add_space(2.0);
    ui.label(
        egui::RichText::new(label)
            .size(9.5)
            .color(theme::MUTED)
            .strong(),
    );
    ui.label(
        egui::RichText::new(value)
            .monospace()
            .size(22.0)
            .strong()
            .color(theme::TEXT),
    );
    if let Some(h) = hint {
        ui.label(
            egui::RichText::new(h)
                .monospace()
                .size(10.5)
                .color(theme::TEXT_2),
        );
    }
    ui.add_space(2.0);
}

fn kv_row(ui: &mut egui::Ui, key: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(key)
                .size(11.0)
                .color(theme::TEXT_2),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(value)
                    .monospace()
                    .size(11.5)
                    .color(theme::TEXT),
            );
        });
    });
}

fn primary_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
    let btn = egui::Button::new(
        egui::RichText::new(text)
            .size(11.0)
            .strong()
            .color(theme::BG),
    )
    .fill(theme::ACCENT)
    .corner_radius(egui::CornerRadius::same(4))
    .min_size(egui::vec2(110.0, 28.0));
    ui.add(btn)
}

/// Like `primary_button` but stretched to the body's full width — used for
/// "hero" actions like Run Fast-Forward.
fn full_width_primary(ui: &mut egui::Ui, text: &str) -> egui::Response {
    let btn = egui::Button::new(
        egui::RichText::new(text)
            .size(11.5)
            .strong()
            .color(theme::BG),
    )
    .fill(theme::ACCENT)
    .corner_radius(egui::CornerRadius::same(5))
    .min_size(egui::vec2(ui.available_width(), 30.0));
    ui.add(btn)
}

fn ghost_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
    let btn = egui::Button::new(
        egui::RichText::new(text)
            .size(11.0)
            .strong()
            .color(theme::TEXT_2),
    )
    .fill(egui::Color32::TRANSPARENT)
    .stroke(egui::Stroke::new(1.0, theme::LINE))
    .corner_radius(egui::CornerRadius::same(4))
    .min_size(egui::vec2(80.0, 28.0));
    ui.add(btn)
}

// ── Drag-value helpers ──────────────────────────────────────────────────────

fn drag_u8 (ui: &mut egui::Ui, label: &str, v: &mut u8,  range: std::ops::RangeInclusive<u8>)  { drag_row(ui, label, |ui| { ui.add(egui::DragValue::new(v).range(*range.start()..=*range.end())); }); }
fn drag_u16(ui: &mut egui::Ui, label: &str, v: &mut u16, range: std::ops::RangeInclusive<u16>) { drag_row(ui, label, |ui| { ui.add(egui::DragValue::new(v).range(*range.start()..=*range.end())); }); }
fn drag_u32(ui: &mut egui::Ui, label: &str, v: &mut u32, range: std::ops::RangeInclusive<u32>) { drag_row(ui, label, |ui| { ui.add(egui::DragValue::new(v).range(*range.start()..=*range.end())); }); }
fn drag_u64(ui: &mut egui::Ui, label: &str, v: &mut u64, range: std::ops::RangeInclusive<u64>) { drag_row(ui, label, |ui| { ui.add(egui::DragValue::new(v).range(*range.start()..=*range.end())); }); }
fn drag_f32(ui: &mut egui::Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>) {
    // Adaptive step + decimals: small ranges (e.g. mutation rates) need fine
    // control with many decimals; larger ranges (e.g. probe distances) want
    // bigger steps and fewer decimals so dragging feels responsive.
    let span = (*range.end() - *range.start()).abs();
    let (speed, decimals) = if span <= 0.2 {
        (0.0005f64, 4)
    } else if span <= 2.0 {
        (0.01, 3)
    } else if span <= 20.0 {
        (0.05, 2)
    } else {
        (0.5, 1)
    };
    drag_row(ui, label, |ui| {
        ui.add(
            egui::DragValue::new(v)
                .range(*range.start()..=*range.end())
                .speed(speed)
                .fixed_decimals(decimals),
        );
    });
}

fn drag_row(ui: &mut egui::Ui, label: &str, body: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .size(11.0)
                .color(theme::TEXT_2),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            body(ui);
        });
    });
}
