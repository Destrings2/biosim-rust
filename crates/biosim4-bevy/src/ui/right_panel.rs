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
/// Width of the vertical icon strip on the left edge of the right panel.
const TAB_STRIP_WIDTH: f32 = 38.0;
/// Icon button height inside the vertical strip.
const TAB_BUTTON_HEIGHT: f32 = 38.0;

/// Names for `SimConfig.barrier_type` — mirrors the match arms in
/// `biosim4_core::barriers::create_barrier`.
const BARRIER_TYPE_OPTIONS: &[(u8, &str)] = &[
    (0, "None"),
    (1, "Three floaters"),
    (2, "Vertical bar"),
    (3, "Horizontal bar"),
    (4, "Staggered blocks"),
    (5, "Left/right walls"),
    (6, "Five blocks"),
    (7, "Horizontal strips"),
];

/// Names for `SimConfig.genome_comparison_method` — mirrors the dispatch in
/// `biosim4_core::analysis::genetic_diversity`.
const GENOME_COMPARISON_OPTIONS: &[(u8, &str)] =
    &[(0, "Jaro-Winkler"), (1, "Hamming bits"), (2, "Hamming bytes")];

/// Display labels for `SimConfig.topology` — mirrors the variants of
/// [`biosim4_core::topology::Topology`]. "Sphere" is the user-facing
/// name for the "wraps both axes" case; topologically it's a flat torus
/// but the simulator semantic is "no edges in any direction".
const TOPOLOGY_OPTIONS: &[(biosim4_core::topology::Topology, &str)] = &[
    (biosim4_core::topology::Topology::Plane, "Plane (bounded)"),
    (biosim4_core::topology::Topology::TorusX, "Torus X (wraps E↔W)"),
    (biosim4_core::topology::Topology::TorusY, "Torus Y (wraps N↔S)"),
    (biosim4_core::topology::Topology::Sphere, "Sphere (wraps both)"),
];

pub fn draw_right_panel(
    mut contexts: EguiContexts,
    sim: Res<Sim>,
    mut controls: ResMut<SimControls>,
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
            // Horizontal split inside the right panel: narrow vertical icon
            // strip on the left, body content on the right. Spacing is zeroed
            // so the strip's right border butts against the body.
            // Snapshot the full panel height BEFORE the horizontal split so
            // the strip can claim 100% of it — `horizontal_top` otherwise
            // sizes each child to its content.
            let panel_height = ui.available_height();
            ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
            ui.horizontal_top(|ui| {
                vertical_tab_strip(ui, panel_height, &mut ui_state.right_panel_tab);

                ui.vertical(|ui| {
                    body_header(ui, ui_state.right_panel_tab);
                    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
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
                                    RightPanelTab::Stats => stats_tab(
                                        ui,
                                        &sim,
                                        &mut controls,
                                        &history,
                                        &mut queue,
                                        &mut local_state,
                                    ),
                                    RightPanelTab::Challenge => {
                                        challenge_tab(ui, &sim, &mut queue, &mut local_state)
                                    }
                                    RightPanelTab::Breeds => {
                                        breeds_tab(ui, &sim, &mut queue, &mut local_state)
                                    }
                                    RightPanelTab::Registry => {
                                        registry_tab(ui, &sim, &mut queue, &mut local_state)
                                    }
                                    RightPanelTab::Config => config_tab(
                                        ui,
                                        &sim,
                                        &controls,
                                        &mut queue,
                                        &mut local_state,
                                        &mut ui_state,
                                    ),
                                }
                            });
                    });
                });
            });
        });
}

pub struct RightPanelLocal {
    /// Ordered list of currently-active challenge ids. Order is significant
    /// for `WeightedSum` composition (parallel to `weights`).
    active_challenges: Vec<String>,
    /// Composition mode: `"Any"` / `"All"` / `"WeightedSum"`. Stored as a
    /// string so the segmented control can compare without unwrapping an enum.
    composition_mode: String,
    /// Per-challenge weight used when `composition_mode == "WeightedSum"`.
    composition_weights: std::collections::HashMap<String, f32>,
    /// `WeightedSum` pass threshold (0..=1).
    composition_threshold: f32,
    /// Active challenge ids whose card is expanded (description + params shown).
    /// Newly-added challenges are auto-expanded.
    expanded_challenges: std::collections::HashSet<String>,
    /// Live param values, keyed by challenge id. Persists across tab
    /// switches so half-filled forms don't reset.
    challenge_params: std::collections::HashMap<String, serde_json::Value>,
    sensor_filter: String,
    action_filter: String,
    edit_config: Option<SimConfig>,
    /// Pending fast-forward target — number of generations to simulate.
    ff_gens: u32,
    /// Currently-highlighted breed id in the picker (not necessarily applied).
    selected_breed: String,
}

impl Default for RightPanelLocal {
    fn default() -> Self {
        Self {
            active_challenges: Vec::new(),
            // `All` matches the common case of stacking challenges as
            // conjunctive selection pressure — every challenge must pass.
            composition_mode: "All".to_string(),
            composition_weights: Default::default(),
            composition_threshold: 0.5,
            expanded_challenges: Default::default(),
            challenge_params: Default::default(),
            sensor_filter: String::new(),
            action_filter: String::new(),
            edit_config: None,
            ff_gens: 100,
            selected_breed: String::new(),
        }
    }
}

/// Vertical icon strip docked at the left edge of the right panel. Each tab
/// is a painted icon with an accent-bar indicator on the left when active —
/// keeps the panel header free to show the active tab's name without eating
/// horizontal space, and makes adding more tabs (Breeds, …) a one-line change.
fn vertical_tab_strip(ui: &mut egui::Ui, full_height: f32, current: &mut RightPanelTab) {
    egui::Frame::default()
        .fill(theme::BG)
        .stroke(egui::Stroke::new(1.0, theme::LINE))
        .inner_margin(egui::Margin { left: 0, right: 0, top: 6, bottom: 6 })
        .show(ui, |ui| {
            // Force the strip to the full panel height so the BG column
            // extends from the topbar all the way to the canvas-bottom edge.
            ui.set_min_size(egui::vec2(TAB_STRIP_WIDTH, full_height));
            ui.set_width(TAB_STRIP_WIDTH);
            ui.spacing_mut().item_spacing.y = 2.0;
            ui.vertical_centered(|ui| {
                for &tab in RightPanelTab::ALL {
                    tab_icon_button(ui, tab, current);
                }
            });
        });
}

fn tab_icon_button(ui: &mut egui::Ui, tab: RightPanelTab, current: &mut RightPanelTab) {
    let active = *current == tab;
    let color = if active { theme::ACCENT } else { theme::TEXT_2 };
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(TAB_STRIP_WIDTH - 2.0, TAB_BUTTON_HEIGHT),
        egui::Sense::click(),
    );
    // Hover/active backdrop.
    if active {
        ui.painter().rect_filled(
            rect.shrink2(egui::vec2(4.0, 4.0)),
            egui::CornerRadius::same(5),
            theme::ACCENT_SOFT,
        );
        // Accent bar on the inner edge.
        let bar = egui::Rect::from_min_max(
            egui::pos2(rect.left(), rect.top() + 6.0),
            egui::pos2(rect.left() + 2.0, rect.bottom() - 6.0),
        );
        ui.painter().rect_filled(bar, 0.0, theme::ACCENT);
    } else if resp.hovered() {
        ui.painter().rect_filled(
            rect.shrink2(egui::vec2(4.0, 4.0)),
            egui::CornerRadius::same(5),
            theme::PANEL_2,
        );
    }
    let icon_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(20.0, 20.0));
    theme::paint_icon(ui.painter(), icon_rect, tab.icon(), color);

    let _ = resp.clone().on_hover_text(tab.tooltip());
    if resp.clicked() {
        *current = tab;
    }
}

/// Title bar at the top of the body area — shows the active tab name and a
/// hairline below it. This is what makes adding tabs cheap: the strip carries
/// the iconography, the header carries the name.
fn body_header(ui: &mut egui::Ui, tab: RightPanelTab) {
    egui::Frame::default()
        .fill(theme::BG_2)
        .inner_margin(egui::Margin { left: BODY_INSET, right: BODY_INSET, top: 12, bottom: 8 })
        .show(ui, |ui| {
            ui.label(egui::RichText::new(tab.label()).size(11.0).strong().color(theme::TEXT));
        });
    // Hairline under the header.
    let avail_w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(avail_w, 1.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 0.0, theme::LINE);
}

// ── Stats ───────────────────────────────────────────────────────────────────

fn stats_tab(
    ui: &mut egui::Ui,
    sim: &Sim,
    controls: &mut SimControls,
    history: &SimHistory,
    queue: &mut SimCommandQueue,
    local: &mut RightPanelLocal,
) {
    // Hero numbers
    hero_number(
        ui,
        "GENERATION",
        format!("{}", sim.state.generation),
        Some(format!("step {} / {}", sim.state.sim_step, sim.state.config.steps_per_generation)),
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
    let topology_label = TOPOLOGY_OPTIONS
        .iter()
        .find(|(t, _)| *t == sim.state.config.topology)
        .map(|(_, name)| *name)
        .unwrap_or("?");
    kv_row(ui, "Topology", topology_label);
    kv_row(
        ui,
        "Barrier type",
        crate::ui::widgets::enum_label(BARRIER_TYPE_OPTIONS, sim.state.config.barrier_type),
    );

    section(ui, "REGISTRY");
    kv_row(
        ui,
        "Sensors",
        &format!("{} / {} active", sim.state.sensors.enabled_count(), sim.state.sensors.count()),
    );
    kv_row(
        ui,
        "Actions",
        &format!("{} / {} active", sim.state.actions.enabled_count(), sim.state.actions.count()),
    );

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
    ui.add_space(2.0);
    kv_row(ui, "Threads", &format!("{}", controls.num_threads));
    kv_row(ui, "FPS", &format!("{:.0}", controls.fps));
    kv_row(ui, "Steps/frame", &crate::sim::format_spf(controls.speed));

    section(ui, "FAST FORWARD");
    ui.label(
        egui::RichText::new("Run N generations at full speed with rendering paused.")
            .size(11.0)
            .color(theme::TEXT_2),
    );
    ui.add_space(6.0);

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
            ui.add(egui::DragValue::new(&mut local.ff_gens).range(1..=100_000).speed(1.0));
        });
    });

    ui.add_space(4.0);
    let n = local.ff_gens;
    let r = full_width_primary_with_icon(ui, theme::Icon::Play, &format!("RUN {n} GENS"));
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
            "Stack one or more survival challenges and pick how they combine. The active set decides which agents seed the next generation.",
        )
        .size(11.0)
        .color(theme::TEXT_2),
    );
    ui.add_space(8.0);

    let schemas = sim.state.challenges.schema_list();
    let Some(arr) = schemas.as_array() else { return };

    // ── ACTIVE CHALLENGES ─────────────────────────────────────────────
    section(ui, "ACTIVE CHALLENGES");

    // Pending mutations collected during the render pass (can't mutate
    // `local.active_challenges` while iterating it as a reference).
    let mut remove_id: Option<String> = None;
    let mut to_add: Option<String> = None;

    if local.active_challenges.is_empty() {
        ui.add_space(4.0);
        egui::Frame::default()
            .fill(theme::BG)
            .stroke(egui::Stroke::new(1.0, theme::LINE))
            .corner_radius(egui::CornerRadius::same(4))
            .inner_margin(egui::Margin::symmetric(10, 10))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("No challenges active.")
                        .size(11.0)
                        .strong()
                        .color(theme::TEXT_2),
                );
                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new(
                        "Every agent survives. Add a challenge below to apply selection pressure.",
                    )
                    .size(10.5)
                    .color(theme::MUTED)
                    .italics(),
                );
            });
    } else {
        // Clone the active list so we can mutate params/expansion state
        // inside the loop without a borrow conflict. Strings are cheap.
        let active_ids: Vec<String> = local.active_challenges.clone();
        let total = active_ids.len();
        for (idx, id) in active_ids.iter().enumerate() {
            let entry = match arr
                .iter()
                .find(|e| e.get("id").and_then(|v| v.as_str()) == Some(id.as_str()))
            {
                Some(e) => e,
                None => continue, // stale id (registry changed) — skip silently
            };
            let name = entry.get("name").and_then(|v| v.as_str()).unwrap_or(id);
            let description = entry.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let is_expanded = local.expanded_challenges.contains(id);
            render_active_challenge_card(
                ui,
                id,
                name,
                description,
                entry,
                is_expanded,
                idx,
                total,
                &mut local.expanded_challenges,
                &mut local.challenge_params,
                &mut remove_id,
            );
            ui.add_space(4.0);
        }
    }

    if let Some(rid) = remove_id {
        local.active_challenges.retain(|x| x != &rid);
        local.expanded_challenges.remove(&rid);
        local.composition_weights.remove(&rid);
    }

    // ── ADD PICKER ────────────────────────────────────────────────────
    ui.add_space(4.0);
    let already_active: std::collections::HashSet<&str> =
        local.active_challenges.iter().map(|s| s.as_str()).collect();
    let available: Vec<(&str, &str)> = arr
        .iter()
        .filter_map(|e| {
            let id = e.get("id").and_then(|v| v.as_str())?;
            if already_active.contains(id) {
                return None;
            }
            let name = e.get("name").and_then(|v| v.as_str()).unwrap_or(id);
            Some((id, name))
        })
        .collect();

    if available.is_empty() {
        ui.label(
            egui::RichText::new("All challenges added.").size(10.5).color(theme::MUTED).italics(),
        );
    } else {
        let body_w = ui.available_width();
        egui::ComboBox::from_id_salt("challenge_add_picker")
            .selected_text(
                egui::RichText::new("+ ADD CHALLENGE").size(11.0).strong().color(theme::ACCENT),
            )
            .width(body_w - 4.0)
            .height(360.0)
            .show_ui(ui, |ui| {
                ui.style_mut().spacing.item_spacing.y = 0.0;
                for (id, name) in &available {
                    if ui
                        .selectable_label(
                            false,
                            egui::RichText::new(*name).size(11.5).color(theme::TEXT),
                        )
                        .clicked()
                    {
                        to_add = Some(id.to_string());
                    }
                }
            });
    }

    if let Some(id) = to_add {
        local.active_challenges.push(id.clone());
        local.expanded_challenges.insert(id);
    }

    // ── COMPOSITION ──────────────────────────────────────────────────
    ui.add_space(8.0);
    section(ui, "COMPOSITION");
    render_composition_picker(ui, local);

    // ── ACTIONS ──────────────────────────────────────────────────────
    ui.add_space(10.0);
    ui.horizontal(|ui| {
        let apply_label = if local.active_challenges.is_empty() {
            "APPLY (none)".to_string()
        } else {
            format!("APPLY ({})", local.active_challenges.len())
        };
        let apply = primary_button(ui, &apply_label)
            .on_hover_text(if local.active_challenges.is_empty() {
                "Clear all survival pressure".to_string()
            } else {
                format!(
                    "Activate {} challenge(s) with {} composition",
                    local.active_challenges.len(),
                    local.composition_mode
                )
            })
            .clicked();
        if apply {
            send_challenge_config(local, queue);
        }
        if ghost_button(ui, "CLEAR").on_hover_text("Remove all active challenges").clicked() {
            local.active_challenges.clear();
            local.expanded_challenges.clear();
            local.composition_weights.clear();
            let cfg = serde_json::json!({
                "active": [],
                "composition": "Any",
                "params": {},
            });
            queue.items.push(SimCommand::SetChallenge(cfg.to_string()));
        }
    });

    // ── WORLD EDITS ──────────────────────────────────────────────────
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

/// Render a single active-challenge card: header row (chevron + name + remove)
/// and, when expanded, the description + per-param editor. Pending edits go
/// out through `remove_id`; the caller applies them after the loop.
#[allow(clippy::too_many_arguments)]
fn render_active_challenge_card(
    ui: &mut egui::Ui,
    id: &str,
    name: &str,
    description: &str,
    entry: &serde_json::Value,
    is_expanded: bool,
    idx: usize,
    total: usize,
    expanded_challenges: &mut std::collections::HashSet<String>,
    challenge_params: &mut std::collections::HashMap<String, serde_json::Value>,
    remove_id: &mut Option<String>,
) {
    egui::Frame::default()
        .fill(theme::PANEL)
        .stroke(egui::Stroke::new(1.0, theme::LINE))
        .corner_radius(egui::CornerRadius::same(5))
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            // Header row.
            ui.horizontal(|ui| {
                let chev_icon =
                    if is_expanded { theme::Icon::ChevDown } else { theme::Icon::ChevRight };
                if icon_button_small(ui, chev_icon, theme::TEXT_2, "Toggle details").clicked() {
                    if is_expanded {
                        expanded_challenges.remove(id);
                    } else {
                        expanded_challenges.insert(id.to_string());
                    }
                }

                // Position chip — handy when WeightedSum composition makes
                // order matter, and a subtle visual ordinal otherwise.
                ui.label(
                    egui::RichText::new(format!("{}/{}", idx + 1, total))
                        .size(9.5)
                        .monospace()
                        .color(theme::MUTED),
                );

                ui.label(egui::RichText::new(name).size(12.0).strong().color(theme::TEXT));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if icon_button_small(ui, theme::Icon::Kill, theme::BAD, "Remove").clicked() {
                        *remove_id = Some(id.to_string());
                    }
                });
            });

            if is_expanded {
                if !description.is_empty() {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(description).size(10.5).color(theme::TEXT_2));
                }

                let properties = entry
                    .get("schema")
                    .and_then(|s| s.get("properties"))
                    .and_then(|p| p.as_object());

                if let Some(props) = properties {
                    if !props.is_empty() {
                        ui.add_space(6.0);
                        // Seed param state from schema defaults on first visit.
                        let params = challenge_params.entry(id.to_string()).or_insert_with(|| {
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
        });
}

/// Three-way segmented control + contextual help + (for `WeightedSum`)
/// per-challenge weights and the pass threshold.
fn render_composition_picker(ui: &mut egui::Ui, local: &mut RightPanelLocal) {
    // Segmented row: [ Any ][ All ][ Weighted ]
    ui.horizontal(|ui| {
        let modes: [(&str, &str); 3] =
            [("Any", "Any"), ("All", "All"), ("WeightedSum", "Weighted")];
        let avail = ui.available_width();
        // Subtract small inter-button gaps from the width budget.
        let gap = 4.0;
        let btn_w = ((avail - gap * (modes.len() - 1) as f32) / modes.len() as f32).max(40.0);
        ui.spacing_mut().item_spacing.x = gap;
        for (key, label) in modes {
            let active = local.composition_mode == key;
            let btn = egui::Button::new(
                egui::RichText::new(label).size(11.0).strong().color(if active {
                    theme::BG
                } else {
                    theme::TEXT_2
                }),
            )
            .fill(if active { theme::ACCENT } else { egui::Color32::TRANSPARENT })
            .stroke(egui::Stroke::new(1.0, if active { theme::ACCENT } else { theme::LINE }))
            .corner_radius(egui::CornerRadius::same(4))
            .min_size(egui::vec2(btn_w, 26.0));
            if ui.add(btn).clicked() {
                local.composition_mode = key.to_string();
            }
        }
    });

    let count = local.active_challenges.len();
    let help = match local.composition_mode.as_str() {
        "All" => {
            if count == 0 {
                "Agents will need to pass every active challenge.".to_string()
            } else {
                format!("Agents must pass ALL {count} challenges to survive.")
            }
        }
        "WeightedSum" => {
            "Weighted average of per-challenge scores must reach the threshold.".to_string()
        }
        _ => {
            if count == 0 {
                "Agents will need to pass at least one active challenge.".to_string()
            } else {
                format!("Agents must pass AT LEAST ONE of the {count} challenges.")
            }
        }
    };
    ui.add_space(4.0);
    ui.label(egui::RichText::new(help).size(10.5).color(theme::MUTED).italics());

    // WeightedSum-only: per-challenge weight inputs + threshold.
    if local.composition_mode == "WeightedSum" && !local.active_challenges.is_empty() {
        ui.add_space(6.0);
        egui::Frame::default()
            .fill(theme::BG)
            .stroke(egui::Stroke::new(1.0, theme::LINE))
            .corner_radius(egui::CornerRadius::same(4))
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 5.0;
                for id in local.active_challenges.clone() {
                    let weight = local.composition_weights.entry(id.clone()).or_insert(1.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(&id).size(10.5).color(theme::TEXT_2));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add(
                                egui::DragValue::new(weight)
                                    .range(0.0..=10.0)
                                    .speed(0.05)
                                    .fixed_decimals(2),
                            );
                        });
                    });
                }
                ui.add_space(2.0);
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 1.0),
                    egui::Sense::hover(),
                );
                ui.painter().rect_filled(rect, 0.0, theme::LINE);
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("threshold").size(10.5).strong().color(theme::TEXT_2),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add(
                            egui::DragValue::new(&mut local.composition_threshold)
                                .range(0.0..=1.0)
                                .speed(0.01)
                                .fixed_decimals(2),
                        );
                    });
                });
            });
    }
}

/// Build the `ChallengeConfig` JSON from the panel's local state and queue
/// it for the sim thread. The composition serializes via serde's external
/// tagging — unit variants as bare strings (`"Any"`, `"All"`), the struct
/// variant as `{"WeightedSum": {"weights": [...], "threshold": ...}}`.
fn send_challenge_config(local: &RightPanelLocal, queue: &mut SimCommandQueue) {
    let mut params_map = serde_json::Map::new();
    for id in &local.active_challenges {
        if let Some(p) = local.challenge_params.get(id) {
            params_map.insert(id.clone(), p.clone());
        }
    }
    let composition = match local.composition_mode.as_str() {
        "All" => serde_json::Value::String("All".to_string()),
        "WeightedSum" => {
            let weights: Vec<f32> = local
                .active_challenges
                .iter()
                .map(|id| local.composition_weights.get(id).copied().unwrap_or(1.0))
                .collect();
            serde_json::json!({
                "WeightedSum": {
                    "weights": weights,
                    "threshold": local.composition_threshold,
                }
            })
        }
        _ => serde_json::Value::String("Any".to_string()),
    };
    let cfg = serde_json::json!({
        "active": local.active_challenges,
        "composition": composition,
        "params": params_map,
    });
    queue.items.push(SimCommand::SetChallenge(cfg.to_string()));
}

/// Compact 18×18 icon-only button used inside card headers. Tint flips to
/// `theme::TEXT` on hover so the affordance is obvious without needing
/// a background fill.
fn icon_button_small(
    ui: &mut egui::Ui,
    icon: theme::Icon,
    color: egui::Color32,
    hover: &str,
) -> egui::Response {
    const SIZE: f32 = 18.0;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(SIZE, SIZE), egui::Sense::click());
    let tint = if resp.hovered() { theme::TEXT } else { color };
    theme::paint_icon(ui.painter(), rect, icon, tint);
    resp.on_hover_text(hover)
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
        ui.label(egui::RichText::new(title).size(11.0).color(theme::TEXT_2));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| match ty {
            "number" => {
                let min = prop.get("minimum").and_then(|v| v.as_f64()).unwrap_or(-1e9);
                let max = prop.get("maximum").and_then(|v| v.as_f64()).unwrap_or(1e9);
                let mut v = obj
                    .get(key)
                    .and_then(|v| v.as_f64())
                    .unwrap_or_else(|| prop.get("default").and_then(|v| v.as_f64()).unwrap_or(0.0));
                let speed = ((max - min).abs() / 200.0).max(0.001);
                let r = ui.add(
                    egui::DragValue::new(&mut v).range(min..=max).speed(speed).fixed_decimals(3),
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
                let mut v = obj.get(key).and_then(|v| v.as_bool()).unwrap_or_else(|| {
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
                if ui.add(egui::TextEdit::singleline(&mut v).desired_width(120.0)).changed() {
                    obj.insert(key.to_string(), serde_json::Value::String(v));
                }
            }
            "array" => render_array_field(ui, key, prop, obj),
            _ => {
                ui.label(
                    egui::RichText::new(format!("({ty})")).size(10.0).color(theme::MUTED).italics(),
                );
            }
        });
    });
}

/// Editor for a JSON-Schema `type: "array"` property.
///
/// Special-cases a 3-element `u8` array as an RGB color picker — that's the
/// shape the wanderers / future predator challenges use for `color`, and
/// `egui::Ui::color_edit_button_srgb` is a much better affordance than
/// three drag fields. For other arrays (e.g. waypoint lists) it falls back
/// to a fixed-length row of typed editors driven by `items.type`.
fn render_array_field(
    ui: &mut egui::Ui,
    key: &str,
    prop: &serde_json::Value,
    obj: &mut serde_json::Map<String, serde_json::Value>,
) {
    let items = prop.get("items");
    let item_type = items.and_then(|i| i.get("type")).and_then(|v| v.as_str()).unwrap_or("integer");
    let min_items = prop.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let max_items = prop.get("maxItems").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
    let item_max =
        items.and_then(|i| i.get("maximum")).and_then(|v| v.as_i64()).unwrap_or(i64::MAX);

    // Pull current value (or schema default) into a mutable Vec we mutate
    // in place before writing back.
    let mut arr: Vec<serde_json::Value> = obj
        .get(key)
        .and_then(|v| v.as_array())
        .cloned()
        .or_else(|| prop.get("default").and_then(|v| v.as_array()).cloned())
        .unwrap_or_default();

    // RGB color shortcut: 3-element integer array with each item in 0..=255.
    let is_rgb = item_type == "integer" && item_max == 255 && min_items == 3 && max_items == 3;

    let mut changed = false;
    if is_rgb {
        let mut rgb: [u8; 3] = [
            arr.first().and_then(|v| v.as_u64()).unwrap_or(0).min(255) as u8,
            arr.get(1).and_then(|v| v.as_u64()).unwrap_or(0).min(255) as u8,
            arr.get(2).and_then(|v| v.as_u64()).unwrap_or(0).min(255) as u8,
        ];
        if ui.color_edit_button_srgb(&mut rgb).changed() {
            arr = vec![
                serde_json::Value::Number(rgb[0].into()),
                serde_json::Value::Number(rgb[1].into()),
                serde_json::Value::Number(rgb[2].into()),
            ];
            changed = true;
        }
    } else {
        // Generic fixed-length array. Keep length pinned to minItems; longer
        // sequences (waypoint lists, etc.) would need add/remove affordances
        // that aren't worth building until a challenge actually wants them.
        let target_len = arr.len().clamp(min_items.max(1), max_items.max(min_items));
        while arr.len() < target_len {
            arr.push(serde_json::Value::Number(0.into()));
        }
        ui.horizontal(|ui| {
            for slot in arr.iter_mut().take(target_len) {
                match item_type {
                    "integer" => {
                        let mut v = slot.as_i64().unwrap_or(0);
                        if ui.add(egui::DragValue::new(&mut v).range(0..=item_max)).changed() {
                            *slot = serde_json::Value::Number(v.into());
                            changed = true;
                        }
                    }
                    "number" => {
                        let mut v = slot.as_f64().unwrap_or(0.0);
                        if ui.add(egui::DragValue::new(&mut v).speed(0.1)).changed() {
                            if let Some(n) = serde_json::Number::from_f64(v) {
                                *slot = serde_json::Value::Number(n);
                                changed = true;
                            }
                        }
                    }
                    _ => {
                        ui.label(
                            egui::RichText::new(format!("[{item_type}]"))
                                .size(10.0)
                                .color(theme::MUTED)
                                .italics(),
                        );
                    }
                }
            }
        });
    }

    if changed {
        obj.insert(key.to_string(), serde_json::Value::Array(arr));
    }
}

// ── Breeds ──────────────────────────────────────────────────────────────────

fn breeds_tab(
    ui: &mut egui::Ui,
    sim: &Sim,
    queue: &mut SimCommandQueue,
    local: &mut RightPanelLocal,
) {
    ui.add_space(2.0);
    ui.label(
        egui::RichText::new(
            "Breeds are curated sensor + action presets. Applying one rewrites the enabled sets; the new wiring kicks in at the next generation rollover.",
        )
        .size(11.0)
        .color(theme::TEXT_2),
    );
    ui.add_space(10.0);

    let breeds = sim.state.breeds.list();
    if breeds.is_empty() {
        ui.label(
            egui::RichText::new("No breeds registered.").size(11.0).color(theme::MUTED).italics(),
        );
        return;
    }

    // Seed the highlight on first visit so the right pane has content.
    if local.selected_breed.is_empty() {
        local.selected_breed = breeds[0].id.clone();
    }

    // Dropdown picker — scales to any number of breeds without burning
    // vertical space. The dropdown popup is height-capped so a hundred
    // breeds would internally scroll.
    let current_name = breeds
        .iter()
        .find(|b| b.id == local.selected_breed)
        .map(|b| b.name.clone())
        .unwrap_or_else(|| "—".to_string());
    let body_w = ui.available_width();
    egui::ComboBox::from_id_salt("breed_dd")
        .selected_text(egui::RichText::new(current_name).size(12.0).color(theme::TEXT).strong())
        .width(body_w - 4.0)
        .height(360.0)
        .show_ui(ui, |ui| {
            ui.style_mut().spacing.item_spacing.y = 0.0;
            for breed in breeds {
                let selected = breed.id == local.selected_breed;
                if ui
                    .selectable_label(
                        selected,
                        egui::RichText::new(&breed.name).size(11.5).color(if selected {
                            theme::ACCENT
                        } else {
                            theme::TEXT
                        }),
                    )
                    .clicked()
                {
                    local.selected_breed = breed.id.clone();
                }
            }
        });

    // Detail card for the highlighted breed.
    if let Some(breed) = sim.state.breeds.get(&local.selected_breed) {
        ui.add_space(10.0);
        section(ui, "DETAILS");
        ui.label(egui::RichText::new(&breed.description).size(11.0).color(theme::TEXT_2));
        ui.add_space(6.0);

        egui::Frame::default()
            .fill(theme::BG)
            .stroke(egui::Stroke::new(1.0, theme::LINE))
            .corner_radius(egui::CornerRadius::same(4))
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 4.0;
                kv_row(ui, "Sensors", &format!("{}", breed.sensors.len()));
                kv_row(ui, "Actions", &format!("{}", breed.actions.len()));
                kv_row(
                    ui,
                    "Challenge",
                    if breed.challenge.is_some() { "embedded" } else { "(unchanged)" },
                );
            });

        ui.add_space(8.0);
        ui.collapsing(
            egui::RichText::new("INCLUDED SENSORS").size(10.0).color(theme::MUTED).strong(),
            |ui| {
                ui.label(
                    egui::RichText::new(breed.sensors.join(", "))
                        .monospace()
                        .size(10.5)
                        .color(theme::TEXT_2),
                );
            },
        );
        ui.collapsing(
            egui::RichText::new("INCLUDED ACTIONS").size(10.0).color(theme::MUTED).strong(),
            |ui| {
                ui.label(
                    egui::RichText::new(breed.actions.join(", "))
                        .monospace()
                        .size(10.5)
                        .color(theme::TEXT_2),
                );
            },
        );

        ui.add_space(10.0);
        let id = breed.id.clone();
        if primary_button(ui, "APPLY")
            .on_hover_text(format!("Apply breed `{id}` and commit on next generation"))
            .clicked()
        {
            queue.items.push(SimCommand::ApplyBreed(id));
        }
    }
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
                    .desired_width(130.0),
            );
        });
        ui.add_space(2.0);
        registry_list(
            ui,
            &local.sensor_filter,
            sim.state.sensors.iter().map(|(_, s, e)| (s.id().to_string(), s.name().to_string(), e)),
            |id, on| {
                queue.items.push(SimCommand::SetSensorEnabled(id, on));
            },
        );
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
                    .desired_width(130.0),
            );
        });
        ui.add_space(2.0);
        registry_list(
            ui,
            &local.action_filter,
            sim.state.actions.iter().map(|(_, a, e)| (a.id().to_string(), a.name().to_string(), e)),
            |id, on| {
                queue.items.push(SimCommand::SetActionEnabled(id, on));
            },
        );
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
                    ui.label(egui::RichText::new(name).size(11.5).color(if enabled {
                        theme::TEXT
                    } else {
                        theme::MUTED
                    }));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(&id).monospace().size(10.0).color(theme::MUTED),
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
    ui_state: &mut UiState,
) {
    use crate::ui::widgets;

    if local.edit_config.is_none() {
        local.edit_config = Some(sim.state.config.clone());
    }
    let Some(cfg) = local.edit_config.as_mut() else { return };

    // SimConfig fields that don't affect a Bevy run are intentionally
    // omitted: `max_generations` (CLI only), `save_video` / `video_stride`
    // (no video pipeline), `genome_analysis_stride` / `display_sample_genomes`
    // (stdout-only). Surfacing them would just lie about having an effect.

    // ─── WORLD ─────────────────────────────────────────────────────────────
    widgets::section_header(ui, "WORLD", Some(&format!("{}×{} grid", cfg.size_x, cfg.size_y)));
    widgets::slider_field_u16(
        ui,
        "Grid width",
        Some("Cells horizontally"),
        &mut cfg.size_x,
        32..=512,
        &[64, 128, 256],
    );
    widgets::slider_field_u16(
        ui,
        "Grid height",
        Some("Cells vertically"),
        &mut cfg.size_y,
        32..=512,
        &[64, 128, 256],
    );
    let grid_cells = cfg.size_x as u32 * cfg.size_y as u32;
    let pop_pct =
        if grid_cells == 0 { 0.0 } else { cfg.population as f32 / grid_cells as f32 * 100.0 };
    widgets::slider_field_u32(
        ui,
        "Population",
        Some(&format!("{pop_pct:.1}% of grid")),
        &mut cfg.population,
        50..=20_000,
        &[500, 1_000, 2_500, 5_000],
    );
    widgets::stepper_field(ui, "Threads", Some("Rayon worker count"), &mut cfg.num_threads, 1..=64);
    widgets::stepper_field(
        ui,
        "Signal layers",
        Some("Pheromone channels"),
        &mut cfg.signal_layers,
        1..=4,
    );
    if widgets::seed_field(ui, "RNG seed", None, &mut cfg.rng_seed) {
        cfg.rng_seed = generate_seed();
    }
    widgets::enum_field(
        ui,
        "Topology",
        Some("Which edges wrap"),
        &mut cfg.topology,
        TOPOLOGY_OPTIONS,
    );

    // ─── GENETICS ──────────────────────────────────────────────────────────
    let length_tally = if cfg.genome_initial_length_min == cfg.genome_initial_length_max {
        format!("{} genes · {} neurons", cfg.genome_initial_length_max, cfg.max_number_neurons)
    } else {
        format!(
            "{}–{} genes · {} neurons",
            cfg.genome_initial_length_min, cfg.genome_initial_length_max, cfg.max_number_neurons,
        )
    };
    widgets::section_header(ui, "GENETICS", Some(&length_tally));
    // Initial-length bounds drawn independently. Each agent's starting
    // genome length is sampled from [min, max] — equal bounds give a
    // fixed-length seed, unequal bounds give variable-length seeds.
    let prev_min = cfg.genome_initial_length_min;
    let prev_max = cfg.genome_initial_length_max;
    widgets::stepper_field(
        ui,
        "Min length",
        Some("Lower bound for initial genome"),
        &mut cfg.genome_initial_length_min,
        1..=512,
    );
    widgets::stepper_field(
        ui,
        "Max length",
        Some("Upper bound for initial genome"),
        &mut cfg.genome_initial_length_max,
        1..=512,
    );
    // Auto-push the unedited side so the pair always satisfies min ≤ max.
    if cfg.genome_initial_length_min != prev_min
        && cfg.genome_initial_length_min > cfg.genome_initial_length_max
    {
        cfg.genome_initial_length_max = cfg.genome_initial_length_min;
    }
    if cfg.genome_initial_length_max != prev_max
        && cfg.genome_initial_length_max < cfg.genome_initial_length_min
    {
        cfg.genome_initial_length_min = cfg.genome_initial_length_max;
    }
    widgets::slider_field_u16(
        ui,
        "Max genome length",
        Some("Upper bound after mutations"),
        &mut cfg.genome_max_length,
        8..=2_048,
        &[128, 256, 512],
    );
    widgets::stepper_field(
        ui,
        "Neurons",
        Some("Hidden layer width"),
        &mut cfg.max_number_neurons,
        1..=64,
    );

    // ─── EVOLUTION ─────────────────────────────────────────────────────────
    widgets::section_header(ui, "EVOLUTION", None);
    widgets::slider_field_u32(
        ui,
        "Steps / generation",
        Some("Sim ticks per epoch"),
        &mut cfg.steps_per_generation,
        30..=5_000,
        &[100, 300, 500, 1_000],
    );
    widgets::slider_field_f32(
        ui,
        "Point mutation",
        Some("Per-gene bit-flip rate"),
        &mut cfg.point_mutation_rate,
        0.0..=0.5,
        |v| format!("{v:.3}"),
    );
    widgets::slider_field_f32(
        ui,
        "Gene insert/delete",
        Some("Per-genome length-change rate"),
        &mut cfg.gene_insertion_deletion_rate,
        0.0..=0.5,
        |v| format!("{v:.4}"),
    );
    widgets::slider_field_f32(
        ui,
        "Deletion ratio",
        Some("0 = insert only, 1 = delete only"),
        &mut cfg.deletion_ratio,
        0.0..=1.0,
        |v| format!("{v:.2}"),
    );

    // ─── REPRODUCTION ──────────────────────────────────────────────────────
    widgets::section_header(ui, "REPRODUCTION", None);
    widgets::toggle_field(
        ui,
        "Sexual reproduction",
        Some("Uniform crossover between two parents"),
        &mut cfg.sexual_reproduction,
    );
    widgets::slider_field_u32(
        ui,
        "Tournament size",
        Some("k for tournament(k); 1=uniform random, 3=default, 5+=strong"),
        &mut cfg.tournament_size,
        1..=16,
        &[1, 2, 3, 5, 8],
    );
    widgets::slider_field_u32(
        ui,
        "Elitism count",
        Some("Top-N survivors copied unchanged each gen"),
        &mut cfg.elitism_count,
        0..=64,
        &[0, 1, 2, 4, 8, 16],
    );
    widgets::toggle_field(
        ui,
        "Adaptive mutation",
        Some("Each lineage evolves its own mutation rate"),
        &mut cfg.adaptive_mutation,
    );
    widgets::slider_field_f32(
        ui,
        "Mutation jitter τ",
        Some("Inheritance scale for adaptive mutation"),
        &mut cfg.mutation_rate_jitter,
        0.0..=1.0,
        |v| format!("{v:.2}"),
    );
    widgets::slider_field_f32(
        ui,
        "Bloat penalty",
        Some("Parsimony pressure on dead-end gene count"),
        &mut cfg.bloat_penalty_weight,
        0.0..=0.5,
        |v| format!("{v:.3}"),
    );
    widgets::toggle_field(ui, "Kill enabled", Some("Peeps can kill"), &mut cfg.kill_enable);

    // ─── AGENT DEFAULTS ────────────────────────────────────────────────────
    widgets::section_header(ui, "AGENT DEFAULTS", None);
    widgets::slider_field_f32(
        ui,
        "Responsiveness",
        Some("Action-level scalar"),
        &mut cfg.responsiveness,
        0.0..=2.0,
        |v| format!("{v:.2}"),
    );
    widgets::slider_field_f32(
        ui,
        "Response k-factor",
        Some("Curve sharpness around r=1"),
        &mut cfg.responsiveness_curve_k_factor,
        0.0..=10.0,
        |v| format!("{v:.2}"),
    );
    widgets::slider_field_f32(
        ui,
        "Pop. sensor radius",
        Some("Density-sensor cell reach"),
        &mut cfg.population_sensor_radius,
        0.5..=16.0,
        |v| format!("{v:.2}"),
    );
    widgets::slider_field_f32(
        ui,
        "Signal sensor radius",
        Some("Pheromone-sensor cell reach"),
        &mut cfg.signal_sensor_radius,
        0.5..=16.0,
        |v| format!("{v:.2}"),
    );
    widgets::stepper_field(
        ui,
        "Long probe distance",
        Some("Cells ahead for long-probe sensor"),
        &mut cfg.long_probe_distance,
        1..=64,
    );
    widgets::stepper_field(
        ui,
        "Short probe distance",
        Some("Cells ahead for barrier probe"),
        &mut cfg.short_probe_barrier_distance,
        1..=32,
    );

    // ─── ENERGY ────────────────────────────────────────────────────────────
    widgets::section_header(ui, "ENERGY", None);
    widgets::toggle_field(
        ui,
        "Enable energy",
        Some("Food + per-step cost"),
        &mut cfg.enable_energy,
    );
    widgets::slider_field_f32(
        ui,
        "Energy / step cost",
        Some("Drain per tick"),
        &mut cfg.energy_per_step_cost,
        0.0..=0.05,
        |v| format!("{v:.4}"),
    );
    widgets::slider_field_f32(
        ui,
        "Food regen rate",
        Some("Per-cell regrowth per tick"),
        &mut cfg.food_regen_rate,
        0.0..=0.05,
        |v| format!("{v:.4}"),
    );
    widgets::slider_field_f32(
        ui,
        "Food initial density",
        Some("Fraction of cells seeded with food"),
        &mut cfg.food_initial_density,
        0.0..=1.0,
        |v| format!("{v:.2}"),
    );

    // ─── ENVIRONMENT ───────────────────────────────────────────────────────
    widgets::section_header(ui, "ENVIRONMENT", None);
    widgets::enum_field_u8(
        ui,
        "Barrier type",
        Some("Procedural obstacle layout"),
        &mut cfg.barrier_type,
        BARRIER_TYPE_OPTIONS,
    );

    // ─── ANALYSIS ──────────────────────────────────────────────────────────
    widgets::section_header(ui, "ANALYSIS", None);
    widgets::enum_field_u8(
        ui,
        "Genome comparison",
        Some("Drives the diversity stat"),
        &mut cfg.genome_comparison_method,
        GENOME_COMPARISON_OPTIONS,
    );

    ui.add_space(8.0);

    // Capture intent before the closure so we don't hold `&mut local.edit_config`
    // (via `cfg`) and `&mut local` simultaneously.
    let cfg_snapshot = cfg.clone();
    let cur = &sim.state.config;
    let needs_reset = cfg_snapshot.size_x != cur.size_x
        || cfg_snapshot.size_y != cur.size_y
        || cfg_snapshot.signal_layers != cur.signal_layers
        || cfg_snapshot.rng_seed != cur.rng_seed;
    let apply_label = if needs_reset { "APPLY  ·  RESET" } else { "APPLY" };
    let apply_hint = if needs_reset {
        "size_x / size_y / signal_layers / rng_seed change requires reinitializing the grid — current run will be discarded."
    } else {
        "Patch the running simulation in place. Per-step values take effect immediately; mutation, selection, and barrier settings take effect at the next generation rollover."
    };
    let mut apply = false;
    let mut discard = false;
    ui.horizontal(|ui| {
        // `draw_right_panel` zeros item_spacing at the panel root so the
        // tab strip butts against the body — that cascades down to here,
        // making APPLY/DISCARD touch. Restore a gap locally.
        ui.spacing_mut().item_spacing.x = 6.0;
        if primary_button(ui, apply_label).on_hover_text(apply_hint).clicked() {
            apply = true;
        }
        if ghost_button(ui, "DISCARD").clicked() {
            discard = true;
        }
    });
    if apply {
        queue.items.push(SimCommand::Recreate(cfg_snapshot));
        if !needs_reset {
            // The Recreate handler patches `sim.state.config` in place;
            // mutation-rate / breed / barrier-type changes are read at
            // the next gen rollover, so surface that to the user.
            ui_state.toast =
                Some(crate::ui::Toast::new("Config patched. Takes effect next generation."));
        }
    }
    if discard {
        local.edit_config = Some(sim.state.config.clone());
    }

    ui.add_space(6.0);
    let spf = crate::sim::format_spf(controls.speed);
    let plural = if (controls.speed - 1.0).abs() < f32::EPSILON { "" } else { "s" };
    ui.label(
        egui::RichText::new(format!(
            "Running on {} threads at {} step{}/frame.",
            controls.num_threads, spf, plural,
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
        ui.label(egui::RichText::new(text).size(10.0).color(theme::MUTED).strong());
        let (rect, _) = ui
            .allocate_exact_size(egui::vec2(ui.available_width() - 4.0, 1.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 0.0, theme::LINE);
    });
    ui.add_space(2.0);
}

fn hero_number(ui: &mut egui::Ui, label: &str, value: String, hint: Option<String>) {
    ui.add_space(2.0);
    ui.label(egui::RichText::new(label).size(9.5).color(theme::MUTED).strong());
    ui.label(egui::RichText::new(value).monospace().size(22.0).strong().color(theme::TEXT));
    if let Some(h) = hint {
        ui.label(egui::RichText::new(h).monospace().size(10.5).color(theme::TEXT_2));
    }
    ui.add_space(2.0);
}

fn kv_row(ui: &mut egui::Ui, key: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(key).size(11.0).color(theme::TEXT_2));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(value).monospace().size(11.5).color(theme::TEXT));
        });
    });
}

fn primary_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
    let btn = egui::Button::new(egui::RichText::new(text).size(11.0).strong().color(theme::BG))
        .fill(theme::ACCENT)
        .corner_radius(egui::CornerRadius::same(4))
        .min_size(egui::vec2(110.0, 28.0));
    ui.add(btn)
}

/// Full-width hero button with a painted leading icon. Used for actions like
/// "▶ RUN 100 GENS" where we want the playback semantics but can't rely on
/// the default font having the play glyph.
///
/// Allocates an EXACT 32px-tall row up front rather than using a Frame's
/// `set_min_size`. When the parent has lots of leftover vertical space
/// (e.g. the Stats tab before any generation has run), a min-sized Frame
/// would stretch to fill it.
fn full_width_primary_with_icon(
    ui: &mut egui::Ui,
    icon: theme::Icon,
    text: &str,
) -> egui::Response {
    const H: f32 = 32.0;
    let width = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, H), egui::Sense::click());
    let painter = ui.painter();
    painter.rect_filled(rect, egui::CornerRadius::same(5), theme::ACCENT);
    // Layout: icon + 8px gap + label, group centered horizontally.
    let icon_size = 14.0_f32;
    let label_galley =
        painter.layout_no_wrap(text.to_string(), egui::FontId::proportional(11.5), theme::BG);
    let gap = 8.0;
    let total_w = icon_size + gap + label_galley.size().x;
    let group_left = rect.center().x - total_w * 0.5;
    let icon_center = egui::pos2(group_left + icon_size * 0.5, rect.center().y);
    theme::paint_icon(
        painter,
        egui::Rect::from_center_size(icon_center, egui::vec2(icon_size, icon_size)),
        icon,
        theme::BG,
    );
    let text_pos =
        egui::pos2(group_left + icon_size + gap, rect.center().y - label_galley.size().y * 0.5);
    painter.galley(text_pos, label_galley, theme::BG);
    resp
}

fn ghost_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
    let btn = egui::Button::new(egui::RichText::new(text).size(11.0).strong().color(theme::TEXT_2))
        .fill(egui::Color32::TRANSPARENT)
        .stroke(egui::Stroke::new(1.0, theme::LINE))
        .corner_radius(egui::CornerRadius::same(4))
        .min_size(egui::vec2(80.0, 28.0));
    ui.add(btn)
}

/// Time-derived 64-bit value used when the user clicks the seed-regen icon.
/// Not crypto-grade; the simulation re-seeds its RNG from `cfg.rng_seed` on
/// reset, and nanosecond noise is plenty of entropy for picking a fresh run.
fn generate_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0xC0FFEE)
}
