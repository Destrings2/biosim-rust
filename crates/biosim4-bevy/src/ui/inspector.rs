//! Agent inspector window — opens when an agent is clicked with the Inspect
//! tool. Two-column layout: metadata + heading mini-map on the left, neural
//! network on the right.
//!
//! The network section has two display modes:
//!   - **LAYERED**: sensors stacked left, neurons middle, actions right.
//!     Fast, deterministic, no animation. Best for reading the topology.
//!   - **FORCE**: spring-driven layout — sensors pinned to the left and
//!     actions to the right, but neurons float freely under repulsion and
//!     edge-spring forces. Surfaces recurrent / intra-layer connections that
//!     the layered view collapses onto each other.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use biosim4_core::genome::gene::{SINK_ACTION, SOURCE_SENSOR};

use crate::sim::{Sim, SimControls};
use crate::theme;

const WIN_WIDTH: f32 = 860.0;
const WIN_HEIGHT: f32 = 600.0;
const LEFT_COL: f32 = 220.0;
const RIGHT_COL: f32 = 180.0;

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum LayoutMode {
    #[default]
    Layered,
    Force,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NodeKind {
    Sensor,
    Neuron,
    Action,
}

#[derive(Clone)]
struct NodeView {
    pos: egui::Vec2,
    vel: egui::Vec2,
    idx: u16,
    kind: NodeKind,
    pinned: bool,
}

#[derive(Clone)]
struct ForceState {
    agent_id: u32,
    nodes: Vec<NodeView>,
    /// Last rect we computed positions for — when it changes we rescale the
    /// node positions to keep them inside the new area.
    last_rect: egui::Rect,
    /// Number of simulation iterations elapsed; we can stop requesting
    /// repaints once movement falls below a threshold.
    settled_frames: u32,
}

impl Default for ForceState {
    fn default() -> Self {
        Self {
            agent_id: 0,
            nodes: Vec::new(),
            // egui::Rect has no `Default` — use ZERO so the rect-change
            // detector treats the first frame as a (no-op) resize.
            last_rect: egui::Rect::ZERO,
            settled_frames: 0,
        }
    }
}

#[derive(Clone, Default)]
struct InspectorState {
    mode: LayoutMode,
    force: ForceState,
    /// Click-drag pan offset for the net canvas, in screen pixels. Applied
    /// only at draw time so the force-directed physics keeps settling
    /// against the original rect — pan is a pure viewport transform.
    pan: egui::Vec2,
    /// Tracks which agent the pan offset belongs to; cleared automatically
    /// when the user switches to another agent.
    pan_agent_id: u32,
}

pub fn draw_agent_inspector(
    mut contexts: EguiContexts,
    sim: Res<Sim>,
    mut controls: ResMut<SimControls>,
) {
    let Some(agent_id) = controls.selected_agent else {
        return;
    };
    let Ok(ctx) = contexts.ctx_mut() else { return };

    let agent_ref = sim.state.population.get(agent_id);
    let mut open = true;
    egui::Window::new(format!("AGENT  #{agent_id}"))
        .open(&mut open)
        .resizable(true)
        .default_width(WIN_WIDTH)
        .default_height(WIN_HEIGHT)
        .min_width(540.0)
        .min_height(380.0)
        .frame(
            egui::Frame::default()
                .fill(theme::PANEL)
                .stroke(egui::Stroke::new(1.0, theme::LINE))
                .corner_radius(egui::CornerRadius::same(6))
                .shadow(theme::float_shadow())
                .inner_margin(egui::Margin::same(14)),
        )
        .show(ctx, |ui| {
            let Some(a) = agent_ref else {
                ui.label(egui::RichText::new("Agent not found.").color(theme::BAD));
                return;
            };
            if !a.alive {
                ui.label(egui::RichText::new("Agent is no longer alive.").color(theme::BAD));
                return;
            }

            ui.horizontal_top(|ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2(LEFT_COL, ui.available_height()),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        egui::ScrollArea::vertical()
                            .auto_shrink([false, false])
                            .id_salt("inspector_left_col")
                            .show(ui, |ui| {
                                left_column(ui, &sim, a);
                            });
                    },
                );
                column_divider(ui);
                // Reserve room for: column_divider (16) + auto item-spacing inserted by the
                // outer allocate_ui_with_layout AFTER the middle column (item_spacing.x) +
                // the right column itself. Forgetting the item-spacing causes the layout to
                // overshoot the window by ~8 px/frame, which egui's Resize widget then takes
                // as a growth signal — making the window expand without bound.
                let spacing = ui.spacing().item_spacing.x;
                let mid_w = (ui.available_width() - RIGHT_COL - 16.0 - spacing).max(60.0);
                ui.allocate_ui_with_layout(
                    egui::vec2(mid_w, ui.available_height()),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        net_section(ui, &sim, a);
                    },
                );
                column_divider(ui);
                ui.allocate_ui_with_layout(
                    egui::vec2(RIGHT_COL, ui.available_height()),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        egui::ScrollArea::vertical()
                            .auto_shrink([false, false])
                            .id_salt("inspector_right_col")
                            .show(ui, |ui| {
                                actions_panel(ui, &sim, a);
                            });
                    },
                );
            });
        });

    if !open {
        controls.selected_agent = None;
    }
}

fn left_column(ui: &mut egui::Ui, sim: &Sim, a: &biosim4_core::agent::Agent) {
    // Big color swatch with the agent's genome-derived RGB.
    let (swatch, _) =
        ui.allocate_exact_size(egui::vec2(LEFT_COL - 8.0, 64.0), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(swatch, 6.0, theme::rgb(a.color));
    painter.rect_stroke(
        swatch,
        6.0,
        egui::Stroke::new(1.0, theme::LINE_2),
        egui::StrokeKind::Outside,
    );
    painter.line_segment(
        [
            egui::pos2(swatch.left() + 6.0, swatch.top() + 6.0),
            egui::pos2(swatch.right() - 6.0, swatch.top() + 6.0),
        ],
        egui::Stroke::new(1.0, egui::Color32::from_white_alpha(36)),
    );

    ui.add_space(8.0);
    ui.label(
        egui::RichText::new(format!("AGENT  #{}", a.id))
            .monospace()
            .size(14.0)
            .strong()
            .color(theme::TEXT),
    );
    ui.label(
        egui::RichText::new(format!("at ({:>3}, {:>3})", a.loc.x, a.loc.y))
            .monospace()
            .size(11.0)
            .color(theme::TEXT_2),
    );

    ui.add_space(10.0);
    kv(ui, "AGE", format!("{}", a.age));
    kv(ui, "GENOME", format!("{} genes", a.genome.len()));
    kv(ui, "RESPNS", format!("{:.2}", a.responsiveness));
    kv(ui, "ENERGY", format!("{:.2}", a.energy));
    kv(ui, "OSC_PERIOD", format!("{}", a.osc_period));
    kv(ui, "LONGPROBE", format!("{}", a.long_probe_dist));
    kv(ui, "HEADING", format!("{:?}", a.heading.0));

    ui.add_space(10.0);
    section_label(ui, "MEMORY");
    egui::Frame::default()
        .fill(theme::BG_2)
        .stroke(egui::Stroke::new(1.0, theme::LINE))
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            for (i, m) in a.memory.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("M{i}"))
                            .monospace()
                            .size(10.0)
                            .color(theme::MUTED)
                            .strong(),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("{m:+.3}"))
                                .monospace()
                                .size(11.0)
                                .color(theme::TEXT),
                        );
                    });
                });
            }
        });

    ui.add_space(10.0);
    section_label(ui, "POSITION");
    minimap(ui, sim, a.loc.x, a.loc.y, a.color);
}

/// Look up the agent's action-level vector from the last step's scratch.
/// Returns `None` if the simulation hasn't stepped yet, or if the agent was
/// born after the last `alive_ids` snapshot.
fn agent_action_levels(sim: &Sim, agent_id: u32) -> Option<&[f32]> {
    let pos = sim
        .state
        .scratch
        .alive_ids
        .iter()
        .position(|&id| id == agent_id)?;
    sim.state
        .scratch
        .per_agent_action_levels
        .get(pos)
        .map(|v| v.as_slice())
}

/// "ACTIONS · N DRIVEN" panel — lists every action with non-trivial output
/// from the most recent Phase 1, sorted by absolute weight. Each row shows
/// the action name, signed weight, and a colored magnitude bar.
fn actions_panel(ui: &mut egui::Ui, sim: &Sim, agent: &biosim4_core::agent::Agent) {
    let levels = match agent_action_levels(sim, agent.id) {
        Some(l) if !l.is_empty() => l,
        _ => {
            section_label(ui, "ACTIONS");
            ui.label(
                egui::RichText::new("step the simulation to see action weights")
                    .size(10.5)
                    .color(theme::MUTED)
                    .italics(),
            );
            return;
        }
    };

    // Filter near-zero (numeric noise), then sort by |weight| descending so
    // the most-driven action is first.
    const EPS: f32 = 0.005;
    let mut entries: Vec<(u16, f32)> = levels
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u16, l))
        .filter(|(_, l)| l.abs() > EPS)
        .collect();
    entries.sort_by(|(_, a), (_, b)| {
        b.abs()
            .partial_cmp(&a.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    section_label(ui, &format!("ACTIONS · {} DRIVEN", entries.len()));

    // Normalise the bar lengths against the strongest output in this frame
    // so even modest weights have a readable bar. Floor at 0.25 so a single
    // dominant action doesn't compress everything else to invisibility.
    let max_abs = entries
        .iter()
        .map(|(_, w)| w.abs())
        .fold(0.0_f32, f32::max)
        .max(0.25);

    egui::Frame::default()
        .fill(theme::BG_2)
        .stroke(egui::Stroke::new(1.0, theme::LINE))
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 6.0;
            for (idx, w) in entries {
                let name = sim.state.actions.name(idx);
                action_row(ui, name, w, max_abs);
            }
        });
}

fn action_row(ui: &mut egui::Ui, name: &str, weight: f32, max_abs: f32) {
    let row_w = ui.available_width();
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.y = 2.0;

        // Name + signed value
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(name).size(11.0).color(theme::TEXT_2));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let value_color = if weight >= 0.0 {
                    theme::ACCENT
                } else {
                    theme::BAD
                };
                ui.label(
                    egui::RichText::new(format!("{:+.2}", weight))
                        .monospace()
                        .size(11.0)
                        .color(value_color),
                );
            });
        });

        // Magnitude bar — single-line, centered around the midpoint so
        // negative weights extend leftward and positive ones extend right.
        // Gives an instant visual read of sign + magnitude.
        let (rect, _) = ui.allocate_exact_size(egui::vec2(row_w, 3.0), egui::Sense::hover());
        let painter = ui.painter();
        painter.rect_filled(rect, 1.5, theme::BG);
        let mag = (weight.abs() / max_abs).clamp(0.0, 1.0);
        let center_x = rect.center().x;
        let half_w = (rect.width() * 0.5 - 1.0).max(0.0);
        let bar_w = half_w * mag;
        let color = if weight >= 0.0 {
            theme::ACCENT
        } else {
            theme::BAD
        };
        let bar_rect = if weight >= 0.0 {
            egui::Rect::from_min_size(
                egui::pos2(center_x, rect.top()),
                egui::vec2(bar_w, rect.height()),
            )
        } else {
            egui::Rect::from_min_size(
                egui::pos2(center_x - bar_w, rect.top()),
                egui::vec2(bar_w, rect.height()),
            )
        };
        painter.rect_filled(bar_rect, 1.5, color);
        // Thin midline so users can see the zero point even when no bar is drawn.
        painter.line_segment(
            [
                egui::pos2(center_x, rect.top()),
                egui::pos2(center_x, rect.bottom()),
            ],
            egui::Stroke::new(1.0, theme::LINE_2),
        );
    });
}

fn minimap(ui: &mut egui::Ui, sim: &Sim, x: i16, y: i16, color: [u8; 3]) {
    let size = (LEFT_COL - 8.0).min(140.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 4.0, theme::BG);
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, theme::LINE),
        egui::StrokeKind::Outside,
    );
    let sx = sim.state.config.size_x as f32;
    let sy = sim.state.config.size_y as f32;
    let nx = x as f32 / sx;
    let ny = 1.0 - (y as f32 / sy);
    let dot = egui::pos2(
        rect.left() + 2.0 + nx * (rect.width() - 4.0),
        rect.top() + 2.0 + ny * (rect.height() - 4.0),
    );
    painter.circle_filled(dot, 3.0, theme::rgb(color));
    painter.circle_stroke(
        dot,
        5.0,
        egui::Stroke::new(1.0, theme::rgb(color).gamma_multiply(0.4)),
    );
}

fn net_section(ui: &mut egui::Ui, sim: &Sim, agent: &biosim4_core::agent::Agent) {
    // Pull persistent inspector state (layout mode + force positions) out of
    // egui memory keyed by agent id so each agent gets its own settled layout.
    let state_id = egui::Id::new(("inspector_state", agent.id));
    let mut state: InspectorState = ui
        .ctx()
        .data_mut(|d| d.get_temp::<InspectorState>(state_id))
        .unwrap_or_default();

    // Reset the viewport pan when the user switches to a new agent so the
    // graph reappears centered in the canvas.
    if state.pan_agent_id != agent.id {
        state.pan = egui::Vec2::ZERO;
        state.pan_agent_id = agent.id;
    }

    // ── Mode toggle row
    ui.horizontal(|ui| {
        mode_button(ui, &mut state.mode, LayoutMode::Layered, "LAYERED");
        mode_button(ui, &mut state.mode, LayoutMode::Force, "FORCE");

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Reset pan + (in force mode) re-seed positions. Always offered
            // since pan can take you off-screen in either layout mode.
            let reset = ui
                .button(
                    egui::RichText::new("⟲  RECENTER")
                        .size(10.5)
                        .color(theme::TEXT_2)
                        .strong(),
                )
                .on_hover_text("Reset pan and (in FORCE mode) re-seed node positions")
                .clicked();
            if reset {
                state.pan = egui::Vec2::ZERO;
                if state.mode == LayoutMode::Force {
                    state.force = ForceState::default();
                    state.force.agent_id = agent.id;
                }
            }
        });
    });
    ui.add_space(4.0);

    // ── Canvas (click-and-drag sensing so we can pan the contents)
    let canvas_h = (ui.available_height() - 6.0).max(260.0);
    let canvas_w = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(canvas_w, canvas_h),
        egui::Sense::click_and_drag(),
    );

    if resp.dragged() {
        state.pan += resp.drag_delta();
    }
    if resp.hovered() && !resp.dragged() {
        resp.clone().on_hover_cursor(egui::CursorIcon::Grab);
    }
    if resp.dragged() {
        resp.clone().on_hover_cursor(egui::CursorIcon::Grabbing);
    }

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 6.0, theme::BG);
    painter.rect_stroke(
        rect,
        6.0,
        egui::Stroke::new(1.0, theme::LINE),
        egui::StrokeKind::Outside,
    );

    // Collect distinct node indices used by the network.
    let (sensors, neurons, actions) = collect_used_ids(&agent.nnet);

    let pan = state.pan;

    // Dispatch to the chosen layout. Both call into the same draw routine so
    // the visual output (edges, nodes, labels) is consistent. The pan is
    // applied as a draw-time offset inside the column-position helpers, so
    // the underlying physics state stays in absolute screen space.
    match state.mode {
        LayoutMode::Layered => {
            draw_layered(
                ui, &painter, rect, pan, sim, agent, &sensors, &neurons, &actions,
            );
        }
        LayoutMode::Force => {
            draw_force(
                ui,
                &painter,
                rect,
                pan,
                sim,
                agent,
                &sensors,
                &neurons,
                &actions,
                &mut state.force,
            );
        }
    }

    // ── Legend (shared)
    let legend_y = rect.bottom() - 14.0;
    let nnet = &agent.nnet;
    painter.text(
        egui::pos2(rect.left() + 18.0, legend_y),
        egui::Align2::LEFT_CENTER,
        format!(
            "{} edges  ·  {}/{} neurons driven",
            nnet.connections.len(),
            nnet.neurons.iter().filter(|n| n.driven).count(),
            nnet.neurons.len(),
        ),
        egui::FontId::monospace(10.0),
        theme::MUTED,
    );
    painter.text(
        egui::pos2(rect.right() - 18.0, legend_y),
        egui::Align2::RIGHT_CENTER,
        "green = +  ·  red = –  ·  thickness ∝ |w|",
        egui::FontId::monospace(10.0),
        theme::MUTED,
    );

    // Persist state back to egui memory.
    ui.ctx().data_mut(|d| d.insert_temp(state_id, state));
}

fn mode_button(ui: &mut egui::Ui, current: &mut LayoutMode, mode: LayoutMode, label: &str) {
    let active = *current == mode;
    let btn = egui::Button::new(
        egui::RichText::new(label)
            .size(10.5)
            .color(if active { theme::ACCENT } else { theme::TEXT_2 })
            .strong(),
    )
    .fill(if active {
        theme::ACCENT_SOFT
    } else {
        egui::Color32::TRANSPARENT
    })
    .stroke(egui::Stroke::new(
        1.0,
        if active { theme::ACCENT } else { theme::LINE },
    ))
    .corner_radius(egui::CornerRadius::same(4))
    .min_size(egui::vec2(72.0, 22.0));
    if ui.add(btn).clicked() {
        *current = mode;
    }
}

fn collect_used_ids(nnet: &biosim4_core::genome::NeuralNet) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut sensors: Vec<u16> = Vec::new();
    let mut actions: Vec<u16> = Vec::new();
    let mut neurons: Vec<u16> = Vec::new();
    for g in &nnet.connections {
        let src = g.source_num() as u16;
        let snk = g.sink_num() as u16;
        if g.source_type() == SOURCE_SENSOR {
            if !sensors.contains(&src) {
                sensors.push(src);
            }
        } else if !neurons.contains(&src) {
            neurons.push(src);
        }
        if g.sink_type() == SINK_ACTION {
            if !actions.contains(&snk) {
                actions.push(snk);
            }
        } else if !neurons.contains(&snk) {
            neurons.push(snk);
        }
    }
    sensors.sort();
    actions.sort();
    neurons.sort();
    (sensors, neurons, actions)
}

// ── Layered layout (unchanged from before, factored out) ────────────────────

fn draw_layered(
    _ui: &egui::Ui,
    painter: &egui::Painter,
    rect: egui::Rect,
    pan: egui::Vec2,
    sim: &Sim,
    agent: &biosim4_core::agent::Agent,
    sensors: &[u16],
    neurons: &[u16],
    actions: &[u16],
) {
    let pad = 18.0;
    // Column headers stay fixed to the canvas — they're UI chrome, not part
    // of the pannable contents.
    let col_y = rect.top() + 26.0;
    painter.text(
        egui::pos2(rect.left() + pad, col_y),
        egui::Align2::LEFT_TOP,
        "SENSORS",
        egui::FontId::monospace(10.0),
        theme::MUTED,
    );
    painter.text(
        egui::pos2(rect.center().x, col_y),
        egui::Align2::CENTER_TOP,
        "NEURONS",
        egui::FontId::monospace(10.0),
        theme::MUTED,
    );
    painter.text(
        egui::pos2(rect.right() - pad, col_y),
        egui::Align2::RIGHT_TOP,
        "ACTIONS",
        egui::FontId::monospace(10.0),
        theme::MUTED,
    );

    let inner_w = (rect.width() - 2.0 * pad).max(0.0);
    let gutter = (inner_w * 0.32).clamp(96.0, 160.0);
    let inner = rect.shrink2(egui::vec2(pad, 50.0));
    let sensor_x = inner.left() + gutter;
    let neuron_x = inner.center().x;
    let action_x = inner.right() - gutter;
    let label_max_chars = ((gutter - 14.0) / 6.0).max(4.0) as usize;

    let column_positions = |xs: f32, n: usize| -> Vec<egui::Pos2> {
        if n == 0 {
            return Vec::new();
        }
        let h = inner.height();
        let step = (h / (n.max(1) as f32)).min(40.0);
        let total = step * (n as f32 - 1.0).max(0.0);
        let start = inner.center().y - total * 0.5;
        (0..n)
            .map(|i| egui::pos2(xs, start + i as f32 * step) + pan)
            .collect()
    };
    let sensor_pos = column_positions(sensor_x, sensors.len());
    let neuron_pos = column_positions(neuron_x, neurons.len());
    let action_pos = column_positions(action_x, actions.len());

    draw_edges(
        painter,
        &agent.nnet,
        sensors,
        neurons,
        actions,
        &sensor_pos,
        &neuron_pos,
        &action_pos,
    );
    draw_nodes(
        painter,
        sim,
        &agent.nnet,
        sensors,
        neurons,
        actions,
        &sensor_pos,
        &neuron_pos,
        &action_pos,
        label_max_chars,
    );
}

// ── Force-directed layout ──────────────────────────────────────────────────

const REPULSION_STRENGTH: f32 = 9_000.0;
const SPRING_STRENGTH: f32 = 0.04;
const SPRING_LENGTH: f32 = 100.0;
const DAMPING: f32 = 0.85;
const MAX_VELOCITY: f32 = 60.0;
const ITERATIONS_PER_FRAME: usize = 4;
const SETTLED_VELOCITY: f32 = 0.2;

fn draw_force(
    ui: &egui::Ui,
    painter: &egui::Painter,
    rect: egui::Rect,
    pan: egui::Vec2,
    sim: &Sim,
    agent: &biosim4_core::agent::Agent,
    sensors: &[u16],
    neurons: &[u16],
    actions: &[u16],
    state: &mut ForceState,
) {
    let pad = 18.0;
    let inner = rect.shrink2(egui::vec2(pad, 50.0));
    // Same gutter math as LAYERED — sensor labels extend leftward from the
    // node, action labels extend rightward, so we need to keep the pinned
    // columns inset by enough room for the longest label or they clip off
    // the painter rect.
    let gutter = (inner.width() * 0.32).clamp(96.0, 160.0);
    let header_y = rect.top() + 26.0;
    painter.text(
        egui::pos2(rect.left() + pad, header_y),
        egui::Align2::LEFT_TOP,
        "SENSORS",
        egui::FontId::monospace(10.0),
        theme::MUTED,
    );
    painter.text(
        egui::pos2(rect.center().x, header_y),
        egui::Align2::CENTER_TOP,
        "NEURONS · springs",
        egui::FontId::monospace(10.0),
        theme::MUTED,
    );
    painter.text(
        egui::pos2(rect.right() - pad, header_y),
        egui::Align2::RIGHT_TOP,
        "ACTIONS",
        egui::FontId::monospace(10.0),
        theme::MUTED,
    );

    // (Re)build node list if the agent changed or this is the first render.
    let needs_rebuild = state.agent_id != agent.id || state.nodes.is_empty();
    if needs_rebuild {
        state.agent_id = agent.id;
        state.nodes = build_initial_nodes(inner, gutter, sensors, neurons, actions);
        state.settled_frames = 0;
    }

    // If the canvas resized, rescale node positions to the new rect.
    if state.last_rect != rect {
        if state.last_rect.width() > 1.0 && state.last_rect.height() > 1.0 {
            let scale = egui::vec2(
                rect.width() / state.last_rect.width(),
                rect.height() / state.last_rect.height(),
            );
            for n in state.nodes.iter_mut() {
                n.pos = (n.pos - state.last_rect.min.to_vec2()) * scale + rect.min.to_vec2();
            }
        }
        state.last_rect = rect;
    }

    // Re-pin sensors / actions inside the gutter so labels have room.
    let n_sensors = sensors.len();
    let n_actions = actions.len();
    pin_column(
        &mut state.nodes,
        NodeKind::Sensor,
        n_sensors,
        inner.left() + gutter,
        inner.top(),
        inner.bottom(),
    );
    pin_column(
        &mut state.nodes,
        NodeKind::Action,
        n_actions,
        inner.right() - gutter,
        inner.top(),
        inner.bottom(),
    );

    // Build edge list once per frame (cheap; ~50 edges typical).
    let edges = build_edges(&agent.nnet, sensors, neurons, actions, &state.nodes);

    // Run a few physics iterations per frame so the layout settles smoothly.
    let mut max_v = 0.0_f32;
    for _ in 0..ITERATIONS_PER_FRAME {
        step_forces(&mut state.nodes, &edges, inner);
        for n in &state.nodes {
            if !n.pinned {
                max_v = max_v.max(n.vel.length());
            }
        }
    }

    // Keep requesting repaints while the layout is still moving.
    if max_v > SETTLED_VELOCITY {
        state.settled_frames = 0;
        ui.ctx().request_repaint();
    } else {
        state.settled_frames = state.settled_frames.saturating_add(1);
    }

    // Lay out positions for the draw routine, sourced from `state.nodes`.
    // The pan offset is added here (draw-time only) so the physics state
    // stays in absolute screen coords and continues to settle correctly.
    let sensor_pos = column_pos_for_kind(&state.nodes, NodeKind::Sensor, sensors, pan);
    let neuron_pos = column_pos_for_kind(&state.nodes, NodeKind::Neuron, neurons, pan);
    let action_pos = column_pos_for_kind(&state.nodes, NodeKind::Action, actions, pan);
    // Match LAYERED: ~6 px per mono char at 10 px font, with the 12 px gap
    // between the node circle and the label text baked in.
    let label_max_chars = ((gutter - 14.0) / 6.0).max(4.0) as usize;

    draw_edges(
        painter,
        &agent.nnet,
        sensors,
        neurons,
        actions,
        &sensor_pos,
        &neuron_pos,
        &action_pos,
    );
    draw_nodes(
        painter,
        sim,
        &agent.nnet,
        sensors,
        neurons,
        actions,
        &sensor_pos,
        &neuron_pos,
        &action_pos,
        label_max_chars,
    );

    // Visualise settling state in the bottom-left.
    let badge = if state.settled_frames > 20 {
        egui::RichText::new("● settled")
            .size(9.5)
            .color(theme::ACCENT)
    } else {
        egui::RichText::new(format!("◌ relaxing  ·  v = {:.2}", max_v))
            .size(9.5)
            .color(theme::WARN)
    };
    painter.text(
        egui::pos2(rect.left() + pad, rect.top() + 12.0),
        egui::Align2::LEFT_TOP,
        badge.text().to_owned(),
        egui::FontId::monospace(9.5),
        if state.settled_frames > 20 {
            theme::ACCENT
        } else {
            theme::WARN
        },
    );
}

fn build_initial_nodes(
    inner: egui::Rect,
    gutter: f32,
    sensors: &[u16],
    neurons: &[u16],
    actions: &[u16],
) -> Vec<NodeView> {
    let mut nodes = Vec::with_capacity(sensors.len() + neurons.len() + actions.len());

    // Pinned columns — inset by `gutter` from each edge so labels have room.
    for (i, &idx) in sensors.iter().enumerate() {
        let y = lerp_y(
            inner,
            i as f32 / (sensors.len().max(1) as f32 - 1.0).max(1.0),
        );
        nodes.push(NodeView {
            pos: egui::vec2(inner.left() + gutter, y),
            vel: egui::Vec2::ZERO,
            idx,
            kind: NodeKind::Sensor,
            pinned: true,
        });
    }
    for (i, &idx) in actions.iter().enumerate() {
        let y = lerp_y(
            inner,
            i as f32 / (actions.len().max(1) as f32 - 1.0).max(1.0),
        );
        nodes.push(NodeView {
            pos: egui::vec2(inner.right() - gutter, y),
            vel: egui::Vec2::ZERO,
            idx,
            kind: NodeKind::Action,
            pinned: true,
        });
    }
    // Neurons start jittered around the middle so the simulation has gradient
    // to climb. Deterministic seed from neuron index — same layout every run.
    for &idx in neurons {
        let theta = (idx as f32) * 2.399_963_1; // golden angle scatter
        let r = 40.0 + (idx as f32 * 7.0) % 30.0;
        let x = inner.center().x + r * theta.cos();
        let y = inner.center().y + r * theta.sin();
        nodes.push(NodeView {
            pos: egui::vec2(x, y),
            vel: egui::Vec2::ZERO,
            idx,
            kind: NodeKind::Neuron,
            pinned: false,
        });
    }
    nodes
}

fn lerp_y(inner: egui::Rect, t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    inner.top() + 20.0 + t * (inner.height() - 40.0)
}

fn pin_column(
    nodes: &mut [NodeView],
    kind: NodeKind,
    count: usize,
    x: f32,
    y_top: f32,
    y_bot: f32,
) {
    let mut k = 0usize;
    for n in nodes.iter_mut() {
        if n.kind == kind {
            let t = if count <= 1 {
                0.5
            } else {
                k as f32 / (count as f32 - 1.0)
            };
            let y = y_top + 20.0 + t * (y_bot - y_top - 40.0).max(0.0);
            n.pos = egui::vec2(x, y);
            n.vel = egui::Vec2::ZERO;
            n.pinned = true;
            k += 1;
        }
    }
}

fn build_edges(
    nnet: &biosim4_core::genome::NeuralNet,
    sensors: &[u16],
    neurons: &[u16],
    actions: &[u16],
    nodes: &[NodeView],
) -> Vec<(usize, usize, f32)> {
    let mut edges = Vec::with_capacity(nnet.connections.len());
    let find = |kind: NodeKind, idx: u16, slice: &[u16]| -> Option<usize> {
        let pos_in_slice = slice.iter().position(|&i| i == idx)?;
        // The `nodes` Vec is built in sensor → action → neuron order, so we
        // need a bit of arithmetic to map (kind, slot) to global index.
        let base = match kind {
            NodeKind::Sensor => 0,
            NodeKind::Action => sensors.len(),
            NodeKind::Neuron => sensors.len() + actions.len(),
        };
        let _ = nodes;
        Some(base + pos_in_slice)
    };
    for g in &nnet.connections {
        let src_idx = g.source_num() as u16;
        let snk_idx = g.sink_num() as u16;
        let from = if g.source_type() == SOURCE_SENSOR {
            find(NodeKind::Sensor, src_idx, sensors)
        } else {
            find(NodeKind::Neuron, src_idx, neurons)
        };
        let to = if g.sink_type() == SINK_ACTION {
            find(NodeKind::Action, snk_idx, actions)
        } else {
            find(NodeKind::Neuron, snk_idx, neurons)
        };
        if let (Some(a), Some(b)) = (from, to) {
            edges.push((a, b, g.weight_as_float()));
        }
    }
    edges
}

fn step_forces(nodes: &mut [NodeView], edges: &[(usize, usize, f32)], bounds: egui::Rect) {
    let n = nodes.len();
    let mut forces = vec![egui::Vec2::ZERO; n];

    // Pairwise repulsion (Coulomb-ish, 1/r²).
    for i in 0..n {
        for j in (i + 1)..n {
            let d = nodes[j].pos - nodes[i].pos;
            let dist_sq = d.length_sq().max(25.0);
            let dist = dist_sq.sqrt();
            let f = REPULSION_STRENGTH / dist_sq;
            let dir = d / dist;
            forces[i] -= dir * f;
            forces[j] += dir * f;
        }
    }

    // Springs along edges, with strength scaled by absolute weight.
    for &(a, b, w) in edges {
        let d = nodes[b].pos - nodes[a].pos;
        let dist = d.length().max(0.01);
        let dir = d / dist;
        // Stronger weights ⇒ tighter springs so important connections cluster.
        let k = SPRING_STRENGTH * (0.5 + w.abs() * 0.25);
        let displacement = dist - SPRING_LENGTH;
        let f = dir * (k * displacement);
        forces[a] += f;
        forces[b] -= f;
    }

    // Integrate. Pinned nodes ignore forces entirely.
    for (i, node) in nodes.iter_mut().enumerate() {
        if node.pinned {
            continue;
        }
        node.vel = (node.vel + forces[i]) * DAMPING;
        let speed = node.vel.length();
        if speed > MAX_VELOCITY {
            node.vel *= MAX_VELOCITY / speed;
        }
        node.pos += node.vel;
        // Keep neurons inside the inner rect.
        let margin = 12.0;
        node.pos.x = node
            .pos
            .x
            .clamp(bounds.left() + margin, bounds.right() - margin);
        node.pos.y = node
            .pos
            .y
            .clamp(bounds.top() + margin, bounds.bottom() - margin);
    }
}

fn column_pos_for_kind(
    nodes: &[NodeView],
    kind: NodeKind,
    ids: &[u16],
    pan: egui::Vec2,
) -> Vec<egui::Pos2> {
    ids.iter()
        .map(|&idx| {
            nodes
                .iter()
                .find(|n| n.kind == kind && n.idx == idx)
                .map(|n| egui::pos2(n.pos.x, n.pos.y) + pan)
                .unwrap_or(egui::pos2(0.0, 0.0))
        })
        .collect()
}

// ── Shared edge + node drawing (used by both layout modes) ─────────────────

fn draw_edges(
    painter: &egui::Painter,
    nnet: &biosim4_core::genome::NeuralNet,
    sensors: &[u16],
    neurons: &[u16],
    actions: &[u16],
    sensor_pos: &[egui::Pos2],
    neuron_pos: &[egui::Pos2],
    action_pos: &[egui::Pos2],
) {
    for gene in &nnet.connections {
        let weight = gene.weight_as_float();
        let abs_w = weight.abs();
        let src_idx = gene.source_num() as u16;
        let snk_idx = gene.sink_num() as u16;
        let from = if gene.source_type() == SOURCE_SENSOR {
            sensors
                .iter()
                .position(|&i| i == src_idx)
                .map(|i| sensor_pos[i])
        } else {
            neurons
                .iter()
                .position(|&i| i == src_idx)
                .map(|i| neuron_pos[i])
        };
        let to = if gene.sink_type() == SINK_ACTION {
            actions
                .iter()
                .position(|&i| i == snk_idx)
                .map(|i| action_pos[i])
        } else {
            neurons
                .iter()
                .position(|&i| i == snk_idx)
                .map(|i| neuron_pos[i])
        };
        let (Some(a), Some(b)) = (from, to) else {
            continue;
        };

        let alpha = (abs_w / 4.0).clamp(0.15, 1.0);
        let color = if weight >= 0.0 {
            theme::ACCENT
        } else {
            theme::BAD
        };
        let stroke_color = color.gamma_multiply(alpha);
        let thickness = (abs_w * 0.4 + 0.7).min(2.8);

        // Self-loop or near-overlapping nodes: arc to one side instead of
        // a degenerate line.
        let dist = (b - a).length();
        if dist < 4.0 {
            let r = 12.0;
            let c = egui::pos2(a.x + r, a.y);
            painter.circle_stroke(c, r, egui::Stroke::new(thickness, stroke_color));
            continue;
        }

        // Quadratic bezier with a perpendicular offset so recurrent or
        // intra-layer edges don't overlap straight inter-layer ones.
        let dir = (b - a) / dist;
        let perp = egui::vec2(-dir.y, dir.x);
        let bend = (dist * 0.15).min(28.0);
        let ctrl = egui::pos2((a.x + b.x) * 0.5, (a.y + b.y) * 0.5) + perp * bend;

        let steps = 24;
        let mut prev = a;
        for s in 1..=steps {
            let t = s as f32 / steps as f32;
            let omt = 1.0 - t;
            let p = egui::pos2(
                omt * omt * a.x + 2.0 * omt * t * ctrl.x + t * t * b.x,
                omt * omt * a.y + 2.0 * omt * t * ctrl.y + t * t * b.y,
            );
            painter.line_segment([prev, p], egui::Stroke::new(thickness, stroke_color));
            prev = p;
        }

        // Arrow head at the sink, oriented along the local curve tangent.
        let raw_tangent = b - prev;
        let dir = if raw_tangent.length_sq() > 1e-6 {
            raw_tangent.normalized()
        } else {
            (b - a).normalized()
        };
        let perp_h = egui::vec2(-dir.y, dir.x);
        let head_size = 5.0;
        let tip = b - dir * 8.0;
        let left = tip - dir * head_size + perp_h * head_size * 0.6;
        let right = tip - dir * head_size - perp_h * head_size * 0.6;
        painter.add(egui::Shape::convex_polygon(
            vec![tip, left, right],
            stroke_color,
            egui::Stroke::NONE,
        ));
    }
}

fn draw_nodes(
    painter: &egui::Painter,
    sim: &Sim,
    nnet: &biosim4_core::genome::NeuralNet,
    sensors: &[u16],
    neurons: &[u16],
    actions: &[u16],
    sensor_pos: &[egui::Pos2],
    neuron_pos: &[egui::Pos2],
    action_pos: &[egui::Pos2],
    label_max_chars: usize,
) {
    // Sensors (left)
    for (i, &idx) in sensors.iter().enumerate() {
        let p = sensor_pos[i];
        node(painter, p, theme::WARN, 6.5);
        let name = sim.state.sensors.name(idx);
        painter.text(
            egui::pos2(p.x - 12.0, p.y),
            egui::Align2::RIGHT_CENTER,
            short(name, label_max_chars),
            egui::FontId::monospace(10.0),
            theme::TEXT_2,
        );
    }
    // Neurons (middle / floating)
    for (i, &idx) in neurons.iter().enumerate() {
        let p = neuron_pos[i];
        let (out, driven) = nnet
            .neurons
            .get(idx as usize)
            .map(|n| (n.output, n.driven))
            .unwrap_or((0.0, false));
        let intensity = out.abs().min(1.0);
        let fill = if driven {
            theme::ACCENT.gamma_multiply((intensity * 0.9 + 0.2).clamp(0.25, 1.0))
        } else {
            theme::LINE_2
        };
        node(painter, p, fill, 8.0);
        if driven {
            painter.circle_stroke(
                p,
                11.0,
                egui::Stroke::new(1.0, theme::ACCENT.gamma_multiply(intensity * 0.6 + 0.1)),
            );
        }
        painter.text(
            egui::pos2(p.x, p.y - 16.0),
            egui::Align2::CENTER_BOTTOM,
            format!("N{idx}"),
            egui::FontId::monospace(9.5),
            theme::TEXT_2,
        );
    }
    // Actions (right)
    for (i, &idx) in actions.iter().enumerate() {
        let p = action_pos[i];
        node(painter, p, theme::ACCENT, 6.5);
        let name = sim.state.actions.name(idx);
        painter.text(
            egui::pos2(p.x + 12.0, p.y),
            egui::Align2::LEFT_CENTER,
            short(name, label_max_chars),
            egui::FontId::monospace(10.0),
            theme::TEXT_2,
        );
    }
}

fn node(painter: &egui::Painter, p: egui::Pos2, fill: egui::Color32, r: f32) {
    painter.circle_filled(p, r, fill);
    painter.circle_stroke(p, r, egui::Stroke::new(1.0, theme::LINE_2));
}

fn short(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max.saturating_sub(1)).collect();
        t.push('…');
        t
    }
}

fn column_divider(ui: &mut egui::Ui) {
    ui.add_space(12.0);
    let painter = ui.painter();
    let avail = ui.available_rect_before_wrap();
    painter.line_segment(
        [
            egui::pos2(avail.left(), avail.top()),
            egui::pos2(avail.left(), avail.bottom()),
        ],
        egui::Stroke::new(1.0, theme::LINE),
    );
    ui.add_space(4.0);
}

fn section_label(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(10.0)
            .color(theme::MUTED)
            .strong(),
    );
}

fn kv(ui: &mut egui::Ui, key: &str, value: String) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(key)
                .size(10.5)
                .color(theme::MUTED)
                .strong(),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(value)
                    .monospace()
                    .size(11.0)
                    .color(theme::TEXT),
            );
        });
    });
}
