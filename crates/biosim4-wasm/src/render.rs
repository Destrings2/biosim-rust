//! Frame rendering: produce flat RGBA byte buffers suitable for `ImageData`.
//!
//! The simulation grid uses bottom-left origin (Y=0 at bottom) like the
//! original C++. Canvas `ImageData` uses top-left origin. We flip Y so the
//! frontend can pass the buffer directly into `new ImageData(buf, w, h)`.

use biosim4_core::grid::{BARRIER, EMPTY};
use biosim4_core::sim_state::SimulationState;
use biosim4_core::types::Coord;

const COLOR_EMPTY:   [u8; 3] = [0, 0, 0];
const COLOR_BARRIER: [u8; 3] = [80, 80, 80];

/// Render the current grid into `buf` as a row-major RGBA byte buffer of size
/// `size_x * size_y * 4`. Y is flipped so row 0 is the top of the world.
/// `buf` is resized to fit; pass the same buffer across calls to reuse the
/// allocation (the WASM frontend's render loop calls this 60×/sec).
///
/// Encoding:
/// - empty cells   → `COLOR_EMPTY` (black)
/// - barrier cells → `COLOR_BARRIER` (gray)
/// - agent cells   → the agent's `color` (genome-derived RGB)
///
/// Alpha is always 255.
pub fn render_frame_into(state: &SimulationState, buf: &mut Vec<u8>) {
    let sx = state.config.size_x as usize;
    let sy = state.config.size_y as usize;
    let needed = sx * sy * 4;
    if buf.len() != needed {
        buf.resize(needed, 0);
    }

    for y in 0..sy {
        // Flip Y: canvas row 0 is the top of the world.
        let world_y = sy - 1 - y;
        let row_base = y * sx * 4;
        for x in 0..sx {
            let cell = state.grid.at(Coord::new(x as i16, world_y as i16));
            let rgb = match cell {
                EMPTY   => COLOR_EMPTY,
                BARRIER => COLOR_BARRIER,
                id      => state
                    .population
                    .get(id)
                    .map(|a| a.color)
                    .unwrap_or(COLOR_EMPTY),
            };
            let off = row_base + x * 4;
            buf[off]     = rgb[0];
            buf[off + 1] = rgb[1];
            buf[off + 2] = rgb[2];
            buf[off + 3] = 255;
        }
    }
}

/// Allocating wrapper around [`render_frame_into`]. Used in tests.
#[cfg(test)]
pub fn render_frame(state: &SimulationState) -> Vec<u8> {
    let mut buf = Vec::new();
    render_frame_into(state, &mut buf);
    buf
}

/// Render a single signal layer into `buf` as a tinted alpha mask. Same shape
/// as [`render_frame_into`]; magnitude 0..255 becomes alpha.
pub fn render_signal_layer_into(state: &SimulationState, layer: u8, tint: [u8; 3], buf: &mut Vec<u8>) {
    let sx = state.config.size_x as usize;
    let sy = state.config.size_y as usize;
    let needed = sx * sy * 4;
    if buf.len() != needed {
        buf.resize(needed, 0);
    }

    if layer >= state.signals.layer_count() {
        // Zero-fill (transparent everywhere) so callers don't see stale data.
        buf.iter_mut().for_each(|b| *b = 0);
        return;
    }

    for y in 0..sy {
        let world_y = sy - 1 - y;
        let row_base = y * sx * 4;
        for x in 0..sx {
            let mag = state.signals.get(layer, Coord::new(x as i16, world_y as i16));
            let off = row_base + x * 4;
            buf[off]     = tint[0];
            buf[off + 1] = tint[1];
            buf[off + 2] = tint[2];
            buf[off + 3] = mag;
        }
    }
}

#[cfg(test)]
pub fn render_signal_layer(state: &SimulationState, layer: u8, tint: [u8; 3]) -> Vec<u8> {
    let mut buf = Vec::new();
    render_signal_layer_into(state, layer, tint, &mut buf);
    buf
}

// ── Tests ────────────────────────────────────────────────────────────────
//
// Render is just a pure function over `SimulationState`, so these tests run
// natively (no wasm runtime required).
#[cfg(test)]
mod tests {
    use super::*;
    use biosim4_core::sim_config::SimConfig;
    use biosim4_core::sim_state::SimulationState;

    fn small_config() -> SimConfig {
        SimConfig {
            size_x: 8,
            size_y: 8,
            population: 4,
            steps_per_generation: 5,
            rng_seed: 42,
            barrier_type: 0,
            ..SimConfig::default()
        }
    }

    #[test]
    fn render_frame_has_correct_byte_count_and_alpha() {
        let state = SimulationState::new(small_config());
        let buf = render_frame(&state);
        let sx = state.config.size_x as usize;
        let sy = state.config.size_y as usize;

        assert_eq!(buf.len(), sx * sy * 4, "buffer must be size_x * size_y * 4 RGBA bytes");
        // Every pixel must have alpha=255 (opaque); empties are black with alpha 255 too.
        for i in 0..(sx * sy) {
            assert_eq!(buf[i * 4 + 3], 255, "pixel {} alpha should be 255", i);
        }
    }

    #[test]
    fn render_frame_marks_alive_agents_with_their_color() {
        let state = SimulationState::new(small_config());
        let sx = state.config.size_x as usize;
        let sy = state.config.size_y as usize;
        let buf = render_frame(&state);

        // For every alive agent, the pixel at its (flipped) location must
        // carry the agent's color.
        let mut found = 0;
        for agent in state.population.iter_alive() {
            let canvas_y = sy - 1 - agent.loc.y as usize;
            let off = (canvas_y * sx + agent.loc.x as usize) * 4;
            assert_eq!(buf[off],     agent.color[0], "R mismatch at agent {}", agent.id);
            assert_eq!(buf[off + 1], agent.color[1], "G mismatch at agent {}", agent.id);
            assert_eq!(buf[off + 2], agent.color[2], "B mismatch at agent {}", agent.id);
            found += 1;
        }
        assert_eq!(found, state.population.alive_count(),
                   "should have walked every alive agent");
    }

    #[test]
    fn render_frame_y_axis_is_flipped() {
        // Build a small state, place an agent at world y=0 (bottom), and verify
        // it shows up on the LAST canvas row (top of canvas == top of world).
        let mut cfg = small_config();
        cfg.population = 0;
        let mut state = SimulationState::new(cfg);

        // Manually inject an agent at (0,0) — bottom-left in world coords.
        use biosim4_core::agent::Agent;
        use biosim4_core::genome::ops::make_random_genome;
        use biosim4_core::genome::neural_net::create_wiring;
        use biosim4_core::types::Coord;

        let wcfg = state.wiring_config();
        let genome = make_random_genome(&state.config, &mut state.rng);
        let nnet = create_wiring(&genome, wcfg);
        let id = state.population.next_id();
        let mut a = Agent::new(id, Coord::new(0, 0), genome, nnet);
        a.color = [255, 0, 0]; // bright red so we can spot it
        let assigned = state.population.spawn(a);
        state.grid.set(Coord::new(0, 0), assigned);

        let sx = state.config.size_x as usize;
        let sy = state.config.size_y as usize;
        let buf = render_frame(&state);

        // World (0,0) → canvas row (sy-1), col 0
        let last_row_first_col = ((sy - 1) * sx + 0) * 4;
        assert_eq!(buf[last_row_first_col],     255, "agent at world y=0 should be on bottom canvas row");
        assert_eq!(buf[last_row_first_col + 1], 0);
        assert_eq!(buf[last_row_first_col + 2], 0);

        // And the top canvas row at col 0 should NOT be red (it's empty)
        let first_row_first_col = (0 * sx + 0) * 4;
        assert_eq!(buf[first_row_first_col],     0);
        assert_eq!(buf[first_row_first_col + 1], 0);
        assert_eq!(buf[first_row_first_col + 2], 0);
    }

    #[test]
    fn render_signal_layer_zero_when_layer_out_of_range() {
        let state = SimulationState::new(small_config());
        // Only 1 layer is created by default. Layer 5 is out of range.
        let buf = render_signal_layer(&state, 5, [0, 0, 255]);
        let sx = state.config.size_x as usize;
        let sy = state.config.size_y as usize;
        assert_eq!(buf.len(), sx * sy * 4);
        // All zeros (alpha 0 = transparent)
        assert!(buf.iter().all(|&b| b == 0), "out-of-range layer must produce all-zero buffer");
    }

    #[test]
    fn render_signal_layer_intensity_maps_to_alpha() {
        let mut state = SimulationState::new(small_config());
        // Drop a signal at world (3, 4)
        let center = biosim4_core::types::Coord::new(3, 4);
        state.signals.increment(0, center, &state.grid);

        let buf = render_signal_layer(&state, 0, [0, 0, 255]);
        let sx = state.config.size_x as usize;
        let sy = state.config.size_y as usize;

        let canvas_y = sy - 1 - center.y as usize;
        let off = (canvas_y * sx + center.x as usize) * 4;
        assert_eq!(buf[off],     0);
        assert_eq!(buf[off + 1], 0);
        assert_eq!(buf[off + 2], 255);
        assert!(buf[off + 3] > 0, "signal center should have non-zero alpha");
    }
}

