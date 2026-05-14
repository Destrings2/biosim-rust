// biosim4 GPU step shader — one module, multiple compute entries.
//
// All phases of `step_one` run as GPU compute dispatches against the same
// set of state buffers. The CPU only owns the host copy at fast-forward
// boundaries (upload at start, download at generation end for spawn_new_
// generation, re-upload, repeat).
//
// Entries (in dispatch order per step):
//   1. clear_step_scratch    — zero action_levels, move_queue_count,
//                              death_queue_count.
//   2. phase1_sensors_ff     — evaluate sensors and run neural feed-
//                              forward for every alive agent. One thread
//                              per agent.
//   3. phase2_actions        — apply the resulting action levels for every
//                              alive agent. Pushes to move/death queues
//                              and emits signals atomically.
//   4. drain_deaths          — mark dead and free the cell.
//   5. drain_moves           — atomic CAS into the grid; on success, move
//                              the agent and free its old cell.
//   6. signal_fade           — decrement every signal cell by 1 (floor 0).
//
// Determinism: same contract as the CPU non-deterministic mode. Per-agent
// sensor RNG is reproducible via `phase1_seed_for`; phase-2 action draws
// use per-agent RNG state that lives on GPU.

// ── Constants ──────────────────────────────────────────────────────────────
const MAX_NEURONS: u32 = 32u;
const EMPTY: u32        = 0u;
const BARRIER: u32      = 0xFFFFFFFFu;
// Kill barrier — agents that try to move into one die instead of being
// blocked. See `drain_moves` for the kill semantics.
const KILL_BARRIER: u32 = 0xFFFFFFFEu;
const SOURCE_SENSOR_BIT: u32 = 1u;
const SINK_ACTION_BIT:   u32 = 2u;

const SIGNAL_MAX: u32 = 255u;

// Compass ordinals (matches biosim4_core::types::Compass).
const DIR_N:  u32 = 0u;
const DIR_NE: u32 = 1u;
const DIR_E:  u32 = 2u;
const DIR_SE: u32 = 3u;
const DIR_S:  u32 = 4u;
const DIR_SW: u32 = 5u;
const DIR_W:  u32 = 6u;
const DIR_NW: u32 = 7u;
const DIR_CENTER: u32 = 8u;

// Sensor ids — must match `sensors_index_table` in `state.rs`.
// Any sensor not in this list will produce 0.0 on the GPU.
const SENSOR_LOC_X:           u32 = 0u;
const SENSOR_LOC_Y:           u32 = 1u;
const SENSOR_BOUNDARY_DIST_X: u32 = 2u;
const SENSOR_BOUNDARY_DIST_Y: u32 = 3u;
const SENSOR_BOUNDARY_DIST:   u32 = 4u;
const SENSOR_LAST_MOVE_X:     u32 = 5u;
const SENSOR_LAST_MOVE_Y:     u32 = 6u;
const SENSOR_OSC1:            u32 = 7u;
const SENSOR_AGE:             u32 = 8u;
const SENSOR_RANDOM:          u32 = 9u;
const SENSOR_MEMORY_0:        u32 = 10u;
const SENSOR_MEMORY_1:        u32 = 11u;
const SENSOR_MEMORY_2:        u32 = 12u;
const SENSOR_MEMORY_3:        u32 = 13u;
const SENSOR_BARRIER_FWD:     u32 = 14u;
const SENSOR_BARRIER_LR:      u32 = 15u;
const SENSOR_POPULATION:      u32 = 16u;
const SENSOR_POPULATION_FWD:  u32 = 17u;
const SENSOR_POPULATION_LR:   u32 = 18u;
const SENSOR_KILL_BARRIER_FWD: u32 = 19u;
const SENSOR_SIGNAL0:         u32 = 20u;
const SENSOR_SIGNAL0_FWD:     u32 = 21u;
const SENSOR_SIGNAL0_LR:      u32 = 22u;
const SENSOR_SIGNAL1:         u32 = 23u;
const SENSOR_SIGNAL1_FWD:     u32 = 24u;
const SENSOR_SIGNAL1_LR:      u32 = 25u;
const SENSOR_SIGNAL2:         u32 = 26u;
const SENSOR_SIGNAL2_FWD:     u32 = 27u;
const SENSOR_SIGNAL2_LR:      u32 = 28u;
const SENSOR_LONGPROBE_POP_FWD: u32 = 29u;
const SENSOR_LONGPROBE_BAR_FWD: u32 = 30u;
const SENSOR_GENETIC_SIM_FWD: u32 = 31u;
const SENSOR_ENERGY_LEVEL:    u32 = 32u;
const SENSOR_FOOD_HERE:       u32 = 33u;
const SENSOR_FOOD_FWD:        u32 = 34u;
const SENSOR_FOOD_LR:         u32 = 35u;

// Action ids — must match `actions_index_table` in `state.rs`.
const ACTION_MOVE_X:           u32 = 0u;
const ACTION_MOVE_Y:           u32 = 1u;
const ACTION_MOVE_FORWARD:     u32 = 2u;
const ACTION_MOVE_RL:          u32 = 3u;
const ACTION_MOVE_RANDOM:      u32 = 4u;
const ACTION_MOVE_REVERSE:     u32 = 5u;
const ACTION_MOVE_LEFT:        u32 = 6u;
const ACTION_MOVE_RIGHT:       u32 = 7u;
const ACTION_MOVE_EAST:        u32 = 8u;
const ACTION_MOVE_WEST:        u32 = 9u;
const ACTION_MOVE_NORTH:       u32 = 10u;
const ACTION_MOVE_SOUTH:       u32 = 11u;
const ACTION_SET_RESPONSIVENESS: u32 = 12u;
const ACTION_SET_OSC_PERIOD:   u32 = 13u;
const ACTION_SET_LONGPROBE:    u32 = 14u;
const ACTION_EMIT_SIGNAL0:     u32 = 15u;
const ACTION_WRITE_MEM0:       u32 = 16u;
const ACTION_WRITE_MEM1:       u32 = 17u;
const ACTION_WRITE_MEM2:       u32 = 18u;
const ACTION_WRITE_MEM3:       u32 = 19u;
const ACTION_KILL_FORWARD:     u32 = 20u;
const ACTION_EMIT_SIGNAL1:     u32 = 21u;
const ACTION_EMIT_SIGNAL2:     u32 = 22u;

// ── Types ──────────────────────────────────────────────────────────────────

struct Agent {
    loc:               vec2<i32>,           // 0
    last_move:         vec2<i32>,           // 8
    heading:           u32,                 // 16
    age:               u32,                 // 20
    osc_period:        u32,                 // 24
    long_probe_dist:   u32,                 // 28
    responsiveness:    f32,                 // 32
    energy:            f32,                 // 36
    alive:             u32,                 // 40
    color:             u32,                 // 44
    challenge_bits:    u32,                 // 48
    rng_state:         u32,                 // 52
    _pad1:             vec2<u32>,           // 56
    memory:            vec4<f32>,           // 64
    neuron_outputs:    array<f32, 32>,      // 80..208
    _pad2:             array<f32, 12>,      // 208..256
}

struct AgentNet {
    conn_start:   u32,
    conn_end:     u32,
    neuron_count: u32,
    driven_mask:  u32,
}

struct Connection {
    source_idx: u32,
    sink_idx:   u32,
    weight:     f32,
    // bit 0 = sensor source, bit 1 = action sink, bits 2..9 = sensor_id (
    // if sensor source), bits 10..17 = action_id (if action sink).
    flags:      u32,
}

struct MoveReq {
    agent_id: u32,
    dest:     vec2<i32>,
    old:      vec2<i32>,
}

struct Params {
    num_population:        u32, // buffer slot count
    sim_step:              u32,
    generation:            u32,
    size_x:                u32,
    size_y:                u32,
    steps_per_generation:  u32,
    sensor_count:          u32,
    action_count:          u32,
    pop_radius:            f32, // for population_density
    rng_seed_lo:           u32,
    rng_seed_hi:           u32,
    short_probe_distance:  u32,
    signal_layers:         u32,
    food_regen_rate:       f32, // unused by sensors; kept for layout parity
    energy_per_step_cost:  f32, // unused by sensors; kept for layout parity
    _pad:                  u32,
}

// ── Storage bindings ───────────────────────────────────────────────────────

@group(0) @binding(0)  var<storage, read_write> agents:             array<Agent>;
@group(0) @binding(1)  var<storage, read>       agent_nets:         array<AgentNet>;
@group(0) @binding(2)  var<storage, read>       connections:        array<Connection>;
@group(0) @binding(3)  var<storage, read_write> grid:               array<atomic<u32>>;
@group(0) @binding(4)  var<storage, read_write> signals:            array<atomic<u32>>;
@group(0) @binding(5)  var<storage, read_write> action_levels:      array<f32>;
@group(0) @binding(6)  var<storage, read_write> move_queue_count:   atomic<u32>;
@group(0) @binding(7)  var<storage, read_write> move_queue:         array<MoveReq>;
@group(0) @binding(8)  var<storage, read_write> death_queue_count:  atomic<u32>;
@group(0) @binding(9)  var<storage, read_write> death_queue:        array<u32>;
@group(0) @binding(10) var<uniform>             params:             Params;
// Read-only food density grid (cells laid out row-major, same indexing as grid).
@group(0) @binding(11) var<storage, read>       food:               array<f32>;
// Flat genome data: one u32 per gene, packed back-to-back across agents.
@group(0) @binding(12) var<storage, read>       genome_data:        array<u32>;
// Per-agent genome offsets/lengths. `genome_offsets[2*id] = start`,
// `genome_offsets[2*id+1] = len`. Index by agent slot id.
@group(0) @binding(13) var<storage, read>       genome_offsets:     array<u32>;

// ── Helpers ────────────────────────────────────────────────────────────────

fn grid_index(loc: vec2<i32>) -> u32 {
    return u32(loc.y) * params.size_x + u32(loc.x);
}

fn signal_index(layer: u32, loc: vec2<i32>) -> u32 {
    return layer * params.size_x * params.size_y
        + u32(loc.y) * params.size_x + u32(loc.x);
}

fn in_bounds(loc: vec2<i32>) -> bool {
    return loc.x >= 0 && loc.y >= 0
        && u32(loc.x) < params.size_x && u32(loc.y) < params.size_y;
}

// SplitMix64-style 32-bit hash for stateless per-agent RNG (sensor seeding).
fn splitmix_seed(seed_lo: u32, seed_hi: u32, generation: u32, sim_step: u32, agent_id: u32) -> u32 {
    var z: u32 = seed_lo
        ^ (0x9E3779B9u * (generation + 1u))
        ^ (0xBF58476Du * (sim_step + 1u))
        ^ agent_id
        ^ seed_hi;
    z = (z ^ (z >> 16u)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13u)) * 0xC2B2AE35u;
    z = z ^ (z >> 16u);
    return z;
}

// LCG step on a per-agent rng state (for phase-2 action draws).
fn rng_next(state: ptr<function, u32>) -> u32 {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

fn rng_unit(state: ptr<function, u32>) -> f32 {
    let r = rng_next(state);
    return f32(r) / 4294967296.0;
}

// Map a heading u32 to a unit-step vec2<i32>. Compass ordinals match
// biosim4_core::types::Dir.
fn heading_to_step(h: u32) -> vec2<i32> {
    switch h {
        case 0u: { return vec2<i32>( 0,  1); } // N
        case 1u: { return vec2<i32>( 1,  1); } // NE
        case 2u: { return vec2<i32>( 1,  0); } // E
        case 3u: { return vec2<i32>( 1, -1); } // SE
        case 4u: { return vec2<i32>( 0, -1); } // S
        case 5u: { return vec2<i32>(-1, -1); } // SW
        case 6u: { return vec2<i32>(-1,  0); } // W
        case 7u: { return vec2<i32>(-1,  1); } // NW
        default: { return vec2<i32>( 0,  0); }
    }
}

fn rotate_cw_step(step: vec2<i32>) -> vec2<i32> {
    return vec2<i32>(step.y, -step.x);
}
fn rotate_ccw_step(step: vec2<i32>) -> vec2<i32> {
    return vec2<i32>(-step.y, step.x);
}

// Jaro-Winkler similarity over up to 20 genes per genome (matches CPU
// `jaro_winkler` in genome/ops.rs). Returns a value in [0, 1].
fn genome_similarity_jw(a_start: u32, a_len: u32, b_start: u32, b_len: u32) -> f32 {
    if (a_len == 0u && b_len == 0u) { return 1.0; }
    if (a_len == 0u || b_len == 0u) { return 0.0; }
    let n = min(a_len, 20u);
    let m = min(b_len, 20u);
    let big = max(n, m);
    var window: u32 = 1u;
    if (big / 2u >= 1u) { window = big / 2u - 1u; if (window < 1u) { window = 1u; } }
    var a_match: u32 = 0u; // bitmask up to 20 bits
    var b_match: u32 = 0u;
    var matches: u32 = 0u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let lo = select(i - window, 0u, i < window);
        var hi = i + window + 1u;
        if (hi > m) { hi = m; }
        for (var j: u32 = lo; j < hi; j = j + 1u) {
            if ((b_match & (1u << j)) != 0u) { continue; }
            if (genome_data[a_start + i] == genome_data[b_start + j]) {
                a_match = a_match | (1u << i);
                b_match = b_match | (1u << j);
                matches = matches + 1u;
                break;
            }
        }
    }
    if (matches == 0u) { return 0.0; }
    var transpositions: u32 = 0u;
    var k: u32 = 0u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        if ((a_match & (1u << i)) == 0u) { continue; }
        loop {
            if ((b_match & (1u << k)) != 0u) { break; }
            k = k + 1u;
        }
        if (genome_data[a_start + i] != genome_data[b_start + k]) {
            transpositions = transpositions + 1u;
        }
        k = k + 1u;
    }
    let mf = f32(matches);
    let jaro = (mf / f32(n)
        + mf / f32(m)
        + (mf - f32(transpositions / 2u)) / mf) / 3.0;
    // Winkler prefix bonus over the first 4 genes.
    var prefix: u32 = 0u;
    let plimit = min(min(n, m), 4u);
    for (var i: u32 = 0u; i < plimit; i = i + 1u) {
        if (genome_data[a_start + i] == genome_data[b_start + i]) {
            prefix = prefix + 1u;
        } else {
            break;
        }
    }
    return min(jaro + f32(prefix) * 0.1 * (1.0 - jaro), 1.0);
}

// Weighted ray-sameness between an offset vector and a direction step
// vector. Matches the CPU `Coord::ray_sameness_dir`.
fn ray_sameness(offset: vec2<i32>, dir_step: vec2<i32>) -> f32 {
    let ox = f32(offset.x); let oy = f32(offset.y);
    let dx = f32(dir_step.x); let dy = f32(dir_step.y);
    let off_mag = sqrt(ox * ox + oy * oy);
    let dir_mag = sqrt(dx * dx + dy * dy);
    if (off_mag == 0.0 || dir_mag == 0.0) { return 0.0; }
    return clamp((ox * dx + oy * dy) / (off_mag * dir_mag), -1.0, 1.0);
}

// Forward-vs-backward signal weighted density along a direction.
// Mirrors CPU `signal_density_along_axis`.
fn signal_density_axis(layer: u32, loc: vec2<i32>, dir_step: vec2<i32>, radius: f32) -> f32 {
    let r = i32(ceil(radius));
    var fwd_sum: f32 = 0.0;
    var bwd_sum: f32 = 0.0;
    for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
        for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
            if (dx == 0 && dy == 0) { continue; }
            let d2 = f32(dx * dx + dy * dy);
            if (d2 > radius * radius) { continue; }
            let p = loc + vec2<i32>(dx, dy);
            if (!in_bounds(p)) { continue; }
            let mag = f32(atomicLoad(&signals[signal_index(layer, p)])) / 255.0;
            let s = ray_sameness(vec2<i32>(dx, dy), dir_step);
            if (s >= 0.0) { fwd_sum = fwd_sum + mag * s; }
            else          { bwd_sum = bwd_sum + mag * (-s); }
        }
    }
    return clamp((fwd_sum - bwd_sum) / (radius * radius), -1.0, 1.0) * 0.5 + 0.5;
}

// Forward-weighted food density along a direction. Mirrors CPU
// `FoodLayer::get_density_fwd` — returns 0.5 when nothing nearby.
fn food_density_axis(loc: vec2<i32>, dir_step: vec2<i32>, radius: f32) -> f32 {
    let r = i32(ceil(radius));
    var fwd_sum: f32 = 0.0;
    var bwd_sum: f32 = 0.0;
    for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
        for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
            if (dx == 0 && dy == 0) { continue; }
            let d2 = f32(dx * dx + dy * dy);
            if (d2 > radius * radius) { continue; }
            let p = loc + vec2<i32>(dx, dy);
            if (!in_bounds(p)) { continue; }
            let mag = food[grid_index(p)];
            let s = ray_sameness(vec2<i32>(dx, dy), dir_step);
            if (s >= 0.0) { fwd_sum = fwd_sum + mag * s; }
            else          { bwd_sum = bwd_sum + mag * (-s); }
        }
    }
    let total = fwd_sum + bwd_sum;
    if (total == 0.0) { return 0.5; }
    return clamp(fwd_sum / total, 0.0, 1.0);
}

fn evaluate_sensor(sensor_id: u32, agent_id: u32, rng: ptr<function, u32>) -> f32 {
    let agent = agents[agent_id];
    switch sensor_id {
        case 0u: { // LOC_X
            return f32(agent.loc.x) / f32(params.size_x - 1u);
        }
        case 1u: { // LOC_Y
            return f32(agent.loc.y) / f32(params.size_y - 1u);
        }
        case 2u: { // BOUNDARY_DIST_X
            let x = f32(agent.loc.x);
            let sx = f32(params.size_x - 1u);
            return clamp(min(x, sx - x) / (sx * 0.5), 0.0, 1.0);
        }
        case 3u: { // BOUNDARY_DIST_Y
            let y = f32(agent.loc.y);
            let sy = f32(params.size_y - 1u);
            return clamp(min(y, sy - y) / (sy * 0.5), 0.0, 1.0);
        }
        case 4u: { // BOUNDARY_DIST
            let x = f32(agent.loc.x); let y = f32(agent.loc.y);
            let sx = f32(params.size_x - 1u); let sy = f32(params.size_y - 1u);
            let dx = min(x, sx - x); let dy = min(y, sy - y);
            return clamp(min(dx, dy) / (min(sx, sy) * 0.5), 0.0, 1.0);
        }
        case 5u: { // LAST_MOVE_X
            return (f32(agent.last_move.x) + 1.0) * 0.5;
        }
        case 6u: { // LAST_MOVE_Y
            return (f32(agent.last_move.y) + 1.0) * 0.5;
        }
        case 7u: { // OSC1
            let period = max(agent.osc_period, 1u);
            let phase = f32(params.sim_step % period) / f32(period);
            return (sin(phase * 6.28318530718) + 1.0) * 0.5;
        }
        case 8u: { // AGE
            return clamp(f32(agent.age) / f32(params.steps_per_generation), 0.0, 1.0);
        }
        case 9u: { // RANDOM
            return rng_unit(rng);
        }
        case 10u: { return (agent.memory.x + 1.0) * 0.5; }
        case 11u: { return (agent.memory.y + 1.0) * 0.5; }
        case 12u: { return (agent.memory.z + 1.0) * 0.5; }
        case 13u: { return (agent.memory.w + 1.0) * 0.5; }
        case 14u: { // BARRIER_FWD — short probe forward for barrier
            let step = heading_to_step(agent.heading);
            for (var i: i32 = 1; i <= i32(params.short_probe_distance); i = i + 1) {
                let p = agent.loc + step * i;
                if (!in_bounds(p)) { return 1.0; }
                if (atomicLoad(&grid[grid_index(p)]) == BARRIER) {
                    return f32(i) / f32(params.short_probe_distance);
                }
            }
            return 1.0;
        }
        case 15u: { // BARRIER_LR
            let fwd = heading_to_step(agent.heading);
            let left = rotate_ccw_step(fwd);
            let right = rotate_cw_step(fwd);
            var sum: f32 = 0.0;
            var count: i32 = 0;
            for (var s: i32 = 0; s < 2; s = s + 1) {
                let step = select(right, left, s == 0);
                for (var i: i32 = 1; i <= i32(params.short_probe_distance); i = i + 1) {
                    let p = agent.loc + step * i;
                    if (!in_bounds(p)) { sum = sum + 1.0; count = count + 1; break; }
                    if (atomicLoad(&grid[grid_index(p)]) == BARRIER) {
                        sum = sum + f32(i) / f32(params.short_probe_distance);
                        count = count + 1;
                        break;
                    }
                }
            }
            if (count == 0) { return 1.0; }
            return sum / f32(count);
        }
        case 16u: { // POPULATION density in radius
            let r = i32(ceil(params.pop_radius));
            var occupied: u32 = 0u;
            var total: u32 = 0u;
            for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
                for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
                    let d2 = f32(dx*dx + dy*dy);
                    if (d2 > params.pop_radius * params.pop_radius) { continue; }
                    let p = agent.loc + vec2<i32>(dx, dy);
                    if (!in_bounds(p)) { continue; }
                    total = total + 1u;
                    let cell = atomicLoad(&grid[grid_index(p)]);
                    if (cell != EMPTY && cell != BARRIER && cell != KILL_BARRIER) { occupied = occupied + 1u; }
                }
            }
            if (total == 0u) { return 0.0; }
            return f32(occupied) / f32(total);
        }
        case 17u: { // POPULATION_FWD — agents along heading axis
            let step = heading_to_step(agent.heading);
            let r = i32(ceil(params.pop_radius));
            var occupied: u32 = 0u;
            var total: u32 = 0u;
            for (var i: i32 = 1; i <= r; i = i + 1) {
                let p = agent.loc + step * i;
                if (!in_bounds(p)) { break; }
                total = total + 1u;
                let cell = atomicLoad(&grid[grid_index(p)]);
                if (cell != EMPTY && cell != BARRIER && cell != KILL_BARRIER) { occupied = occupied + 1u; }
            }
            if (total == 0u) { return 0.0; }
            return f32(occupied) / f32(total);
        }
        case 18u: { // POPULATION_LR — agents perpendicular to heading
            let fwd = heading_to_step(agent.heading);
            let left = rotate_ccw_step(fwd);
            let right = rotate_cw_step(fwd);
            let r = i32(ceil(params.pop_radius));
            var occupied: u32 = 0u;
            var total: u32 = 0u;
            for (var s: i32 = 0; s < 2; s = s + 1) {
                let step = select(right, left, s == 0);
                for (var i: i32 = 1; i <= r; i = i + 1) {
                    let p = agent.loc + step * i;
                    if (!in_bounds(p)) { break; }
                    total = total + 1u;
                    let cell = atomicLoad(&grid[grid_index(p)]);
                    if (cell != EMPTY && cell != BARRIER && cell != KILL_BARRIER) { occupied = occupied + 1u; }
                }
            }
            if (total == 0u) { return 0.0; }
            return f32(occupied) / f32(total);
        }
        case 19u: { // KILL_BARRIER_FWD — short probe forward for kill barriers
            let step = heading_to_step(agent.heading);
            if (step.x == 0 && step.y == 0) { return 0.0; }
            let max = i32(params.short_probe_distance);
            for (var i: i32 = 1; i <= max; i = i + 1) {
                let p = agent.loc + step * i;
                if (!in_bounds(p)) { return 0.0; }
                if (atomicLoad(&grid[grid_index(p)]) == KILL_BARRIER) {
                    return 1.0 - f32(i - 1) / f32(max);
                }
            }
            return 0.0;
        }
        case 20u: { // SIGNAL0 (here) — current cell, layer 0
            let v = atomicLoad(&signals[signal_index(0u, agent.loc)]);
            return f32(v) / 255.0;
        }
        case 21u: { // SIGNAL0_FWD — average signal along heading axis
            let step = heading_to_step(agent.heading);
            if (step.x == 0 && step.y == 0) { return 0.0; }
            let max = i32(params.short_probe_distance);
            var sum: u32 = 0u;
            var count: i32 = 0;
            for (var i: i32 = 1; i <= max; i = i + 1) {
                let p = agent.loc + step * i;
                if (!in_bounds(p)) { break; }
                sum = sum + atomicLoad(&signals[signal_index(0u, p)]);
                count = count + 1;
            }
            if (count == 0) { return 0.0; }
            return f32(sum) / (f32(count) * 255.0);
        }
        case 22u: { // SIGNAL0_LR — average signal perpendicular to heading
            let fwd = heading_to_step(agent.heading);
            let left = rotate_ccw_step(fwd);
            let right = rotate_cw_step(fwd);
            let max = i32(params.short_probe_distance);
            var sum: u32 = 0u;
            var count: i32 = 0;
            for (var s: i32 = 0; s < 2; s = s + 1) {
                let step = select(right, left, s == 0);
                for (var i: i32 = 1; i <= max; i = i + 1) {
                    let p = agent.loc + step * i;
                    if (!in_bounds(p)) { break; }
                    sum = sum + atomicLoad(&signals[signal_index(0u, p)]);
                    count = count + 1;
                }
            }
            if (count == 0) { return 0.0; }
            return f32(sum) / (f32(count) * 255.0);
        }
        case 23u: { // SIGNAL1 here
            if (params.signal_layers < 2u) { return 0.0; }
            let v = atomicLoad(&signals[signal_index(1u, agent.loc)]);
            return f32(v) / 255.0;
        }
        case 24u: { // SIGNAL1_FWD
            if (params.signal_layers < 2u) { return 0.0; }
            let step = heading_to_step(agent.heading);
            if (step.x == 0 && step.y == 0) { return 0.0; }
            let max = i32(params.short_probe_distance);
            var sum: u32 = 0u;
            var count: i32 = 0;
            for (var i: i32 = 1; i <= max; i = i + 1) {
                let p = agent.loc + step * i;
                if (!in_bounds(p)) { break; }
                sum = sum + atomicLoad(&signals[signal_index(1u, p)]);
                count = count + 1;
            }
            if (count == 0) { return 0.0; }
            return f32(sum) / (f32(count) * 255.0);
        }
        case 25u: { // SIGNAL1_LR
            if (params.signal_layers < 2u) { return 0.0; }
            let fwd = heading_to_step(agent.heading);
            let left = rotate_ccw_step(fwd);
            let right = rotate_cw_step(fwd);
            let max = i32(params.short_probe_distance);
            var sum: u32 = 0u;
            var count: i32 = 0;
            for (var s: i32 = 0; s < 2; s = s + 1) {
                let step = select(right, left, s == 0);
                for (var i: i32 = 1; i <= max; i = i + 1) {
                    let p = agent.loc + step * i;
                    if (!in_bounds(p)) { break; }
                    sum = sum + atomicLoad(&signals[signal_index(1u, p)]);
                    count = count + 1;
                }
            }
            if (count == 0) { return 0.0; }
            return f32(sum) / (f32(count) * 255.0);
        }
        case 26u: { // SIGNAL2 here
            if (params.signal_layers < 3u) { return 0.0; }
            let v = atomicLoad(&signals[signal_index(2u, agent.loc)]);
            return f32(v) / 255.0;
        }
        case 27u: { // SIGNAL2_FWD
            if (params.signal_layers < 3u) { return 0.0; }
            let step = heading_to_step(agent.heading);
            if (step.x == 0 && step.y == 0) { return 0.0; }
            let max = i32(params.short_probe_distance);
            var sum: u32 = 0u;
            var count: i32 = 0;
            for (var i: i32 = 1; i <= max; i = i + 1) {
                let p = agent.loc + step * i;
                if (!in_bounds(p)) { break; }
                sum = sum + atomicLoad(&signals[signal_index(2u, p)]);
                count = count + 1;
            }
            if (count == 0) { return 0.0; }
            return f32(sum) / (f32(count) * 255.0);
        }
        case 28u: { // SIGNAL2_LR
            if (params.signal_layers < 3u) { return 0.0; }
            let fwd = heading_to_step(agent.heading);
            let left = rotate_ccw_step(fwd);
            let right = rotate_cw_step(fwd);
            let max = i32(params.short_probe_distance);
            var sum: u32 = 0u;
            var count: i32 = 0;
            for (var s: i32 = 0; s < 2; s = s + 1) {
                let step = select(right, left, s == 0);
                for (var i: i32 = 1; i <= max; i = i + 1) {
                    let p = agent.loc + step * i;
                    if (!in_bounds(p)) { break; }
                    sum = sum + atomicLoad(&signals[signal_index(2u, p)]);
                    count = count + 1;
                }
            }
            if (count == 0) { return 0.0; }
            return f32(sum) / (f32(count) * 255.0);
        }
        case 29u: { // LONGPROBE_POP_FWD — distance to nearest agent ahead, normalized
            let step = heading_to_step(agent.heading);
            if (step.x == 0 && step.y == 0) { return 0.0; }
            let max = max(i32(agent.long_probe_dist), 1);
            for (var i: i32 = 1; i <= max; i = i + 1) {
                let p = agent.loc + step * i;
                if (!in_bounds(p)) { return 1.0; }
                let cell = atomicLoad(&grid[grid_index(p)]);
                if (cell != EMPTY && cell != BARRIER && cell != KILL_BARRIER) {
                    return 1.0 - f32(i) / f32(max);
                }
            }
            return 0.0;
        }
        case 30u: { // LONGPROBE_BAR_FWD — distance to nearest barrier ahead
            let step = heading_to_step(agent.heading);
            if (step.x == 0 && step.y == 0) { return 0.0; }
            let max = max(i32(agent.long_probe_dist), 1);
            for (var i: i32 = 1; i <= max; i = i + 1) {
                let p = agent.loc + step * i;
                if (!in_bounds(p) || atomicLoad(&grid[grid_index(p)]) == BARRIER) {
                    return 1.0 - f32(i) / f32(max);
                }
            }
            return 0.0;
        }
        case 31u: { // GENETIC_SIM_FWD — Jaro-Winkler to first agent within 4 cells ahead
            let step = heading_to_step(agent.heading);
            if (step.x == 0 && step.y == 0) { return 0.0; }
            let a_off = genome_offsets[agent_id * 2u];
            let a_len = genome_offsets[agent_id * 2u + 1u];
            for (var i: i32 = 1; i <= 4; i = i + 1) {
                let p = agent.loc + step * i;
                if (!in_bounds(p)) { break; }
                let cell = atomicLoad(&grid[grid_index(p)]);
                if (cell == EMPTY || cell == BARRIER || cell == KILL_BARRIER) { continue; }
                let b_off = genome_offsets[cell * 2u];
                let b_len = genome_offsets[cell * 2u + 1u];
                return genome_similarity_jw(a_off, a_len, b_off, b_len);
            }
            return 0.0;
        }
        case 32u: { // ENERGY_LEVEL — agent.energy clamped to [0,1]
            return clamp(agent.energy, 0.0, 1.0);
        }
        case 33u: { // FOOD_HERE — food at current cell
            return food[grid_index(agent.loc)];
        }
        case 34u: { // FOOD_FWD
            let step = heading_to_step(agent.heading);
            return food_density_axis(agent.loc, step, 3.0);
        }
        case 35u: { // FOOD_LR
            let fwd = heading_to_step(agent.heading);
            let left = rotate_ccw_step(fwd);
            let right = rotate_cw_step(fwd);
            let lv = food_density_axis(agent.loc, left, 3.0);
            let rv = food_density_axis(agent.loc, right, 3.0);
            return clamp((lv + rv) * 0.5, 0.0, 1.0);
        }
        default: { return 0.0; }
    }
}

// ── Phase 1: sensors + feed-forward (one thread per agent) ────────────────

@compute @workgroup_size(64)
fn phase1_sensors_ff(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.num_population) { return; }
    if (agents[agent_id].alive == 0u) { return; }

    let net = agent_nets[agent_id];
    let neuron_count = net.neuron_count;
    let driven_mask = net.driven_mask;

    var sensor_rng: u32 = splitmix_seed(
        params.rng_seed_lo, params.rng_seed_hi,
        params.generation, params.sim_step, agent_id,
    );

    var neuron_accum: array<f32, 32>;
    for (var n: u32 = 0u; n < neuron_count; n = n + 1u) {
        neuron_accum[n] = 0.0;
    }

    // Clear action levels for this agent.
    let abase = agent_id * params.action_count;
    for (var a: u32 = 0u; a < params.action_count; a = a + 1u) {
        action_levels[abase + a] = 0.0;
    }

    var neurons_computed: bool = false;

    for (var c: u32 = net.conn_start; c < net.conn_end; c = c + 1u) {
        let conn = connections[c];
        let is_sensor_src = (conn.flags & SOURCE_SENSOR_BIT) != 0u;
        let is_action_snk = (conn.flags & SINK_ACTION_BIT)   != 0u;

        // Mid-loop pivot: finalize neuron outputs before first action sink.
        if (is_action_snk && !neurons_computed) {
            for (var n: u32 = 0u; n < neuron_count; n = n + 1u) {
                if ((driven_mask & (1u << n)) != 0u) {
                    agents[agent_id].neuron_outputs[n] = tanh(neuron_accum[n]);
                }
            }
            neurons_computed = true;
        }

        var src_val: f32 = 0.0;
        if (is_sensor_src) {
            // Sensor id is encoded in conn.source_idx (registry index).
            src_val = evaluate_sensor(conn.source_idx, agent_id, &sensor_rng);
        } else {
            src_val = agents[agent_id].neuron_outputs[conn.source_idx];
        }

        let contribution = conn.weight * src_val;
        if (is_action_snk) {
            action_levels[abase + conn.sink_idx]
                = action_levels[abase + conn.sink_idx] + contribution;
        } else {
            neuron_accum[conn.sink_idx]
                = neuron_accum[conn.sink_idx] + contribution;
        }
    }

    if (!neurons_computed) {
        for (var n: u32 = 0u; n < neuron_count; n = n + 1u) {
            if ((driven_mask & (1u << n)) != 0u) {
                agents[agent_id].neuron_outputs[n] = tanh(neuron_accum[n]);
            }
        }
    }
}

// ── Phase 2: action execution + age (one thread per agent) ────────────────

fn responsiveness_curve(r: f32) -> f32 {
    // Matches the CPU `responsiveness_curve` (k=2 default sigmoid-ish).
    let k = 2.0;
    let r2 = clamp(r, 0.0, 1.0);
    let m = (r2 - 0.5) * 2.0;
    let s = 1.0 / (1.0 + exp(-m * k));
    return s;
}

fn prob2bool(p: f32, rng: ptr<function, u32>) -> bool {
    return rng_unit(rng) < clamp(p, 0.0, 1.0);
}

fn try_enqueue_move(agent_id: u32, old_loc: vec2<i32>, dest: vec2<i32>) {
    if (!in_bounds(dest)) { return; }
    let q = atomicAdd(&move_queue_count, 1u);
    move_queue[q] = MoveReq(agent_id, dest, old_loc);
}

fn emit_signal(layer: u32, center: vec2<i32>) {
    // Center +2, neighbors +1, saturating at SIGNAL_MAX. Mirrors CPU
    // `Signals::increment`.
    let layer_off = layer * params.size_x * params.size_y;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let d2 = f32(dx * dx + dy * dy);
            if (d2 > 1.5 * 1.5) { continue; }
            let p = center + vec2<i32>(dx, dy);
            if (!in_bounds(p)) { continue; }
            let delta: u32 = select(1u, 2u, dx == 0 && dy == 0);
            let idx = layer_off + u32(p.y) * params.size_x + u32(p.x);
            // Saturate at SIGNAL_MAX via CAS loop.
            loop {
                let prev = atomicLoad(&signals[idx]);
                let new_v = min(prev + delta, SIGNAL_MAX);
                if (new_v == prev) { break; }
                let r = atomicCompareExchangeWeak(&signals[idx], prev, new_v);
                if (r.exchanged) { break; }
            }
        }
    }
}

@compute @workgroup_size(64)
fn phase2_actions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.num_population) { return; }
    if (agents[agent_id].alive == 0u) { return; }

    let abase = agent_id * params.action_count;
    let resp = responsiveness_curve(agents[agent_id].responsiveness);
    var rng_state: u32 = agents[agent_id].rng_state
        ^ splitmix_seed(params.rng_seed_lo ^ 0xDEADBEEFu, params.rng_seed_hi,
                        params.generation, params.sim_step, agent_id);

    let loc = agents[agent_id].loc;
    let fwd = heading_to_step(agents[agent_id].heading);
    let left = rotate_ccw_step(fwd);
    let right = rotate_cw_step(fwd);

    // Movement is accumulated as a vec2 then resolved at the end. Matches
    // the CPU behavior where multiple move-axis actions compete.
    var move_x_drive: f32 = 0.0;
    var move_y_drive: f32 = 0.0;

    for (var a: u32 = 0u; a < params.action_count; a = a + 1u) {
        let level = action_levels[abase + a] * resp;
        let p = tanh(level);
        if (abs(p) < 0.0001) { continue; }
        switch a {
            case 0u: { move_x_drive = move_x_drive + p; }                                     // MOVE_X
            case 1u: { move_y_drive = move_y_drive + p; }                                     // MOVE_Y
            case 2u: { move_x_drive = move_x_drive + p * f32(fwd.x);
                       move_y_drive = move_y_drive + p * f32(fwd.y); }                        // MOVE_FORWARD
            case 3u: {
                let s = select(right, left, p > 0.0);
                move_x_drive = move_x_drive + abs(p) * f32(s.x);
                move_y_drive = move_y_drive + abs(p) * f32(s.y);
            }                                                                                 // MOVE_RL
            case 4u: {                                                                        // MOVE_RANDOM
                if (prob2bool(abs(p), &rng_state)) {
                    let dx = i32(rng_next(&rng_state) % 3u) - 1;
                    let dy = i32(rng_next(&rng_state) % 3u) - 1;
                    move_x_drive = move_x_drive + f32(dx);
                    move_y_drive = move_y_drive + f32(dy);
                }
            }
            case 5u: { move_x_drive = move_x_drive - p * f32(fwd.x);
                       move_y_drive = move_y_drive - p * f32(fwd.y); }                        // MOVE_REVERSE
            case 6u: { move_x_drive = move_x_drive + p * f32(left.x);
                       move_y_drive = move_y_drive + p * f32(left.y); }                        // MOVE_LEFT
            case 7u: { move_x_drive = move_x_drive + p * f32(right.x);
                       move_y_drive = move_y_drive + p * f32(right.y); }                       // MOVE_RIGHT
            case 8u: { move_x_drive = move_x_drive + p; }                                     // MOVE_EAST
            case 9u: { move_x_drive = move_x_drive - p; }                                     // MOVE_WEST
            case 10u: { move_y_drive = move_y_drive + p; }                                    // MOVE_NORTH
            case 11u: { move_y_drive = move_y_drive - p; }                                    // MOVE_SOUTH
            case 12u: { agents[agent_id].responsiveness = clamp((tanh(level) + 1.0) * 0.5, 0.0, 1.0); } // SET_RESPONSIVENESS
            case 13u: { agents[agent_id].osc_period = u32(clamp((tanh(level) + 1.0) * 0.5 * 100.0 + 1.0, 1.0, 1024.0)); }
            case 14u: { agents[agent_id].long_probe_dist = u32(clamp((tanh(level) + 1.0) * 0.5 * 32.0 + 1.0, 1.0, 64.0)); }
            case 15u: { if (prob2bool(abs(p), &rng_state)) { emit_signal(0u, loc); } }        // EMIT_SIGNAL0
            case 16u: { agents[agent_id].memory.x = tanh(level); }
            case 17u: { agents[agent_id].memory.y = tanh(level); }
            case 18u: { agents[agent_id].memory.z = tanh(level); }
            case 19u: { agents[agent_id].memory.w = tanh(level); }
            case 20u: { // KILL_FORWARD — probabilistically kill the agent
                       // one cell ahead. Mirrors CPU `KillForward`.
                if (prob2bool(abs(p), &rng_state)) {
                    let dest_k = loc + fwd;
                    if (in_bounds(dest_k)) {
                        let cell = atomicLoad(&grid[grid_index(dest_k)]);
                        if (cell != EMPTY && cell != BARRIER && cell != KILL_BARRIER) {
                            // cell holds the victim's agent_id
                            let q = atomicAdd(&death_queue_count, 1u);
                            death_queue[q] = cell;
                        }
                    }
                }
            }
            case 21u: { // EMIT_SIGNAL1
                if (params.signal_layers >= 2u && prob2bool(abs(p), &rng_state)) {
                    emit_signal(1u, loc);
                }
            }
            case 22u: { // EMIT_SIGNAL2
                if (params.signal_layers >= 3u && prob2bool(abs(p), &rng_state)) {
                    emit_signal(2u, loc);
                }
            }
            default: {}
        }
    }

    // Decide a destination cell from accumulated drives. Probabilistic
    // step on each axis so small drives don't always move (matches CPU
    // `move_x` semantics).
    var dx: i32 = 0;
    var dy: i32 = 0;
    if (prob2bool(abs(tanh(move_x_drive)), &rng_state)) {
        dx = select(1, -1, move_x_drive < 0.0);
    }
    if (prob2bool(abs(tanh(move_y_drive)), &rng_state)) {
        dy = select(1, -1, move_y_drive < 0.0);
    }
    if (dx != 0 || dy != 0) {
        let dest = loc + vec2<i32>(dx, dy);
        try_enqueue_move(agent_id, loc, dest);
    }

    // Age and stash rng back so subsequent steps stay independent.
    agents[agent_id].age = agents[agent_id].age + 1u;
    agents[agent_id].rng_state = rng_state;
}

// ── Drain death queue (one thread per entry) ───────────────────────────────

@compute @workgroup_size(64)
fn drain_deaths(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = atomicLoad(&death_queue_count);
    if (i >= count) { return; }
    let agent_id = death_queue[i];
    if (agents[agent_id].alive == 0u) { return; }
    agents[agent_id].alive = 0u;
    let loc = agents[agent_id].loc;
    atomicStore(&grid[grid_index(loc)], EMPTY);
}

// ── Drain move queue (atomic CAS into grid) ────────────────────────────────

@compute @workgroup_size(64)
fn drain_moves(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = atomicLoad(&move_queue_count);
    if (i >= count) { return; }
    let req = move_queue[i];
    if (agents[req.agent_id].alive == 0u) { return; }

    let target_idx = grid_index(req.dest);
    let cell = atomicLoad(&grid[target_idx]);
    // Kill barrier: agent dies on contact. The kill barrier cell stays put
    // so subsequent agents also die. Old cell is freed.
    if (cell == KILL_BARRIER) {
        agents[req.agent_id].alive = 0u;
        atomicStore(&grid[grid_index(req.old)], EMPTY);
        return;
    }

    let result = atomicCompareExchangeWeak(&grid[target_idx], EMPTY, req.agent_id);
    if (result.exchanged) {
        // Won the race. Update agent loc + last_move + heading, free old cell.
        let old = req.old;
        let new_loc = req.dest;
        agents[req.agent_id].loc = new_loc;
        let delta = new_loc - old;
        agents[req.agent_id].last_move = delta;
        // Pick a heading that best matches the move (8-way).
        var best_h: u32 = agents[req.agent_id].heading;
        var best_dot: i32 = -999;
        for (var h: u32 = 0u; h < 8u; h = h + 1u) {
            let s = heading_to_step(h);
            let dot = s.x * delta.x + s.y * delta.y;
            if (dot > best_dot) { best_dot = dot; best_h = h; }
        }
        agents[req.agent_id].heading = best_h;
        atomicStore(&grid[grid_index(old)], EMPTY);
    }
}

// ── Signal fade (one thread per cell across all layers) ───────────────────

@compute @workgroup_size(64)
fn signal_fade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.size_x * params.size_y * params.signal_layers;
    if (i >= total) { return; }
    loop {
        let prev = atomicLoad(&signals[i]);
        if (prev == 0u) { break; }
        let new_v = prev - 1u;
        let r = atomicCompareExchangeWeak(&signals[i], prev, new_v);
        if (r.exchanged) { break; }
    }
}

// ── Clear scratch (action_levels + queue counters) ─────────────────────────

@compute @workgroup_size(64)
fn clear_step_scratch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.num_population * params.action_count;
    if (i < total) {
        action_levels[i] = 0.0;
    }
    if (i == 0u) {
        atomicStore(&move_queue_count, 0u);
        atomicStore(&death_queue_count, 0u);
    }
}
