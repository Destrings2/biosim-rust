//! Neural network compilation and feed-forward evaluation.
//!
//! # Network representation
//!
//! A [`NeuralNet`] is compiled from a [`Genome`](super::ops::Genome) by
//! [`create_wiring`]. It holds a flat list of resolved [`Gene`] connections
//! and a list of internal [`Neuron`] states. Connections are sorted:
//! neuron→neuron first, neuron/sensor→action last. This ordering is required
//! by [`feed_forward`]'s single-pass tanh strategy (see below).
//!
//! # `create_wiring` algorithm (6 steps)
//!
//! 1. **Remap indices** — raw gene source/sink numbers are taken modulo
//!    `sensor_count`, `action_count`, or `max_neurons` so the genome is valid
//!    regardless of the current registry configuration.
//!
//! 2. **Build node map** — for each unique neuron index, count its total
//!    outputs, self-inputs, and other-inputs.
//!
//! 3. **Iterative cull** — neurons with `num_outputs == 0` are removed.
//!    When a neuron is removed, the `num_outputs` of neurons that fed into it
//!    are decremented. The loop repeats until stable. This eliminates hidden
//!    neurons that produce no effect on any action.
//!
//! 4. **Assign sequential indices** — surviving neurons receive compact
//!    indices 0..N (sorted by original ID for determinism).
//!
//! 5. **Build connection list** — connections referencing culled neurons are
//!    dropped. Remaining connections are split into neuron→neuron and →action
//!    lists and concatenated in that order.
//!
//! 6. **Build neuron list** — each surviving neuron initializes `output = 0.5`
//!    and `driven = true` if it receives any non-self input.
//!
//! # `feed_forward` ordering invariant
//!
//! The two connection lists are walked in sequence: first every
//! neuron→neuron connection accumulates into a per-neuron sum; then `tanh()`
//! is applied to each driven neuron's sum; then every →action connection is
//! evaluated against the now-finalised neuron outputs. Un-driven neurons skip
//! `tanh` and keep their persistent `output` (initialised to `0.5`, never
//! updated), contributing a constant bias to any action they connect to.
//!
//! Applying `tanh` neuron-by-neuron during the first walk would produce
//! incorrect results — each neuron's accumulator must be fully summed before
//! clamping.

use std::collections::HashMap;
use crate::genome::gene::Gene;

/// Parameters needed to wire a genome into a NeuralNet.
#[derive(Clone, Copy, Debug)]
pub struct WiringConfig {
    pub sensor_count: u16,
    pub action_count: u16,
    pub max_neurons: u16,
}

#[derive(Clone, Debug, Default)]
pub struct Neuron {
    /// Current activation, persisted across simulation steps.
    pub output: f32,
    /// True if this neuron receives at least one non-self input.
    pub driven: bool,
}

#[derive(Clone, Debug, Default)]
pub struct NeuralNet {
    /// Connections whose sink is an internal neuron. Walked first by
    /// `feed_forward` to accumulate neuron inputs.
    pub neuron_connections: Vec<Gene>,
    /// Connections whose sink is an action. Walked after the neuron tanh
    /// pass so they see finalised neuron outputs.
    pub action_connections: Vec<Gene>,
    /// Active internal neurons, indexed sequentially 0..N.
    pub neurons: Vec<Neuron>,
}

impl NeuralNet {
    /// Total connection count — sum of both lists. Used by inspectors.
    pub fn connection_count(&self) -> usize {
        self.neuron_connections.len() + self.action_connections.len()
    }

    /// Iterate every connection in `feed_forward` order (neurons first,
    /// then actions). Used by inspectors and graph renderers.
    pub fn all_connections(&self) -> impl Iterator<Item = &Gene> {
        self.neuron_connections.iter().chain(self.action_connections.iter())
    }
}

/// Compile a genome into a NeuralNet.
pub fn create_wiring(genome: &[Gene], cfg: WiringConfig) -> NeuralNet {
    if genome.is_empty() || cfg.sensor_count == 0 || cfg.action_count == 0 {
        return NeuralNet::default();
    }
    let max_n = cfg.max_neurons as u8;

    // Step 1: remap raw indices via modulo
    let remapped: Vec<Gene> = genome.iter().map(|g| {
        let sn = if g.is_sensor_source() {
            g.source_num() % cfg.sensor_count as u8
        } else {
            g.source_num() % max_n
        };
        let sk = if g.is_action_sink() {
            g.sink_num() % cfg.action_count as u8
        } else {
            g.sink_num() % max_n
        };
        Gene::new(g.source_type(), sn, g.sink_type(), sk, g.weight_raw())
    }).collect();

    // Step 2: build node map — count inputs/outputs per neuron index
    #[derive(Default)]
    struct NodeInfo {
        num_outputs: u32,
        num_self_inputs: u32,
        num_other_inputs: u32,
        remapped_idx: u16, // assigned after culling
    }
    let mut nodes: HashMap<u8, NodeInfo> = HashMap::new();

    for g in &remapped {
        if !g.is_sensor_source() {
            // neuron as source
            let e = nodes.entry(g.source_num()).or_default();
            if !g.is_action_sink() && g.source_num() == g.sink_num() {
                e.num_self_inputs += 1;
            } else {
                e.num_outputs += 1;
            }
        }
        if !g.is_action_sink() {
            // neuron as sink
            let e = nodes.entry(g.sink_num()).or_default();
            if !g.is_sensor_source() && g.source_num() == g.sink_num() {
                // already counted above
            } else {
                e.num_other_inputs += 1;
            }
        }
    }

    // Step 3: iteratively cull useless neurons.
    // A neuron is useless if it has no outputs (nothing it connects to that survives).
    // When we remove a neuron, we decrement the output count of neurons that fed into it.
    let mut changed = true;
    while changed {
        changed = false;
        let culled: Vec<u8> = nodes.iter()
            .filter(|(_, n)| n.num_outputs == 0)
            .map(|(&k, _)| k)
            .collect();
        for culled_id in culled {
            nodes.remove(&culled_id);
            changed = true;
            // Decrement num_outputs for every neuron that was feeding into culled_id
            for g in &remapped {
                if !g.is_action_sink() && g.sink_num() == culled_id && !g.is_sensor_source() {
                    let src = g.source_num();
                    if src != culled_id {
                        if let Some(src_node) = nodes.get_mut(&src) {
                            src_node.num_outputs = src_node.num_outputs.saturating_sub(1);
                        }
                    }
                }
            }
        }
    }

    // Step 4: assign sequential indices to surviving neurons
    let mut sorted_ids: Vec<u8> = nodes.keys().copied().collect();
    sorted_ids.sort_unstable();
    for (new_idx, &old_id) in sorted_ids.iter().enumerate() {
        nodes.get_mut(&old_id).unwrap().remapped_idx = new_idx as u16;
    }
    let neuron_count = sorted_ids.len();

    // Step 5: build connection list — neuron→neuron first, then →action
    let mut neuron_to_neuron: Vec<Gene> = Vec::new();
    let mut to_action: Vec<Gene> = Vec::new();

    for g in &remapped {
        let src_valid = if g.is_sensor_source() {
            true // sensors always valid
        } else {
            nodes.contains_key(&g.source_num())
        };
        let sink_valid = if g.is_action_sink() {
            true
        } else {
            nodes.contains_key(&g.sink_num())
        };
        if !src_valid || !sink_valid { continue; }

        let new_src = if g.is_sensor_source() {
            g.source_num()
        } else {
            nodes[&g.source_num()].remapped_idx as u8
        };
        let new_sink = if g.is_action_sink() {
            g.sink_num()
        } else {
            nodes[&g.sink_num()].remapped_idx as u8
        };
        let wired = Gene::new(g.source_type(), new_src, g.sink_type(), new_sink, g.weight_raw());
        if g.is_action_sink() {
            to_action.push(wired);
        } else {
            neuron_to_neuron.push(wired);
        }
    }

    // Step 6: build neuron list
    let neurons = (0..neuron_count).map(|i| {
        let old_id = sorted_ids[i];
        let info = &nodes[&old_id];
        Neuron {
            output: 0.5,
            driven: info.num_other_inputs > 0 || info.num_self_inputs > 0,
        }
    }).collect();

    NeuralNet {
        neuron_connections: neuron_to_neuron,
        action_connections: to_action,
        neurons,
    }
}

/// Allocating wrapper around [`feed_forward`]. Convenient for tests and
/// one-off use; the hot path should use [`feed_forward`] with reused scratch.
pub fn feed_forward_alloc(
    nnet: &mut NeuralNet,
    action_count: u16,
    get_sensor: impl FnMut(u16) -> f32,
) -> Vec<f32> {
    let mut action_accum: Vec<f32> = Vec::new();
    let mut neuron_accum: Vec<f32> = Vec::new();
    feed_forward(nnet, action_count, &mut action_accum, &mut neuron_accum, get_sensor);
    action_accum
}

/// Run one feedforward pass, writing action levels into `action_accum` and
/// using `neuron_accum` as scratch. Both buffers are resized to fit and
/// zero-cleared in-place — pass the same buffers across calls to avoid
/// per-call heap allocations.
pub fn feed_forward(
    nnet: &mut NeuralNet,
    action_count: u16,
    action_accum: &mut Vec<f32>,
    neuron_accum: &mut Vec<f32>,
    mut get_sensor: impl FnMut(u16) -> f32,
) {
    action_accum.clear();
    action_accum.resize(action_count as usize, 0.0);
    neuron_accum.clear();
    neuron_accum.resize(nnet.neurons.len(), 0.0);

    // Phase A: accumulate inputs into neuron sums. Sink is always a neuron,
    // so no per-conn branch on sink type.
    for conn in &nnet.neuron_connections {
        let src_val = if conn.is_sensor_source() {
            get_sensor(conn.source_num() as u16)
        } else {
            nnet.neurons[conn.source_num() as usize].output
        };
        neuron_accum[conn.sink_num() as usize] += conn.weight_as_float() * src_val;
    }

    // Phase B: clamp each driven neuron's accumulator to its output. Un-driven
    // neurons keep their persistent `output` (constant bias).
    for (i, neuron) in nnet.neurons.iter_mut().enumerate() {
        if neuron.driven {
            neuron.output = neuron_accum[i].tanh();
        }
    }

    // Phase C: drive actions from finalised sensor/neuron outputs. Sink is
    // always an action.
    for conn in &nnet.action_connections {
        let src_val = if conn.is_sensor_source() {
            get_sensor(conn.source_num() as u16)
        } else {
            nnet.neurons[conn.source_num() as usize].output
        };
        action_accum[conn.sink_num() as usize] += conn.weight_as_float() * src_val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gene(src_t: u8, src_n: u8, sink_t: u8, sink_n: u8, w: i16) -> Gene {
        Gene::new(src_t, src_n, sink_t, sink_n, w)
    }

    #[test]
    fn single_sensor_to_action() {
        let cfg = WiringConfig { sensor_count: 1, action_count: 1, max_neurons: 2 };
        // sensor 0 → action 0, weight = +4096 (≈ 0.5 float)
        let genome = vec![make_gene(1, 0, 1, 0, 4096)];
        let mut nnet = create_wiring(&genome, cfg);
        let mut levels = Vec::new();
        let mut scratch = Vec::new();
        feed_forward(&mut nnet, 1, &mut levels, &mut scratch, |_| 1.0);
        assert!(levels[0].abs() > 0.0, "expected non-zero action level");
    }

    #[test]
    fn neuron_culled_when_no_output() {
        let cfg = WiringConfig { sensor_count: 1, action_count: 1, max_neurons: 4 };
        // neuron 0 → neuron 1 (no action output from either) — both should be culled
        let genome = vec![
            make_gene(0, 0, 0, 1, 1000), // neuron 0 → neuron 1
        ];
        let nnet = create_wiring(&genome, cfg);
        assert_eq!(nnet.connection_count(), 0);
        assert_eq!(nnet.neurons.len(), 0);
    }
}
