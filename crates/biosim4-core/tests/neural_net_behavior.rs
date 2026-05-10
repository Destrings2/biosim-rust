//! Neural net wiring and feed-forward behavior. The C++ feedForward.cpp has
//! subtle invariants (neuron→neuron connections sorted before action sinks,
//! tanh applied once per step, undriven neurons keep previous output) that
//! are easy to break in a port.

use biosim4_core::genome::{
    gene::Gene,
    neural_net::{create_wiring, feed_forward_alloc, WiringConfig},
};

fn make_gene(src_type: u8, src_num: u8, sink_type: u8, sink_num: u8, weight: i16) -> Gene {
    Gene::new(src_type, src_num, sink_type, sink_num, weight)
}

#[test]
fn empty_genome_has_no_connections() {
    let cfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: 5 };
    let nnet = create_wiring(&[], cfg);
    assert_eq!(nnet.connections.len(), 0);
    assert_eq!(nnet.neurons.len(), 0);
}

#[test]
fn sensor_to_action_connection_survives_culling() {
    let cfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: 5 };
    let g = vec![make_gene(1, 0, 1, 0, 4096)];  // sensor 0 → action 0
    let nnet = create_wiring(&g, cfg);
    assert_eq!(nnet.connections.len(), 1, "direct sensor→action should survive");
    assert_eq!(nnet.neurons.len(), 0, "no neurons needed");
}

#[test]
fn dangling_neuron_chain_is_culled() {
    // Sensor 0 → neuron 0 → neuron 1 → (nothing). All must be culled.
    let cfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: 5 };
    let g = vec![
        make_gene(1, 0, 0, 0, 4096),  // sensor 0 → neuron 0
        make_gene(0, 0, 0, 1, 4096),  // neuron 0 → neuron 1
    ];
    let nnet = create_wiring(&g, cfg);
    assert_eq!(nnet.connections.len(), 0, "dangling chain should be fully culled");
    assert_eq!(nnet.neurons.len(), 0);
}

#[test]
fn neuron_with_action_output_survives() {
    // Sensor 0 → neuron 0 → action 0
    let cfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: 5 };
    let g = vec![
        make_gene(1, 0, 0, 0, 4096),  // sensor 0 → neuron 0
        make_gene(0, 0, 1, 0, 4096),  // neuron 0 → action 0
    ];
    let nnet = create_wiring(&g, cfg);
    assert_eq!(nnet.connections.len(), 2);
    assert_eq!(nnet.neurons.len(), 1);
}

#[test]
fn connections_ordered_neuron_to_neuron_before_action_sinks() {
    // The feed_forward function depends on this ordering for the tanh-application
    // boundary. If a neuron→action connection appears before a neuron→neuron, the
    // neuron output gets latched too early.
    let cfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: 5 };
    let g = vec![
        make_gene(0, 0, 1, 0, 4096),  // neuron 0 → action 0
        make_gene(1, 0, 0, 0, 4096),  // sensor 0 → neuron 0
        make_gene(0, 0, 0, 1, 4096),  // neuron 0 → neuron 1
        make_gene(0, 1, 1, 0, 4096),  // neuron 1 → action 0
    ];
    let nnet = create_wiring(&g, cfg);
    // First half should be all neuron-sink, second half should be all action-sink
    let mut seen_action = false;
    for c in &nnet.connections {
        if c.is_action_sink() {
            seen_action = true;
        } else if seen_action {
            panic!("found neuron-sink connection {:?} after action-sink", c);
        }
    }
}

#[test]
fn feed_forward_with_constant_sensor_produces_finite_actions() {
    let cfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: 5 };
    let g = vec![
        make_gene(1, 0, 0, 0, 4096),  // sensor 0 → neuron 0
        make_gene(0, 0, 1, 0, 4096),  // neuron 0 → action 0
        make_gene(1, 1, 1, 1, -3000), // sensor 1 → action 1 (negative weight)
    ];
    let mut nnet = create_wiring(&g, cfg);
    let levels = feed_forward_alloc(&mut nnet, 17, |idx| {
        if idx == 0 { 0.7 } else if idx == 1 { 0.3 } else { 0.5 }
    });
    assert_eq!(levels.len(), 17);
    for (i, l) in levels.iter().enumerate() {
        assert!(l.is_finite(), "action {} produced non-finite level: {}", i, l);
    }
    assert!(levels[0].abs() > 0.0, "action 0 should receive non-zero signal");
    assert!(levels[1] != 0.0, "action 1 should receive non-zero signal");
}

#[test]
fn feed_forward_inactive_actions_are_zero() {
    // Only action 5 has a connection; the rest must be exactly 0.
    let cfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: 5 };
    let g = vec![make_gene(1, 0, 1, 5, 4096)];
    let mut nnet = create_wiring(&g, cfg);
    let levels = feed_forward_alloc(&mut nnet, 17, |_| 1.0);
    for (i, l) in levels.iter().enumerate() {
        if i == 5 {
            assert_ne!(*l, 0.0);
        } else {
            assert_eq!(*l, 0.0, "action {} should be 0, got {}", i, l);
        }
    }
}

#[test]
fn create_wiring_respects_max_neurons() {
    // With max_neurons=2, even if the genome references 7-bit neuron indices
    // (0..127), the wiring must collapse them to ≤2 distinct neurons.
    let cfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: 2 };
    let g = vec![
        make_gene(1, 0, 0, 0, 4096),
        make_gene(1, 0, 0, 5, 4096),
        make_gene(1, 0, 0, 9, 4096),
        make_gene(0, 0, 1, 0, 4096),
        make_gene(0, 5, 1, 1, 4096),
        make_gene(0, 9, 1, 2, 4096),
    ];
    let nnet = create_wiring(&g, cfg);
    assert!(nnet.neurons.len() <= 2, "neurons exceed max: {}", nnet.neurons.len());
}

// ── Recurrent neuron latching (cross-step) ────────────────────────────────

/// A self-recurrent neuron (neuron0 → neuron0) should preserve its output
/// between `feed_forward` calls, creating a memory effect across simulation
/// steps. This is the "latching" invariant from the C++ feedForward.cpp.
///
/// Without latching the self-loop would always read the initial output (0.5)
/// and give a much weaker action signal on step 2.
#[test]
fn recurrent_neuron_output_latches_across_steps() {
    // weight 8192 / 8192.0 = 1.0
    let cfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: 5 };
    let g = vec![
        make_gene(1, 0, 0, 0, 8192),  // sensor 0 → neuron 0 (weight +1.0)
        make_gene(0, 0, 0, 0, 8192),  // neuron 0 → neuron 0 self-loop (weight +1.0)
        make_gene(0, 0, 1, 0, 8192),  // neuron 0 → action 0 (weight +1.0)
    ];
    let mut nnet = create_wiring(&g, cfg);

    assert_eq!(nnet.neurons.len(), 1, "expected exactly one internal neuron");
    assert!(nnet.neurons[0].driven,
        "neuron0 has both sensor input and self-input, so driven=true");
    // Initial output is 0.5 (set in create_wiring)
    assert!((nnet.neurons[0].output - 0.5).abs() < 1e-6);

    // ── Step 1: sensor = 1.0 ──
    // neuron_accum = 1.0*sensor + 1.0*initial_output = 1.0 + 0.5 = 1.5
    // neuron.output ← tanh(1.5) ≈ 0.9051
    let _levels1 = feed_forward_alloc(&mut nnet, 17, |_| 1.0);
    let v1 = nnet.neurons[0].output;
    assert!(v1 > 0.8,
        "after step with sensor=1.0, neuron should have large positive output, got {}", v1);

    // ── Step 2: sensor = 0.0 ──
    // With latching:    neuron_accum = 1.0*0.0 + 1.0*v1  ≈ 0.905  → action ≈ tanh(0.905) ≈ 0.72
    // Without latching: neuron_accum = 1.0*0.0 + 1.0*0.5 = 0.5    → action ≈ tanh(0.5)   ≈ 0.46
    let levels2 = feed_forward_alloc(&mut nnet, 17, |_| 0.0);
    assert!(
        levels2[0] > 0.6,
        "step-2 action (sensor=0) should reflect latched output (expect >0.6 with latch, ~0.46 without), got {}",
        levels2[0]
    );
    assert!(levels2[0].is_finite());
}

/// A neuron that receives no external inputs (sensor or other-neuron) is
/// "undriven". Its output is never updated — it acts as a constant bias of
/// 0.5 injected into every downstream action regardless of sensor values.
#[test]
fn undriven_neuron_acts_as_constant_bias() {
    // Only connection: neuron0 → action0. Nothing feeds neuron0.
    // neuron0.driven = false  →  output stays at the initial 0.5 forever.
    let cfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: 5 };
    let g = vec![
        make_gene(0, 0, 1, 0, 8192),  // neuron 0 → action 0 (weight +1.0)
    ];
    let mut nnet = create_wiring(&g, cfg);

    assert_eq!(nnet.neurons.len(), 1, "expected 1 neuron");
    assert!(!nnet.neurons[0].driven, "neuron with no inputs must be undriven");
    assert!((nnet.neurons[0].output - 0.5).abs() < 1e-6, "initial output should be 0.5");

    // Different sensor values must not change action 0
    let levels1 = feed_forward_alloc(&mut nnet, 17, |_| 0.0);
    let levels2 = feed_forward_alloc(&mut nnet, 17, |_| 1.0);

    assert!(
        (levels1[0] - 0.5).abs() < 1e-5,
        "undriven neuron contributes constant 0.5 to action, got {}", levels1[0]
    );
    assert!(
        (levels2[0] - levels1[0]).abs() < 1e-5,
        "undriven neuron output must be identical across steps: {} vs {}", levels1[0], levels2[0]
    );
    // Neuron output itself must remain 0.5 (never updated)
    assert!(
        (nnet.neurons[0].output - 0.5).abs() < 1e-6,
        "undriven neuron.output must still be 0.5, got {}", nnet.neurons[0].output
    );
}

/// A pure self-loop (neuron0 → neuron0 with no action output) must be culled
/// because it has no forward path to any action sink.
#[test]
fn pure_self_loop_neuron_is_culled() {
    let cfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: 5 };
    let g = vec![
        make_gene(0, 0, 0, 0, 8192),  // neuron 0 → neuron 0 only — no action output
    ];
    let nnet = create_wiring(&g, cfg);
    assert_eq!(nnet.connections.len(), 0,
        "self-loop with no downstream action must be fully culled");
    assert_eq!(nnet.neurons.len(), 0);
}

/// Three-step decay: sensor fires only on step 1, then goes silent.
/// The recurrent connection should produce a decaying but nonzero signal
/// for at least two subsequent silent steps.
#[test]
fn recurrent_output_decays_across_multiple_silent_steps() {
    let cfg = WiringConfig { sensor_count: 21, action_count: 17, max_neurons: 5 };
    // Self-loop weight 0.5 so the decay is gradual rather than explosive
    let g = vec![
        make_gene(1, 0, 0, 0, 8192),   // sensor 0 → neuron 0  (weight +1.0)
        make_gene(0, 0, 0, 0, 4096),   // neuron 0 → neuron 0  (weight +0.5)
        make_gene(0, 0, 1, 0, 8192),   // neuron 0 → action 0  (weight +1.0)
    ];
    let mut nnet = create_wiring(&g, cfg);

    // Prime with a single strong sensor pulse
    let _prime = feed_forward_alloc(&mut nnet, 17, |_| 1.0);

    // Two silent steps — both must still produce a nonzero action signal
    let step1 = feed_forward_alloc(&mut nnet, 17, |_| 0.0);
    let step2 = feed_forward_alloc(&mut nnet, 17, |_| 0.0);

    assert!(step1[0].abs() > 1e-3,
        "one step after sensor pulse, action should still be nonzero (got {})", step1[0]);
    assert!(step2[0].abs() > 1e-3,
        "two steps after pulse, action should still be nonzero (got {})", step2[0]);
    // Signal must decay (not amplify unboundedly)
    assert!(
        step2[0].abs() <= step1[0].abs() + 1e-3,
        "action should not grow after sensor goes silent: step1={} step2={}", step1[0], step2[0]
    );
}
