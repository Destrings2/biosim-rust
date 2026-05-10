// Mirrors the Rust DTOs serialized through serde-wasm-bindgen as plain
// JS objects. Keep in sync with `biosim4-wasm/src/lib.rs` and
// `biosim4-core/src/agent.rs`.

export interface SimStats {
  generation: number;
  sim_step: number;
  steps_per_generation: number;
  alive_count: number;
  population: number;
  sensor_count: number;
  action_count: number;
  challenge_count: number;
}

export interface EpochResult {
  generation: number;          // generation that just ended
  next_generation: number;     // newly spawned generation
  survivors: number;
  population: number;
  diversity: number;
  survival_rate: number;
}

export interface AgentSnapshot {
  id: number;
  x: number;
  y: number;
  heading: number;             // Compass enum ordinal
  color: [number, number, number];
  age: number;
  alive: boolean;
  breed_id: number;
  responsiveness: number;
  genome_length: number;
}

export interface RegistryEntry {
  index: number;
  id: string;
  name: string;
}

export interface ChallengeSchema {
  id: string;
  name: string;
  description: string;
  schema: Record<string, unknown>;
}

export interface SimConfig {
  size_x: number;
  size_y: number;
  population: number;
  steps_per_generation: number;
  rng_seed: number;
  kill_enable: boolean;
  barrier_type: number;
  point_mutation_rate: number;
  max_number_neurons: number;
  // ...plus other SimConfig fields. Treat as `any` to avoid drift.
  [k: string]: unknown;
}

export interface NetNode {
  kind: "sensor" | "neuron" | "action";
  index: number;
  label: string;
}

export interface NetEdge {
  from_kind: "sensor" | "neuron";
  from_index: number;
  to_kind: "action" | "neuron";
  to_index: number;
  weight: number;
}

export interface NeuronState {
  index: number;
  output: number;
  driven: boolean;
}

export interface NetworkSnapshot {
  id: number;
  color: [number, number, number];
  age: number;
  genome_length: number;
  responsiveness: number;
  nodes: NetNode[];
  edges: NetEdge[];
  neuron_states: NeuronState[];
}

export type ChallengeComposition = "Any" | "All" | { WeightedSum: { weights: number[]; threshold: number } };

export interface ChallengeConfig {
  active: string[];
  composition: ChallengeComposition;
  params: Record<string, Record<string, unknown>>;
}
