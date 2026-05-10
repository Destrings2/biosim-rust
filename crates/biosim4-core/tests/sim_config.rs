//! SimConfig serde / patch tests. The frontend ships JSON; correctness here
//! determines whether the WASM API can configure the simulator at all.

use biosim4_core::sim_config::SimConfig;

#[test]
fn default_config_serializes_to_json() {
    let cfg = SimConfig::default();
    let s = serde_json::to_string(&cfg).expect("default config should serialize");
    assert!(s.contains("size_x"));
    assert!(s.contains("population"));
    assert!(s.contains("steps_per_generation"));
}

#[test]
fn default_config_roundtrip_through_json() {
    let cfg = SimConfig::default();
    let s = serde_json::to_string(&cfg).unwrap();
    let parsed: SimConfig = serde_json::from_str(&s).unwrap();
    assert_eq!(cfg.size_x, parsed.size_x);
    assert_eq!(cfg.size_y, parsed.size_y);
    assert_eq!(cfg.population, parsed.population);
    assert_eq!(cfg.rng_seed, parsed.rng_seed);
    assert_eq!(cfg.max_number_neurons, parsed.max_number_neurons);
}

#[test]
fn from_json_with_invalid_json_returns_err() {
    let r = SimConfig::from_json("{ not valid json");
    assert!(r.is_err());
}

#[test]
fn from_json_with_missing_required_field_returns_err() {
    // Empty object is missing many required fields
    let r = SimConfig::from_json("{}");
    assert!(r.is_err(), "config without required fields should fail to parse");
}

#[test]
fn patch_json_overrides_only_specified_fields() {
    let mut cfg = SimConfig::default();
    let original_size_y = cfg.size_y;
    let original_seed = cfg.rng_seed;

    cfg.patch_json(r#"{"size_x": 256}"#).expect("patch should apply cleanly");
    assert_eq!(cfg.size_x, 256, "patched field should update");
    assert_eq!(cfg.size_y, original_size_y, "non-patched field should be preserved");
    assert_eq!(cfg.rng_seed, original_seed, "non-patched field should be preserved");
}

#[test]
fn patch_json_with_multiple_fields() {
    let mut cfg = SimConfig::default();
    cfg.patch_json(r#"{"size_x": 200, "size_y": 100, "population": 999}"#)
        .expect("multi-field patch should apply");
    assert_eq!(cfg.size_x, 200);
    assert_eq!(cfg.size_y, 100);
    assert_eq!(cfg.population, 999);
}

#[test]
fn patch_json_with_invalid_json_returns_err() {
    let mut cfg = SimConfig::default();
    let r = cfg.patch_json("{ broken");
    assert!(r.is_err());
}
