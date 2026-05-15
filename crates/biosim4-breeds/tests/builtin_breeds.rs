//! Smoke tests for the built-in breeds.
//!
//! The breeds reference sensor / action ids by string, so a typo only shows
//! up at apply-time. These tests run every built-in breed through `apply()`
//! against a freshly-registered registry pair — if any id has drifted, the
//! apply errors and the test fails with the offending name.

use biosim4_breeds::register_builtin_breeds;
use biosim4_core::registry::{ActionRegistry, BreedRegistry, ChallengeRegistry, SensorRegistry};

fn registries() -> (SensorRegistry, ActionRegistry, ChallengeRegistry, BreedRegistry) {
    let mut sensors = SensorRegistry::new();
    let mut actions = ActionRegistry::new();
    let mut challenges = ChallengeRegistry::new();
    let mut breeds = BreedRegistry::new();
    biosim4_sensors::register_builtin_sensors(&mut sensors);
    biosim4_actions::register_builtin_actions(&mut actions);
    biosim4_challenges::register_builtin_challenges(&mut challenges);
    register_builtin_breeds(&mut breeds);
    (sensors, actions, challenges, breeds)
}

#[test]
fn every_builtin_breed_applies_cleanly() {
    let (sensors_proto, actions_proto, challenges_proto, breeds) = registries();
    assert!(breeds.count() >= 5, "expected at least 5 built-in breeds");
    for breed in breeds.list() {
        // Fresh registries each iteration so an earlier breed can't mask a
        // later one's bad reference.
        let mut s = SensorRegistry::new();
        let mut a = ActionRegistry::new();
        let mut c = ChallengeRegistry::new();
        biosim4_sensors::register_builtin_sensors(&mut s);
        biosim4_actions::register_builtin_actions(&mut a);
        biosim4_challenges::register_builtin_challenges(&mut c);
        if let Err(e) = breed.apply(&mut s, &mut a, &mut c) {
            panic!("built-in breed `{}` failed to apply: {e}", breed.id);
        }
    }
    // Sanity: prove the "prototype" registries still have all builtins so a
    // future refactor of `registries()` doesn't silently neuter the test.
    assert!(sensors_proto.count() >= 30);
    assert!(actions_proto.count() >= 15);
    assert!(!challenges_proto.schema_list().as_array().unwrap().is_empty());
}

#[test]
fn default_breed_mirrors_launch_state() {
    // The default `SimConfig` has `enable_energy = false` and
    // `signal_layers = 1`, so `apply_feature_enables` turns off these
    // ids at every generation boundary. The `default` breed must match
    // that mask — otherwise applying it would silently re-enable sensors
    // the underlying config can't actually feed.
    const GATED_SENSORS: &[&str] = &[
        "signal1",
        "signal1_fwd",
        "signal1_lr",
        "signal2",
        "signal2_fwd",
        "signal2_lr",
        "energy_level",
        "food_here",
        "food_fwd",
        "food_lr",
    ];
    const GATED_ACTIONS: &[&str] = &["emit_signal1", "emit_signal2"];

    let (_, _, _, breeds) = registries();
    let breed = breeds.get("default").expect("default registered");
    for id in GATED_SENSORS {
        assert!(
            !breed.sensors.iter().any(|s| s == id),
            "default breed must NOT include feature-gated sensor `{id}`",
        );
    }
    for id in GATED_ACTIONS {
        assert!(
            !breed.actions.iter().any(|a| a == id),
            "default breed must NOT include feature-gated action `{id}`",
        );
    }

    let mut s = SensorRegistry::new();
    let mut a = ActionRegistry::new();
    let mut c = ChallengeRegistry::new();
    biosim4_sensors::register_builtin_sensors(&mut s);
    biosim4_actions::register_builtin_actions(&mut a);
    biosim4_challenges::register_builtin_challenges(&mut c);
    breeds.apply("default", &mut s, &mut a, &mut c).expect("default applies");

    for id in GATED_SENSORS {
        assert!(!s.is_enabled(id), "sensor `{id}` should be disabled (pending) after default");
    }
    for id in GATED_ACTIONS {
        assert!(!a.is_enabled(id), "action `{id}` should be disabled (pending) after default");
    }

    // Simulate the generation-boundary commit so we can check `enabled_count`
    // (which only reflects committed state). The runtime path that runs this
    // commit lives in `spawn::spawn_new_generation`.
    s.commit_enabled();
    a.commit_enabled();
    let total_sensors = s.count() as usize;
    let total_actions = a.count() as usize;
    assert_eq!(s.enabled_count() as usize, total_sensors - GATED_SENSORS.len());
    assert_eq!(a.enabled_count() as usize, total_actions - GATED_ACTIONS.len());
}

#[test]
fn default_breed_is_first_in_registration_order() {
    // The dropdown lands on the first-registered breed, so `default` MUST
    // be first if we want a freshly-launched UI to point at the actual
    // runtime baseline.
    let (_, _, _, breeds) = registries();
    let first = breeds.list().first().expect("at least one breed");
    assert_eq!(first.id, "default", "expected `default` first, got `{}`", first.id);
}

#[test]
fn applying_minimal_strictly_narrows_the_sensor_set() {
    let (_, _, _, breeds) = registries();
    let mut s = SensorRegistry::new();
    let mut a = ActionRegistry::new();
    let mut c = ChallengeRegistry::new();
    biosim4_sensors::register_builtin_sensors(&mut s);
    biosim4_actions::register_builtin_actions(&mut a);
    biosim4_challenges::register_builtin_challenges(&mut c);

    let total = s.count();
    breeds.apply("minimal", &mut s, &mut a, &mut c).expect("minimal applies");
    // `enabled_count` only changes at commit (gen boundary). Pre-commit, the
    // change is reflected in the disabled set via `is_enabled`.
    let pending_enabled = (0..total).filter(|i| s.is_enabled(s.id(*i))).count();
    assert!(pending_enabled < total as usize, "minimal should disable some sensors (pending)");
    assert!(pending_enabled >= 3, "minimal should still leave a handful enabled (pending)");
    s.commit_enabled();
    assert_eq!(
        s.enabled_count() as usize,
        pending_enabled,
        "post-commit enabled_count should match the pending picture",
    );
}

#[test]
fn unknown_breed_returns_error() {
    let (_, _, _, breeds) = registries();
    let mut s = SensorRegistry::new();
    let mut a = ActionRegistry::new();
    let mut c = ChallengeRegistry::new();
    biosim4_sensors::register_builtin_sensors(&mut s);
    biosim4_actions::register_builtin_actions(&mut a);
    biosim4_challenges::register_builtin_challenges(&mut c);

    let err = breeds.apply("does_not_exist", &mut s, &mut a, &mut c).unwrap_err();
    assert!(err.contains("does_not_exist"), "error should name the missing breed: {err}");
}
