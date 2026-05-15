//! Pluggable registry system for sensors, actions, and challenges.
//!
//! Each registry holds a `Vec<Box<dyn Trait>>` and exposes an enable/disable
//! lifecycle. See [`sensor::SensorRegistry`] for the canonical description of
//! the pending/commit pattern — [`action::ActionRegistry`] mirrors it exactly.
//! [`challenge::ChallengeRegistry`] uses a different activation model
//! (explicit active list rather than enable/disable) but the same JSON
//! configuration interface.

pub mod action;
pub mod challenge;
pub mod sensor;

pub use action::{Action, ActionContext, ActionRegistry};
pub use challenge::{Challenge, ChallengeComposition, ChallengeConfig, ChallengeRegistry};
pub use sensor::{Sensor, SensorContext, SensorRegistry};
