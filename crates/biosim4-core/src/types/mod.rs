//! Spatial primitives: [`Coord`], [`Dir`], [`Polar`].
//!
//! All coordinates use `i16` — the simulation grid is integer-only.
//! Floating-point is used only for distance comparisons and normalization
//! within individual operations, never for persistent state.

pub mod coord;
pub mod dir;

pub use coord::{Coord, Polar};
pub use dir::{Compass, Dir};
