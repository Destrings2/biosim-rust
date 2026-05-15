//! Direction type and 9-value compass.
//!
//! `Compass` is a 9-variant enum: SW=0, S=1, SE=2, W=3, CENTER=4, E=5,
//! NW=6, N=7, NE=8. The ordinal layout matches the original C++ implementation
//! and is relied upon by `ROTATIONS` indexing.
//!
//! `Dir::rotate(n)` uses a pre-computed 64-entry lookup table (8 directions ×
//! 8 step offsets) instead of arithmetic, avoiding the modular arithmetic edge
//! cases that come with CENTER. Positive `n` is clockwise; 8 steps is a full
//! circle. `rotate` on `CENTER` is a no-op.
//!
//! `as_normalized_coord` returns unit offsets (-1/0/1 per axis) for each of
//! the 9 compass values.

use super::coord::Coord;
use serde::{Deserialize, Serialize};

/// Nine-value compass: the 8 cardinal and ordinal directions plus center.
///
/// The numeric layout matches the original C++ implementation and is required
/// by the rotation lookup table in [`Dir::rotate`]. Do not reorder variants.
///
/// Use [`Compass::ALL8`] to iterate the 8 non-center directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Compass {
    /// South-west (down-left).
    SW = 0,
    /// South (down).
    S = 1,
    /// South-east (down-right).
    SE = 2,
    /// West (left).
    W = 3,
    /// No direction. Used as a default heading before an agent moves.
    CENTER = 4,
    /// East (right).
    E = 5,
    /// North-west (up-left).
    NW = 6,
    /// North (up).
    N = 7,
    /// North-east (up-right).
    NE = 8,
}

impl Compass {
    /// All 8 non-center compass values. Useful for iterating every movement direction.
    pub const ALL8: [Compass; 8] = [
        Compass::SW,
        Compass::S,
        Compass::SE,
        Compass::W,
        Compass::E,
        Compass::NW,
        Compass::N,
        Compass::NE,
    ];
}

impl From<u8> for Compass {
    fn from(v: u8) -> Self {
        match v % 9 {
            0 => Compass::SW,
            1 => Compass::S,
            2 => Compass::SE,
            3 => Compass::W,
            4 => Compass::CENTER,
            5 => Compass::E,
            6 => Compass::NW,
            7 => Compass::N,
            _ => Compass::NE,
        }
    }
}

/// A simulation direction, wrapping one of 9 [`Compass`] values.
///
/// [`Dir::default()`] returns `CENTER`. Use [`Dir::rotate`] to step clockwise
/// or counter-clockwise. Use [`Dir::as_normalized_coord`] to get the unit
/// offset vector for movement or sensor probing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dir(pub Compass);

// Rotation lookup: rotations[dir * 8 + steps] gives rotated dir (8 steps = 360°, positive = CW).
// Pre-computed for the 8 non-center directions.
const ROTATIONS: [u8; 64] = [
    // SW(0) rotated 0..7 CW
    0, 3, 6, 7, 8, 5, 2, 1, // S(1)
    1, 0, 3, 6, 7, 8, 5, 2, // SE(2)
    2, 1, 0, 3, 6, 7, 8, 5, // W(3)
    3, 6, 7, 8, 5, 2, 1, 0, // E(5) — stored at index 4 (skip CENTER)
    5, 2, 1, 0, 3, 6, 7, 8, // NW(6)
    6, 7, 8, 5, 2, 1, 0, 3, // N(7)
    7, 8, 5, 2, 1, 0, 3, 6, // NE(8)
    8, 5, 2, 1, 0, 3, 6, 7,
];

/// Map from Compass ordinal (0..8) to rotation-table row index (skipping CENTER=4).
fn compass_to_row(c: Compass) -> usize {
    match c {
        Compass::SW => 0,
        Compass::S => 1,
        Compass::SE => 2,
        Compass::W => 3,
        Compass::E => 4,
        Compass::NW => 5,
        Compass::N => 6,
        Compass::NE => 7,
        Compass::CENTER => 0, // fallback; shouldn't rotate CENTER
    }
}

impl Dir {
    /// Construct a `Dir` from a [`Compass`] variant.
    pub fn new(c: Compass) -> Self {
        Dir(c)
    }

    /// Return `Dir(Compass::CENTER)`.
    pub fn center() -> Self {
        Dir(Compass::CENTER)
    }

    /// Rotate by `n` steps clockwise (negative = counter-clockwise). 8 steps = full circle.
    pub fn rotate(&self, n: i32) -> Self {
        if self.0 == Compass::CENTER {
            return *self;
        }
        let row = compass_to_row(self.0);
        let steps = ((n % 8) + 8) as usize % 8;
        Dir(Compass::from(ROTATIONS[row * 8 + steps]))
    }

    /// Rotate 90° clockwise (2 steps).
    pub fn rotate90cw(&self) -> Self {
        self.rotate(2)
    }

    /// Rotate 90° counter-clockwise (2 steps).
    pub fn rotate90ccw(&self) -> Self {
        self.rotate(-2)
    }

    /// Rotate 180° (4 steps), returning the opposite direction.
    pub fn rotate180(&self) -> Self {
        self.rotate(4)
    }

    /// Unit offset vector for this direction. Each component is -1, 0, or 1.
    pub fn as_normalized_coord(&self) -> Coord {
        match self.0 {
            Compass::SW => Coord::new(-1, -1),
            Compass::S => Coord::new(0, -1),
            Compass::SE => Coord::new(1, -1),
            Compass::W => Coord::new(-1, 0),
            Compass::CENTER => Coord::new(0, 0),
            Compass::E => Coord::new(1, 0),
            Compass::NW => Coord::new(-1, 1),
            Compass::N => Coord::new(0, 1),
            Compass::NE => Coord::new(1, 1),
        }
    }

    /// Choose one of the 8 non-center directions uniformly at random.
    pub fn random8(rng: &mut impl rand::Rng) -> Self {
        Dir(Compass::ALL8[rng.gen_range(0..8)])
    }
}

impl Default for Dir {
    fn default() -> Self {
        Dir(Compass::CENTER)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotate_east_cw_gives_se() {
        let e = Dir(Compass::E);
        assert_eq!(e.rotate(1), Dir(Compass::SE));
    }
    #[test]
    fn rotate_full_circle_identity() {
        for &c in &Compass::ALL8 {
            let d = Dir(c);
            assert_eq!(d.rotate(8), d);
        }
    }
    #[test]
    fn rotate_ccw_undoes_cw() {
        let n = Dir(Compass::N);
        assert_eq!(n.rotate(3).rotate(-3), n);
    }
}
