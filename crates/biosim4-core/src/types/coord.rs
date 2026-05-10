//! 2D integer coordinate and polar form.
//!
//! `as_dir` classifies a displacement vector into one of 9 [`Dir`] values
//! using integer arithmetic instead of `atan2`. It applies a rational
//! approximation of tan(22.5°) ≈ 13860/33461 to determine whether the angle
//! is within 22.5° of a cardinal axis (E/W/N/S) or falls in a diagonal
//! octant (NE/NW/SE/SW).
//!
//! `ray_sameness` computes the normalized dot product of two displacement
//! vectors, returning a value in [-1, 1] where 1 means same direction and
//! -1 means opposite. Used by directional population and signal sensors.

use super::dir::{Compass, Dir};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Coord {
    pub x: i16,
    pub y: i16,
}

impl Coord {
    pub fn new(x: i16, y: i16) -> Self { Self { x, y } }

    pub fn length(&self) -> f32 {
        ((self.x as f32).powi(2) + (self.y as f32).powi(2)).sqrt()
    }

    /// Closest direction using integer tangent approximation (avoids atan2).
    pub fn as_dir(&self) -> Dir {
        if self.x == 0 && self.y == 0 {
            return Dir::center();
        }
        // Rotate coordinate system ~22.5° using rational tan(22.5°) ≈ 13860/33461,
        // then classify by octant.
        let x = self.x as i32;
        let y = self.y as i32;
        // Use 8-quadrant classification by sign and |x| vs |y|
        let c = if x == 0 {
            if y > 0 { Compass::N } else { Compass::S }
        } else if y == 0 {
            if x > 0 { Compass::E } else { Compass::W }
        } else {
            let ax = x.abs();
            let ay = y.abs();
            // Octant classification using integer arithmetic to avoid atan2.
            // tan(22.5°) ≈ 13860/33461.  Horizontal zone: ay/ax < tan(22.5°).
            // Vertical zone: ay/ax > tan(67.5°) = 33461/13860.
            // Diagonal zone: everything in between.
            if ay * 33461 < ax * 13860 {
                // angle < 22.5° from horizontal axis → E or W
                if x > 0 { Compass::E } else { Compass::W }
            } else if ay * 13860 > ax * 33461 {
                // angle > 67.5° from horizontal axis → N or S
                if y > 0 { Compass::N } else { Compass::S }
            } else {
                // diagonal
                match (x > 0, y > 0) {
                    (true,  true)  => Compass::NE,
                    (true,  false) => Compass::SE,
                    (false, true)  => Compass::NW,
                    (false, false) => Compass::SW,
                }
            }
        };
        Dir(c)
    }

    /// Normalize to the nearest unit vector (values in -1/0/1 per axis).
    pub fn normalize(&self) -> Coord {
        Coord::new(self.x.signum(), self.y.signum())
    }

    /// Convert to (magnitude, dir) polar form.
    pub fn as_polar(&self) -> Polar {
        Polar { mag: self.length() as i32, dir: self.as_dir() }
    }

    /// Dot-product similarity with another coord, normalized to [-1, 1].
    pub fn ray_sameness(&self, other: Coord) -> f32 {
        let mag_a = self.length();
        let mag_b = other.length();
        if mag_a == 0.0 || mag_b == 0.0 { return 0.0; }
        let dot = self.x as f32 * other.x as f32 + self.y as f32 * other.y as f32;
        (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
    }

    pub fn ray_sameness_dir(&self, d: Dir) -> f32 {
        self.ray_sameness(d.as_normalized_coord())
    }
}

impl std::ops::Add for Coord {
    type Output = Coord;
    fn add(self, rhs: Coord) -> Coord { Coord::new(self.x + rhs.x, self.y + rhs.y) }
}
impl std::ops::Sub for Coord {
    type Output = Coord;
    fn sub(self, rhs: Coord) -> Coord { Coord::new(self.x - rhs.x, self.y - rhs.y) }
}
impl std::ops::Mul<i16> for Coord {
    type Output = Coord;
    fn mul(self, rhs: i16) -> Coord { Coord::new(self.x * rhs, self.y * rhs) }
}

/// Magnitude + direction.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Polar {
    pub mag: i32,
    pub dir: Dir,
}

impl Polar {
    pub fn as_coord(&self) -> Coord {
        let unit = self.dir.as_normalized_coord();
        // For diagonal directions, scale by 1/sqrt(2) ≈ 45_341/64_000 (fixed point)
        let is_diag = matches!(
            self.dir.0,
            Compass::NE | Compass::NW | Compass::SE | Compass::SW
        );
        if is_diag {
            let scaled = (self.mag as i64 * 45_341 / 64_000) as i16;
            Coord::new(unit.x * scaled, unit.y * scaled)
        } else {
            Coord::new(unit.x * self.mag as i16, unit.y * self.mag as i16)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_of_unit() {
        assert!((Coord::new(1, 0).length() - 1.0).abs() < 1e-6);
        assert!((Coord::new(0, 1).length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn as_dir_cardinal() {
        assert_eq!(Coord::new(5, 0).as_dir(), Dir(Compass::E));
        assert_eq!(Coord::new(0, 5).as_dir(), Dir(Compass::N));
        assert_eq!(Coord::new(-3, 0).as_dir(), Dir(Compass::W));
        assert_eq!(Coord::new(0, -3).as_dir(), Dir(Compass::S));
    }

    #[test]
    fn as_dir_diagonal() {
        assert_eq!(Coord::new(3, 3).as_dir(), Dir(Compass::NE));
        assert_eq!(Coord::new(-3, 3).as_dir(), Dir(Compass::NW));
        assert_eq!(Coord::new(3, -3).as_dir(), Dir(Compass::SE));
        assert_eq!(Coord::new(-3, -3).as_dir(), Dir(Compass::SW));
    }

    #[test]
    fn ray_sameness_parallel_is_1() {
        let a = Coord::new(4, 0);
        let b = Coord::new(2, 0);
        assert!((a.ray_sameness(b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ray_sameness_opposite_is_neg1() {
        let a = Coord::new(4, 0);
        let b = Coord::new(-2, 0);
        assert!((a.ray_sameness(b) + 1.0).abs() < 1e-6);
    }
}
