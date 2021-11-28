use std::f32::consts::TAU;
use Compass::{Center, East, North, NorthEast, NorthWest, South, SouthEast, SouthWest, West};

//<editor-fold desc="Constants">
// Constants for converting between types
const DIR_ROTATE_RIGHT: [Compass; 9] = [West, SouthWest, South, NorthWest, Center, SouthEast, North, NorthEast, East];
const DIR_ROTATE_LEFT: [Compass; 9] = [South, SouthEast, East, SouthWest, Center, NorthEast, West, NorthWest, North];
const COORD_DIR_CONVERSION : [Compass; 8] = [East, NorthEast, North, NorthWest, West, SouthWest, South, SouthEast];
const TAU_SEGMENT : f32 = TAU/2.0;
const COMPASS_TO_RADIANS : [f32; 9] =
    [
        5.*TAU_SEGMENT, 6.*TAU_SEGMENT, 7.*TAU_SEGMENT, 4.*TAU_SEGMENT,
        0., 0.*TAU_SEGMENT, 3.*TAU_SEGMENT, 2.*TAU_SEGMENT, 1.*TAU_SEGMENT
    ];
//</editor-fold>

//<editor-fold desc="Compass enum">
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Compass {
    SouthWest = 0,
    South,
    SouthEast,
    West,
    Center,
    East,
    NorthWest,
    North,
    NorthEast
}
//</editor-fold>

//<editor-fold desc="Dir implementation">
/// Abstract type for 8 directions plus center.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Dir(Compass);

impl Dir {
    /// Returns a new `Dir` rotated _steps_ steps. A full rotation is 8 steps.
    ///
    /// # Arguments
    /// * `steps` - How many steps to rotate, positive values represent rotations to the right, negative values represent
    /// rotations to the left
    pub fn rotate(&self, steps: i8) -> Dir {
        let mut direction = self.0;
        let mut direction_index = direction as usize;
        let rotator = if steps < 0 {&DIR_ROTATE_LEFT} else {&DIR_ROTATE_RIGHT};
        for _ in 0..steps {
            direction = rotator[direction_index];
            direction_index = direction as usize;
        }
        return Dir(direction)
    }

    pub fn rotate90deg_cw(&self) -> Dir {
        return self.rotate(2);
    }

    pub fn rotate90deg_ccw(&self) -> Dir {
        return self.rotate(-2);
    }

    pub fn rotate180deg(&self) -> Dir {
        return self.rotate(4);
    }
}

//<editor-fold desc="Type conversions for Dir">
impl From<Coord> for Dir {
    fn from(c: Coord) -> Self {
        if c.0 == 0 && c.1 == 0 {
            return Dir(Compass::Center)
        }

        let f_x = c.0 as f32;
        let f_y = c.1 as f32;
        let mut angle = f_y.atan2(f_x);
        if angle < 0.0 {
            angle = TAU + angle;
        }

        angle += TAU / 16.0;
        if angle > TAU {
            angle -= TAU
        }

        let slice : usize = (angle / (TAU / 8.0)) as usize;
        /*
        We have to convert slice values:
            3  2  1
            4     0
            5  6  7
        into Dir8Compass value:
            6  7  8
            3  4  5
            0  1  2
        */
        return Dir(COORD_DIR_CONVERSION[slice]);
    }
}

impl From<Polar> for Dir {
    fn from(p: Polar) -> Self {
        return p.direction
    }
}
//</editor-fold>
//</editor-fold>

//<editor-fold desc="Coord implementation">
/// i16 pair, absolute location or difference of locations
///
/// # Arithmetic
/// * Coord + Dir
/// * Coord + Coord
/// * Coord + Polar
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Coord(i16, i16);

impl Coord {
    pub fn length(&self) -> f32 {
        let (f_x, f_y) = (self.0 as f32, self.1 as f32);

        return f32::sqrt(f_x*f_x + f_y*f_y)
    }

    pub fn normalize(&self) -> Coord {
        let dir : Dir = self.clone().into();
        return dir.into()
    }

    pub fn is_normalized(&self) -> bool {
        return self.0 >= -1 && self.0 <= 1 && self.1 >= -1 && self.1 <= 1;
    }

    pub fn ray_sameness(&self, other: Coord) -> f32 {
        let first_magnitude = self.length();
        let second_magnitude = self.length();
        if first_magnitude == 0.0 || second_magnitude == 0.0 {
            return 1.0;
        }

        let dot = self.0 as f32 * other.0 as f32 + self.1 as f32 * other.1 as f32;
        let cos_angle = dot / (first_magnitude * second_magnitude);
        //Assert delta of result.
        assert!(cos_angle >= -1.0001 && cos_angle <= 1.0001);
        //Clip value
        f32::min(1.0, f32::max(-1.0, cos_angle))
    }

    pub fn ray_sameness_dir(&self, other: Dir) -> f32 {
        return self.ray_sameness(other.into());
    }
}

//<editor-fold desc="Operator overload for Coord">
// Coord + Coord
impl std::ops::Add<Coord> for Coord {
    type Output = Coord;

    fn add(self, rhs: Coord) -> Coord {
        return Coord(self.0 + rhs.0, self.1 + rhs.1);
    }
}

// Coord - Coord
impl std::ops::Sub<Coord> for Coord {
    type Output = Coord;

    fn sub(self, rhs: Coord) -> Coord {
        return Coord(self.0 - rhs.0, self.1 - rhs.1);
    }
}

// Coord + Dir
impl std::ops::Add<Dir> for Coord {
    type Output = Coord;

    fn add(self, rhs: Dir) -> Coord {
        let coord_dir : Dir = rhs.into();
        return self + coord_dir
    }
}

// Coord - Dir
impl std::ops::Sub<Dir> for Coord {
    type Output = Coord;

    fn sub(self, rhs: Dir) -> Coord {
        let coord_dir : Dir = rhs.into();
        return self - coord_dir;
    }
}

// Coord * i16
impl std::ops::Mul<i16> for Coord {
    type Output = Coord;

    fn mul(self, rhs: i16) -> Coord {
        return Coord(self.0 * rhs, self.1 * rhs);
    }
}
//</editor-fold>

//<editor-fold desc="Type conversions for Coord">
impl From<Dir> for Coord {
    fn from(d: Dir) -> Self {
        let direction_index = d.0 as i8;
        let x = (direction_index % 3) - 1;
        let y = (direction_index / 3) - 1;
        return Coord(x as i16, y as i16);
    }
}

impl From<Polar> for Coord {
    fn from(p: Polar) -> Self {
        if p.direction.0 == Center {
            return Coord(0,0);
        }

        let x = (p.magnitude as f32 * f32::cos(COMPASS_TO_RADIANS[p.direction.0 as usize])) + 0.5;
        let y = (p.magnitude as f32 * f32::sin(COMPASS_TO_RADIANS[p.direction.0 as usize])) + 0.5;
        return Coord(x as i16, y as i16);
    }
}
//</editor-fold>
//</editor-fold>

//<editor-fold desc="Polar implementation">
/// Polar magnitudes are signed 32-bit integers so that they can extend across any 2D
/// area defined by the Coord class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Polar {
    pub magnitude: u32,
    pub direction: Dir
}

impl Polar {
    fn new(magnitude: u32, direction: Dir) -> Self {
        return Polar {
            magnitude,
            direction
        }
    }
}

//<editor-fold desc="Operator overload for Polar">
// Polar + Polar
impl std::ops::Add<Polar> for Polar {
    type Output = Polar;

    fn add(self, rhs: Polar) -> Polar {
        let first_coord : Coord = self.into();
        let second_coord : Coord = rhs.into();
        let result = first_coord + second_coord;
        return result.into();
    }
}

// Polar - Polar
impl std::ops::Sub<Polar> for Polar {
    type Output = Polar;

    fn sub(self, rhs: Polar) -> Polar {
        let first_coord : Coord = self.into();
        let second_coord : Coord = rhs.into();
        let result = first_coord - second_coord;
        return result.into();
    }
}

// Polar * Polar (dot product)
impl std::ops::Mul<Polar> for Polar {
    type Output = i32;

    fn mul(self, rhs: Polar) -> i32 {
        let first_coord : Coord = self.into();
        let second_coord : Coord = rhs.into();
        let result = first_coord.0 * second_coord.0 + first_coord.1 * second_coord.1;
        return result as i32;
    }
}
//</editor-fold>

//<editor-fold desc="Type conversions for Polar">
impl From<Coord> for Polar {
    fn from(c: Coord) -> Self {
        Polar {
            magnitude: c.length() as u32,
            direction: c.into()
        }
    }
}

impl From<Dir> for Polar {
    fn from(d: Dir) -> Self {
        return Polar {direction: d, magnitude: 1}
    }
}
//</editor-fold>

//</editor-fold>

//<editor-fold desc="Unit tests">
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_dir_rotation_left() {
        let dir = Dir(Compass::North);

        assert_eq!(dir.rotate(1).0, Compass::NorthEast);
        assert_eq!(dir.rotate(2).0, Compass::East);
        assert_eq!(dir.rotate(3).0, Compass::SouthEast);
        assert_eq!(dir.rotate(8).0, Compass::North);
        assert_eq!(dir.rotate(9).0, Compass::NorthEast);
    }

    #[test]
    fn test_dir_rotation_right() {
        let dir = Dir(Compass::South);

        assert_eq!(dir.rotate(1).0, Compass::SouthWest);
        assert_eq!(dir.rotate(2).0, Compass::West);
        assert_eq!(dir.rotate(3).0, Compass::NorthWest);
        assert_eq!(dir.rotate(8).0, Compass::South);
        assert_eq!(dir.rotate(9).0, Compass::SouthWest);
    }

    #[test]
    fn test_coord_into_dir() {
        let center : Dir = Coord(0,0).into();
        let south : Dir = Coord(0,-1).into();
        let west : Dir = Coord(-1,0).into();
        let north_east : Dir = Coord(1,1).into();

        assert_eq!(center.0, Compass::Center);
        assert_eq!(south.0, Compass::South);
        assert_eq!(west.0, Compass::West);
        assert_eq!(north_east.0, Compass::NorthEast);
    }

    #[test]
    fn test_dir_into_coord() {
        let north : Coord = Dir(Compass::North).into();
        let south : Coord = Dir(Compass::South).into();
        let north_west : Coord = Dir(Compass::NorthWest).into();


        assert_eq!(north, Coord(0,1));
        assert_eq!(south, Coord(0,-1));
        assert_eq!(north_west, Coord(-1,1));
    }
}
//</editor-fold>