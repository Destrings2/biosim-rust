//! Direction / Compass / Coord invariants.
//! These are foundational — every sensor and action that mentions "forward",
//! "left", "right", or "near" depends on this layer being correct.

use biosim4_core::types::{Compass, Coord, Dir};

#[test]
fn rotate_full_circle_returns_identity() {
    for &c in &Compass::ALL8 {
        let d = Dir(c);
        assert_eq!(d.rotate(8), d, "rotating 8 positions (full circle) should be identity");
    }
}

#[test]
fn rotate_zero_is_identity() {
    for &c in &Compass::ALL8 {
        let d = Dir(c);
        assert_eq!(d.rotate(0), d, "rotate(0) should be identity");
    }
}

#[test]
fn rotate_90cw_four_times_full_circle() {
    for &c in &Compass::ALL8 {
        let d = Dir(c);
        let rotated = d.rotate90cw().rotate90cw().rotate90cw().rotate90cw();
        assert_eq!(rotated, d, "rotate90cw × 4 should return original");
    }
}

#[test]
fn rotate_180_twice_is_identity() {
    for &c in &Compass::ALL8 {
        let d = Dir(c);
        assert_eq!(d.rotate180().rotate180(), d);
    }
}

#[test]
fn rotate_cw_and_ccw_are_inverses() {
    for &c in &Compass::ALL8 {
        let d = Dir(c);
        assert_eq!(d.rotate90cw().rotate90ccw(), d, "cw then ccw should be identity for {:?}", c);
        assert_eq!(d.rotate90ccw().rotate90cw(), d, "ccw then cw should be identity for {:?}", c);
    }
}

#[test]
fn east_rotate90cw_is_south() {
    assert_eq!(Dir(Compass::E).rotate90cw(), Dir(Compass::S));
}

#[test]
fn north_rotate90cw_is_east() {
    assert_eq!(Dir(Compass::N).rotate90cw(), Dir(Compass::E));
}

#[test]
fn rotate_center_stays_center() {
    let c = Dir::center();
    assert_eq!(c.rotate(1), c);
    assert_eq!(c.rotate(8), c);
}

#[test]
fn as_normalized_coord_for_each_compass() {
    assert_eq!(Dir(Compass::E).as_normalized_coord(), Coord::new(1, 0));
    assert_eq!(Dir(Compass::W).as_normalized_coord(), Coord::new(-1, 0));
    assert_eq!(Dir(Compass::N).as_normalized_coord(), Coord::new(0, 1));
    assert_eq!(Dir(Compass::S).as_normalized_coord(), Coord::new(0, -1));
    assert_eq!(Dir(Compass::NE).as_normalized_coord(), Coord::new(1, 1));
    assert_eq!(Dir(Compass::NW).as_normalized_coord(), Coord::new(-1, 1));
    assert_eq!(Dir(Compass::SE).as_normalized_coord(), Coord::new(1, -1));
    assert_eq!(Dir(Compass::SW).as_normalized_coord(), Coord::new(-1, -1));
    assert_eq!(Dir(Compass::CENTER).as_normalized_coord(), Coord::new(0, 0));
}

#[test]
fn coord_as_dir_zero_is_center() {
    assert_eq!(Coord::new(0, 0).as_dir(), Dir::center());
}

#[test]
fn coord_as_dir_all_eight_octants() {
    // Pure cardinal
    assert_eq!(Coord::new(10, 0).as_dir(), Dir(Compass::E));
    assert_eq!(Coord::new(-10, 0).as_dir(), Dir(Compass::W));
    assert_eq!(Coord::new(0, 10).as_dir(), Dir(Compass::N));
    assert_eq!(Coord::new(0, -10).as_dir(), Dir(Compass::S));
    // Pure diagonal
    assert_eq!(Coord::new(7, 7).as_dir(), Dir(Compass::NE));
    assert_eq!(Coord::new(-7, 7).as_dir(), Dir(Compass::NW));
    assert_eq!(Coord::new(7, -7).as_dir(), Dir(Compass::SE));
    assert_eq!(Coord::new(-7, -7).as_dir(), Dir(Compass::SW));
}

#[test]
fn coord_subtraction_works() {
    let a = Coord::new(5, 3);
    let b = Coord::new(2, 1);
    assert_eq!(a - b, Coord::new(3, 2));
}
