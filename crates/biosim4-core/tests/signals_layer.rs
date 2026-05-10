//! Signals layer (pheromone field): increment / fade / get_density math.
//! The C++ uses these for stigmergic communication; bugs in saturating arithmetic
//! cause subtle drift over many steps.

use biosim4_core::{
    grid::Grid,
    signals_layer::{Signals, SIGNAL_MAX},
    types::Coord,
};

#[test]
fn new_signals_layer_starts_at_zero() {
    let g = Grid::new(10, 10);
    let s = Signals::new(1, 10, 10);
    for x in 0..10i16 {
        for y in 0..10i16 {
            assert_eq!(s.get(0, Coord::new(x, y)), 0);
        }
    }
    let _ = g;
}

#[test]
fn increment_center_gets_plus_2_neighbors_get_plus_1() {
    let grid = Grid::new(10, 10);
    let mut s = Signals::new(1, 10, 10);
    let center = Coord::new(5, 5);
    s.increment(0, center, &grid);
    assert_eq!(s.get(0, center), 2, "center should get +2");
    // 4 cardinal neighbors get +1
    assert_eq!(s.get(0, Coord::new(4, 5)), 1);
    assert_eq!(s.get(0, Coord::new(6, 5)), 1);
    assert_eq!(s.get(0, Coord::new(5, 4)), 1);
    assert_eq!(s.get(0, Coord::new(5, 6)), 1);
    // Diagonals also within radius 1.5 get +1
    assert_eq!(s.get(0, Coord::new(4, 4)), 1);
    assert_eq!(s.get(0, Coord::new(6, 6)), 1);
}

#[test]
fn fade_decreases_all_values_by_one_with_floor() {
    let grid = Grid::new(10, 10);
    let mut s = Signals::new(1, 10, 10);
    s.increment(0, Coord::new(5, 5), &grid);
    s.fade(0);
    assert_eq!(s.get(0, Coord::new(5, 5)), 1, "center after fade should be 2-1=1");
    assert_eq!(s.get(0, Coord::new(4, 5)), 0, "neighbor after fade should be 1-1=0");
    s.fade(0);
    assert_eq!(s.get(0, Coord::new(5, 5)), 0, "center after second fade should saturate at 0");
}

#[test]
fn increment_saturates_at_signal_max() {
    let grid = Grid::new(5, 5);
    let mut s = Signals::new(1, 5, 5);
    let center = Coord::new(2, 2);
    // Many increments should never overflow u8
    for _ in 0..1000 {
        s.increment(0, center, &grid);
    }
    assert_eq!(s.get(0, center), SIGNAL_MAX, "saturating add should cap at SIGNAL_MAX");
}

#[test]
fn increment_at_corner_does_not_panic() {
    let grid = Grid::new(10, 10);
    let mut s = Signals::new(1, 10, 10);
    s.increment(0, Coord::new(0, 0), &grid);
    s.increment(0, Coord::new(9, 9), &grid);
    assert_eq!(s.get(0, Coord::new(0, 0)), 2);
    assert_eq!(s.get(0, Coord::new(9, 9)), 2);
}

#[test]
fn get_density_normalized_to_unit_interval() {
    let grid = Grid::new(10, 10);
    let mut s = Signals::new(1, 10, 10);
    // Empty layer → 0
    let d_empty = s.get_density(0, Coord::new(5, 5), 2.0, &grid);
    assert!(d_empty.abs() < 1e-6);

    // Saturate the entire neighborhood
    for x in 0..10 {
        for y in 0..10 {
            for _ in 0..200 {
                s.increment(0, Coord::new(x, y), &grid);
            }
        }
    }
    let d_full = s.get_density(0, Coord::new(5, 5), 2.0, &grid);
    assert!((d_full - 1.0).abs() < 1e-3, "full saturation density should be ≈1.0, got {}", d_full);
}

#[test]
fn zero_fill_clears_all_layers() {
    let grid = Grid::new(5, 5);
    let mut s = Signals::new(2, 5, 5);
    s.increment(0, Coord::new(2, 2), &grid);
    s.increment(1, Coord::new(2, 2), &grid);
    s.zero_fill();
    for layer in 0..2 {
        for x in 0..5i16 {
            for y in 0..5i16 {
                assert_eq!(s.get(layer, Coord::new(x, y)), 0);
            }
        }
    }
}
