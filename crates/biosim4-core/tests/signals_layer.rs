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
fn increment_center_gets_plus_3_neighbors_get_plus_1() {
    let grid = Grid::new(10, 10);
    let s = Signals::new(1, 10, 10);
    let center = Coord::new(5, 5);
    s.increment(0, center, &grid);
    assert_eq!(s.get(0, center), 3, "center receives +1 (neighborhood) +2 (explicit) = +3");
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
    assert_eq!(s.get(0, Coord::new(5, 5)), 2, "center after fade should be 3-1=2");
    assert_eq!(s.get(0, Coord::new(4, 5)), 0, "neighbor after fade should be 1-1=0");
    s.fade(0);
    s.fade(0);
    assert_eq!(s.get(0, Coord::new(5, 5)), 0, "center after three fades should saturate at 0");
}

#[test]
fn increment_saturates_at_signal_max() {
    let grid = Grid::new(5, 5);
    let s = Signals::new(1, 5, 5);
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
    let s = Signals::new(1, 10, 10);
    s.increment(0, Coord::new(0, 0), &grid);
    s.increment(0, Coord::new(9, 9), &grid);
    assert_eq!(s.get(0, Coord::new(0, 0)), 3);
    assert_eq!(s.get(0, Coord::new(9, 9)), 3);
}

#[test]
fn get_density_normalized_to_unit_interval() {
    let grid = Grid::new(10, 10);
    let s = Signals::new(1, 10, 10);
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

#[test]
fn diffusion_reduces_magnitude_over_time() {
    // After an increment, repeated fades must bring all cells monotonically
    // toward zero. The center starts at 3 and neighbors at 1, so three
    // fades zero everything.
    let grid = Grid::new(10, 10);
    let mut s = Signals::new(1, 10, 10);
    let center = Coord::new(5, 5);
    s.increment(0, center, &grid);

    assert_eq!(s.get(0, center), 3, "center should start at 3");

    s.fade(0);
    assert_eq!(s.get(0, center), 2);
    s.fade(0);
    assert_eq!(s.get(0, center), 1);
    s.fade(0);
    assert_eq!(s.get(0, center), 0);

    // All cells must now be 0 (neighbors were at 1 and floored at 0 after
    // the first fade; the center reached 0 on the third).
    for x in 0..10i16 {
        for y in 0..10i16 {
            assert_eq!(s.get(0, Coord::new(x, y)), 0, "cell ({x},{y}) should be 0 after 3 fades");
        }
    }
}
