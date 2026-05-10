use crate::types::Coord;
use crate::grid::{visit_neighborhood, Grid};

pub const SIGNAL_MAX: u8 = 255;

/// Multi-layer pheromone grid. Each cell is a u8 magnitude (0..SIGNAL_MAX).
pub struct Signals {
    /// layers[layer][x][y]
    layers: Vec<Vec<Vec<u8>>>,
    pub size_x: u16,
    pub size_y: u16,
}

impl Signals {
    pub fn new(num_layers: u8, size_x: u16, size_y: u16) -> Self {
        let layers = (0..num_layers)
            .map(|_| vec![vec![0u8; size_y as usize]; size_x as usize])
            .collect();
        Self { layers, size_x, size_y }
    }

    pub fn zero_fill(&mut self) {
        for layer in self.layers.iter_mut() {
            for col in layer.iter_mut() {
                col.fill(0);
            }
        }
    }

    pub fn layer_count(&self) -> u8 { self.layers.len() as u8 }

    pub fn get(&self, layer: u8, loc: Coord) -> u8 {
        self.layers[layer as usize][loc.x as usize][loc.y as usize]
    }

    /// Increment center by +2 and all neighbors within radius 1.5 by +1, clamped to SIGNAL_MAX.
    pub fn increment(&mut self, layer: u8, center: Coord, grid: &Grid) {
        let l = &mut self.layers[layer as usize];
        let add = |l: &mut Vec<Vec<u8>>, loc: Coord, v: u8| {
            if loc.x >= 0 && loc.y >= 0
                && (loc.x as u16) < grid.size_x
                && (loc.y as u16) < grid.size_y
            {
                let cell = &mut l[loc.x as usize][loc.y as usize];
                *cell = cell.saturating_add(v);
            }
        };

        // Center gets +2
        add(l, center, 2);

        // Neighbors within radius 1.5 get +1
        for dx in -1i16..=1 {
            for dy in -1i16..=1 {
                if dx == 0 && dy == 0 { continue; }
                let d2 = (dx * dx + dy * dy) as f32;
                if d2 <= 1.5 * 1.5 {
                    add(l, Coord::new(center.x + dx, center.y + dy), 1);
                }
            }
        }
    }

    /// Decrement all values in a layer by 1 (floor 0) to simulate pheromone decay.
    pub fn fade(&mut self, layer: u8) {
        for col in self.layers[layer as usize].iter_mut() {
            for cell in col.iter_mut() {
                *cell = cell.saturating_sub(1);
            }
        }
    }

    /// Get the total signal density in a neighborhood (sum / count, normalized 0..1).
    pub fn get_density(&self, layer: u8, center: Coord, radius: f32, grid: &Grid) -> f32 {
        let mut sum = 0u32;
        let mut count = 0u32;
        visit_neighborhood(grid, center, radius, |loc| {
            sum += self.get(layer, loc) as u32;
            count += 1;
        });
        if count == 0 { return 0.0; }
        (sum as f32 / count as f32) / SIGNAL_MAX as f32
    }
}
