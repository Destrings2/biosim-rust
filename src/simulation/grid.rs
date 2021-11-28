//<editor-fold desc="Column implementation">
struct Column {
    data: Vec<u16>,
}

impl Column {
    fn new(size: usize) -> Column {
        Column {
            data: vec![0; size],
        }
    }

    fn size(&self) -> usize {
        self.data.len()
    }
}

impl std::ops::Index<usize> for Column {
    type Output = u16;

    fn index(&self, index: usize) -> &u16 {
        return &self.data[index];
    }
}
//</editor-fold>

pub struct Grid {
    pub width: u16,
    pub height: u16,
    data: Vec<Column>,
}

impl Grid {
    pub fn new(width: u16, height: u16) -> Grid {
        let mut data = Vec::new();
        for _ in 0..width {
            data.push(Column::new(height as usize));
        }
        return Grid { width, height, data};
        }

    pub fn get(&self, x: u16, y: u16) -> u16 {
        self.data[x as usize][y as usize]
    }
}