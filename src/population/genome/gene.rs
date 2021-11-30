use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gene {
    pub encoding: u16,
    pub weight: i16
}

pub const SENSOR: bool = true;
pub const ACTION: bool = true;
pub const NEURON: bool = false;

// This is quite more messy than the C++ version, as Rust doesn't have bitfields.
/// Each gene specifies one synaptic connection in a neural net. Each connection has an input (source),
/// which is either a sensor or another neuron, and an output (sink) which is either an action or another neuron.
/// Each gene has a weight, which is a floating point value derived from a signed 16-bit integer.
/// The signed integer weight is scaled to a small range, then cubed to provide fine resolution near zero.
impl Gene {
    //<editor-fold desc="Bitfield manipulation">
    pub fn weight_as_float(&self) -> f32 {
        return self.weight as f32 / 8192.0
    }

    pub fn set_bit(&mut self, bit: u8, value: bool) {
        if value {
            self.encoding |= 1 << bit;
        } else {
            self.encoding &= !(1 << bit);
        }
    }

    pub fn get_bit(&self, bit: u8) -> bool {
        return self.encoding & (1 << bit) != 0;
    }

    // Source type is in bit 16
    pub fn get_source_type(&self) -> bool { // SENSOR or NEURON
        return ((self.encoding & 0b1000000000000000) >> 15) == 1;
    }

    pub fn set_source_type(&mut self, flag: bool) {
        self.encoding = (self.encoding & !0b1000000000000000) | ((flag as u16) << 15);
    }

    // Source num is in bit 15-9
    pub fn get_source_num(&self) -> u8 {
        return ((self.encoding & 0b0111111100000000) >> 8) as u8;
    }

    pub fn set_source_num(&mut self, num: u8) {
        self.encoding = (self.encoding & !0b0111111100000000) | ((num as u16) << 8);
    }

    // Destination type is in bit 8
    pub fn get_sink_type(&self) -> bool { // NEURON or ACTION
        return ((self.encoding & 0b0000000010000000) >> 7) == 1;
    }

    pub fn set_sink_type(&mut self, flag: bool) {
        self.encoding = (self.encoding & !0b0000000010000000) | ((flag as u16) << 7);
    }

    // Destination num is in bit 7-1
    pub fn get_sink_num(&self) -> u8 {
        return (self.encoding & 0b0000000001111111) as u8;
    }

    pub fn set_sink_num(&mut self, num: u8) {
        self.encoding = (self.encoding & !0b0000000001111111) | (num as u16);
    }
    //</editor-fold>

    pub fn make_encoding(source_type: bool, source_num: u8, sink_type: bool, sink_num: u8) -> u16 {
        return (source_type as u16) << 15 | (source_num as u16) << 8 | (sink_type as u16) << 7 | (sink_num as u16);
    }

    pub fn make_random_weight() -> i16 {
        let mut rng = rand::thread_rng();
        return rng.gen()
    }

    pub fn make_random_encoding() -> u16 {
        let mut rng = rand::thread_rng();
        return rng.gen()
    }

    pub fn make_random_gene() -> Gene {
        return Gene {
            encoding: Gene::make_random_encoding(),
            weight: Gene::make_random_weight()
        };
    }

    pub fn new(source_type: bool, source_num: u8, sink_type: bool, sink_num: u8, weight: i16) -> Gene {
        return Gene {
            encoding: Gene::make_encoding(source_type, source_num, sink_type, sink_num),
            weight
        };
    }

    pub fn hex_string(&self) -> String {
        return format!("{:x}{:x}", self.encoding, self.weight);
    }

    pub fn bit_string(&self) -> String {
        return format!("{:b}{:b}", self.encoding, self.weight);
    }
}

impl ToString for Gene {
    fn to_string(&self) -> String {
        return format!("Gene(source_type={}, source_num={}, sink_type={}, sink_num={}, weight={})",
                       self.get_source_type(), self.get_source_num(), self.get_sink_type(), self.get_sink_num(), self.weight);
    }
}

//<editor-fold desc="Unit tests">
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bit_field() {
        let mut gene = Gene::new(false, 16, false, 25, 1);
        assert_eq!(gene.get_source_type(), false);
        assert_eq!(gene.get_source_num(), 16);
        assert_eq!(gene.get_sink_type(), false);
        assert_eq!(gene.get_sink_num(), 25);

        gene.set_sink_num(99);
        gene.set_source_num(35);
        gene.set_sink_type(false);
        gene.set_source_type(true);
        assert_eq!(gene.get_source_type(), true);
        assert_eq!(gene.get_source_num(), 35);
        assert_eq!(gene.get_sink_type(), false);
        assert_eq!(gene.get_sink_num(), 99);
    }
}
//</editor-fold>