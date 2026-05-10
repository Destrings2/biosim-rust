//! Gene encoding and accessors.
//!
//! A `Gene` is a 32-bit packed value representing one synaptic connection:
//!
//! ```text
//! bit  31:    source_type  (0 = neuron, 1 = sensor)
//! bits 30-24: source_num   (7 bits, 0..127)
//! bit  23:    sink_type    (0 = neuron, 1 = action)
//! bits 22-16: sink_num     (7 bits, 0..127)
//! bits 15-0:  weight       (signed i16, scaled to ≈ -4.0..4.0 by ÷ 8192)
//! ```
//!
//! Raw indices are remapped modulo `sensor_count`/`action_count`/`max_neurons`
//! during `create_wiring`, so the 7-bit range is irrelevant to the actual
//! registry size. A random `u32` is a valid (if arbitrary) gene.

/// A single synaptic connection gene encoded in 4 bytes:
///   bits 31:   source type (0=neuron, 1=sensor)
///   bits 30-24: source num (7 bits)
///   bit  23:   sink type (0=neuron, 1=action)
///   bits 22-16: sink num (7 bits)
///   bits 15-0:  weight (signed i16)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gene(pub u32);

pub const SOURCE_SENSOR: u8 = 1;
pub const SOURCE_NEURON: u8 = 0;
pub const SINK_ACTION:  u8 = 1;
pub const SINK_NEURON:  u8 = 0;

impl Gene {
    pub fn new(source_type: u8, source_num: u8, sink_type: u8, sink_num: u8, weight: i16) -> Self {
        let w = weight as u16 as u32;
        let raw = ((source_type as u32 & 1) << 31)
            | ((source_num as u32 & 0x7F) << 24)
            | ((sink_type as u32 & 1) << 23)
            | ((sink_num as u32 & 0x7F) << 16)
            | w;
        Gene(raw)
    }

    pub fn from_raw(raw: u32) -> Self { Gene(raw) }

    pub fn source_type(&self) -> u8 { ((self.0 >> 31) & 1) as u8 }
    pub fn source_num(&self)  -> u8 { ((self.0 >> 24) & 0x7F) as u8 }
    pub fn sink_type(&self)   -> u8 { ((self.0 >> 23) & 1) as u8 }
    pub fn sink_num(&self)    -> u8 { ((self.0 >> 16) & 0x7F) as u8 }
    pub fn weight_raw(&self)  -> i16 { (self.0 & 0xFFFF) as u16 as i16 }

    /// Weight scaled to approximately -4.0..4.0
    pub fn weight_as_float(&self) -> f32 {
        self.weight_raw() as f32 / 8192.0
    }

    pub fn is_sensor_source(&self) -> bool { self.source_type() == SOURCE_SENSOR }
    pub fn is_action_sink(&self)   -> bool { self.sink_type()   == SINK_ACTION }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_fields() {
        let g = Gene::new(1, 42, 0, 15, -1000);
        assert_eq!(g.source_type(), 1);
        assert_eq!(g.source_num(),  42);
        assert_eq!(g.sink_type(),   0);
        assert_eq!(g.sink_num(),    15);
        assert_eq!(g.weight_raw(), -1000);
    }
    #[test]
    fn weight_float_range() {
        let max = Gene::new(0, 0, 0, 0, i16::MAX);
        let min = Gene::new(0, 0, 0, 0, i16::MIN);
        assert!(max.weight_as_float() > 3.9);
        assert!(min.weight_as_float() < -3.9);
    }
}
