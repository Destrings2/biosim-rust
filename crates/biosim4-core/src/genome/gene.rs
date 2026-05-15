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

/// A single synaptic connection encoded in 32 bits.
///
/// Bit layout:
/// ```text
/// bit  31:    source_type  — 0 = internal neuron, 1 = sensor
/// bits 30-24: source_num   — 7-bit index (remapped mod sensor_count or max_neurons)
/// bit  23:    sink_type    — 0 = internal neuron, 1 = action
/// bits 22-16: sink_num     — 7-bit index (remapped mod action_count or max_neurons)
/// bits 15-0:  weight       — signed i16; divide by 8192 to get ≈ -4.0..4.0
/// ```
///
/// Any random `u32` is a valid gene. Raw indices are remapped during
/// [`create_wiring`](super::neural_net::create_wiring), so the actual index
/// range doesn't matter at the genome level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gene(pub u32);

/// `source_type` value indicating the source is a registered sensor.
pub const SOURCE_SENSOR: u8 = 1;
/// `source_type` value indicating the source is an internal neuron.
pub const SOURCE_NEURON: u8 = 0;
/// `sink_type` value indicating the sink is a registered action.
pub const SINK_ACTION: u8 = 1;
/// `sink_type` value indicating the sink is an internal neuron.
pub const SINK_NEURON: u8 = 0;

impl Gene {
    /// Construct a gene from its logical fields. Only the low bits of each
    /// argument are used: 1 bit for type fields, 7 bits for num fields.
    pub fn new(source_type: u8, source_num: u8, sink_type: u8, sink_num: u8, weight: i16) -> Self {
        let w = weight as u16 as u32;
        let raw = ((source_type as u32 & 1) << 31)
            | ((source_num as u32 & 0x7F) << 24)
            | ((sink_type as u32 & 1) << 23)
            | ((sink_num as u32 & 0x7F) << 16)
            | w;
        Gene(raw)
    }

    /// Wrap a raw `u32` as a gene without interpreting its fields.
    pub fn from_raw(raw: u32) -> Self {
        Gene(raw)
    }

    /// Source node type: [`SOURCE_SENSOR`] (1) or [`SOURCE_NEURON`] (0).
    pub fn source_type(&self) -> u8 {
        ((self.0 >> 31) & 1) as u8
    }

    /// 7-bit source index. Remapped modulo `sensor_count` or `max_neurons` during wiring.
    pub fn source_num(&self) -> u8 {
        ((self.0 >> 24) & 0x7F) as u8
    }

    /// Sink node type: [`SINK_ACTION`] (1) or [`SINK_NEURON`] (0).
    pub fn sink_type(&self) -> u8 {
        ((self.0 >> 23) & 1) as u8
    }

    /// 7-bit sink index. Remapped modulo `action_count` or `max_neurons` during wiring.
    pub fn sink_num(&self) -> u8 {
        ((self.0 >> 16) & 0x7F) as u8
    }

    /// Raw signed 16-bit weight. Use [`weight_as_float`](Self::weight_as_float)
    /// for the scaled floating-point value.
    pub fn weight_raw(&self) -> i16 {
        (self.0 & 0xFFFF) as u16 as i16
    }

    /// Weight scaled to approximately -4.0..4.0
    pub fn weight_as_float(&self) -> f32 {
        self.weight_raw() as f32 / crate::constants::GENE_WEIGHT_SCALE
    }

    /// Return `true` if the source node is a sensor (not an internal neuron).
    pub fn is_sensor_source(&self) -> bool {
        self.source_type() == SOURCE_SENSOR
    }

    /// Return `true` if the sink node is an action (not an internal neuron).
    pub fn is_action_sink(&self) -> bool {
        self.sink_type() == SINK_ACTION
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_fields() {
        let g = Gene::new(1, 42, 0, 15, -1000);
        assert_eq!(g.source_type(), 1);
        assert_eq!(g.source_num(), 42);
        assert_eq!(g.sink_type(), 0);
        assert_eq!(g.sink_num(), 15);
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
