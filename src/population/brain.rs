mod sensor_actions;

use crate::population::genome::Genome;

struct Neuron {
    output: f64,
    driven: bool
}

impl Neuron {
    pub const fn initial_neuron_output() -> f64 {
        return 0.5;
    }
}

/// An individual's "brain" is a neural net specified by a set
/// of Genes where each Gene specifies one connection in the neural net (see
/// Genome comments above). Each neuron has a single output which is
/// connected to a set of sinks where each sink is either an action output
/// or another neuron. Each neuron has a set of input sources where each
/// source is either a sensor or another neuron. There is no concept of
/// layers in the net: it's a free-for-all topology with forward, backwards,
/// and sideways connection allowed. Weighted connections are allowed
/// directly from any source to any action.
///
/// Currently, the genome does not specify the activation function used in
/// the neurons. (May be hardcoded to std::tanh() !!!)
///
/// When the input is a sensor, the input value to the sink is the raw
/// sensor value of type float and depends on the sensor. If the output
/// is an action, the source's output value is interpreted by the action
/// node and whether the action occurs or not depends on the action's
/// implementation.
///
/// In the genome, neurons are identified by 15-bit unsigned indices,
/// which are reinterpreted as values in the range 0..p.genomeMaxLength-1
/// by taking the 15-bit index modulo the max number of allowed neurons.
/// In the neural net, the neurons that end up connected get new indices
/// assigned sequentially starting at 0.
pub struct NeuralNet {
    connections: Genome,
    neurons: Vec<Neuron>,
}
