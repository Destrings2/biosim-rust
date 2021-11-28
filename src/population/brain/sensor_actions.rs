pub mod sensor_implementation;
pub mod action_implementation;

//<editor-fold desc="Constants">
const SENSOR_MIN: f32 = 0.0;
const SENSOR_MAX: f32 = 1.0;

const NEURON_MIN: f32 = 0.0;
const NEURON_MAX: f32 = 1.0;

const ACTION_MIN: f32 = 0.0;
const ACTION_MAX: f32 = 1.0;
const ACTION_RANGE: f32 = ACTION_MAX - ACTION_MIN;
//</editor-fold>

//<editor-fold desc="Sensor implementation">
pub enum Sensor {
    LocX = 0,             // I distance from left edge
    LocY,             // I distance from bottom
    BoundaryDistX,   // I X distance to nearest edge of world
    BoundaryDist,     // I distance to nearest edge of world
    BoundaryDistY,   // I Y distance to nearest edge of world
    GeneticSimFwd,   // I genetic similarity forward
    LastMoveDirX,   // I +- amount of X movement in last movement
    LastMoveDirY,   // I +- amount of Y movement in last movement
    LongProbePopFwd, // W long look for population forward
    LongProbeBarFwd, // W long look for barriers forward
    Population,        // W population density in neighborhood
    PopulationFwd,    // W population density in the forward-reverse axis
    PopulationLR,     // W population density in the left-right axis
    Osc1,              // I oscillator +-value
    Age,               // I
    BarrierFwd,       // W neighborhood barrier distance forward-reverse axis
    BarrierLR,        // W neighborhood barrier distance left-right axis
    Rnd,            //   random sensor value, uniform distribution
    Signal0,           // W strength of signal0 in neighborhood
    Signal0Fwd,       // W strength of signal0 in the forward-reverse axis
    Signal0LR        // W strength of signal0 in the left-right axis
}

pub const ENABLED_SENSORS: [Sensor; 21] =
    [
        Sensor::LocX, Sensor::LocY, Sensor::BoundaryDistX, Sensor::BoundaryDist, Sensor::BoundaryDistY,
        Sensor::GeneticSimFwd, Sensor::LastMoveDirX, Sensor::LastMoveDirY, Sensor::LongProbePopFwd,
        Sensor::LongProbeBarFwd, Sensor::Population, Sensor::PopulationFwd, Sensor::PopulationLR,
        Sensor::Osc1, Sensor::Age, Sensor::BarrierFwd, Sensor::BarrierLR, Sensor::Rnd, Sensor::Signal0,
        Sensor::Signal0Fwd, Sensor::Signal0LR
    ];

impl Sensor {
    pub fn get_name(&self) -> String {
        match self {
            Sensor::LocX => { "loc X" }
            Sensor::LocY => { "loc Y" }
            Sensor::BoundaryDistX => { "boundary dist X" }
            Sensor::BoundaryDist => { "boundary dist" }
            Sensor::BoundaryDistY => { "boundary dist Y" }
            Sensor::GeneticSimFwd => { "genetic simimlarity fwd" }
            Sensor::LastMoveDirX => { "last move dir X" }
            Sensor::LastMoveDirY => { "last move dir Y" }
            Sensor::LongProbePopFwd => { "long probe population fwd" }
            Sensor::LongProbeBarFwd => { "long probe barrier fwd" }
            Sensor::Population => { "population" }
            Sensor::PopulationFwd => { "population fwd" }
            Sensor::PopulationLR => { "population left-right" }
            Sensor::Osc1 => { "oscillator 1" }
            Sensor::Age => { "age" }
            Sensor::BarrierFwd => { "barrier fwd" }
            Sensor::BarrierLR => { "barrier left-right" }
            Sensor::Rnd => { "random" }
            Sensor::Signal0 => { "signal 0" }
            Sensor::Signal0Fwd => { "signal 0 fwd" }
            Sensor::Signal0LR => { "signal 0 left-right" }
        }.to_string()
    }
}

impl ToString for Sensor {
    fn to_string(&self) -> String {
        match self {
            Sensor::LocX => { "Lx" }
            Sensor::LocY => { "Ly" }
            Sensor::BoundaryDistX => { "EDx" }
            Sensor::BoundaryDist => { "ED" }
            Sensor::BoundaryDistY => { "EDy" }
            Sensor::GeneticSimFwd => { "Gen" }
            Sensor::LastMoveDirX => { "LMx" }
            Sensor::LastMoveDirY => { "LMy" }
            Sensor::LongProbePopFwd => { "LPf" }
            Sensor::LongProbeBarFwd => { "LPb" }
            Sensor::Population => { "Pop" }
            Sensor::PopulationFwd => { "Pfd" }
            Sensor::PopulationLR => { "Plr" }
            Sensor::Osc1 => { "Osc" }
            Sensor::Age => { "Age" }
            Sensor::BarrierFwd => { "Bfd" }
            Sensor::BarrierLR => { "Blr" }
            Sensor::Rnd => { "Rnd" }
            Sensor::Signal0 => { "Sg" }
            Sensor::Signal0Fwd => { "Sfd" }
            Sensor::Signal0LR => { "Slr" }
        }.to_string()
    }
}
//</editor-fold>

//<editor-fold desc="Action Implementation">
// I means the action affects the individual internally (Indiv)
// W means the action also affects the environment (Peeps or Grid)
pub enum Action {
    MoveX = 0,                   // W +- X component of movement
    MoveY,                   // W +- Y component of movement
    MoveForward,             // W continue last direction
    MoveRL,                  // W +- component of movement
    MoveRandom,              // W
    SetOscillatorPeriod,    // I
    SetLongProbeDist,       // I
    SetResponsiveness,       // I
    EmitSignal0,             // W
    MoveEast,                // W
    MoveWest,                // W
    MoveNorth,               // W
    MoveSouth,               // W
    MoveLeft,                // W
    MoveRight,               // W
    MoveReverse,             // W
    KillForward             // W
}

pub const ENABLED_ACTIONS: [Action; 16] =
    [
        Action::MoveX, Action::MoveY, Action::MoveForward, Action::MoveRL, Action::MoveRandom,
        Action::SetOscillatorPeriod, Action::SetLongProbeDist, Action::SetResponsiveness,
        Action::EmitSignal0, Action::MoveEast, Action::MoveWest, Action::MoveNorth, Action::MoveSouth,
        Action::MoveLeft, Action::MoveRight, Action::MoveReverse
    ];

impl Action {
    fn get_name(&self) -> String {
        match self {
            Action::MoveX => { "move X" }
            Action::MoveY => { "move Y" }
            Action::MoveForward => { "move forward" }
            Action::MoveRL => { "move R-L" }
            Action::MoveRandom => { "move random" }
            Action::SetOscillatorPeriod => { "set oscillator period" }
            Action::SetLongProbeDist => { "set long probe dist" }
            Action::SetResponsiveness => { "set responsiveness" }
            Action::EmitSignal0 => { "emit signal 0" }
            Action::MoveEast => { "move east" }
            Action::MoveWest => { "move west" }
            Action::MoveNorth => { "move north" }
            Action::MoveSouth => { "move south" }
            Action::MoveLeft => { "move left" }
            Action::MoveRight => { "move right" }
            Action::MoveReverse => { "move reverse" }
            Action::KillForward => { "kill forward" }
        }.to_string()
    }
}

impl ToString for Action {
    fn to_string(&self) -> String {
        match self {
            Action::MoveX => {"MvX"}
            Action::MoveY => {"MvY"}
            Action::MoveForward => {"MvF"}
            Action::MoveRL => {"MRL"}
            Action::MoveRandom => {"Mrn"}
            Action::SetOscillatorPeriod => {"OSC"}
            Action::SetLongProbeDist => {"LPD"}
            Action::SetResponsiveness => {"Res"}
            Action::EmitSignal0 => {"SG"}
            Action::MoveEast => {"MvE"}
            Action::MoveWest => {"MvW"}
            Action::MoveNorth => {"MvN"}
            Action::MoveSouth => {"MvS"}
            Action::MoveLeft => {"MvL"}
            Action::MoveRight => {"MvR"}
            Action::MoveReverse => {"Mrv"}
            Action::KillForward => {"Klf"}
        }.to_string()
    }
}
//</editor-fold>