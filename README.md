# biosim-rust
## What?

This is a (in-progess) rust implementation of (davidmiller/biosim4)[https://github.com/davidrmiller/biosim4].

## Why?

I found C++ really unsightly and bloated. I also find the code and topic very interesting, so this is a way to learn more about the implementation while avoiding C++.

## Progress

This is not a direct rewrite, as I'm improving the code where possible. The original program structre is as follows:

```
├── analysis.cpp
├── basicTypes.cpp DONE
├── basicTypes.h DONE
├── createBarrier.cpp
├── endOfGeneration.cpp
├── endOfSimStep.cpp
├── executeActions.cpp
├── feedForward.cpp
├── genome-compare.cpp
├── genome-neurons.h IN PROGRESS
├── genome.cpp IN PROGRESS
├── getSensor.cpp
├── grid.cpp
├── grid.h
├── imageWriter.cpp
├── imageWriter.h
├── indiv.cpp IN PROGRESS
├── indiv.h IN PROGRESS
├── main.cpp
├── params.cpp IN PROGRESS
├── params.h IN PROGRESS
├── peeps.cpp
├── peeps.h
├── random.cpp
├── random.h
├── sensors-actions.h
├── signals.cpp
├── signals.h
├── simulator.cpp
├── simulator.h
├── spawnNewGeneration.cpp
├── survival-criteria.cpp
├── unitTestBasicTypes.cpp
├── unitTestConnectNeuralNetWiringFromGenome.cpp
└── unitTestGridVisitNeighborhood.cpp
```
