# Geodesics Integrator tool for Photons in GRMHD using Adaptive Mesh Refinement

The integration of null geodesic equations within the 3+1 formalism of general relativity plays a key role in the process of computing images via ray tracing and analyzing spectra in GRMHD simulations. The objective of this project is to develop a geodesic integrator for photons in a generic GRMHD background using Adaptive Mesh Refinement (AMR), and to accelerate the integration with GPU computing. The implementation will utilize the C++ programming language in conjunction with the AMReX framework, which has been built with CUDA support enabled. The project will focus both on the accurate numerical integration of null geodesic and on evaluating GPU-accelerated performance. The purpose of this tool is to serve as a post-processing module for the visualization and analysis of GRMHD simulation data. Additionally, it represents a critical first step toward the development of a full Monte Carlo radiation transport code, where fast and efficient geodesic integration is essential.

**Keywords: Null geodesics, GRMHD simulations, Ray tracing in curved spacetime, Monte Carlo radiation transport, GPU acceleration, Adaptive Mesh Refinement (AMR), AMReX framework, Parallel numerical integration.**

## Index

- [Geodesics Integrator X](#geodesics-integrator-x)
    - [Photons Container](#photons-container)
    - [Photons Solver Utilities](#photons-solver-utilities)
- [Files Structure](#files-structure)

## Geodesics Integrator X

The geodesics integrator thorn aims to solve the geodesic equations in the 3+1 description of General Relativity using AMReX and running it on GPU. This thorn contains the following utilities:

### Photons Container

This folder contains the `PhotonsContainer.hxx` header file where the Particle container for photons class, that inherits from `BaseParticleContainer` abstract class (This class is implemented in [ParticlesUtilities](https://github.com/Jh0mpis/ParticlesUtilities/)), is defined. This class contains the method for compute the right hand side part for the Rung-Kutta method evolution.

This also contains the `Photons.hxx` file that contains the definition of the struct used for the Photons evolution.

> [!NOTE]
> Contains the `test.cxx` file, this files contain an example of how to use the PhotonsContainer and its methods, can be used as an example in the future or has to be removed and included in the documentation.

### Photons Solver Utilities

This folder includes the utilities needed to compute the right hand side of the differential equations. The files included there are:

- `Discretizer.hxx`: This file contains the discretization of the first derivatives on each direction at orders 2, 4, 6 and 8.
- `Interpolator.hxx`: This file contains different interpolators.
- `Utilities.hxx`: This file contains different product of symmetric 3x3 matrices and vectors.

## Files Structure

```bash
GeodesicIntegratorX/
├── PhotonsContainer/
│   ├── configuration.ccl
│   ├── interface.ccl
│   ├── param.ccl
│   ├── schedule.ccl
│   └── src/
│       ├── make.code.defn
│       ├── PhotonsContainer.hxx
│       ├── Photons.hxx
│       └── test.cxx
├── PhotonSolverUtilities/
│   ├── configuration.ccl
│   ├── doc/
│   ├── interface.ccl
│   ├── param.ccl
│   ├── schedule.ccl
│   └── src/
│       ├── include/
│       │   ├── Discretizer.hxx
│       │   ├── Interpolator.hxx
│       │   ├── make.code.defn
│       │   └── Utilities.hxx
│       └── make.code.defn
└── README.md
```
