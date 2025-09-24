# Geodesics Integrator tool for Photons in GRMHD using Adaptive Mesh Refinement

The integration of null geodesic equations within the 3+1 formalism of general relativity plays a key role in the process of computing images via ray tracing and analyzing spectra in GRMHD simulations. The objective of this project is to develop a geodesic integrator for photons in a generic GRMHD background using Adaptive Mesh Refinement (AMR), and to accelerate the integration with GPU computing. The implementation will utilize the C++ programming language in conjunction with the AMReX framework, which has been built with CUDA support enabled. The project will focus both on the accurate numerical integration of null geodesic and on evaluating GPU-accelerated performance. The purpose of this tool is to serve as a post-processing module for the visualization and analysis of GRMHD simulation data. Additionally, it represents a critical first step toward the development of a full Monte Carlo radiation transport code, where fast and efficient geodesic integration is essential.

**Keywords: Null geodesics, GRMHD simulations, Ray tracing in curved spacetime, Monte Carlo radiation transport, GPU acceleration, Adaptive Mesh Refinement (AMR), AMReX framework, Parallel numerical integration.**

## Index

The following readme presents a description of the project, to see the full documentation of the code you can check [GeodesicIntegratorX:Geodesics Integrator tool for Photons in GRMHD using Adaptive Mesh Refinement using amrex.](https://jh0mpis.github.io/GeodesicIntegratorX/index.html)

- [Geodesics Integrator X](#geodesics-integrator-x)
    - [Photons Container](#photons-container)
    - [Photons Solver Utilities](#photons-solver-utilities)
- [Files Structure](#files-structure)
- [How to run it](#how-to-run-it)
    - [Installation with a compiled Einstein Toolkit](#installation-with-a-compiled-einstein-toolkit)
    - [Parameters file](#parameters-file)
    - [Create-run and running it](#create-run-and-running-it)

## Geodesics Integrator X

The geodesics integrator thorn aims to solve the geodesic equations in the 3+1 description of General Relativity using AMReX and running it on GPU. This thorn contains the following utilities:

### Photons Container

This folder contains the `PhotonsContainer.hxx` header file where the Particle container for photons class, that inherits from `BaseParticleContainer` abstract class (This class is implemented in [ParticlesUtilities](https://github.com/Jh0mpis/ParticlesUtilities/)), is defined. This class contains the method for compute the right hand side part for the Rung-Kutta method evolution.

This also contains the `Photons.hxx` file that contains the definition of the struct used for the Photons evolution.

Finally, `PhotonsInitializers.hxx` contains different ways of initialize the system of photons, so far random initializations.

> [!NOTE]
> Contains the `test.cxx` file, this files contain an example of how to use the PhotonsContainer and its methods, can be used as an example in the future or has to be removed and included in the documentation.

### Photons Solver Utilities

This folder includes the utilities needed to compute the right hand side of the differential equations. The files included there are:

- `Discretizer.hxx`: This file contains the discretization of the first derivatives on each direction at orders 2, 4, 6 and 8.
- `Interpolator.hxx`: This file contains different interpolators like barycentric interpolator.
- `Utilities.hxx`: This file contains different product of symmetric 3x3 matrices and vectors.

## Files Structure

```bash
GeodesicIntegratorX/
├── PhotonsContainer
│   ├── configuration.ccl
│   ├── interface.ccl
│   ├── par
│   │   └── PhotonsContainer.par
│   ├── param.ccl
│   ├── schedule.ccl
│   └── src
│       ├── make.code.defn
│       ├── PhotonsContainer.hxx
│       ├── Photons.hxx
│       ├── PhotonsInitializers.hxx
│       └── test.cxx
├── PhotonSolverUtilities
│   ├── configuration.ccl
│   ├── doc
│   ├── interface.ccl
│   ├── param.ccl
│   ├── schedule.ccl
│   └── src
│       ├── include
│       │   ├── Discretizer.hxx
│       │   ├── Interpolator.hxx
│       │   ├── make.code.defn
│       │   └── Utilities.hxx
│       └── make.code.defn
└── README.md
```

##  How to run it

> [!NOTE]
> This way of build and run the code is just provisional. This could change in further versions.

### Installation with a compiled Einstein Toolkit

In order to compile the new thorn into a Einstein Toolkit build we can use the same approach used in [CreatingANewThorn-HeatEqn.ipynb](https://github.com/EinsteinToolkit/jupyter-et/blob/master/tutorial-server/notebooks/CreatingANewThorn-HeatEqn.ipynb) at the `EinsteinToolkit\jupyter-et` github repository. The first step is to clone the repository and its dependency into `Cactus/arrangements/`:

```bash
git clone -b dev git@github.com:Jh0mpis/ParticlesUtilities.git
git clone -b dev git@github.com:Jh0mpis/GeodesicIntegratorX.git
```

Next step is to include the Thorn and its utilities into the `Cactus/configs/sim/ThornList` file and add the following lines:

```bash
# Null GeodesicIntegratorX Utilities
ParticlesUtilities/ParticlesContainer
GeodesicIntegratorX/PhotonSolverUtilities
GeodesicIntegratorX/PhotonsContainer
```

After this change we need to re-build `Cactus` by running

```bash
./simfactory/bin/sim build -j2
```

This is going to compile the new thorns and all its dependencies.

### Parameters file

In order to run the `test.cxx` defined inside the `GeodesicIntegratorX` thorn we need to use a `param.par` file. The `param.ccl` file allows to define the following parameters:

| Parameter        | Type    | Description                                        | Range                                    | Default             |
| ---------------- | ------- | -------------------------------------------------- | ---------------------------------------- | ------------------- |
| photons_per_cell | INTEGER | Number of photons per cell unit on the simulation. | [1, $\infty$)                            | 1                   |
| black_hole_mass  | REAL    | Schwarzschild metric black hole mass.              | [0, $\infty$)                            | 1.0                 |
| initial_metric   | KEYWORD | Metric to be initialized.                          | "none", "minkowski", "iso schwarzschild" | "iso schwarzschild" |

***Shared with `ParticlesContainer`:***

| Parameter           | Type    | Description                                                                                          | Range                          | Default |
| ------------------- | ------- | ---------------------------------------------------------------------------------------------------- | ------------------------------ | ------- |
| particle_plot_every | INTEGER | The number of step where the simulation should print the particles data using amrex print utilities. | [0, $\infty$) 0 means no print | 0       |
| particle_tsv_every  | INTEGER | The number of step where the simulation should print the particles data using ascii.                 | [0, $\infty$) 0 means no print | 0       |

An example of how to define those parameters along with the CarpetX parameters can be seen into [./PhotonsContainer/par/PhotonsContainer.par](./PhotonsContainer/par/PhotonsContainer.par).

### Create-run and running it

Once we have everything setted up we can create and run the simulation using the simfactory `create-run` command:

```bash
rm -rf /simulations/folder/path/run-name # If exists any previous run
./simfactory/bin/sim create-run run-name --parfile=/parfile/path/parfile-name.par
```
