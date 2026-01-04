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
        - [Initialization](#initialization)
    - [Create-run and running it](#create-run-and-running-it)

## Geodesics Integrator X

The geodesics integrator thorn aims to solve the geodesic equations in the 3+1 description of General Relativity using AMReX and both CPU and GPU.

Using the four ADM parameters used for 3+1 numerical relativity simulations ($\alpha$, $\vec{\beta}$, $K_{\mu\nu}$ and $\gamma_{\mu\nu}$) we can define the following tensors for any particle in GR:

$$p^\mu = E(n^\alpha + V^\alpha),$$

where $p^\mu$ is the particle's four momentum and $n^\mu$ is the 4-velocity of the Euclidean Observer, then $E = -p_\nu n^\nu$ and $V^\mu$ is the particle energy and velocity respectively. Using the 4-momentum conservation equation 

$$p^\mu \nabla_\mu p^\nu = 0,$$

along with the ADM properties we can find the following differential equation system for the position $X^i$, velocity $V_i = V^j\gamma_{ji}$ and $\ln(E)$ for each particle, this is:

$$\frac{d X^i}{dt} = \alpha \gamma^{ij}V_j - \beta^i,$$
$$\frac{d V_i}{dt} = -\partial_i \alpha  + \left(\gamma^{kj}V_k\partial_j\alpha  - \alpha K_{jk}\gamma^{jl}\gamma^{km}V_mV_l\right)V_i + \frac{1}{2}\alpha \gamma^{jl}\gamma^{km}V_lV_m\partial_i\gamma_{jk} + V_j\partial_i\beta^j,$$
$$\frac{d\ln E}{dt} = \alpha K_{jk}\gamma^{lj}\gamma^{mk}V_lV_m - V_l\gamma^{lj}\partial_j\alpha.$$

Those are the variables that we are evolving for the particles and the equations we are solving.

This thorn contains the following utilities:

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
│   ├── doc
│   ├── interface.ccl
│   ├── par
│   │   ├── HydroFile.par
│   │   ├── PhotonsContainer.par
│   │   └── PhotonsMultiGrid.par
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
│       ├── Discretizer.hxx
│       ├── Interpolator.hxx
│       ├── make.code.defn
│       └── Utilities.hxx
└── README.md
```

##  How to run it

In order to run and include the thorn inside your EinsteinToolkit project you have to compile it first and then build it using a compiled EinsteinToolkit.

### Installation with a compiled Einstein Toolkit

You could add this thorn to your project by adding the following lines to your EinsteinToolkit thornlist:

```bash
# PUtilX <- Dependency
!TARGET   = $ARR
!TYPE     = git
!URL      = https://github.com/Jh0mpis/ParticlesUtilities.git
!REPO_BRANCH = dev
!REPO_PATH = $2
!CHECKOUT =
ParticlesUtilities/ParticlesContainer

# GInX thorn
!TARGET   = $ARR
!TYPE     = git
!URL      = https://github.com/Jh0mpis/GeodesicIntegratorX.git
!REPO_BRANCH = dev
!REPO_PATH = $2
!CHECKOUT =
GeodesicIntegratorX/PhotonSolverUtilities
GeodesicIntegratorX/PhotonsContainer
```

and then you can execute the EinsteinToolkit re-build command that you usually use, for instance:

```bash
./simfactory/bin/sim build -j4 ...
```

### Parameters file

Al the parameters are defined inside of the `PhotonsContainer` thorn, plus the one shared with the `BaseParticleContainer` Thorn, check [ParticlesUtilities](https://github.com/Jh0mpis/ParticlesUtilities/) for more details.

#### Initialization

The parameters used for the initialization of the particles are the following ones:

| Parameter           | Type        | Description                                                         | Range                                                                         | Default             |
| ------------------- | ----------- | -------------------------------------------------- | ---------------------------------------- | ------------------- |
| num_photons         | INTEGER     | Total number of photons in the simulation.                          | any                  | 0    |
| initializer         | KEYWORD     | Available initializer function to be called.                        | box, spherical, none | none |
| init_params_d       | REAL[10]    | double type parameters needed to pass to the initializer function.  | any                  | 0.0  |
| init_params_i       | INTEGER[10] | integer type parameters needed to pass to the initializer function. | any                  | 0    |


#### Variables shared with ParticlesUtilities

The thorn shares the following parameters with the ParticlesUtilities thorn, check [ParticlesUtilities](https://github.com/Jh0mpis/ParticlesUtilities/) for more info.

| Parameter           | Type    | Description                                                                                          | Range                          | Default |
| ------------------- | ------- | ---------------------------------------------------------------------------------------------------- | ------------------------------ | ------- |
| particle_plot_every | INTEGER | The number of step where the simulation should print the particles data using amrex print utilities. | [0, $\infty$) 0 means no print | 0       |
| particle_tsv_every  | INTEGER | The number of step where the simulation should print the particles data using ascii.                 | [0, $\infty$) 0 means no print | 0       |
| banned_regions      | INTEGER | Number of banned regions for the simulation.                         | [0, 10]                               | 0                    |
| region_<n>_position | REAL[3] | Size 3 array containing the coordinates of the banned region center. | any                                   | 0.0                  |
| region_<n>_radius   | REAL | Banned region's radius.                                                 | any                                   | 0.0                  |

### Create-run and running it

Once we have everything setted up we can create and run the simulation using the simfactory `create-run` command:

```bash
./simfactory/bin/sim create-run run-name --parfile=/parfile/path/parfile-name.par other-options
```

Once you create your run, you could submit a new one by running 

```bash
./simfactory/bin/sim run run-name --parfile=/parfile/path/parfile-name.par other-options
```

It is going to create a new simulation on the same folder created before. If any of the previous commands fails you can always run the executable directly.
