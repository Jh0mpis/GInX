# Geodesics Integrator tool for particles in GRMHD using Adaptive Mesh Refinement

The integration of null geodesic equations within the 3+1 formalism of general relativity plays a key role in the process of computing images via ray tracing and analyzing spectra in GRMHD simulations. The objective of this project is to develop a geodesic integrator for particles in a generic GRMHD background  using the Fast-Light approximation with Adaptive Mesh Refinement (AMR), and to accelerate the integration with GPU computing. The implementation will utilize the C++ programming language in conjunction with the AMReX framework, which has been built with GPU support enabled. The project will focus both on the accurate numerical integration of null geodesic and on evaluating GPU-accelerated performance. The purpose of this tool is to serve as a post-processing module for the visualization and analysis of GRMHD simulation data. Additionally, it represents a critical first step toward the development of a full Monte Carlo radiation transport code, where fast and efficient geodesic integration is essential.

**Keywords: Null geodesics, GRMHD simulations, Ray tracing in curved spacetime, Monte Carlo radiation transport, GPU acceleration, Adaptive Mesh Refinement (AMR), AMReX framework, Parallel numerical integration.**

## Index

The following README file presents a description of the project, to see the full documentation of the code you can check [GInX:Geodesics Integrator tool for particles in GRMHD using Adaptive Mesh Refinement using amrex.](https://jh0mpis.github.io/GInX/)

- [GInX](#ginx)
    - [Base Particles Container](#base-particles-container)
    - [Particles Container](#particles-container)
    - [Particles Solver Utilities](#particles-solver-utilities)
- [Files Structure](#files-structure)
- [How to run it](#how-to-run-it)
    - [Installation with a compiled Einstein Toolkit](#installation-with-a-compiled-einstein-toolkit)
    - [Create-run and running it](#create-run-and-running-it)

## GInX

GInX is a null and timelike geodesic integrator using the fast light approximation. This code is written in C++17, is open source. It uses the barycentric Lagrange interpolation and a fourth-order Runge-Kutta method to accurately integrate the geodesics. The code's accuracy has been verified using conserved quantities in Schwarzschild and Kerr spacetimes. The code uses AMReX to enable parallel execution on both GPU and CPU architectures and to support Adaptive Mesh Refinement (AMR). Its performance was evaluated by measuring strong and weak scaling on CINECA's Leonardo, TACC Frontera and Vista clusters. It is designed to be easily integrated with others Einstein Toolkit thorns.

Using the four parameters used for 3+1 numerical relativity simulations ($\alpha$, $\vec{\beta}$, $K_{\mu\nu}$ and $\gamma_{\mu\nu}$) we can define particle's 4-momentum as

$$p^\mu = E(n^\alpha + V^\alpha),$$

where $p^\mu$ is the particle's four momentum and $n^\mu$ is the 4-velocity of the Euclidean Observer, then $E = -p_\nu n^\nu$ and $V^\mu$ is the particle energy and velocity respectively. Using the 4-momentum conservation equation 

$$p^\mu \nabla_\mu p^\nu = 0,$$

along with the ADM properties we can find the following differential equation system for the position $X^i$, velocity $V_i = V^j\gamma_{ji}$ and $\ln(E)$ for each particle, this is:

$$\frac{d X^i}{dt} = \alpha \gamma^{ij}V_j - \beta^i,$$
$$\frac{d V_i}{dt} = -\partial_i \alpha  + \left(\gamma^{kj}V_k\partial_j\alpha  - \alpha K_{jk}\gamma^{jl}\gamma^{km}V_mV_l\right)V_i + \frac{1}{2}\alpha \gamma^{jl}\gamma^{km}V_lV_m\partial_i\gamma_{jk} + V_j\partial_i\beta^j,$$
$$\frac{d\ln E}{dt} = \alpha K_{jk}\gamma^{lj}\gamma^{mk}V_lV_m - V_l\gamma^{lj}\partial_j\alpha.$$

Those are the variables that we are evolving for the particles and the equations we are solving.

GInX provides the following thorns

### Base Particles Container

This thorn contain several utilities for the evolution of particles using the AMReX Particles container approach. The BaseParticlesContainer thorn is an interface thorn that contains the common classes and functions for differential equation associated to systems of particles.

The particles container thorn contains the definition of the `BaseParticleContainer` templated class. This abstract class defines the basic methods that have to be defined on each of the derived classes, this class extends from the `amrex::AmrParticleContainer`.

> [!NOTE]
> For an extended explanation you can check the README file inside the thorn's folder.

### Particles Container

This folder contains the `ParticlesContainer.hxx` header file where the Particle container for particles is defined. It inherits from `BaseParticleContainer` abstract class and define the abstract method for the null particles normalization, i.e., 

$$V_iV^i = 1$$ 

and timelike particles, i.e., 

$$V_iV^i = 1 - \frac{m^2}{E^2}.$$

This also contains the `Photons.hxx` file that contains the definition of the struct used for the Photons evolution.

Finally, `ParticlesInitializers.hxx` contains different ways of initialize the system of photons, so far random initializations.

The `ParticlesEvolution.cxx` file contains the definitions related with the particle's evolution.

> [!NOTE]
> For an extended explanation you can check the README file inside the thorn's folder.

### Particles Solver Utilities

This folder includes the utilities needed to compute the right hand side of the differential equations. The files included there are:

- `Interpolator.hxx`: This file contains the definitions of the barycentric interpolation method.
- `Utilities.hxx`: This file contains different product of symmetric 3x3 matrices and vectors.

> [!NOTE]
> For an extended explanation you can check the README file inside the thorn's folder.

## Files Structure

```bash
GInX/
├── docs
├── BaseParticlesContainer
│   ├── configuration.ccl
│   ├── interface.ccl
│   ├── param.ccl
│   ├── README.md
│   ├── schedule.ccl
│   └── src
│       ├── BaseParticleContainer.hxx
│       ├── make.code.defn
│       └── metrics.cxx
├── ParticlesContainer
│   ├── configuration.ccl
│   ├── interface.ccl
│   ├── par
│   │   ├── HydroFile.par
│   │   ├── ParticlesContainer.par
│   │   └── ParticlesMultiGrid.par
│   ├── param.ccl
│   ├── README.md
│   ├── schedule.ccl
│   └── src
│       ├── make.code.defn
│       ├── ParticlesContainer.hxx
│       ├── ParticlesEvolution.cxx
│       ├── Photons.hxx
│       └── PhotonsInitializers.hxx
├── ParticlesSolverUtilities
│   ├── configuration.ccl
│   ├── interface.ccl
│   ├── param.ccl
│   ├── README.md
│   ├── schedule.ccl
│   └── src
│       ├── Interpolator.hxx
│       ├── make.code.defn
│       └── Utilities.hxx
└── README.md
```

##  How to run it

In order to run and include the thorn inside your Einstein Toolkit project you have to compile it first. Then include the thorns into your thornlist.

### Installation with a compiled Einstein Toolkit

You could add this thorn to your project by adding the following lines to your EinsteinToolkit thornlist:

```bash
# GInX thorn
!TARGET   = $ARR
!TYPE     = git
!URL      = https://github.com/Jh0mpis/GInX.git
!REPO_BRANCH = main
!REPO_PATH = $2
!CHECKOUT =
GInX/BaseParticlesContainer
GInX/ParticlesSolverUtilities
GInX/ParticlesContainer
```

and then you can execute the EinsteinToolkit re-build command that you usually use, for instance:

```bash
./simfactory/bin/sim build -j4 ... --clean
```

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
