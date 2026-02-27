 \dir ../PhotonsContainer 
# Photons Container

The thorn implements the geodesic evolution of null and timelike particles in a 3+1 general relativity background. It uses the implementations made in `ParticlesContainer` and `PhotonSolverUtilities` thorns.

## Index

- [Photons Container](#photons-container)
- [Dependencies](#dependencies)
- [Additional thorn features](#additional-thorn-features)
    - [Random particle initialization](#random-particle-initialization)
- [Parameters file](#parameters-file)
    - [Random particle initialization](#random-particle-initialization)
    - [ParticlesContainer shared parameters](#particlesContainer-shared-parameters)

## Photons Container

The Photons container class has the definitions of the class, methods and functions needed to perform the geodesic evolution. It defines the `PhotonsContainer` class derived from the `BaseParticlesContainer` abstract class implemented in the **Particles Container** thorn.

The `PhotonsContainer` class contains the definition of the `evolve`, `compute_rhs`, and `normalize_velocity` methods, all of it needed during the initialization and evolution of the geodesics. It also contains the value of the mass of the particles. It is defined inside of the `PhotonsContainer.hxx` file.

It also contains the definition of two initializers, for the moment, that set up the particles' position and velocity inside of the `PhotonInitializers.hxx` file. Inside the `test.cxx` are the scheduled functions implemented by the thorn.

## Dependencies

The code uses the thorns:

1. ParticlesContainer.
2. PhotonSolverUtilities

You need to have those thorns and its requirements compiled.

## Additional thorn features

The thorn also includes some extra features that can be used for the simulation.

### Random particle initialization

The thorn provides two functions that initialize an arbitrary number of particles with random positions and velocities but fixed energy. The main difference is that one function initializes particles distributed within a sphere of radius $R$, while the other distributes them across the entire rectangular grid. This can be managed by the `num_photons`, `initializer`, `init_params_d[10]`, and `init_params_i[10]` parameters.

## Parameters file

The thorn have the parameters related with the particle initialization and the ones shared with the `ParticlesContainer` thorn.

The most relevant parameter for timelike particles creation is:

| Parameter           | Type    | Description                                                                                          | Range                          | Default |
| ------------------- | ------- | ---------------------------------------------------------------------------------------------------- | ------------------------------ | ------- |
| particles_mass      | REAL    | Particles' mass.                             | 0.0 to infinity | 0.0 (null particles)   |

### Random particle initialization

The thorn has the following parameters for the particles initialization:

| Parameter           | Type    | Description                                                                                          | Range                          | Default |
| ------------------- | ------- | ---------------------------------------------------------------------------------------------------- | ------------------------------ | ------- |
| num_photons         | INTEGER | Total number of photons to be generated      | 0 to infinity | 0       |
| initializer         | KEYWORDS | Available initialization function to be called | box, spherical | none |
| init_params_d[10]         | REAL[10] | Double parameters needed to pass to the initializer function | any | 0.0 |
| init_params_i[10]         | INTEGER[10] | Integer parameters needed to pass to the initializer function | any | 0 |

### ParticlesContainer shared parameters

The thorn shares all the parameters defined on the `ParticlesContainer` thorn. See [ParticlesContainer documentation](../ParticlesContainer/README.md).
