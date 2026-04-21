# RaytracingX

A relativistic raytracing thorn in the CarpetX framework.

Written by Noelle Feist <njf4697<area>@rit.edu> with code adapted from Jhon Sebastián Moreno Triana <github: Jh0mpis> in
collaboration with Manuela Campanelli, Yosef Zlochower, Liwei Ji, and Scott Noble.

This thorn creates a grid of mass-less particles (sometimes referred to as photons in the code) set to move in
different directions originating at a single point, then integrates their corresponding geodesics backwards until
they either reach the simulation boundaries, an event horizon, or the photosphere (WIP). The geodesic integration
is modified from and has [`GInX/BaseParticlesContianer` and `GInX/ParticlesContainer`](https://github.com/Jh0mpis/GInX) as prerequisites. The geodesic initial data and
integration of geodesics follows [Bohn et al. (2015)](https://arxiv.org/pdf/1410.7775). That point is then taken to be
the starting point for integration of the relativistic radiative transfer equation, and the specific intensity is
integrated back to the camera (WIP).

**Any interaction with matter (including radiative transfer) is currently a work in progress.**

## Interaction with GInX

While much of the code is shared with [GInX](https://github.com/Jh0mpis/GInX), only the parameters for particle
output and banned regions from `GInX/BaseParticlesContainer` are used. Additionally, the thorn `ParticlesContainer`
by default runs a test evolution of particles. Thus, it is recommended to set `ParticlesContainer::run_test = false`
in the parameter file.

## Modes of Operation

This code currently operates with two modes, using the fast-light approximation (all raytracing happens in a single
simulation iteration) and numerical spacetime (WIP).

## Camera and Image Setup

The camera is defined by a position `camera_pos`, a velocity `camera_vel`, a pointing direction `camera_point`, and
an upwards direction `camera_up`. Each of these variables should be defined as a 3-vector in the simulation frame in
the parameter file. Note that the camera's position is not evolved over time, so if two snapshots are requested, they
will be from the same position regardless of the camera's velocity.

The edges of the image are defined by the horizontal and vertical field of view parameters `horizontal_fov` and
`vertical_fov` given in degrees. The resolution is given by the number of pixels across the width and height of the 
image `num_pixels_width` and `num_pixels_height`. These parameters should be set in the parameter file.

One mass-less particle is created for each pixel. The initial position of each particle is set to `camera_pos` and 
equations for the initial velocity are given by [Bohn et al. (2015)](https://arxiv.org/pdf/1410.7775). In effect, 
each initial velocity is a normalized 4-vector in the direction from `camera_pos` to the center of each pixel.
Each velocity can be thought of as offset by $d\theta=\frac{\text{FOV}}{\text{num. pixels}}$ in the horizontal and
vertical directions, though this is not how they are calculated. **This is different from traditional raytracing, 
where the image is defined by a rectangular grid set a distance away from the camera.** This image created by this
code code can be thought of as a grid on the surface of a sphere centered on the camera. Due to this, the raw image
is distorted in a 'fish-eye' manner. This distortion can be undone, and an example of this is shown in
`post_processing/debug_image.ipynb`.

Each pixel is given an index from which the image can be constructed. This starts from zero in the top-left corner and
increments left-to-right, then up-to-down. An example is shown in the image below, where the camera is placed at the
black dot, the spatial components of the camera basis vectors given from the Gram-Schmidt process from `camera_point` 
and `camera_up` (with the addition of a third vector in the 'right' direction) are shown in blue, red, and green
respectively.

![Image depicting the image and camera setup.](wireframe.png)

Since the geodesics are evolved backwards, the *actual* initial velocity has the spatial components reversed.

## Evolution Details

Each particle is integrated backwards until it reaches the bounds of the simulation, a banned region, an event horizon, or the photosphere (WIP). The banned regions are defined using the `BaseParticlesContainer` parameters such as the number of banned regions `banned_regions`, the position of the $i$ th banned region `region_i_position`, and the radius of the $i$ th region `region_i_radius`. This thorn adds the parameters `region_i_a` to convert the spherical banned regions into those with the shape of a Kerr BH outer event horizon with appropriate spin. In most cases, it suffices to not use the banned regions; as a particle moves towards an event horizon, it's energy increases exponentially, so any particles with an energy above this thorn's `max_energy` parameter is deleted. As the particles are evolved, the optical depth is also evolved, so any particles with an optical depth of at least 1 is also deleted (WIP).

The actual raytracing is done at specified times and/or interation numbers. The number of images produced is given by the parameters `num_raytrace_times` and `num_raytrace_iterations`, and the parameters `raytrace_at_time` and `raytrace_at_iteration` are vectors that contain all times/iterations in which images should be produced.

Since the raytracing evolves until all particles are deleted, a failsafe maximum number of raytracing steps is given by the parameter `max_iterations`.

## Output Notes

This thorn has three possible outputs at the moment. The first is given by `BaseParticlesContainer`'s `particle_plot_every` parameter, which gives output from the [standard AMReX particle `WritePlotFile` routine](https://github.com/AMReX-Codes/amrex/blob/development/Src/Particle/AMReX_ParticleIO.H#L113), and outputs every `particle_plot_every` raytracing iterations. The second is from `BaseParticlesContainer`'s `particle_tsv_every` parameter, which uses the [standard AMReX particle `WriteAsciiFile` routine](https://github.com/AMReX-Codes/amrex/blob/development/Src/Particle/AMReX_ParticleIO.H#L1141) and outputs every `particle_tsv_every` raytracing iterations. The output is sent to a file named `Photons#####` with the raytracing iteration number replacing the `#`. The output file is read as follows:
```
{number of particles}
0
0
6
0
{x} {y} {z} {particle_id per cpu} {cpu_id} {v_x} {v_y} {v_z} {ln_alphaE} {pixel_id}
{x} {y} {z} {particle_id per cpu} {cpu_id} {v_x} {v_y} {v_z} {ln_alphaE} {pixel_id}
.
.
.
```
An example usage for this output is given in `post_processing/geodesic_visualization.ipynb`. The final output format name is given by the parameter `final_data_file_name` and is only output if the `output_final_data = true`. This output gives the information for each particle before being deleted. There will be one for each processor named `{final_data_file_name}.{cpu_id}.0`. The format is as follows:
```
{pixel_id} {x} {y} {z} {vx} {vy} {vz} {ln_alphaE} {tau} {deletion_reason}
{pixel_id} {x} {y} {z} {vx} {vy} {vz} {ln_alphaE} {tau} {deletion_reason}
.
.
.
```
An example usage for this output is given in `post_processing/debug_image.ipynb`. The `deletion_reason` data gives the reason each particle was deleted, and the values are given by the following table:
| Value | Reason for Deletion |
| ----- | ------------------- |
| $-1$  | $x>x_{max}$         |
| $-2$  | $x<x_{min}$         |
| $-3$  | $y>y_{max}$         |
| $-4$  | $y>y_{min}$         |
| $-5$  | $z>z_{max}$         |
| $-6$  | $z>z_{min}$         |
| $-7$  | $p^0>E_{max}$       |
| $-8-i$| $i$ th Banned Region|
| $-999$| Error               |
