/**
 * \file PhotonsInitializers.hxx
 * \brief PhotonsContainer initialization functions.
 *
 * This File contains examples of how to create different initial conditions for
 * photons, imposing the initial conditions on the velocity.
 */
#ifndef PHOTON_INITIALIZERS
#define PHOTON_INITIALIZERS

#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#include <AMReX_Random.H>
#include <AMReX_RandomEngine.H>
#include <AMReX_Scan.H>
#include <cctk.h>
#include <cmath>
#include <iostream>

#include "AMReX_Array.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParticleContainer.H"
#include "Interpolator.hxx"

namespace photons_init {

using namespace Interpolator;

/**
 * \brief This function random initializes a custom number of particles on each
 * grid cell
 *
 * @param pc The particle container that is going to be initialized.
 * @param number_of_particles_per_cell The number of particles contained on each
 * cell.
 * @param metric The 3 dimensional ADM metric.
 * @param lev AMR discretization level.
 */
template <typename StructType, typename ParticleContainerClass>
void random_photons_initializer(ParticleContainerClass &pc,
                                const int &number_of_particles_per_cell,
                                const amrex::MultiFab &metric, const int &lev) {

  CCTK_INFO("Initializing particles using the random_photons_initializer");

  // Get the with of the discretization on each direction.
  const auto dx = pc.Geom(lev).CellSizeArray();

  // Get the lower and higher value over the ParticleContainer Geometry
  const auto p_lo = pc.Geom(lev).ProbLoArray();
  const auto p_hi = pc.Geom(lev).ProbHiArray();

  amrex::MFIter mfi = pc.MakeMFIter(lev);

  // Iterating over all the tiles of the particle data structure
  for (; mfi.isValid(); ++mfi) {

    // get each tile box
    const amrex::Box &tile_box = mfi.tilebox();

    // Get the tile bounds
    const auto lo = amrex::lbound(tile_box);
    const auto hi = amrex::ubound(tile_box);

    // Get a reference to the particles
    auto &particles = pc.GetParticles(lev);
    auto &particle_tile =
        particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

    // Determines the current size and the required new size
    auto old_size = particle_tile.GetArrayOfStructs().size();
    auto new_size = old_size + number_of_particles_per_cell *
                                   (hi.x - lo.x + 1) * (hi.z - lo.z + 1) *
                                   (hi.y - lo.y + 1);

    // Resize the container once, we do not need to do it one by one
    particle_tile.resize(new_size);

    // Gets raw pointers to the two different ways particle data is stored for
    // performance reasons: Array of Struct (AoS) and Struct of Arrays (SoA)
    typename ParticleContainerClass::ParticleType *p_struct =
        particle_tile.GetArrayOfStructs()().data();
    auto arrdata = particle_tile.GetStructOfArrays().realarray();

    // get the current process id
    int proc_id = amrex::ParallelDescriptor::MyProc();
    auto const metric_array = metric.array(mfi);

    // Start a for loop with Random Number evolution
    amrex::ParallelForRNG(tile_box, [=] AMREX_GPU_DEVICE(
                                        int i, int j, int k,
                                        amrex::RandomEngine const
                                            &engine) noexcept {
      // Calculate cell_id
      int ix = i - lo.x;
      int iy{0}, iz{0};

      int nx = hi.x - lo.x + 1;
      int ny{0}, nz{0};

      unsigned int uix = amrex::min(nx - 1, amrex::max(0, ix));
      unsigned int uiy{0}, uiz{0};

      iy = j - lo.y;
      ny = hi.y - lo.y + 1;
      uiy = amrex::min(ny - 1, amrex::max(0, iy));
      iz = k - lo.z;
      nz = hi.z - lo.z + 1;
      uiz = amrex::min(nz - 1, amrex::max(0, iz));

      unsigned int cell_id = (uix * ny + uiy) * nz + uiz;

      // Retrievers the starting write index (pidx) for the current cell
      // (i, j, k) from the offsets array that was calculated by the
      // exclusive_scan
      int pidx = old_size + number_of_particles_per_cell * cell_id;

      for (int i_part = 0; i_part < number_of_particles_per_cell; i_part++) {
        amrex::Real ratio[AMREX_SPACEDIM];

        // Create particles outside of an sphere of radius 0.5
        // Generate a random position
        ratio[0] =
            (std::abs(p_hi[0] - p_lo[0]) * 0.5 - 0.7) * amrex::Random(engine) +
            0.7;
        ratio[1] = amrex::Random(engine) * M_PI;
        ratio[2] = amrex::Random(engine) * 2. * M_PI;

        typename ParticleContainerClass::ParticleType &p = p_struct[pidx];
        p.id() = pidx + 1;
        p.cpu() = proc_id;

        p.pos(0) = ratio[0] * std::sin(ratio[1]) * std::cos(ratio[2]);
        p.pos(1) = ratio[0] * std::sin(ratio[1]) * std::sin(ratio[2]);
        p.pos(2) = ratio[0] * std::cos(ratio[1]);

        const int i0 = amrex::Math::floor((p.pos(0) - p_lo[0]) / dx[0]);
        const int j0 = amrex::Math::floor((p.pos(1) - p_lo[1]) / dx[1]);
        const int k0 = amrex::Math::floor((p.pos(2) - p_lo[2]) / dx[2]);

        // Interpolate metric
        const amrex::GpuArray<CCTK_REAL, 6> gamma_x = {
            barycentric_cubic_3d<4>(metric_array, i0, j0, k0, p.pos(0),
                                    p.pos(1), p.pos(2), dx, p_lo,
                                    0), // g_11
            barycentric_cubic_3d<4>(metric_array, i0, j0, k0, p.pos(0),
                                    p.pos(1), p.pos(2), dx, p_lo,
                                    1), // g_12 & g_21
            barycentric_cubic_3d<4>(metric_array, i0, j0, k0, p.pos(0),
                                    p.pos(1), p.pos(2), dx, p_lo,
                                    2), // g_13 & g_31
            barycentric_cubic_3d<4>(metric_array, i0, j0, k0, p.pos(0),
                                    p.pos(1), p.pos(2), dx, p_lo,
                                    3), // g_22
            barycentric_cubic_3d<4>(metric_array, i0, j0, k0, p.pos(0),
                                    p.pos(1), p.pos(2), dx, p_lo,
                                    4), // g_23, g_32
            barycentric_cubic_3d<4>(metric_array, i0, j0, k0, p.pos(0),
                                    p.pos(1), p.pos(2), dx, p_lo, 5)}; // g_33

        const CCTK_REAL inv_det_gamma =
            1.0 / (gamma_x[0] * gamma_x[3] * gamma_x[5] +
                   2. * gamma_x[1] * gamma_x[2] * gamma_x[4] -
                   gamma_x[2] * gamma_x[2] * gamma_x[3] -
                   gamma_x[4] * gamma_x[4] * gamma_x[0] -
                   gamma_x[1] * gamma_x[1] * gamma_x[5]);

        const amrex::GpuArray<CCTK_REAL, 6> gamma_inv_x = {
            (gamma_x[3] * gamma_x[5] - gamma_x[4] * gamma_x[4]) * inv_det_gamma,
            (gamma_x[4] * gamma_x[2] - gamma_x[1] * gamma_x[5]) * inv_det_gamma,
            (gamma_x[1] * gamma_x[4] - gamma_x[2] * gamma_x[3]) * inv_det_gamma,
            (gamma_x[0] * gamma_x[5] - gamma_x[2] * gamma_x[2]) * inv_det_gamma,
            (gamma_x[2] * gamma_x[1] - gamma_x[0] * gamma_x[4]) * inv_det_gamma,
            (gamma_x[0] * gamma_x[3] - gamma_x[1] * gamma_x[1]) *
                inv_det_gamma};

        // Compute a random initial velocity
        ratio[1] = 2.0 * amrex::Random(engine) - 1.0;
        ratio[2] = 2.0 * amrex::Random(engine) - 1.0;
        ratio[0] = 2.0 * amrex::Random(engine) - 1.0;

        // Normalizing the velocity.
        const CCTK_REAL v_squared = ratio[0] * ratio[0] * gamma_inv_x[0] +
                                    ratio[1] * ratio[1] * gamma_inv_x[3] +
                                    ratio[2] * ratio[2] * gamma_inv_x[5] +
                                    2.0 * ratio[0] * ratio[1] * gamma_inv_x[1] +
                                    2.0 * ratio[0] * ratio[2] * gamma_inv_x[2] +
                                    2.0 * ratio[1] * ratio[2] * gamma_inv_x[4];
        const CCTK_REAL v = std::sqrt(v_squared);

        // Create the particle and add it to the container
        arrdata[StructType::vx][pidx] = ratio[0] / v;
        arrdata[StructType::vy][pidx] = ratio[1] / v;
        arrdata[StructType::vz][pidx] = ratio[2] / v;

        // Update the particles counter
        ++pidx;
      }
    });
    CCTK_VINFO("%d particles created",
               particle_tile.GetArrayOfStructs().size());
  }

} // random_photons_initializer

/**
 * \brief This function random initializes a custom number of particles on each
 * particle container.
 *
 * @param pc The particle container that is going to be initialized.
 * @param number_of_particles_per_container The number of particles contained on
 * each particle container.
 * @param metric The 3 dimensional ADM metric.
 */
template <typename StructType, typename ParticleContainerClass>
void random_photons_per_container_initializer(
    ParticleContainerClass &pc, const int &number_of_particles_per_container,
    const amrex::MultiFab &metric) {

  CCTK_INFO("Initializing particles using the "
            "random_photons_per_container_initializer");
  int lev = 0;

  // Get the with of the discretization on each direction.
  const auto dx = pc.Geom(lev).CellSizeArray();

  // Get the lower and higher value over the ParticleContainer Geometry
  const auto p_lo = pc.Geom(lev).ProbLoArray();
  const auto p_hi = pc.Geom(lev).ProbHiArray();

  amrex::MFIter mfi = pc.MakeMFIter(lev);

  // Iterating over all the tiles of the particle data structure
  for (; mfi.isValid(); ++mfi) {

    // get each tile box
    const amrex::Box &tile_box = mfi.tilebox();

    // Get the tile bounds
    const auto lo = amrex::lbound(tile_box);
    const auto hi = amrex::ubound(tile_box);

    // Get a reference to the particles
    auto &particles = pc.GetParticles(lev);
    auto &particle_tile = pc.DefineAndReturnParticleTile(lev, mfi);

    // Determines the current size and the required new size
    auto old_size = particle_tile.GetArrayOfStructs().size();
    auto new_size = old_size + number_of_particles_per_container;

    // Resize the container once, we do not need to do it one by one
    particle_tile.resize(new_size);

    // Gets raw pointers to the two different ways particle data is stored for
    // performance reasons: Array of Struct (AoS) and Struct of Arrays (SoA)
    typename ParticleContainerClass::ParticleType *p_struct =
        particle_tile.GetArrayOfStructs()().data();
    auto arrdata = particle_tile.GetStructOfArrays().realarray();

    // get the current process id
    int proc_id = amrex::ParallelDescriptor::MyProc();
    auto const metric_array = metric.array(mfi);

    for (int i = 0; i < number_of_particles_per_container; i++) {
      // Start a for loop with Random Number evolution
      int pidx = old_size + i;

      amrex::Real ratio[AMREX_SPACEDIM];

      // Create particles outside of an sphere of radius 0.5
      // Generate a random position
      ratio[0] =
          (std::abs(p_hi[0] - p_lo[0]) * 0.5 - 0.7) * amrex::Random() + 0.7;
      ratio[1] = amrex::Random() * M_PI;
      ratio[2] = amrex::Random() * 2. * M_PI;

      typename ParticleContainerClass::ParticleType &p = p_struct[pidx];
      p.id() = pidx + 1;
      p.cpu() = proc_id;

      p.pos(0) = ratio[0] * std::sin(ratio[1]) * std::cos(ratio[2]);
      p.pos(1) = ratio[0] * std::sin(ratio[1]) * std::sin(ratio[2]);
      p.pos(2) = ratio[0] * std::cos(ratio[1]);

      const int i0 = amrex::Math::floor((p.pos(0) - p_lo[0]) / dx[0]);
      const int j0 = amrex::Math::floor((p.pos(1) - p_lo[1]) / dx[1]);
      const int k0 = amrex::Math::floor((p.pos(2) - p_lo[2]) / dx[2]);

      // Interpolate metric
      const amrex::GpuArray<CCTK_REAL, 6> gamma_x = {
          barycentric_cubic_3d<5>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo,
                                  0), // g_11
          barycentric_cubic_3d<5>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo,
                                  1), // g_12 & g_21
          barycentric_cubic_3d<5>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo,
                                  2), // g_13 & g_31
          barycentric_cubic_3d<5>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo,
                                  3), // g_22
          barycentric_cubic_3d<5>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo,
                                  4), // g_23, g_32
          barycentric_cubic_3d<5>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo, 5)}; // g_33

      const CCTK_REAL inv_det_gamma =
          1.0 / (gamma_x[0] * gamma_x[3] * gamma_x[5] +
                 2. * gamma_x[1] * gamma_x[2] * gamma_x[4] -
                 gamma_x[2] * gamma_x[2] * gamma_x[3] -
                 gamma_x[4] * gamma_x[4] * gamma_x[0] -
                 gamma_x[1] * gamma_x[1] * gamma_x[5]);

      const amrex::GpuArray<CCTK_REAL, 6> gamma_inv_x = {
          (gamma_x[3] * gamma_x[5] - gamma_x[4] * gamma_x[4]) * inv_det_gamma,
          (gamma_x[4] * gamma_x[2] - gamma_x[1] * gamma_x[5]) * inv_det_gamma,
          (gamma_x[1] * gamma_x[4] - gamma_x[2] * gamma_x[3]) * inv_det_gamma,
          (gamma_x[0] * gamma_x[5] - gamma_x[2] * gamma_x[2]) * inv_det_gamma,
          (gamma_x[2] * gamma_x[1] - gamma_x[0] * gamma_x[4]) * inv_det_gamma,
          (gamma_x[0] * gamma_x[3] - gamma_x[1] * gamma_x[1]) * inv_det_gamma};

      // Compute a random initial Velocity
      ratio[0] = 2.0 * amrex::Random() - 1.0;
      ratio[1] = 2.0 * amrex::Random() - 1.0;
      ratio[2] = 2.0 * amrex::Random() - 1.0;

      // Normalizing the velocity.
      const CCTK_REAL v_squared = ratio[0] * ratio[0] * gamma_inv_x[0] +
                                  ratio[1] * ratio[1] * gamma_inv_x[3] +
                                  ratio[2] * ratio[2] * gamma_inv_x[5] +
                                  2.0 * ratio[0] * ratio[1] * gamma_inv_x[1] +
                                  2.0 * ratio[0] * ratio[2] * gamma_inv_x[2] +
                                  2.0 * ratio[1] * ratio[2] * gamma_inv_x[4];
      const CCTK_REAL v = std::sqrt(v_squared);

      // Create the particle and add it to the container
      arrdata[StructType::vx][pidx] = ratio[0] / v;
      arrdata[StructType::vy][pidx] = ratio[1] / v;
      arrdata[StructType::vz][pidx] = ratio[2] / v;
      arrdata[StructType::ln_E][pidx] = 0;
    }

    CCTK_VINFO("%d particles created",
               particle_tile.GetArrayOfStructs().size());
  }

} // random_photons_per_container_initializer

template <typename StructType, typename ParticleContainerClass>
void random_parallel_photons_per_container_initializer(
    ParticleContainerClass &pc, const int &number_of_particles_per_container,
    const amrex::MultiFab &metric) {

  CCTK_INFO("Initializing particles using the "
            "random_parallel_photons_per_container_initializer");
  int lev = 0;

  // Get the with of the discretization on each direction.
  const auto dx = pc.Geom(lev).CellSizeArray();

  // Get the lower and higher value over the ParticleContainer Geometry
  const auto p_lo = pc.Geom(lev).ProbLoArray();
  const auto p_hi = pc.Geom(lev).ProbHiArray();

  const int n_procs = amrex::ParallelDescriptor::NProcs();
  const int proc_id = amrex::ParallelDescriptor::MyProc();

  const int local_particles_size =
      number_of_particles_per_container / n_procs +
      (proc_id < number_of_particles_per_container % n_procs);

  int total_tiles{0};
  for (amrex::MFIter mfi = pc.MakeMFIter(lev); mfi.isValid(); ++mfi) {
    total_tiles++;
  }

  int current_tile = 0;

  // Iterating over all the tiles of the particle data structure
  for (amrex::MFIter mfi = pc.MakeMFIter(lev); mfi.isValid(); ++mfi) {

    const int particles_per_tile =
        local_particles_size / total_tiles +
        (current_tile < local_particles_size % total_tiles);
    CCTK_VWarn(1, __LINE__, __FILE__, CCTK_THORNSTRING,
               "Number of particles created at tile %d: %d ", current_tile,
               particles_per_tile);

    // get each tile box
    const amrex::Box &tile_box = mfi.tilebox();

    // Get the tile bounds
    const auto lo = amrex::lbound(tile_box);
    const auto hi = amrex::ubound(tile_box);

    const CCTK_REAL tile_x_min = p_lo[0] + lo.x * dx[0];
    const CCTK_REAL tile_x_max = p_lo[0] + (hi.x + 1) * dx[0];
    const CCTK_REAL tile_y_min = p_lo[1] + lo.y * dx[1];
    const CCTK_REAL tile_y_max = p_lo[1] + (hi.y + 1) * dx[1];
    const CCTK_REAL tile_z_min = p_lo[2] + lo.z * dx[2];
    const CCTK_REAL tile_z_max = p_lo[2] + (hi.z + 1) * dx[2];

    // Get a reference to the particles
    auto &particles = pc.GetParticles(lev);
    auto &particle_tile = pc.DefineAndReturnParticleTile(lev, mfi);

    // Determines the current size and the required new size
    auto old_size = particle_tile.GetArrayOfStructs().size();

    // Resize the container once, we do not need to do it one by one
    auto new_size = old_size + particles_per_tile;
    particle_tile.resize(new_size);

    // Gets raw pointers to the two different ways particle data is stored for
    // performance reasons: Array of Struct (AoS) and Struct of Arrays (SoA)
    typename ParticleContainerClass::ParticleType *p_struct =
        particle_tile.GetArrayOfStructs()().data();
    auto arrdata = particle_tile.GetStructOfArrays().realarray();

    for (int i = 0; i < particles_per_tile; i++) {
      // Start a for loop with Random Number evolution
      int pidx = old_size + i;

      // Generate a random position
      const amrex::Real ratio[AMREX_SPACEDIM] = {
          (std::abs(p_hi[0] - p_lo[0]) - 1.4) * 0.5 * amrex::Random() + 0.7,
          amrex::Random() * M_PI, amrex::Random() * 2. * M_PI};
      // M_PI / 2., 0.0};

      typename ParticleContainerClass::ParticleType &p = p_struct[pidx];

      p.id() = ParticleContainerClass::ParticleType::NextID();
      p.cpu() = amrex::ParallelDescriptor::MyProc();

      p.pos(0) = ratio[0] * std::sin(ratio[1]) * std::cos(ratio[2]);
      p.pos(1) = ratio[0] * std::sin(ratio[1]) * std::sin(ratio[2]);
      p.pos(2) = ratio[0] * std::cos(ratio[1]);

      // Create the particle and add it to the container
      arrdata[StructType::vx][pidx] = 0.0;
      arrdata[StructType::vy][pidx] = 0.0;
      arrdata[StructType::vz][pidx] = 0.0;
      arrdata[StructType::ln_E][pidx] = 0;
    }
    current_tile++;
  }

  pc.Redistribute();

  // Iterating over all the tiles of the particle data structure
  for (amrex::MFIter mfi = pc.MakeMFIter(lev); mfi.isValid(); ++mfi) {
    // Get a reference to the particles
    auto &particles = pc.GetParticles(lev);
    auto &particle_tile = pc.DefineAndReturnParticleTile(lev, mfi);

    // Determines the current size and the required new size
    auto current_size = particle_tile.GetArrayOfStructs().size();

    // Gets raw pointers to the two different ways particle data is stored for
    // performance reasons: Array of Struct (AoS) and Struct of Arrays (SoA)
    typename ParticleContainerClass::ParticleType *p_struct =
        particle_tile.GetArrayOfStructs()().data();
    auto arrdata = particle_tile.GetStructOfArrays().realarray();

    // get the current process id
    auto const metric_array = metric.array(mfi);

    for (int i = 0; i < current_size; i++) {

      if (!(arrdata[StructType::vx][i] == 0.0 &&
            arrdata[StructType::vy][i] == 0.0 &&
            arrdata[StructType::vz][i] == 0.0)) {
        continue;
      }

      // Start a for loop with Random Number evolution for the velocity
      const amrex::Real ratio[AMREX_SPACEDIM] = {2.0 * amrex::Random() - 1.0,
                                                 2.0 * amrex::Random() - 1.0,
                                                 2.0 * amrex::Random() - 1.0};

      // Generate a random position
      typename ParticleContainerClass::ParticleType &p = p_struct[i];
      // const amrex::Real ratio[AMREX_SPACEDIM] = {1.0, 0.0, 0.0};

      const int i0 = amrex::Math::floor((p.pos(0) - p_lo[0]) / dx[0]);
      const int j0 = amrex::Math::floor((p.pos(1) - p_lo[1]) / dx[1]);
      const int k0 = amrex::Math::floor((p.pos(2) - p_lo[2]) / dx[2]);

      // Interpolate metric
      const amrex::GpuArray<CCTK_REAL, 6> gamma_x = {
          barycentric_cubic_3d<5>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo, 0), // g_11
          barycentric_cubic_3d<5>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo, 1), // g_12 & g_21
          barycentric_cubic_3d<5>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo, 2), // g_13 & g_31
          barycentric_cubic_3d<5>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo, 3), // g_22
          barycentric_cubic_3d<5>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo, 4), // g_23, g_32
          barycentric_cubic_3d<5>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo, 5)}; // g_33

      const CCTK_REAL inv_det_gamma =
          1.0 / (gamma_x[0] * gamma_x[3] * gamma_x[5] +
                 2. * gamma_x[1] * gamma_x[2] * gamma_x[4] -
                 gamma_x[2] * gamma_x[2] * gamma_x[3] -
                 gamma_x[4] * gamma_x[4] * gamma_x[0] -
                 gamma_x[1] * gamma_x[1] * gamma_x[5]);

      const amrex::GpuArray<CCTK_REAL, 6> gamma_inv_x = {
          (gamma_x[3] * gamma_x[5] - gamma_x[4] * gamma_x[4]) * inv_det_gamma,
          (gamma_x[4] * gamma_x[2] - gamma_x[1] * gamma_x[5]) * inv_det_gamma,
          (gamma_x[1] * gamma_x[4] - gamma_x[2] * gamma_x[3]) * inv_det_gamma,
          (gamma_x[0] * gamma_x[5] - gamma_x[2] * gamma_x[2]) * inv_det_gamma,
          (gamma_x[2] * gamma_x[1] - gamma_x[0] * gamma_x[4]) * inv_det_gamma,
          (gamma_x[0] * gamma_x[3] - gamma_x[1] * gamma_x[1]) * inv_det_gamma};

      // Normalizing the velocity.
      const CCTK_REAL v_squared = ratio[0] * ratio[0] * gamma_inv_x[0] +
                                  ratio[1] * ratio[1] * gamma_inv_x[3] +
                                  ratio[2] * ratio[2] * gamma_inv_x[5] +
                                  2.0 * ratio[0] * ratio[1] * gamma_inv_x[1] +
                                  2.0 * ratio[0] * ratio[2] * gamma_inv_x[2] +
                                  2.0 * ratio[1] * ratio[2] * gamma_inv_x[4];

      const CCTK_REAL v = std::sqrt(v_squared);

      // Create the particle and add it to the container
      arrdata[StructType::vx][i] = ratio[0] / v;
      arrdata[StructType::vy][i] = ratio[1] / v;
      arrdata[StructType::vz][i] = ratio[2] / v;
      arrdata[StructType::ln_E][i] = 0;
    }
  }

  pc.SortParticlesByCell();

  CCTK_VINFO("%d particles created", pc.TotalNumberOfParticles());

} // random_parallel_photons_initializer

} // namespace photons_init

#endif // !PHOTON_INITIALIZERS
