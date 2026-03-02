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
#include "Interpolator.hxx"

using namespace GInX;

/**
 * \brief This function random initializes particles spherically distributed
 * around a sphere of radius real_params[0].
 *
 * @param pc The particle container that is going to be initialized.
 * cell.
 * @param real_params Double type array that contains the real parameters needed
 * to initialize the particles.
 * @param int_params Integer type array that contains the integer parameters
 * needed to initialize the particles.
 */
template <typename StructType, typename ParticleContainerClass>
void random_spherical_initializer(ParticleContainerClass &pc,
                                  const CCTK_REAL *real_params,
                                  const CCTK_INT *int_params) {

  CCTK_INFO("Initializing particles using the "
            "random_spherical_photons_per_container_initializer");

  const CCTK_INT level = 0;
  const CCTK_REAL radius = real_params[0];

  // Get the with of the discretization on each direction.
  const auto dx = pc.Geom(level).CellSizeArray();

  // Get the lower and higher value over the ParticleContainer Geometry
  const auto p_lo = pc.Geom(level).ProbLoArray();
  const auto p_hi = pc.Geom(level).ProbHiArray();

  const int n_procs = amrex::ParallelDescriptor::NProcs();
  const int proc_id = amrex::ParallelDescriptor::MyProc();

  const int total_particles = int_params[0];
  const int local_particles_size =
      total_particles / n_procs + (proc_id < total_particles % n_procs);

  int total_tiles{0};
  for (amrex::MFIter mfi = pc.MakeMFIter(level); mfi.isValid(); ++mfi) {
    total_tiles++;
  }

  int current_tile = 0;

  // Iterating over all the tiles of the particle data structure
  for (amrex::MFIter mfi = pc.MakeMFIter(level); mfi.isValid(); ++mfi) {

    const unsigned int particles_per_tile =
        local_particles_size / total_tiles +
        (current_tile < local_particles_size % total_tiles);

    // get each tile box
    const amrex::Box &tile_box = mfi.tilebox();

    // Get the tile bounds
    const auto lo = amrex::lbound(tile_box);
    const auto hi = amrex::ubound(tile_box);
    // Get a reference to the particles
    auto &particles = pc.GetParticles(level);
    auto &particle_tile = pc.DefineAndReturnParticleTile(level, mfi);

    // Determines the current size and the required new size
    auto old_size = particle_tile.GetArrayOfStructs().size();

    // Resize the container once, we do not need to do it one by one
    auto new_size = old_size + particles_per_tile;
    particle_tile.resize(new_size);

    // Gets raw pointers to the two different ways particle data is stored for
    // performance reasons: Array of Struct (AoS) and Struct of Arrays (SoA)
    auto *p_struct = particle_tile.GetArrayOfStructs()().data();
    auto arrdata = particle_tile.GetStructOfArrays().realarray();

    std::vector<long> particle_ids(particles_per_tile);
    for (int i = 0; i < particles_per_tile; ++i) {
      particle_ids[i] = ParticleContainerClass::ParticleType::NextID();
    }

    // Copy IDs to GPU (AMReX handles this automatically if needed)
    amrex::Gpu::DeviceVector<long> d_particle_ids(particle_ids.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, particle_ids.begin(),
                          particle_ids.end(), d_particle_ids.begin());
    const long *id_ptr = d_particle_ids.data();

    amrex::ParallelForRNG(
        particles_per_tile,
        [=] AMREX_GPU_DEVICE(int i,
                             amrex::RandomEngine const &engine) noexcept {
          // Start a for loop with Random Number evolution
          int pidx = old_size + i;

          // Generate a random position
          const amrex::Real ratio[AMREX_SPACEDIM] = {
              (std::abs(p_hi[0] - p_lo[0]) - (radius) * 2.) * 0.5 *
                      Random(engine) +
                  radius,
              Random(engine) * M_PI, Random(engine) * 2. * M_PI};

          auto &p = p_struct[pidx];

          p.id() = id_ptr[i];
          p.cpu() = proc_id;

          p.pos(0) = ratio[0] * std::sin(ratio[1]) * std::cos(ratio[2]);
          p.pos(1) = ratio[0] * std::sin(ratio[1]) * std::sin(ratio[2]);
          p.pos(2) = ratio[0] * std::cos(ratio[1]);

          // Create the particle and add it to the container
          arrdata[StructType::vx][pidx] = 2. * Random(engine) - 1.0;
          arrdata[StructType::vy][pidx] = 2. * Random(engine) - 1.0;
          arrdata[StructType::vz][pidx] = 2. * Random(engine) - 1.0;
          arrdata[StructType::ln_E][pidx] = 0;
        });
    CCTK_VWarn(1, __LINE__, __FILE__, CCTK_THORNSTRING,
               "Number of particles created at tile %d: %d ", current_tile,
               particles_per_tile);
    current_tile++;
  }

  pc.Redistribute();
  pc.SortParticlesByCell();

  CCTK_VINFO("%d particles created", pc.TotalNumberOfParticles());

} // random_spherical_initializer

/**
 * \brief This function random initializes particles uniformly distributed 
 * in a box avoiding a sphere of radius real_params[0].
 *
 * @param pc The particle container that is going to be initialized.
 * @param real_params Double type array that contains the real parameters needed
 * to initialize the particles.
 * @param int_params Integer type array that contains the integer parameters
 * needed to initialize the particles.
 */
template <typename StructType, typename ParticleContainerClass>
void random_uniform_initializer(ParticleContainerClass &pc,
                                const CCTK_REAL *real_params,
                                const CCTK_INT *int_params) {

  CCTK_INFO("Initializing particles using the "
            "random_uniform_photons_per_container_initializer");

  const CCTK_INT level = 0;
  const CCTK_REAL radius = real_params[0];

  // Get the lower and higher value over the ParticleContainer Geometry
  const auto p_lo = pc.Geom(level).ProbLoArray();
  const auto p_hi = pc.Geom(level).ProbHiArray();

  const int n_procs = amrex::ParallelDescriptor::NProcs();
  const int proc_id = amrex::ParallelDescriptor::MyProc();

  const int total_particles = int_params[0];
  const int local_particles_size =
      total_particles / n_procs + (proc_id < total_particles % n_procs);

  int total_tiles{0};
  for (amrex::MFIter mfi = pc.MakeMFIter(level); mfi.isValid(); ++mfi) {
    total_tiles++;
  }

  int current_tile = 0;

  // Iterating over all the tiles of the particle data structure
  for (amrex::MFIter mfi = pc.MakeMFIter(level); mfi.isValid(); ++mfi) {

    const unsigned int particles_per_tile =
        local_particles_size / total_tiles +
        (current_tile < local_particles_size % total_tiles);

    // get each tile box
    const amrex::Box &tile_box = mfi.tilebox();

    // Get the tile bounds
    const auto lo = amrex::lbound(tile_box);
    const auto hi = amrex::ubound(tile_box);
    // Get a reference to the particles
    auto &particles = pc.GetParticles(level);
    auto &particle_tile = pc.DefineAndReturnParticleTile(level, mfi);

    // Determines the current size and the required new size
    auto old_size = particle_tile.GetArrayOfStructs().size();

    // Resize the container once, we do not need to do it one by one
    auto new_size = old_size + particles_per_tile;
    particle_tile.resize(new_size);

    // Gets raw pointers to the two different ways particle data is stored for
    // performance reasons: Array of Struct (AoS) and Struct of Arrays (SoA)
    auto *p_struct = particle_tile.GetArrayOfStructs()().data();
    auto arrdata = particle_tile.GetStructOfArrays().realarray();

    std::vector<long> particle_ids(particles_per_tile);
    for (int i = 0; i < particles_per_tile; ++i) {
      particle_ids[i] = ParticleContainerClass::ParticleType::NextID();
    }

    // Copy IDs to GPU (AMReX handles this automatically if needed)
    amrex::Gpu::DeviceVector<long> d_particle_ids(particle_ids.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, particle_ids.begin(),
                          particle_ids.end(), d_particle_ids.begin());
    const long *id_ptr = d_particle_ids.data();

    amrex::ParallelForRNG(
        particles_per_tile,
        [=] AMREX_GPU_DEVICE(int i,
                             amrex::RandomEngine const &engine) noexcept {
          // Start a for loop with Random Number evolution
          int pidx = old_size + i;
          auto &p = p_struct[pidx];

          p.id() = id_ptr[i];
          p.cpu() = proc_id;

          do {
            p.pos(0) = Random(engine) * (p_hi[0] - p_lo[0]) + p_lo[0];
            p.pos(1) = Random(engine) * (p_hi[1] - p_lo[1]) + p_lo[1];
            p.pos(2) = Random(engine) * (p_hi[2] - p_lo[2]) + p_lo[2];
          } while ((p.pos(0) * p.pos(0) + p.pos(1) * p.pos(1) +
                    p.pos(2) * p.pos(2)) <
                   (radius) * (radius));
                
          // Create the particle and add it to the container
          arrdata[StructType::vx][pidx] = 2. * Random(engine) - 1.0;
          arrdata[StructType::vy][pidx] = 2. * Random(engine) - 1.0;
          arrdata[StructType::vz][pidx] = 2. * Random(engine) - 1.0;
          arrdata[StructType::ln_E][pidx] = 0;
        });
    CCTK_VWarn(1, __LINE__, __FILE__, CCTK_THORNSTRING,
               "Number of particles created at tile %d: %d ", current_tile,
               particles_per_tile);
    current_tile++;
  }

  pc.Redistribute();
  pc.SortParticlesByCell();

  CCTK_VINFO("%d particles created", pc.TotalNumberOfParticles());

} // random_uniform_initializer

#endif // !PHOTON_INITIALIZERS
