/**
 * @file Initializers.hxx
 *
 * @brief Includes some Initializers functions useful for different initial
 * conditions.
 *
 * On this file different functions that initialize the particles in different
 * ways. Al the functions defined here are templated, so any <Particle>Container
 * class can use this functions (if the struct is well defined). If you want to
 * create a custom initialization this should be templated in the same way.
 */
#ifndef INITIALIZERS_H
#define INITIALIZERS_H

#include <cctk.h>

#include "AMReX_Box.H"
#include "AMReX_Config.H"
#include "AMReX_MFIter.H"
#include "AMReX_REAL.H"
#include "AMReX_Random.H"
#include "AMReX_RandomEngine.H"
#include "AMReX_Scan.H"
#include <array>
#include <iostream>

namespace Initializer {

/**
 * Initialize the particles inside of a sphere with random velocities.
 *
 * @param pc A <Particle>Container instance that extends from the
 * BaseParticlesContainer class defined inside of BaseParticlesContainer.hxx.
 * @param nppc A vector with the number of particles per cell on each directions
 * depending on the AMREX_SPACEDIM.
 */
template <typename StructType, typename ParticleContainerClass>
void spherical_initializer(ParticleContainerClass &pc,
                           const std::array<int, AMREX_SPACEDIM> nppc) {

  // Get the total number of particles depending on the dimensions.
#if (AMREX_SPACEDIM == 1)
  const int num_ppc = nppc[0];
#elif (AMREX_SPACEDIM == 2)
  const int num_ppc = nppc[0] * nppc[1];
#elif (AMREX_SPACEDIM == 3)
  const int num_ppc = nppc[0] * nppc[1] * nppc[2];
#endif

  // Print information message.
  CCTK_INFO("Initializing particles using the spherical_initializer");

  // Data refinement level
  const int lev = 0;

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

    // Define the GPU vectors counts and offsets, this allow us to push new
    // particles into the Particle containers
    //  Number of items per tile
    amrex::Gpu::ManagedVector<unsigned int> counts(tile_box.numPts(), 0);
    unsigned int *p_counts = counts.dataPtr();
    // where each tile should start writing a global array
    amrex::Gpu::ManagedVector<unsigned int> offsets(tile_box.numPts(), 0);
    unsigned int *p_offsets = offsets.dataPtr();

    // Counting pass: figure out exactly how many particles need to be created
    // in each grid cell
    amrex::ParallelFor(
        tile_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          // Select the cells that are inside of a radius lower or equal to 1.0
          amrex::Real xc = p_lo[0] + (i + 0.5) * dx[0];
          amrex::Real yc{0.};
          amrex::Real zc{0.};
#if (AMREX_SPACEDIM == 2)
          yc = p_lo[1] + (j + 0.5) * dx[1];
#elif (AMREX_SPACEDIM == 3)
          yc = p_lo[1] + (j + 0.5) * dx[1];
          zc = p_lo[2] + (k + 0.5) * dx[2];
#endif
          // Compute the distance to the center of the sphere
          amrex::Real rc = std::sqrt(xc * xc + yc * yc + zc * zc);

          if (rc > 1.0)
            return;

          // If the cell is inside we can add num_ppc particles to that cell
          for (int i_part = 0; i_part < num_ppc; i_part++) {
            // Calculates a unique 1D index (cellid) from the 3D cell index
            // (i, j, k). This maps the 3D grid cell to a 1D memory location
            // in the counts array.
            int ix = i - lo.x;
            int iy{0}, iz{0};

            int nx = hi.x - lo.x + 1;
            int ny{0}, nz{0};

            unsigned int uix = amrex::min(nx - 1, amrex::max(0, ix));
            unsigned int uiy{0}, uiz{0};

#if (AMREX_SPACEDIM == 2)
            iy = j - lo.y;
            ny = hi.y - lo.y + 1;
            uiy = amrex::min(ny - 1, amrex::max(0, iy));
#elif (AMREX_SPACEDIM == 3)
            iy = j - lo.y;
            ny = hi.y - lo.y + 1;
            uiy = amrex::min(ny - 1, amrex::max(0, iy));
            iz = k - lo.z;
            nz = hi.z - lo.z + 1;
            uiz = amrex::min(nz - 1, amrex::max(0, iz));
#endif

            // compute the cell_id in 1D
            unsigned int cell_id = (uix * ny + uiy) * nz + uiz;
            // So we can just save it in 1D vectors
            p_counts[cell_id] += 1;
          }
        });

    // Compute the offsets of the global particle array
    amrex::Gpu::exclusive_scan(counts.begin(), counts.end(), offsets.begin());

    // Then we can compute the number of particles to add by using the offsets
    // and the counts
    int num_to_add =
        offsets[tile_box.numPts() - 1] + counts[tile_box.numPts() - 1];

    // Nothing to do
    if (num_to_add == 0)
      continue;

    auto &particles = pc.GetParticles(lev);
    auto &particle_tile =
        particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

    // Determines the current size and the required new size
    auto old_size = particle_tile.GetArrayOfStructs().size();
    auto new_size = old_size + num_to_add;

    // Resize the container once, we don't need to do it one by one
    particle_tile.resize(new_size);

    // Gets raw pointers to the two different ways particle data is stored for
    // performance reasons: Array of Struct (AoS) and Struct of Arrays (SoA)
    typename ParticleContainerClass::ParticleType *p_struct =
        particle_tile.GetArrayOfStructs()().data();
    auto arrdata = particle_tile.GetStructOfArrays().realarray();

    // get the current process id
    int proc_id = amrex::ParallelDescriptor::MyProc();

    // Start a for loop with Random Number evolution
    amrex::ParallelForRNG(
        tile_box,
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                             amrex::RandomEngine const &engine) noexcept {
          // get the center of the cell and compute the distance to the center
          // of the mesh grid.
          amrex::Real xc = p_lo[0] + (i + 0.5) * dx[0];
          amrex::Real yc{0.};
          amrex::Real zc{0.};
#if (AMREX_SPACEDIM == 2)
          yc = p_lo[1] + (j + 0.5) * dx[1];
#elif (AMREX_SPACEDIM == 3)
          yc = p_lo[1] + (j + 0.5) * dx[1];
          zc = p_lo[2] + (k + 0.5) * dx[2];
#endif

          amrex::Real rc = std::sqrt(xc * xc + yc * yc + zc * zc);

          // if the cell is farther than 1.0, return
          if (rc > 1.0)
            return;

          // Calculate cell_id
          int ix = i - lo.x;
          int iy{0}, iz{0};

          int nx = hi.x - lo.x + 1;
          int ny{0}, nz{0};

          unsigned int uix = amrex::min(nx - 1, amrex::max(0, ix));
          unsigned int uiy{0}, uiz{0};

#if (AMREX_SPACEDIM == 2)
          iy = j - lo.y;
          ny = hi.y - lo.y + 1;
          uiy = amrex::min(ny - 1, amrex::max(0, iy));
#elif (AMREX_SPACEDIM == 3)
          iy = j - lo.y;
          ny = hi.y - lo.y + 1;
          uiy = amrex::min(ny - 1, amrex::max(0, iy));
          iz = k - lo.z;
          nz = hi.z - lo.z + 1;
          uiz = amrex::min(nz - 1, amrex::max(0, iz));
#endif

          unsigned int cell_id = (uix * ny + uiy) * nz + uiz;

          // Retrievers the starting write index (pidx) for the current cell
          // (i, j, k) from the offsets array that was calculated by the
          // exclusive_scan
          int pidx = old_size + p_offsets[cell_id];

          for (int i_part = 0; i_part < num_ppc; i_part++) {
            amrex::Real ratio[AMREX_SPACEDIM];

            // Generate a random position
            ratio[0] = amrex::Random(engine);
            amrex::Real x = p_lo[0] + (i + ratio[0]) * dx[0];

#if (AMREX_SPACEDIM == 2)
            ratio[1] = amrex::Random(engine);
            amrex::Real y = p_lo[1] + (j + ratio[1]) * dx[1];
#elif (AMREX_SPACEDIM == 3)
            ratio[1] = amrex::Random(engine);
            ratio[2] = amrex::Random(engine);
            amrex::Real y = p_lo[1] + (j + ratio[1]) * dx[1];
            amrex::Real z = p_lo[2] + (k + ratio[2]) * dx[2];
#endif

            // Compute a random initial momentum taking care of more than one
            // dimensions
            const amrex::Real pt = 1.0;

            amrex::Real costh = AMREX_SPACEDIM > 1 ? Random(engine) * 2 - 1 : 1;
            amrex::Real ph = Random(engine) * (2 * M_PI);
            amrex::Real sinth =
                std::sqrt(amrex::max(amrex::Real(0), 1 - costh * costh));
            amrex::Real cosph = AMREX_SPACEDIM > 2 ? std::cos(ph) : 1;
            amrex::Real sinph = AMREX_SPACEDIM > 2 ? std::sin(ph) : 1;

            // Create the particle and add it to the container
            typename ParticleContainerClass::ParticleType &p = p_struct[pidx];
            p.id() = pidx + 1;
            p.cpu() = proc_id;
            p.pos(0) = x;

            arrdata[StructType::vx][pidx] = pt * sinth * cosph;
#if (AMREX_SPACEDIM == 2)
            p.pos(1) = y;
            arrdata[StructType::vy][pidx] = pt * sinth * sinph;
#elif (AMREX_SPACEDIM == 3)
            p.pos(1) = y;
            p.pos(2) = z;
            arrdata[StructType::vy][pidx] = pt * sinth * sinph;
            arrdata[StructType::vz][pidx] = pt * costh;
#endif

            // Update the particles counter
            ++pidx;
          }
        });
  }

} // Function spherical_initializer

template <typename StructType, typename ParticleContainerClass>
void random_initializer(ParticleContainerClass &pc,
                        const std::array<int, AMREX_SPACEDIM> nppc) {

  // Get the total number of particles depending on the dimensions.
#if (AMREX_SPACEDIM == 1)
  const int num_ppc = nppc[0];
#elif (AMREX_SPACEDIM == 2)
  const int num_ppc = nppc[0] * nppc[1];
#elif (AMREX_SPACEDIM == 3)
  const int num_ppc = nppc[0] * nppc[1] * nppc[2];
#endif

  // Print information message.
  CCTK_INFO("Initializing particles using the random_initializer");

  // Data refinement level
  const int lev = 0;

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
    auto new_size = old_size + num_ppc * (hi.x - lo.x + 1) * (hi.z - lo.z + 1) *
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

    // Start a for loop with Random Number evolution
    amrex::ParallelForRNG(
        tile_box,
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                             amrex::RandomEngine const &engine) noexcept {
          // Calculate cell_id
          int ix = i - lo.x;
          int iy{0}, iz{0};

          int nx = hi.x - lo.x + 1;
          int ny{0}, nz{0};

          unsigned int uix = amrex::min(nx - 1, amrex::max(0, ix));
          unsigned int uiy{0}, uiz{0};

#if (AMREX_SPACEDIM == 2)
          iy = j - lo.y;
          ny = hi.y - lo.y + 1;
          uiy = amrex::min(ny - 1, amrex::max(0, iy));
#elif (AMREX_SPACEDIM == 3)
          iy = j - lo.y;
          ny = hi.y - lo.y + 1;
          uiy = amrex::min(ny - 1, amrex::max(0, iy));
          iz = k - lo.z;
          nz = hi.z - lo.z + 1;
          uiz = amrex::min(nz - 1, amrex::max(0, iz));
#endif

          unsigned int cell_id = (uix * ny + uiy) * nz + uiz;

          // Retrievers the starting write index (pidx) for the current cell
          // (i, j, k) from the offsets array that was calculated by the
          // exclusive_scan
          int pidx = old_size + num_ppc * cell_id;

          for (int i_part = 0; i_part < num_ppc; i_part++) {
            amrex::Real ratio[AMREX_SPACEDIM];

            // Generate a random position
            ratio[0] = amrex::Random(engine);
            amrex::Real x = p_lo[0] + (i + ratio[0]) * dx[0];

#if (AMREX_SPACEDIM == 2)
            ratio[1] = amrex::Random(engine);
            amrex::Real y = p_lo[1] + (j + ratio[1]) * dx[1];
#elif (AMREX_SPACEDIM == 3)
            ratio[1] = amrex::Random(engine);
            ratio[2] = amrex::Random(engine);
            amrex::Real y = p_lo[1] + (j + ratio[1]) * dx[1];
            amrex::Real z = p_lo[2] + (k + ratio[2]) * dx[2];
#endif

            // Compute a random initial momentum taking care of more than one
            // dimensions
            const amrex::Real pt = 1.0;

            amrex::Real costh = AMREX_SPACEDIM > 1 ? Random(engine) * 2 - 1 : 1;
            amrex::Real ph = Random(engine) * (2 * M_PI);
            amrex::Real sinth =
                std::sqrt(amrex::max(amrex::Real(0), 1 - costh * costh));
            amrex::Real cosph = AMREX_SPACEDIM > 2 ? std::cos(ph) : 1;
            amrex::Real sinph = AMREX_SPACEDIM > 2 ? std::sin(ph) : 1;

            // Create the particle and add it to the container
            typename ParticleContainerClass::ParticleType &p = p_struct[pidx];
            p.id() = pidx + 1;
            p.cpu() = proc_id;
            p.pos(0) = x;

            arrdata[StructType::vx][pidx] = pt * sinth * cosph;
#if (AMREX_SPACEDIM == 2)
            p.pos(1) = y;
            arrdata[StructType::vy][pidx] = pt * sinth * sinph;
#elif (AMREX_SPACEDIM == 3)
            p.pos(1) = y;
            p.pos(2) = z;
            arrdata[StructType::vy][pidx] = pt * sinth * sinph;
            arrdata[StructType::vz][pidx] = pt * costh;
#endif

            // Update the particles counter
            ++pidx;
          }
        });
    std::cout << "PARTICLES CREATED: "
              << particle_tile.GetArrayOfStructs().size() << std::endl;
  }

} // Function random_initializer
} // namespace Initializer
#endif // !INITIALIZERS_H
