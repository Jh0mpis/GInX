#ifndef PHOTON_INITIALIZERS
#define PHOTON_INITIALIZERS

#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_MFIter.H>
#include <AMReX_REAL.H>
#include <AMReX_Random.H>
#include <AMReX_RandomEngine.H>
#include <AMReX_Scan.H>
#include <cctk.h>
#include <cmath>

#include "Interpolator.hxx"

namespace photons_init {

using namespace Interpolator;

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
  std::cout <<"\n\n\n" << p_hi[0] << " " << p_hi[1] << " " << p_hi[2] << "\n\n\n";

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
        ratio[0] = (std::abs(p_hi[0] - p_lo[0])*0.5 - 0.7) * amrex::Random(engine) + 0.7;
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
            barycentric_cubic_3d<3>(metric_array, i0, j0, k0, p.pos(0),
                                    p.pos(1), p.pos(2), dx, p_lo,
                                    0), // g_11
            barycentric_cubic_3d<3>(metric_array, i0, j0, k0, p.pos(0),
                                    p.pos(1), p.pos(2), dx, p_lo,
                                    1), // g_12 & g_21
            barycentric_cubic_3d<3>(metric_array, i0, j0, k0, p.pos(0),
                                    p.pos(1), p.pos(2), dx, p_lo,
                                    2), // g_13 & g_31
            barycentric_cubic_3d<3>(metric_array, i0, j0, k0, p.pos(0),
                                    p.pos(1), p.pos(2), dx, p_lo,
                                    3), // g_22
            barycentric_cubic_3d<3>(metric_array, i0, j0, k0, p.pos(0),
                                    p.pos(1), p.pos(2), dx, p_lo,
                                    4), // g_23, g_32
            barycentric_cubic_3d<3>(metric_array, i0, j0, k0, p.pos(0),
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

        // Compute a random initial momentum taking care of more than one
        // dimensions
        ratio[0] = 2.0 * amrex::Random(engine) - 1.0;
        ratio[1] = 2.0 * amrex::Random(engine) - 1.0;
        ratio[2] = 2.0 * amrex::Random(engine) - 1.0;

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

} // namespace photons_init

#endif // !PHOTON_INITIALIZERS
