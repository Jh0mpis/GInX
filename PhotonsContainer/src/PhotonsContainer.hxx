/**
 * \file PhotonsContainer.hxx
 * \brief PhotonsContainer class definition.
 *
 * The following file contains the definition of the PhotonsContainer class and
 * its methods, the class inherits from the abstract BaseContainer class. On
 * this class we have defined the evolution of the particles quantities involved
 * on the dynamics of the photons.
 */

#ifndef PHOTONSCONTAINER_HXX
#define PHOTONSCONTAINER_HXX

// Import libraries
#include <cctk.h>

#include "AMReX_Array.H"
#include "AMReX_CTOParallelForImpl.H"
#include "AMReX_ParallelDescriptor.H"
#include "ConcreteContainer.hxx"
#include "Interpolator.hxx"
#include "Utilities.hxx"
#include "cctk_Types.h"
#include "cctk_core.h"
#include <AMReX_AmrParticles.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Particles.H>
#include <AMReX_REAL.H>
#include <cassert>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace Containers {

// #############################################################################
//                   PhotonsContainer::CLASS INITIALIZATION
// #############################################################################

using namespace Interpolator;
using namespace GInX;

/**
 * \brief PhotonsContainer class definition.
 *
 * The following class define the needed functions to evolve the position and
 * velocity of the photons in the simulation.
 */
template <typename StructType>
class PhotonsContainer
    : public ConcreteContainer<PhotonsContainer<StructType>, StructType> {

public:
  /**
   * \brief Using BaseParticlesContainer constructor
   */
  using Concrete = ConcreteContainer<PhotonsContainer<StructType>, StructType>;
  using Concrete::Concrete;

  ~PhotonsContainer() = default;

  // Computes and normalize the velocity
  void normalize_velocity(const amrex::MultiFab &metric,
                                  const int level) override;

  void redistribute_particles();
}; // PhotonsContainer class

// ##############################################################################
//                   PhotonsContainer::METHODS DECLARATION
// ##############################################################################

/**
 * \brief Normalize the velocity accordingly to the metric on each particle
 * position.
 *
 * This function is made to normalize the velocity given a random initial data
 * using the Photons positions by using that
 *
 *  \f[
 *  P^\mu P_\mu = 0.
 *  \f]
 *
 *  or equivalently
 *
 *  \f[
 *  V^\alpha V_\alpha = V_\alpha V_\beta \gamma^{\alpha\beta} = 1.
 *  \f]
 *
 * @param metric ADM 3 dimension metric.
 * @param Current refinement level.
 */
template <typename StructType>
void PhotonsContainer<StructType>::normalize_velocity(
    const amrex::MultiFab &metric, const int level) {

  // Get the with of the discretization on each direction.
  const auto dx = this->Geom(level).CellSizeArray();
  // Get the lower and higher value over the ParticleContainer Geometry
  const auto p_lo = this->Geom(level).ProbLoArray();
  const auto p_hi = this->Geom(level).ProbHiArray();

  for (amrex::MFIter mfi = this->MakeMFIter(level); mfi.isValid(); ++mfi) {

    // Get a reference to the particles
    auto &particle_tile = this->DefineAndReturnParticleTile(level, mfi);

    // Determines the current size and the required new size
    const auto current_size = particle_tile.GetArrayOfStructs().size();

    // Gets raw pointers to the two different ways particle data is stored for
    // performance reasons: Array of Struct (AoS) and Struct of Arrays (SoA)
    auto *p_struct = particle_tile.GetArrayOfStructs()().data();
    auto arrdata = particle_tile.GetStructOfArrays().realarray();

    // get the current process id
    const auto metric_array = metric.array(mfi);

    amrex::ParallelFor(current_size, [=] AMREX_GPU_DEVICE(int i) noexcept {
      // Start a for loop with Random Number evolution for the velocity
      const amrex::Real ratio[AMREX_SPACEDIM] = {arrdata[StructType::vx][i],
                                                 arrdata[StructType::vy][i],
                                                 arrdata[StructType::vz][i]};

      // Generate a random position
      const auto &p = p_struct[i];

      const int i0 = amrex::Math::floor((p.pos(0) - p_lo[0]) / dx[0]);
      const int j0 = amrex::Math::floor((p.pos(1) - p_lo[1]) / dx[1]);
      const int k0 = amrex::Math::floor((p.pos(2) - p_lo[2]) / dx[2]);

      // Interpolate metric
      amrex::GpuArray<CCTK_REAL, 6> gamma_x;
      interpolate_array<5>(gamma_x, metric_array, i0, j0, k0, p.pos(0),
                           p.pos(1), p.pos(2), dx, p_lo);

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

      arrdata[StructType::vx][i] = ratio[0] / v;
      arrdata[StructType::vy][i] = ratio[1] / v;
      arrdata[StructType::vz][i] = ratio[2] / v;
    });
  }
} // PhotonsContainer::normalize_velocity

template <typename StructType>
void PhotonsContainer<StructType>::redistribute_particles() {
  CCTK_INFO("Redistributing particles");
} // PhotonsContainer::redistribute_particles

} // namespace Containers

#endif // !PHOTONSCONTAINER_HXX
