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
#include "AMReX_GpuLaunchFunctsC.H"
#include "AMReX_ParallelDescriptor.H"
#include "BaseParticleContainer.hxx"
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

using namespace BaseContainer;
using namespace Interpolator;

/**
 * \brief PhotonsContainer class definition.
 *
 * The following class define the needed functions to evolve the position and
 * velocity of the photons in the simmulation.
 */
template <typename StructType>
class PhotonsContainer
    : public BaseParticleContainer<PhotonsContainer<StructType>, StructType> {

public:
  /**
   * \brief Using BaseParticlesContainer constructor
   */
  using Base = BaseParticleContainer<PhotonsContainer<StructType>, StructType>;
  using Base::Base;

  void evolve(const amrex::MultiFab &lapse, const amrex::MultiFab &shift,
              const amrex::MultiFab &metric, const amrex::MultiFab &curv,
              const CCTK_REAL &dt, const int &lev) override;

  // Given differential equation dU/dt = f(U, dU/dx; t) computes f(U, dU/dx;t)
  amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3>
  compute_rhs(const amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3> &u,
              const CCTK_REAL &t, amrex::Array4<CCTK_REAL const> const &lapse,
              amrex::Array4<CCTK_REAL const> const &shift,
              amrex::Array4<CCTK_REAL const> const &metric,
              amrex::Array4<CCTK_REAL const> const &K, const CCTK_REAL dt,
              const amrex::GpuArray<double, 3> &dx, const int lev,
              const amrex::GpuArray<double, 3> &plo);

  // Computes and print all photons velocities
  void check_constants(CCTK_ARGUMENTS, const amrex::MultiFab &metric,
                       const amrex::MultiFab &lapse,
                       const amrex::MultiFab &shift, const int &lev);

  void redistribute_particles();
}; // PhotonsContainer class

// ##############################################################################
//                   PhotonsContainer::METHODS DECLARATION
// ##############################################################################

/**
 * \brief Computes the right hind side of the geodesic differential equation.
 *
 * Given differential equation \f[\frac{d}{dt}U = f\left(U, \frac{dU}{dx};
 * t\right)\f] computes
 * \f[f\left(U, \frac{dU}{dx}; t\right)\f]
 *
 * where \f$U\f$ is a vector that contains \f$(x_u, y_u, z_u, vx_d, vy_d, vz_d,
 * E)\f$. That's why each rhs depends on the other components of the vector
 * \f$U\f$. For the position the differential equation  is:
 *
 * \f[\frac{d}{dt} U[i] = \alpha \gamma^{ij} U[3 + j] - \beta^i\f]
 *
 * Where \f$i,j = 0, 1, 2\f$, \f$\gamma\f$ is the induced metric, \f$\alpha\f$
 * is the lapse function and \f$\beta\f$ is the shift vector.
 *
 * For the Velocity_d the differential equation is:
 *
 * \f{eqnarray*}{
 * \frac{d}{dt}U[3 + i] &= -\partial_i\alpha + \left(\gamma^{kj} U[3 + k]
 * \partial_j\alpha - \alpha K_{jk}\gamma^{jl}\gamma^{km}U[3+l]U[3+m]\right) U[3
 * + i]\\ & +
 * \frac{1}{2}\alpha\gamma^{jl}\gamma^{km}U[3+l]U[3+m]\partial_i\gamma{jk} + U[3
 * + j] \partial_i\beta^j
 * \f}
 *
 * and finally, for the \f$ ln E \f$ the differential equation is:
 *
 * \f[ \dfrac{d}{dt} U[6] = \alpha K_{jk}U[3 + l]U[3 + m]\gamma^{lj}\gamma^{mk}
 * - U[3+l]\gamma^{lj}\partial_j\alpha\f]
 *
 *  Where \f$i, j, k, l, m = 0, 1, 2\f$ and we have been using Einstein
 * notation.
 *
 *  @param u A GpuArray of size n_attributes + the coordinates that contains the
 * varaibles needed to evolve.
 *  @param t Current time t.
 *  @param lapse ADM lapse function.
 *  @param shift Shift vector \beta^i
 *  @param metric 3 dimensional ADM metric.
 *  @param curv Extrinsic curvature.
 *  @param dt Timestep.
 *  @param dx Spacestep
 *  @param lev AMR Level of discretization.
 *  @param plo Physical lower bounds of the whole domain.
 *  @return The right hind side of the differential equation.
 */
template <typename StructType>
amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3>
PhotonsContainer<StructType>::compute_rhs(
    const amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3> &u,
    const CCTK_REAL &t, amrex::Array4<CCTK_REAL const> const &lapse,
    const amrex::Array4<CCTK_REAL const> &shift,
    const amrex::Array4<CCTK_REAL const> &metric,
    const amrex::Array4<CCTK_REAL const> &curv, const CCTK_REAL dt,
    const amrex::GpuArray<double, 3> &dx, const int lev,
    const amrex::GpuArray<double, 3> &plo) {

  amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3> rhs = {
      0., 0., 0., 0., 0., 0., 0.};

  // Compute the cell's index where the particle is at position (u[0], u[1],
  // u[2])
  const unsigned long i0 = amrex::Math::floor((u[0] - plo[0]) / dx[0]);
  const unsigned long j0 = amrex::Math::floor((u[1] - plo[1]) / dx[1]);
  const unsigned long k0 = amrex::Math::floor((u[2] - plo[2]) / dx[2]);

  // Interpolate lapse
  // Interpolate partial lapse
  CCTK_REAL lapse_x;
  amrex::GpuArray<CCTK_REAL, 3> d_lapse_x;
  barycentric_derivative_and_interpolate<5>(lapse_x, d_lapse_x[0], d_lapse_x[1],
                                            d_lapse_x[2], lapse, i0, j0, k0,
                                            u[0], u[1], u[2], dx, plo, 0);

  // Interpolate shift
  // Interpolate partial shift
  amrex::GpuArray<CCTK_REAL, 3> shift_x;
  amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 3>, 3> d_shift_x;
  barycentric_derivative_and_interpolate<5>(
      shift_x[0], d_shift_x[0][0], d_shift_x[1][0], d_shift_x[2][0], shift, i0,
      j0, k0, u[0], u[1], u[2], dx, plo, 0);
  barycentric_derivative_and_interpolate<5>(
      shift_x[1], d_shift_x[0][1], d_shift_x[1][1], d_shift_x[2][1], shift, i0,
      j0, k0, u[0], u[1], u[2], dx, plo, 1);
  barycentric_derivative_and_interpolate<5>(
      shift_x[2], d_shift_x[0][2], d_shift_x[1][2], d_shift_x[2][2], shift, i0,
      j0, k0, u[0], u[1], u[2], dx, plo, 2);

  // Interpolate metric
  // Interpolate partial metric
  amrex::GpuArray<CCTK_REAL, 6> gamma_x;
  amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 6>, 3> d_gamma_x;
  barycentric_derivative_and_interpolate<5>(
      gamma_x[0], d_gamma_x[0][0], d_gamma_x[1][0], d_gamma_x[2][0], metric, i0,
      j0, k0, u[0], u[1], u[2], dx, plo, 0);
  barycentric_derivative_and_interpolate<5>(
      gamma_x[1], d_gamma_x[0][1], d_gamma_x[1][1], d_gamma_x[2][1], metric, i0,
      j0, k0, u[0], u[1], u[2], dx, plo, 1);
  barycentric_derivative_and_interpolate<5>(
      gamma_x[2], d_gamma_x[0][2], d_gamma_x[1][2], d_gamma_x[2][2], metric, i0,
      j0, k0, u[0], u[1], u[2], dx, plo, 2);
  barycentric_derivative_and_interpolate<5>(
      gamma_x[3], d_gamma_x[0][3], d_gamma_x[1][3], d_gamma_x[2][3], metric, i0,
      j0, k0, u[0], u[1], u[2], dx, plo, 3);
  barycentric_derivative_and_interpolate<5>(
      gamma_x[4], d_gamma_x[0][4], d_gamma_x[1][4], d_gamma_x[2][4], metric, i0,
      j0, k0, u[0], u[1], u[2], dx, plo, 4);
  barycentric_derivative_and_interpolate<5>(
      gamma_x[5], d_gamma_x[0][5], d_gamma_x[1][5], d_gamma_x[2][5], metric, i0,
      j0, k0, u[0], u[1], u[2], dx, plo, 5);

  // Interpolate Curvature
  const amrex::GpuArray<CCTK_REAL, 6> curv_x = {
      barycentric_cubic_3d<5>(curv, i0, j0, k0, u[0], u[1], u[2], dx, plo, 0),
      barycentric_cubic_3d<5>(curv, i0, j0, k0, u[0], u[1], u[2], dx, plo, 1),
      barycentric_cubic_3d<5>(curv, i0, j0, k0, u[0], u[1], u[2], dx, plo, 2),
      barycentric_cubic_3d<5>(curv, i0, j0, k0, u[0], u[1], u[2], dx, plo, 3),
      barycentric_cubic_3d<5>(curv, i0, j0, k0, u[0], u[1], u[2], dx, plo, 4),
      barycentric_cubic_3d<5>(curv, i0, j0, k0, u[0], u[1], u[2], dx, plo, 5)};

  // Compute the inverse of the metric.
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

  const amrex::GpuArray<CCTK_REAL, 3> V_down = {u[3], u[4], u[5]};

  // Compute the upper index velocity terms.
  const amrex::GpuArray<CCTK_REAL, 3> V_up = {
      gamma_inv_x[0] * u[3] + gamma_inv_x[1] * u[4] + gamma_inv_x[2] * u[5],
      gamma_inv_x[1] * u[3] + gamma_inv_x[3] * u[4] + gamma_inv_x[4] * u[5],
      gamma_inv_x[2] * u[3] + gamma_inv_x[4] * u[4] + gamma_inv_x[5] * u[5]};

  // Compute the rhs for position
  rhs[0] = lapse_x * V_up[0] - shift_x[0];
  rhs[1] = lapse_x * V_up[1] - shift_x[1];
  rhs[2] = lapse_x * V_up[2] - shift_x[2];

  // Compute the rhs for velocity
  for (int i = 0; i < 3; i++) {
    rhs[3 + i] =
        -d_lapse_x[i] +
        (VecVecMul(d_lapse_x, V_up) -
         lapse_x * VecVecMul(SMatVecMul(curv_x, V_up), V_up)) *
            V_down[i] +
        0.5 * lapse_x * VecVecMul(SMatVecMul(d_gamma_x[i], V_up), V_up) +
        VecVecMul(V_down, d_shift_x[i]);
  }

  // Compute the rhs for energy
  rhs[3 + StructType::ln_E] =
      lapse_x * VecVecMul(SMatVecMul(curv_x, V_up), V_up) -
      VecVecMul(V_up, d_lapse_x);

  return rhs;

} // PhotonsContainer::compute_rhs

/**
 *  \brief Evolving using Runge-Kutta 4.
 *
 *  For the Runge-Kutta 4 we are solving the differential equation
 * \f$\frac{dU}{dt} = f\left(U, \frac{dU}{dx}, t\right)\f$ using:
 *
 *  \f[
 *  U_{n+1} = U_n + \frac{1}{6}\Delta t \left(f_1 + 2f_2 + 2f_3 + f_4\right)
 *  \f]
 *
 *  where:
 *
 *  * \f$f_1 = f(U_n, t),\f$
 *  * \f$f_2 = f\left(U_n + \frac{\Delta t}{2} f_1, t + \frac{\Delta
 * t}{2}\right),\f$
 *  * \f$f_3 = f\left(U_n + \frac{\Delta t}{2} f_2, t + \frac{\Delta
 * t}{2}\right),\f$
 *  * \f$f_4 = f(U_n + \Delta t f_3, t + \Delta t),\f$
 *
 *  And checking if it goes outside of the grid boundaries.
 *
 *  @see compute_rhs()
 *  @param lapse ADM lapse function.
 *  @param shift ADM shift vector.
 *  @param metric ADM 3 dimension metric.
 *  @param curv Extrinsic curvature.
 *  @param dt Timestep.
 *  @param lev AMR Level of discretization.
 */
template <typename StructType>
void PhotonsContainer<StructType>::evolve(const amrex::MultiFab &lapse,
                                          const amrex::MultiFab &shift,
                                          const amrex::MultiFab &metric,
                                          const amrex::MultiFab &curv,
                                          const CCTK_REAL &dt, const int &lev) {

  const auto plo0 = this->Geom(0).ProbLoArray();
  const auto phi0 = this->Geom(0).ProbHiArray();

  const auto dx = this->Geom(lev).CellSizeArray();
  const auto plo = this->Geom(lev).ProbLoArray();

  const CCTK_REAL boundarie_hx = phi0[0] - 0.5 * dx[0];
  const CCTK_REAL boundarie_lx = plo0[0] + 0.5 * dx[0];
  const CCTK_REAL boundarie_hy = phi0[1] - 0.5 * dx[1];
  const CCTK_REAL boundarie_ly = plo0[1] + 0.5 * dx[1];
  const CCTK_REAL boundarie_hz = phi0[2] - 0.5 * dx[2];
  const CCTK_REAL boundarie_lz = plo0[2] + 0.5 * dx[2];

  for (Iterator::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
       ++pti) {

    const int np = pti.numParticles();

    // Get the information relate to the velocities.
    auto &attribs = pti.GetAttributes();
    CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
    CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
    CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
    CCTK_REAL *AMREX_RESTRICT ln_energy = attribs[StructType::ln_E].data();
    auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

    // Get the array of each parameters.
    auto const lapse_array = lapse.array(pti);
    auto const shift_array = shift.array(pti);
    auto const metric_array = metric.array(pti);
    auto const curv_array = curv.array(pti);

    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
      const amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3> U = {
          particles[i].pos(0),         particles[i].pos(1),
          particles[i].pos(2),         attribs[StructType::vx][i],
          attribs[StructType::vy][i],  attribs[StructType::vz][i],
          attribs[StructType::ln_E][i]};

      amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3> U_tmp;
      amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3> partial_sum;

      U_tmp = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      partial_sum = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      // f1 = rhs(u , t) for the runge kutta 4 step
      auto k_odd =
          this->compute_rhs(U, 0.0, lapse_array, shift_array, metric_array,
                            curv_array, dt, dx, lev, plo0);

      U_tmp[0] = U[0] + 0.5 * dt * k_odd[0];
      U_tmp[1] = U[1] + 0.5 * dt * k_odd[1];
      U_tmp[2] = U[2] + 0.5 * dt * k_odd[2];
      U_tmp[3] = U[3] + 0.5 * dt * k_odd[3];
      U_tmp[4] = U[4] + 0.5 * dt * k_odd[4];
      U_tmp[5] = U[5] + 0.5 * dt * k_odd[5];
      U_tmp[6] = U[6] + 0.5 * dt * k_odd[6];

      if ((U_tmp[0] > boundarie_hx) || (U_tmp[0] < boundarie_lx) ||
          (U_tmp[1] > boundarie_hy) || (U_tmp[1] < boundarie_ly) ||
          (U_tmp[2] > boundarie_hz) || (U_tmp[2] < boundarie_lz)) {
        particles[i].id() = -1;
        return;
      }

      // f2 = rhs(u + 0.5 * dt * f1, t) for the runge kutta 4 step
      auto k_even =
          this->compute_rhs(U_tmp, 0.5 * dt, lapse_array, shift_array,
                            metric_array, curv_array, dt, dx, lev, plo0);

      // Update particles with the f1 and f2 from RK4
      U_tmp[0] = U[0] + 0.5 * dt * k_even[0];
      U_tmp[1] = U[1] + 0.5 * dt * k_even[1];
      U_tmp[2] = U[2] + 0.5 * dt * k_even[2];
      U_tmp[3] = U[3] + 0.5 * dt * k_even[3];
      U_tmp[4] = U[4] + 0.5 * dt * k_even[4];
      U_tmp[5] = U[5] + 0.5 * dt * k_even[5];
      U_tmp[6] = U[6] + 0.5 * dt * k_even[6];

      partial_sum[0] = k_odd[0] + 2. * k_even[0];
      partial_sum[1] = k_odd[1] + 2. * k_even[1];
      partial_sum[2] = k_odd[2] + 2. * k_even[2];
      partial_sum[3] = k_odd[3] + 2. * k_even[3];
      partial_sum[4] = k_odd[4] + 2. * k_even[4];
      partial_sum[5] = k_odd[5] + 2. * k_even[5];
      partial_sum[6] = k_odd[6] + 2. * k_even[6];

      if ((U_tmp[0] > boundarie_hx) || (U_tmp[0] < boundarie_lx) ||
          (U_tmp[1] > boundarie_hy) || (U_tmp[1] < boundarie_ly) ||
          (U_tmp[2] > boundarie_hz) || (U_tmp[2] < boundarie_lz)) {
        particles[i].id() = -1;
        return;
      }

      // f3 = rhs(u + 0.5 * dt * f2, t) for the runge kutta 4 step
      k_odd = this->compute_rhs(U_tmp, 0.0, lapse_array, shift_array,
                                metric_array, curv_array, dt, dx, lev, plo0);

      U_tmp[0] = U[0] + dt * k_odd[0];
      U_tmp[1] = U[1] + dt * k_odd[1];
      U_tmp[2] = U[2] + dt * k_odd[2];
      U_tmp[3] = U[3] + dt * k_odd[3];
      U_tmp[4] = U[4] + dt * k_odd[4];
      U_tmp[5] = U[5] + dt * k_odd[5];
      U_tmp[6] = U[6] + dt * k_odd[6];

      if ((U_tmp[0] > boundarie_hx) || (U_tmp[0] < boundarie_lx) ||
          (U_tmp[1] > boundarie_hy) || (U_tmp[1] < boundarie_ly) ||
          (U_tmp[2] > boundarie_hz) || (U_tmp[2] < boundarie_lz)) {
        particles[i].id() = -1;
        return;
      }

      // f4 = rhs(u + dt * f3, t) for the runge kutta 4 step
      k_even = this->compute_rhs(U_tmp, 0.0, lapse_array, shift_array,
                                 metric_array, curv_array, dt, dx, lev, plo0);

      // Update particles with the f3 and f4 from RK4
      particles[i].pos(0) +=
          (1. / 6.) * dt * (2. * k_odd[0] + k_even[0] + partial_sum[0]);
      particles[i].pos(1) +=
          (1. / 6.) * dt * (2. * k_odd[1] + k_even[1] + partial_sum[1]);
      particles[i].pos(2) +=
          (1. / 6.) * dt * (2. * k_odd[2] + k_even[2] + partial_sum[2]);
      vels_x[i] +=
          (1. / 6.) * dt * (2. * k_odd[3] + k_even[3] + partial_sum[3]);
      vels_y[i] +=
          (1. / 6.) * dt * (2. * k_odd[4] + k_even[4] + partial_sum[4]);
      vels_z[i] +=
          (1. / 6.) * dt * (2. * k_odd[5] + k_even[5] + partial_sum[5]);
      ln_energy[i] +=
          (1. / 6.) * dt * (2. * k_odd[6] + k_even[6] + partial_sum[6]);

      // Check if is outside of the grid
      if ((particles[i].pos(0) > boundarie_hx) ||
          (particles[i].pos(0) < boundarie_lx) ||
          (particles[i].pos(1) > boundarie_hy) ||
          (particles[i].pos(1) < boundarie_ly) ||
          (particles[i].pos(2) > boundarie_hz) ||
          (particles[i].pos(2) < boundarie_lz)) {
        particles[i].id() = -1;
        return;
      }
    });
  }
} // PhotonsContainer::evolve

/**
 * \brief Computes and print all photons velocities.
 *
 * This function is for checking the photons velocities and distance between 1.0
 * and v^2.
 *
 * @param metric ADM 3 dimension metric.
 *  @param lev AMR Level of discretization.
 */
template <typename StructType>
void PhotonsContainer<StructType>::check_constants(
    CCTK_ARGUMENTS, const amrex::MultiFab &metric, const amrex::MultiFab &lapse,
    const amrex::MultiFab &shift, const int &lev) {
  // Get the current iteration number.
  const int iteration = cctkGH->cctk_iteration;
  CCTK_VINFO("Checking the constants and the velocity error at iteration %d.",
             iteration);

  // Initializing the files.
  std::ostringstream file_name;
  file_name << "Constants_" << amrex::ParallelDescriptor::MyProc() << "_"
            << iteration;
  std::ofstream vel_file;
  vel_file.open(file_name.str());

  const auto plo0 = this->Geom(0).ProbLoArray();
  const auto phi0 = this->Geom(0).ProbHiArray();

  const auto dx = this->Geom(lev).CellSizeArray();
  const auto p_lo = this->Geom(lev).ProbLoArray();

  for (Iterator::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
       ++pti) {
    const int np = pti.numParticles();

    // Get the velocity information
    auto &attribs = pti.GetAttributes();
    CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
    CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
    CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
    CCTK_REAL *AMREX_RESTRICT ln_energy = attribs[StructType::ln_E].data();
    auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

    auto const metric_array = metric.array(pti);
    auto const lapse_array = lapse.array(pti);
    auto const shift_array = shift.array(pti);

    // foreach particle
    amrex::ParallelFor(np, [=, &vel_file] AMREX_GPU_DEVICE(int i) noexcept {
      const auto p = particles[i];

      // Get the cell's index at the particle's position.
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

      const CCTK_REAL lapse_x = barycentric_cubic_3d<5>(
          lapse_array, i0, j0, k0, p.pos(0), p.pos(1), p.pos(2), dx, p_lo);
      amrex::GpuArray<CCTK_REAL, 3> shift_x = {
          barycentric_cubic_3d<5>(shift_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo, 0),
          barycentric_cubic_3d<5>(shift_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo, 1),
          barycentric_cubic_3d<5>(shift_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo, 2)};

      // Get the inverse metric.
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

      const CCTK_REAL v_squared = vels_x[i] * vels_x[i] * gamma_inv_x[0] +
                                  vels_y[i] * vels_y[i] * gamma_inv_x[3] +
                                  vels_z[i] * vels_z[i] * gamma_inv_x[5] +
                                  2.0 * vels_x[i] * vels_y[i] * gamma_inv_x[1] +
                                  2.0 * vels_x[i] * vels_z[i] * gamma_inv_x[2] +
                                  2.0 * vels_y[i] * vels_z[i] * gamma_inv_x[4];

      const CCTK_REAL E = std::exp(ln_energy[i]);
      const amrex::GpuArray<CCTK_REAL, 3> shift_down =
          SMatVecMul(gamma_x, shift_x);
      const amrex::GpuArray<CCTK_REAL, 3> P_i = {E * vels_x[i], E * vels_y[i],
                                                 E * vels_z[i]};
      const amrex::GpuArray<CCTK_REAL, 3> P_i_up = SMatVecMul(gamma_inv_x, P_i);
      const amrex::GpuArray<CCTK_REAL, 4> P_mu = {
          (-lapse_x * lapse_x + VecVecMul(shift_x, shift_down)) *
                  (E / lapse_x) +
              VecVecMul(shift_down, P_i_up),
          P_i[0], P_i[1], P_i[2]};
      // Write data into a file.
      // V^2 - 1 error
      vel_file << std::setprecision(17) << std::scientific << p.pos(0) << "\t"
               << p.pos(1) << "\t" << p.pos(2) << "\t"
               << amrex::ParallelDescriptor::MyProc() << "\t" << p.id() << "\t"
               << v_squared - 1.0 << "\t"
               << P_mu[0] * (E / lapse_x) + VecVecMul(P_i, P_i_up) << "\t"
               << P_mu[0] << "\t" << p.pos(0) * P_i[1] - p.pos(1) * P_i[0]
               << "\t"
               << p.pos(0) * p.pos(0) + p.pos(1) * p.pos(1) +
                      p.pos(2) * p.pos(2)
               << "\n";
    });
  }

  vel_file.close();

} // PhotonsContainer::check_constants

template <typename StructType>
void PhotonsContainer<StructType>::redistribute_particles() {
  CCTK_INFO("Redistributing particles");
} // PhotonsContainer::redistribute_particles

} // namespace Containers

#endif // !PHOTONSCONTAINER_HXX
