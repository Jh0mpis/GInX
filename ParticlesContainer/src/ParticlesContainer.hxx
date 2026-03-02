/**
 * \file ParticlesContainer.hxx
 * \brief ParticlesContainer class definition.
 *
 * The following file contains the definition of the ParticlesContainer class and
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

namespace GInX {

// #############################################################################
//                   ParticlesContainer::CLASS INITIALIZATION
// #############################################################################

/**
 * \brief ParticlesContainer class definition.
 *
 * The following class define the needed functions to evolve the position and
 * velocity of the photons in the simulation.
 */
template <typename StructType>
class ParticlesContainer
    : public BaseParticleContainer<ParticlesContainer<StructType>,
                                                  StructType> {
protected:
  CCTK_REAL mass = 0.;

public:
  /**
   * \brief Using BaseParticlesContainer constructor
   */
  using Base =
      BaseParticleContainer<ParticlesContainer<StructType>,
                                           StructType>;
  using Base::Base;

  ParticlesContainer(amrex::AmrCore *amr_core, const CCTK_REAL m)
      : Base(amr_core), mass{m} {};

  ~ParticlesContainer() = default;

  // Evolving all the particles on each container
  void evolve(const amrex::MultiFab &lapse, const amrex::MultiFab &shift,
              const amrex::MultiFab &metric, const amrex::MultiFab &curv,
              const CCTK_REAL &dt, const int &lev) override;

  // Given differential equation dU/dt = f(U, dU/dx; t) computes f(U, dU/dx;t)
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
      amrex::GpuArray<CCTK_REAL, 7>
      compute_rhs(const amrex::GpuArray<CCTK_REAL, 7> &u, const CCTK_REAL &t,
                  amrex::Array4<CCTK_REAL const> const &lapse,
                  amrex::Array4<CCTK_REAL const> const &shift,
                  amrex::Array4<CCTK_REAL const> const &metric,
                  amrex::Array4<CCTK_REAL const> const &K, const CCTK_REAL dt,
                  const amrex::GpuArray<double, 3> &dx, const int lev,
                  const amrex::GpuArray<double, 3> &plo);

  // Computes and normalize the velocity
  void normalize_velocity(const amrex::MultiFab &metric, const int level);

  // Get the mass value
  CCTK_INT get_mass() { return this->mass; }

  void redistribute_particles();
}; // ParticlesContainer class

// ##############################################################################
//                   ParticlesContainer::METHODS DECLARATION
// ##############################################################################

/**
 * \brief Computes the right hand side of the geodesic differential equation.
 *
 * Given differential equation \f[\frac{d}{dt}U = f\left(U, \frac{dU}{dx};
 * t\right)\f] computes
 * \f[f\left(U, \frac{dU}{dx}; t\right)\f]
 *
 * where \f$U\f$ is a vector which contains \f$(x, y, z, v_x, v_y, v_z,
 * \ln E)\f$. The differential equation for the particles' position is
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
 * and finally, for the \f$ \ln E \f$ the differential equation is:
 *
 * \f[ \dfrac{d}{dt} U[6] = \alpha K_{jk}U[3 + l]U[3 + m]\gamma^{lj}\gamma^{mk}
 * - U[3+l]\gamma^{lj}\partial_j\alpha\f]
 *
 *  Where \f$i, j, k, l, m = 0, 1, 2\f$ and \f$K_{ij}\f$ is the extrinsic
 * curvature. We have been using Einstein notation.
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
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
    amrex::GpuArray<CCTK_REAL, 7>
    ParticlesContainer<StructType>::compute_rhs(
        const amrex::GpuArray<CCTK_REAL, 7> &u, const CCTK_REAL &t,
        amrex::Array4<CCTK_REAL const> const &lapse,
        const amrex::Array4<CCTK_REAL const> &shift,
        const amrex::Array4<CCTK_REAL const> &metric,
        const amrex::Array4<CCTK_REAL const> &curv, const CCTK_REAL dt,
        const amrex::GpuArray<double, 3> &dx, const int lev,
        const amrex::GpuArray<double, 3> &plo) {

  amrex::GpuArray<CCTK_REAL, 7> rhs = {0., 0., 0., 0., 0., 0., 0.};

  const long int i0 = amrex::Math::floor((u[0] - plo[0]) / dx[0]);
  const long int j0 = amrex::Math::floor((u[1] - plo[1]) / dx[1]);
  const long int k0 = amrex::Math::floor((u[2] - plo[2]) / dx[2]);

  // Interpolate lapse & partial lapse at \vect{x}
  CCTK_REAL lapse_x;
  amrex::GpuArray<CCTK_REAL, 3> d_lapse_x;
  d_interpolate_array<5>(lapse_x, d_lapse_x, lapse, i0, j0, k0, u[0], u[1],
                         u[2], dx, plo);

  // Interpolate shift & partial shift at \vect{x}
  amrex::GpuArray<CCTK_REAL, 3> shift_x;
  amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 3>, 3> d_shift_x;
  d_interpolate_array<5>(shift_x, d_shift_x, shift, i0, j0, k0, u[0], u[1],
                         u[2], dx, plo);

  // Interpolate metric & partial metric at \vect{x}
  amrex::GpuArray<CCTK_REAL, 6> gamma_x;
  amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 6>, 3> d_gamma_x;
  d_interpolate_array<5>(gamma_x, d_gamma_x, metric, i0, j0, k0, u[0], u[1],
                         u[2], dx, plo);

  // Interpolate Curvature at \vect{x}
  amrex::GpuArray<CCTK_REAL, 6> curv_x;
  interpolate_array<5>(curv_x, curv, i0, j0, k0, u[0], u[1], u[2], dx, plo);

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

} // ParticlesContainer::compute_rhs

/**
 *  \brief Evolving using Runge-Kutta 4.
 *
 * We are solving the differential equation
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
 *  While computing we are checking if the particles still in the physical domain.
 *
 *  @see compute_rhs()
 *  @param lapse ADM lapse function.
 *  @param shift ADM shift vector.
 *  @param metric ADM induced metric.
 *  @param curv Extrinsic curvature.
 *  @param dt Timestep.
 *  @param lev Refinement level.
 */
template <typename StructType>
void ParticlesContainer<StructType>::evolve(const amrex::MultiFab &lapse,
                                          const amrex::MultiFab &shift,
                                          const amrex::MultiFab &metric,
                                          const amrex::MultiFab &curv,
                                          const CCTK_REAL &dt, const int &lev) {

  const auto plo0 = this->Geom(0).ProbLoArray();
  const auto phi0 = this->Geom(0).ProbHiArray();

  const auto dx = this->Geom(lev).CellSizeArray();
  const auto plo = this->Geom(lev).ProbLoArray();

  const CCTK_REAL boundarie_hx = phi0[0] - 0.0 * dx[0];
  const CCTK_REAL boundarie_lx = plo0[0] + 0.0 * dx[0];
  const CCTK_REAL boundarie_hy = phi0[1] - 0.0 * dx[1];
  const CCTK_REAL boundarie_ly = plo0[1] + 0.0 * dx[1];
  const CCTK_REAL boundarie_hz = phi0[2] - 0.0 * dx[2];
  const CCTK_REAL boundarie_lz = plo0[2] + 0.0 * dx[2];

  for (ParticleIterator<StructType> pti(*this, lev); pti.isValid();
       ++pti) {

    const int np = pti.numParticles();

    // Get the information relate to the velocities and energy.
    auto &attribs = pti.GetAttributes();
    CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
    CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
    CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
    CCTK_REAL *AMREX_RESTRICT ln_energy = attribs[StructType::ln_E].data();
    auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

    // Get the array of each parameter.
    auto const lapse_array = lapse.array(pti);
    auto const shift_array = shift.array(pti);
    auto const metric_array = metric.array(pti);
    auto const curv_array = curv.array(pti);

    // Needed for GPU
    auto self = this;

    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
      const amrex::GpuArray<CCTK_REAL, 7> U = {
          particles[i].pos(0), particles[i].pos(1), particles[i].pos(2),
          vels_x[i],           vels_y[i],           vels_z[i],
          ln_energy[i]};

      bool out_of_bounds = false;

      amrex::GpuArray<CCTK_REAL, 7> U_tmp = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      // f1 = rhs(u , t) for the runge kutta 4 step
      auto k_odd =
          self->compute_rhs(U, 0.0, lapse_array, shift_array, metric_array,
                            curv_array, dt, dx, lev, plo0);

      U_tmp[0] = U[0] + 0.5 * dt * k_odd[0];
      U_tmp[1] = U[1] + 0.5 * dt * k_odd[1];
      U_tmp[2] = U[2] + 0.5 * dt * k_odd[2];
      U_tmp[3] = U[3] + 0.5 * dt * k_odd[3];
      U_tmp[4] = U[4] + 0.5 * dt * k_odd[4];
      U_tmp[5] = U[5] + 0.5 * dt * k_odd[5];
      U_tmp[6] = U[6] + 0.5 * dt * k_odd[6];

      out_of_bounds |= (U_tmp[0] > boundarie_hx) || (U_tmp[0] < boundarie_lx);
      out_of_bounds |= (U_tmp[1] > boundarie_hy) || (U_tmp[1] < boundarie_ly);
      out_of_bounds |= (U_tmp[2] > boundarie_hz) || (U_tmp[2] < boundarie_lz);

      if (out_of_bounds) {
        particles[i].id() = -1;
        return;
      }

      // f2 = rhs(u + 0.5 * dt * f1, t) for the runge kutta 4 step
      auto k_even =
          self->compute_rhs(U_tmp, 0.5 * dt, lapse_array, shift_array,
                            metric_array, curv_array, dt, dx, lev, plo0);

      // Update particles with the f1 and f2 from RK4
      U_tmp[0] = U[0] + 0.5 * dt * k_even[0];
      U_tmp[1] = U[1] + 0.5 * dt * k_even[1];
      U_tmp[2] = U[2] + 0.5 * dt * k_even[2];
      U_tmp[3] = U[3] + 0.5 * dt * k_even[3];
      U_tmp[4] = U[4] + 0.5 * dt * k_even[4];
      U_tmp[5] = U[5] + 0.5 * dt * k_even[5];
      U_tmp[6] = U[6] + 0.5 * dt * k_even[6];

      particles[i].pos(0) += (1. / 6.) * dt * (k_odd[0] + 2. * k_even[0]);
      particles[i].pos(1) += (1. / 6.) * dt * (k_odd[1] + 2. * k_even[1]);
      particles[i].pos(2) += (1. / 6.) * dt * (k_odd[2] + 2. * k_even[2]);
      vels_x[i] += (1. / 6.) * dt * (k_odd[3] + 2. * k_even[3]);
      vels_y[i] += (1. / 6.) * dt * (k_odd[4] + 2. * k_even[4]);
      vels_z[i] += (1. / 6.) * dt * (k_odd[5] + 2. * k_even[5]);
      ln_energy[i] += (1. / 6.) * dt * (k_odd[6] + 2. * k_even[6]);

      out_of_bounds |= (U_tmp[0] > boundarie_hx) || (U_tmp[0] < boundarie_lx);
      out_of_bounds |= (U_tmp[1] > boundarie_hy) || (U_tmp[1] < boundarie_ly);
      out_of_bounds |= (U_tmp[2] > boundarie_hz) || (U_tmp[2] < boundarie_lz);

      if (out_of_bounds) {
        particles[i].id() = -1;
        return;
      }

      // f3 = rhs(u + 0.5 * dt * f2, t) for the runge kutta 4 step
      k_odd = self->compute_rhs(U_tmp, 0.5 * dt, lapse_array, shift_array,
                                metric_array, curv_array, dt, dx, lev, plo0);

      U_tmp[0] = U[0] + dt * k_odd[0];
      U_tmp[1] = U[1] + dt * k_odd[1];
      U_tmp[2] = U[2] + dt * k_odd[2];
      U_tmp[3] = U[3] + dt * k_odd[3];
      U_tmp[4] = U[4] + dt * k_odd[4];
      U_tmp[5] = U[5] + dt * k_odd[5];
      U_tmp[6] = U[6] + dt * k_odd[6];

      out_of_bounds |= (U_tmp[0] > boundarie_hx) || (U_tmp[0] < boundarie_lx);
      out_of_bounds |= (U_tmp[1] > boundarie_hy) || (U_tmp[1] < boundarie_ly);
      out_of_bounds |= (U_tmp[2] > boundarie_hz) || (U_tmp[2] < boundarie_lz);

      if (out_of_bounds) {
        particles[i].id() = -1;
        return;
      }

      // f4 = rhs(u + dt * f3, t) for the runge kutta 4 step
      k_even = self->compute_rhs(U_tmp, dt, lapse_array, shift_array,
                                 metric_array, curv_array, dt, dx, lev, plo0);

      // Update particles with the f3 and f4 from RK4
      particles[i].pos(0) += (1. / 6.) * dt * (2. * k_odd[0] + k_even[0]);
      particles[i].pos(1) += (1. / 6.) * dt * (2. * k_odd[1] + k_even[1]);
      particles[i].pos(2) += (1. / 6.) * dt * (2. * k_odd[2] + k_even[2]);
      vels_x[i] += (1. / 6.) * dt * (2. * k_odd[3] + k_even[3]);
      vels_y[i] += (1. / 6.) * dt * (2. * k_odd[4] + k_even[4]);
      vels_z[i] += (1. / 6.) * dt * (2. * k_odd[5] + k_even[5]);
      ln_energy[i] += (1. / 6.) * dt * (2. * k_odd[6] + k_even[6]);

      out_of_bounds |= (particles[i].pos(0) > boundarie_hx) ||
                       (particles[i].pos(0) < boundarie_lx);
      out_of_bounds |= (particles[i].pos(1) > boundarie_hy) ||
                       (particles[i].pos(1) < boundarie_ly);
      out_of_bounds |= (particles[i].pos(2) > boundarie_hz) ||
                       (particles[i].pos(2) < boundarie_lz);

      if (out_of_bounds) {
        particles[i].id() = -1;
        return;
      }
    });
  }
} // ParticlesContainer::evolve

/**
 * \brief Normalize the velocity accordingly to the metric on each particle
 * position.
 *
 * This function normalizes the velocity given a random initial data.
 * The invariant mass is described by
 *
 *  \f[
 *  P^\mu P_\mu = 0,
 *  \f]
 *
 *  for null geodesics and 
 *
 *  \f[
 *  P^\mu P_\mu = -m^2,
 *  \f]
 *
 *  for timelike particles.
 *
 *  Implying
 *
 *  \f[
 *  V^\alpha V_\alpha = V_\alpha V_\beta \gamma^{\alpha\beta} = 1,
 *  \f]
 *
 *  for null geodesic and
 *
 *  \f[
 *  V^\alpha V_\alpha = V_\alpha V_\beta \gamma^{\alpha\beta} = 1 -\frac{m^2}{E^2}
 *  \f]
 *
 *  for timelike particles.
 *
 * @param metric ADM 3 dimension metric.
 * @param Current refinement level.
 */
template <typename StructType>
void ParticlesContainer<StructType>::normalize_velocity(
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
    const CCTK_REAL m = this->mass;

    amrex::ParallelFor(current_size, [=] AMREX_GPU_DEVICE(int i) noexcept {
      // Start a for loop with Random Number evolution for the velocity
      const CCTK_REAL ratio[3] = {arrdata[StructType::vx][i],
                                  arrdata[StructType::vy][i],
                                  arrdata[StructType::vz][i]};
      const CCTK_REAL E = std::exp(arrdata[StructType::ln_E][i]);

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
      const CCTK_REAL alpha = std::sqrt(1. - m * m / (E * E));

      arrdata[StructType::vx][i] = ratio[0] * alpha / v;
      arrdata[StructType::vy][i] = ratio[1] * alpha / v;
      arrdata[StructType::vz][i] = ratio[2] * alpha / v;
    });
  }
} // ParticlesContainer::normalize_velocity

template <typename StructType>
void ParticlesContainer<StructType>::redistribute_particles() {
  CCTK_INFO("Redistributing particles");
} // ParticlesContainer::redistribute_particles

} // namespace GInX

#endif // !PHOTONSCONTAINER_HXX
