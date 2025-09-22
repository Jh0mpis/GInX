/**
 * @file PhotonsContainer.hxx
 */

#ifndef PHOTONSCONTAINER_HXX
#define PHOTONSCONTAINER_HXX

// Import libraries
#include <cctk.h>

#include "AMReX_Array.H"
#include "AMReX_GpuLaunchFunctsC.H"
#include "BaseParticleContainer.hxx"
#include "Discretizer.hxx"
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
#include <sstream>
#include <string>

namespace Containers {

// #############################################################################
//                   PhotonsContainer::CLASS INITIALIZATION
// #############################################################################

using namespace BaseContainer;
using namespace Interpolator;
using namespace Discretize;

template <typename StructType>
class PhotonsContainer
    : public BaseParticleContainer<PhotonsContainer<StructType>, StructType> {

public:
  // Using BaseParticlesContainer constructor
  using Base = BaseParticleContainer<PhotonsContainer<StructType>, StructType>;
  using Base::Base;

  void evolve() override;

  void evolveRK2(const amrex::MultiFab &lapse, const amrex::MultiFab &shift,
                 const amrex::MultiFab &metric, const amrex::MultiFab &curv,
                 const CCTK_REAL &dt, const int &lev);

  void evolveRK4(const amrex::MultiFab &lapse, const amrex::MultiFab &shift,
                 const amrex::MultiFab &metric, const amrex::MultiFab &curv,
                 const CCTK_REAL &dt, const int &lev);

  amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3>
  compute_rhs(const amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3> &u,
              const CCTK_REAL &t, amrex::Array4<CCTK_REAL const> const &lapse,
              amrex::Array4<CCTK_REAL const> const &shift,
              amrex::Array4<CCTK_REAL const> const &metric,
              amrex::Array4<CCTK_REAL const> const &K, const CCTK_REAL dt,
              const amrex::GpuArray<double, 3> &dx, const int lev,
              const amrex::GpuArray<double, 3> &plo);

  void compute_rhs() { CCTK_INFO("void computing rhs."); };

  void check_velocity(CCTK_ARGUMENTS, const amrex::MultiFab &metric,
                      const int &lev);

  void redistribute_particles();
}; // PhotonsContainer class

// ##############################################################################
//                   PhotonsContainer::METHODS DECLARATION
// ##############################################################################

// I'm assuming an equation dU/dt = F(u, t), where U is a vector that contains
// (x_u, y_u, z_u, vx_d, vy_d, vz_d). That's why each rhs depends on the other
// components of the vector U. For the position the differential equation  is:
//
// dU[i] / dt = alpha * gamma[i][j] *  U[3 + j] - beta_u[i];
//
// Where i,j = 0, 1, 2, gamma is the induced metric with the two indices
// up, alpha is the lapse function and beta is the shift vector.
//
// For the Velocity_d the differential equation is:
//
// dU[i + 3] /dt = - first_derivative<i>(alpha) +
// (first_derivative<j>(alpha) * gamma[j][k] * U[3 + k] - alpha * K[j][k] *
// gamma[j][l] * gamma[k][m] * U[3 + l] * U[3 + m]) * U[i + 3]
//  + first_derivative<i>(beta[k]) * U[3 + k]
//  - 0.5 * alpha * first_derivative<i>(gamma[j][k]) * U[j + 3] * U[k + 3]
//
//  Where i, j, k, l, m = 0, 1, 2.
template <typename StructType>
amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3>
PhotonsContainer<StructType>::compute_rhs(
    const amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3> &u,
    const CCTK_REAL &t, amrex::Array4<CCTK_REAL const> const &lapse,
    amrex::Array4<CCTK_REAL const> const &shift,
    amrex::Array4<CCTK_REAL const> const &metric,
    amrex::Array4<CCTK_REAL const> const &curv, const CCTK_REAL dt,
    const amrex::GpuArray<double, 3> &dx, const int lev,
    const amrex::GpuArray<double, 3> &plo) {

  amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3> rhs = {
      0., 0., 0., 0., 0., 0., 0.};

  const int i0 = amrex::Math::floor((u[0] - plo[0]) / dx[0]);
  const int j0 = amrex::Math::floor((u[1] - plo[1]) / dx[1]);
  const int k0 = amrex::Math::floor((u[2] - plo[2]) / dx[2]);

  // Interpolate lapse
  const CCTK_REAL lapse_x =
      barycentric_cubic_3d<3>(lapse, i0, j0, k0, u[0], u[1], u[2], dx, plo);

  // Interpolate shift
  const amrex::GpuArray<CCTK_REAL, 3> shift_x = {
      barycentric_cubic_3d<3>(shift, i0, j0, k0, u[0], u[1], u[2], dx, plo, 0),
      barycentric_cubic_3d<3>(shift, i0, j0, k0, u[0], u[1], u[2], dx, plo, 1),
      barycentric_cubic_3d<3>(shift, i0, j0, k0, u[0], u[1], u[2], dx, plo, 2)};

  // Interpolate metric
  const amrex::GpuArray<CCTK_REAL, 6> gamma_x = {
      barycentric_cubic_3d<3>(metric, i0, j0, k0, u[0], u[1], u[2], dx, plo,
                              0), // g_11
      barycentric_cubic_3d<3>(metric, i0, j0, k0, u[0], u[1], u[2], dx, plo,
                              1), // g_12 & g_21
      barycentric_cubic_3d<3>(metric, i0, j0, k0, u[0], u[1], u[2], dx, plo,
                              2), // g_13 & g_31
      barycentric_cubic_3d<3>(metric, i0, j0, k0, u[0], u[1], u[2], dx, plo,
                              3), // g_22
      barycentric_cubic_3d<3>(metric, i0, j0, k0, u[0], u[1], u[2], dx, plo,
                              4), // g_23, g_32
      barycentric_cubic_3d<3>(metric, i0, j0, k0, u[0], u[1], u[2], dx, plo,
                              5)}; // g_33

  // Interpolate Curvature
  const amrex::GpuArray<CCTK_REAL, 6> curv_x = {
      barycentric_cubic_3d<3>(curv, i0, j0, k0, u[0], u[1], u[2], dx, plo,
                              0), // K_11
      barycentric_cubic_3d<3>(curv, i0, j0, k0, u[0], u[1], u[2], dx, plo,
                              1), // K_12 & g_21
      barycentric_cubic_3d<3>(curv, i0, j0, k0, u[0], u[1], u[2], dx, plo,
                              2), // K_13 & K_31
      barycentric_cubic_3d<3>(curv, i0, j0, k0, u[0], u[1], u[2], dx, plo,
                              3), // K_22
      barycentric_cubic_3d<3>(curv, i0, j0, k0, u[0], u[1], u[2], dx, plo,
                              4), // K_23, K_32
      barycentric_cubic_3d<3>(curv, i0, j0, k0, u[0], u[1], u[2], dx, plo,
                              5)}; // K_33

  // Interpolate partial lapse
  amrex::GpuArray<CCTK_REAL, 3> d_lapse_x; //= {0, 0, 0};
  scalar_barycentric_derivative<3, 2>(d_lapse_x, lapse, i0, j0, k0, u[0], u[1],
                                      u[2], dx, plo);

  // Interpolate partial shift
  amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 3>, 3>
      d_shift_x; // = {
                 // {{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}}};
  vector_barycentric_derivative<3, 2>(d_shift_x, shift, i0, j0, k0, u[0], u[1],
                                      u[2], dx, plo);

  // Interpolate partial metric
  amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 6>, 3> d_gamma_x; //= {
  /*{{0., 0., 0., 0., 0., 0.},
   {0., 0., 0., 0., 0., 0.},
   {0., 0., 0., 0., 0., 0.}}};*/
  smatrix_barycentric_derivative<3, 2>(d_gamma_x, metric, i0, j0, k0, u[0],
                                       u[1], u[2], dx, plo);

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

  const CCTK_REAL &V1 = u[3];
  const CCTK_REAL &V2 = u[4];
  const CCTK_REAL &V3 = u[5];

  const amrex::GpuArray<CCTK_REAL, 3> V_up = {
      gamma_inv_x[0] * V1 + gamma_inv_x[1] * V2 + gamma_inv_x[2] * V3,
      gamma_inv_x[1] * V1 + gamma_inv_x[3] * V2 + gamma_inv_x[4] * V3,
      gamma_inv_x[2] * V1 + gamma_inv_x[4] * V2 + gamma_inv_x[5] * V3};

  // Compute the rhs for position
  rhs[0] = lapse_x * V_up[0] - shift_x[0];
  rhs[1] = lapse_x * V_up[1] - shift_x[1];
  rhs[2] = lapse_x * V_up[2] - shift_x[2];

  const amrex::GpuArray<CCTK_REAL, 3> V = {V1, V2, V3};

  // Compute the rhs for velocity
  for (int i = 0; i < 3; i++) {
    CCTK_REAL t1 = -d_lapse_x[i];

    CCTK_REAL t2 = (VecVecMul(d_lapse_x, V_up) -
                    lapse_x * VecVecMul(SMatVecMul(curv_x, V_up), V_up)) *
                   V[i];
    CCTK_REAL t3 =
        0.5 * lapse_x * VecVecMul(SMatVecMul(d_gamma_x[i], V_up), V_up);
    CCTK_REAL t4 = VecVecMul(V, d_shift_x[i]);
    rhs[3 + i] = t1 + t2 + t3 + t4;
  }

  // Compute the rhs for energy
  rhs[3 + StructType::E] = 0;

  return rhs;

} // PhotonsContainer::get_rhs

template <typename StructType>
void PhotonsContainer<StructType>::redistribute_particles() {
  CCTK_INFO("Redistributing particles");
} // PhotonsContainer::redistribute_particles

template <typename StructType> void PhotonsContainer<StructType>::evolve() {
  CCTK_INFO("Evolving");
}

template <typename StructType>
void PhotonsContainer<StructType>::evolveRK2(const amrex::MultiFab &lapse,
                                             const amrex::MultiFab &shift,
                                             const amrex::MultiFab &metric,
                                             const amrex::MultiFab &curv,
                                             const CCTK_REAL &dt,
                                             const int &lev) {
  CCTK_INFO("Evolving using RK2");

  const auto plo0 = this->Geom(0).ProbLoArray();
  const auto phi0 = this->Geom(0).ProbHiArray();

  const auto dx = this->Geom(lev).CellSizeArray();
  const auto plo = this->Geom(lev).ProbLoArray();
  // for each particle:

  for (Iterator::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
       ++pti) {
    const int np = pti.numParticles();

    auto &attribs = pti.GetAttributes();
    CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
    CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
    CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
    auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

    auto const lapse_array = lapse.array(pti);
    auto const shift_array = shift.array(pti);
    auto const metric_array = metric.array(pti);
    auto const curv_array = curv.array(pti);

    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
      amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3> U = {
          particles[i].pos(0),        particles[i].pos(1),
          particles[i].pos(2),        attribs[StructType::vx][i],
          attribs[StructType::vy][i], attribs[StructType::vz][i],
          attribs[StructType::E][i]};

      // rhs(u , t) for the runge kutta 2 step
      auto rhs_1 =
          this->compute_rhs(U, 0.0, lapse_array, shift_array, metric_array,
                            curv_array, dt, dx, lev, plo);

      U[0] += 0.5 * dt * rhs_1[0];
      U[1] += 0.5 * dt * rhs_1[1];
      U[2] += 0.5 * dt * rhs_1[2];
      U[3] += 0.5 * dt * rhs_1[3];
      U[4] += 0.5 * dt * rhs_1[4];
      U[5] += 0.5 * dt * rhs_1[5];
      U[6] += 0.5 * dt * rhs_1[6];

      // rhs(u + 0.5 * dt * rhs(u,t), t) for the runge kutta 2 step
      rhs_1 = this->compute_rhs(U, 0.0 + 0.5 * dt, lapse_array, shift_array,
                                metric_array, curv_array, dt, dx, lev, plo);

      // Runge kutta 2 step
      particles[i].pos(0) += dt * rhs_1[0];
      particles[i].pos(1) += dt * rhs_1[1];
      particles[i].pos(2) += dt * rhs_1[2];
      vels_x[i] += dt * rhs_1[3];
      vels_y[i] += dt * rhs_1[4];
      vels_z[i] += dt * rhs_1[5];

      // Check if is outside of the grid
      if (particles[i].pos(0) > (phi0[0] - dx[0]) ||
          particles[i].pos(0) < (plo0[0] + dx[0]) ||
          particles[i].pos(1) > (phi0[1] - dx[1]) ||
          particles[i].pos(1) < (plo0[1] + dx[1]) ||
          particles[i].pos(2) > (phi0[2] - dx[2]) ||
          particles[i].pos(2) < (plo0[2] + dx[2])) {
        particles[i].id() = -1;
        return;
      }
    });
  }

  this->Redistribute();
} // PhotonsContainer::evolveRK2

template <typename StructType>
void PhotonsContainer<StructType>::evolveRK4(const amrex::MultiFab &lapse,
                                             const amrex::MultiFab &shift,
                                             const amrex::MultiFab &metric,
                                             const amrex::MultiFab &curv,
                                             const CCTK_REAL &dt,
                                             const int &lev) {
  CCTK_INFO("Evolving particles using Runge-Kutta 4.");

  const auto plo0 = this->Geom(0).ProbLoArray();
  const auto phi0 = this->Geom(0).ProbHiArray();

  const auto dx = this->Geom(lev).CellSizeArray();
  const auto plo = this->Geom(lev).ProbLoArray();

  for (Iterator::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
       ++pti) {
    const int np = pti.numParticles();

    auto &attribs = pti.GetAttributes();
    CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
    CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
    CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
    auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

    auto const lapse_array = lapse.array(pti);
    auto const shift_array = shift.array(pti);
    auto const metric_array = metric.array(pti);
    auto const curv_array = curv.array(pti);

    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
      amrex::GpuArray<CCTK_REAL, StructType::n_attributes + 3> U = {
          particles[i].pos(0),        particles[i].pos(1),
          particles[i].pos(2),        attribs[StructType::vx][i],
          attribs[StructType::vy][i], attribs[StructType::vz][i],
          attribs[StructType::E][i]};

      // f1 = rhs(u , t) for the runge kutta 4 step
      auto rhs_1 =
          this->compute_rhs(U, 0.0, lapse_array, shift_array, metric_array,
                            curv_array, dt, dx, lev, plo);

      U[0] += 0.5 * dt * rhs_1[0];
      U[1] += 0.5 * dt * rhs_1[1];
      U[2] += 0.5 * dt * rhs_1[2];
      U[3] += 0.5 * dt * rhs_1[3];
      U[4] += 0.5 * dt * rhs_1[4];
      U[5] += 0.5 * dt * rhs_1[5];
      U[6] += 0.5 * dt * rhs_1[6];

      if (U[0] > (phi0[0] - 0.5 * dx[0]) || U[0] < (plo0[0] + 0.5 * dx[0]) ||
          U[1] > (phi0[1] - 0.5 * dx[1]) || U[1] < (plo0[1] + 0.5 * dx[1]) ||
          U[2] > (phi0[2] - 0.5 * dx[2]) || U[2] < (plo0[2] + 0.5 * dx[2])) {
        particles[i].id() = -1;
        return;
      }

      // f2 = rhs(u + 0.5 * dt * f1, t) for the runge kutta 4 step
      auto rhs_2 =
          this->compute_rhs(U, 0.0 + 0.5 * dt, lapse_array, shift_array,
                            metric_array, curv_array, dt, dx, lev, plo);

      // Update particles with the f1 and f2 from RK4
      particles[i].pos(0) += (1. / 6.) * dt * (rhs_1[0] + 2. * rhs_2[0]);
      particles[i].pos(1) += (1. / 6.) * dt * (rhs_1[1] + 2. * rhs_2[1]);
      particles[i].pos(2) += (1. / 6.) * dt * (rhs_1[2] + 2. * rhs_2[2]);
      vels_x[i] += (1. / 6.) * dt * (rhs_1[3] + 2. * rhs_2[3]);
      vels_y[i] += (1. / 6.) * dt * (rhs_1[4] + 2. * rhs_2[4]);
      vels_z[i] += (1. / 6.) * dt * (rhs_1[5] + 2. * rhs_2[5]);

      U[0] += 0.5 * dt * rhs_2[0];
      U[1] += 0.5 * dt * rhs_2[1];
      U[2] += 0.5 * dt * rhs_2[2];
      U[3] += 0.5 * dt * rhs_2[3];
      U[4] += 0.5 * dt * rhs_2[4];
      U[5] += 0.5 * dt * rhs_2[5];
      U[6] += 0.5 * dt * rhs_2[6];

      if (particles[i].pos(0) > (phi0[0] - 0.5 * dx[0]) ||
          particles[i].pos(0) < (plo0[0] + 0.5 * dx[0]) ||
          particles[i].pos(1) > (phi0[1] - 0.5 * dx[1]) ||
          particles[i].pos(1) < (plo0[1] + 0.5 * dx[1]) ||
          particles[i].pos(2) > (phi0[2] - 0.5 * dx[2]) ||
          particles[i].pos(2) < (plo0[2] + 0.5 * dx[2])) {
        particles[i].id() = -1;
        return;
      }

      if (U[0] > (phi0[0] - 0.5 * dx[0]) || U[0] < (plo0[0] + 0.5 * dx[0]) ||
          U[1] > (phi0[1] - 0.5 * dx[1]) || U[1] < (plo0[1] + 0.5 * dx[1]) ||
          U[2] > (phi0[2] - 0.5 * dx[2]) || U[2] < (plo0[2] + 0.5 * dx[2])) {
        particles[i].id() = -1;
        return;
      }

      // f3 = rhs(u + 0.5 * dt * f2, t) for the runge kutta 4 step
      rhs_1 = this->compute_rhs(U, 0.0, lapse_array, shift_array, metric_array,
                                curv_array, dt, dx, lev, plo);

      U[0] += dt * rhs_1[0];
      U[1] += dt * rhs_1[1];
      U[2] += dt * rhs_1[2];
      U[3] += dt * rhs_1[3];
      U[4] += dt * rhs_1[4];
      U[5] += dt * rhs_1[5];
      U[6] += dt * rhs_1[6];

      if (U[0] > (phi0[0] - 0.5 * dx[0]) || U[0] < (plo0[0] + 0.5 * dx[0]) ||
          U[1] > (phi0[1] - 0.5 * dx[1]) || U[1] < (plo0[1] + 0.5 * dx[1]) ||
          U[2] > (phi0[2] - 0.5 * dx[2]) || U[2] < (plo0[2] + 0.5 * dx[2])) {
        particles[i].id() = -1;
        return;
      }

      // f4 = rhs(u + dt * f3, t) for the runge kutta 4 step
      rhs_2 = this->compute_rhs(U, 0.0, lapse_array, shift_array, metric_array,
                                curv_array, dt, dx, lev, plo);

      // Update particles with the f3 and f4 from RK4
      particles[i].pos(0) += (1. / 6.) * dt * (2. * rhs_1[0] + rhs_2[0]);
      particles[i].pos(1) += (1. / 6.) * dt * (2. * rhs_1[1] + rhs_2[1]);
      particles[i].pos(2) += (1. / 6.) * dt * (2. * rhs_1[2] + rhs_2[2]);
      vels_x[i] += (1. / 6.) * dt * (2. * rhs_1[3] + rhs_2[3]);
      vels_y[i] += (1. / 6.) * dt * (2. * rhs_1[4] + rhs_2[4]);
      vels_z[i] += (1. / 6.) * dt * (2. * rhs_1[5] + rhs_2[5]);

      // Check if is outside of the grid
      if (particles[i].pos(0) > (phi0[0] - 0.5 * dx[0]) ||
          particles[i].pos(0) < (plo0[0] + 0.5 * dx[0]) ||
          particles[i].pos(1) > (phi0[1] - 0.5 * dx[1]) ||
          particles[i].pos(1) < (plo0[1] + 0.5 * dx[1]) ||
          particles[i].pos(2) > (phi0[2] - 0.5 * dx[2]) ||
          particles[i].pos(2) < (plo0[2] + 0.5 * dx[2])) {
        particles[i].id() = -1;
        return;
      }
    });
  }

  this->Redistribute();
} // PhotonsContainer::evolve

template <typename StructType>
void PhotonsContainer<StructType>::check_velocity(CCTK_ARGUMENTS,
                                                  const amrex::MultiFab &metric,
                                                  const int &lev) {
  const int iteration = cctkGH->cctk_iteration;
  CCTK_VINFO("Checking the velocity error at iteration %d.", iteration);
  std::ostringstream file_name;

  file_name << "Velocity_" << iteration << ".dat";
  std::ofstream vel_file;
  vel_file.open(file_name.str());

  const auto plo0 = this->Geom(0).ProbLoArray();
  const auto phi0 = this->Geom(0).ProbHiArray();

  const auto dx = this->Geom(lev).CellSizeArray();
  const auto p_lo = this->Geom(lev).ProbLoArray();

  for (Iterator::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
       ++pti) {
    const int np = pti.numParticles();

    auto &attribs = pti.GetAttributes();
    CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
    CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
    CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
    auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

    auto const metric_array = metric.array(pti);

    amrex::ParallelFor(np, [=, &vel_file] AMREX_GPU_DEVICE(int i) noexcept {
      const auto p = particles[i];

      const int i0 = amrex::Math::floor((p.pos(0) - p_lo[0]) / dx[0]);
      const int j0 = amrex::Math::floor((p.pos(1) - p_lo[1]) / dx[1]);
      const int k0 = amrex::Math::floor((p.pos(2) - p_lo[2]) / dx[2]);

      // Interpolate metric
      const amrex::GpuArray<CCTK_REAL, 6> gamma_x = {
          barycentric_cubic_3d<3>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo,
                                  0), // g_11
          barycentric_cubic_3d<3>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo,
                                  1), // g_12 & g_21
          barycentric_cubic_3d<3>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo,
                                  2), // g_13 & g_31
          barycentric_cubic_3d<3>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo,
                                  3), // g_22
          barycentric_cubic_3d<3>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
                                  p.pos(2), dx, p_lo,
                                  4), // g_23, g_32
          barycentric_cubic_3d<3>(metric_array, i0, j0, k0, p.pos(0), p.pos(1),
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

      const CCTK_REAL v_squared = vels_x[i] * vels_x[i] * gamma_inv_x[0] +
                                  vels_y[i] * vels_y[i] * gamma_inv_x[3] +
                                  vels_z[i] * vels_z[i] * gamma_inv_x[5] +
                                  2.0 * vels_x[i] * vels_y[i] * gamma_inv_x[1] +
                                  2.0 * vels_x[i] * vels_z[i] * gamma_inv_x[2] +
                                  2.0 * vels_y[i] * vels_z[i] * gamma_inv_x[4];

      vel_file << p.id() << "\t" << std::abs(v_squared - 1.0) << "\n";
    });
  }

  vel_file.close();

} // PhotonsContainer::check_velocity

} // namespace Containers

#endif // !PHOTONSCONTAINER_HXX
