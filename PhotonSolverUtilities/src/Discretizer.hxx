/**
 * \file Discretizer.hxx
 * \brief First derivatives computations.
 *
 * This file contains the interpolation of the first order derivative for
 * scalar, vectorial and matrices functions.
 */
#ifndef DISCRETIZER_HXX
#define DISCRETIZER_HXX

#include <cctk.h>

#include "AMReX_Array.H"
#include "AMReX_Box.H"
#include "AMReX_Config.H"
#include "Interpolator.hxx"
#include "cctk_Types.h"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

/**
 * \brief Partial derivatives namespace.
 */
namespace Discretize {

using namespace Interpolator;

/**
 * Computes the first derivative of a scalar function.
 *
 * For compute the first derivative of a scalar function on a discretized grid
 * we can compute the values of the function in the following way:
 *
 * \f[
 *   \frac{\partial f}{\partial x_i} \approx \frac{f(x+\Delta x) - f(x + \Delta
 * x)}{2\Delta x_i}
 * \f]
 *
 * This function fills a size 3 vector with the derivatives of the scalar
 * function on each direction.
 *
 * @param df Empty vector of size 3 that contains the derivative on each
 * direction.
 * @param f Array with the values of the scalar function on each grid point.
 * @param f_xyz Interpolation of the value \f$f(x, y, z)\f$.
 * @param i0 Basis cell's index on the x direction.
 * @param j0 Basis cell's index on the x direction.
 * @param k0 Basis cell's index on the x direction.
 * @param x The x coordinate to evaluate.
 * @param y The y coordinate to evaluate.
 * @param z The z coordinate to evaluate.
 * @param dx The grid's cell \f$\Delta x\f$ vector.
 * @param plo The lower bound of the entire domain.
 */
template <int DERIVATIVE_ORDER>
AMREX_GPU_HOST_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
scalar_barycentric_derivative(amrex::GpuArray<CCTK_REAL, 3> &df,
                              amrex::Array4<CCTK_REAL const> const &f,
                              int const &i0, int const &j0, int const &k0,
                              CCTK_REAL const &x, CCTK_REAL const &y,
                              CCTK_REAL const &z,
                              const amrex::GpuArray<double, 3> &dx,
                              const amrex::GpuArray<double, 3> &plo) {
  // Computing the f(x-dx, y, z) value
  const int ip_half_0 = amrex::Math::floor((x + 0.5 * dx[0] - plo[0]) / dx[0]);
  const int jp_half_0 = amrex::Math::floor((y + 0.5 * dx[1] - plo[1]) / dx[1]);
  const int kp_half_0 = amrex::Math::floor((z + 0.5 * dx[2] - plo[2]) / dx[2]);
  const int im_half_0 = amrex::Math::floor((x - 0.5 * dx[0] - plo[0]) / dx[0]);
  const int jm_half_0 = amrex::Math::floor((y - 0.5 * dx[1] - plo[1]) / dx[1]);
  const int km_half_0 = amrex::Math::floor((z - 0.5 * dx[2] - plo[2]) / dx[2]);

  const CCTK_REAL p_half_x = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, ip_half_0, j0, k0, x + 0.5 * dx[0], y, z, dx, plo);
  const CCTK_REAL m_half_x = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, im_half_0, j0, k0, x - 0.5 * dx[0], y, z, dx, plo);
  // Computing the f(x, y-dy, z) value
  const CCTK_REAL p_half_y = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jp_half_0, k0, x, y + 0.5 * dx[1], z, dx, plo);
  const CCTK_REAL m_half_y = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jm_half_0, k0, x, y - 0.5 * dx[1], z, dx, plo);
  // Computing the f(x, y, z-dz) value
  const CCTK_REAL p_half_z = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, kp_half_0, x, y, z + 0.5 * dx[2], dx, plo);
  const CCTK_REAL m_half_z = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, km_half_0, x, y, z - 0.5 * dx[2], dx, plo);

  // fill the gradient
  df[0] = (p_half_x - m_half_x) / (dx[0]);
  df[1] = (p_half_y - m_half_y) / (dx[1]);
  df[2] = (p_half_z - m_half_z) / (dx[2]);
} // scalar_barycentric_derivative

/**
 * Computes the first derivative of a vectorial function.
 *
 * For compute the first derivative of a vectorial function on a discretized
 * grid we can compute the values of the function in the following way:
 *
 * \f[
 *   \frac{\partial \vec{f}}{\partial x_i} \approx \frac{\vec{f}(x+\Delta x) -
 * \vec{f}(x + \Delta x)}{2\Delta x_i}
 * \f]
 *
 * This function fills a size 3 vector with the derivatives of each component of
 * the vectorial function on each direction.
 *
 * @param df Empty vecto of size 3 that contains the derivative on each
 * direction of the components of the \f$\vec{f}\f$.
 * @param f Array with the values of the vectorial function on each grid point.
 * @param f_xyz Interpolation of the value \f$\vec{f}(x, y, z)\f$.
 * @param i0 Basis cell's index on the x direction.
 * @param j0 Basis cell's index on the x direction.
 * @param k0 Basis cell's index on the x direction.
 * @param x The x coordinate to evaluate.
 * @param y The y coordinate to evaluate.
 * @param z The z coordinate to evaluate.
 * @param dx The grid's cell \f$\Delta x\f$ vector.
 * @param plo The lower bound of the entire domain.
 */
template <int DERIVATIVE_ORDER>
AMREX_GPU_HOST_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
vector_barycentric_derivative(
    amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 3>, 3> &df,
    amrex::Array4<CCTK_REAL const> const &f, int const &i0, int const &j0,
    int const &k0, CCTK_REAL const &x, CCTK_REAL const &y, CCTK_REAL const &z,
    const amrex::GpuArray<double, 3> &dx,
    const amrex::GpuArray<double, 3> &plo) {

  // Computing the f(x-dx, y, z) value
  const int ip_half_0 = amrex::Math::floor((x + 0.5 * dx[0] - plo[0]) / dx[0]);
  const int jp_half_0 = amrex::Math::floor((y + 0.5 * dx[1] - plo[1]) / dx[1]);
  const int kp_half_0 = amrex::Math::floor((z + 0.5 * dx[2] - plo[2]) / dx[2]);
  const int im_half_0 = amrex::Math::floor((x - 0.5 * dx[0] - plo[0]) / dx[0]);
  const int jm_half_0 = amrex::Math::floor((y - 0.5 * dx[1] - plo[1]) / dx[1]);
  const int km_half_0 = amrex::Math::floor((z - 0.5 * dx[2] - plo[2]) / dx[2]);

  const CCTK_REAL p_half_x_0 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, ip_half_0, j0, k0, x + 0.5 * dx[0], y, z, dx, plo, 0);
  const CCTK_REAL m_half_x_0 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, im_half_0, j0, k0, x - 0.5 * dx[0], y, z, dx, plo, 0);
  const CCTK_REAL p_half_y_0 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jp_half_0, k0, x, y + 0.5 * dx[1], z, dx, plo, 0);
  const CCTK_REAL m_half_y_0 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jm_half_0, k0, x, y - 0.5 * dx[1], z, dx, plo, 0);
  const CCTK_REAL p_half_z_0 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, kp_half_0, x, y, z + 0.5 * dx[2], dx, plo, 0);
  const CCTK_REAL m_half_z_0 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, km_half_0, x, y, z - 0.5 * dx[2], dx, plo, 0);

  const CCTK_REAL p_half_x_1 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, ip_half_0, j0, k0, x + 0.5 * dx[0], y, z, dx, plo, 1);
  const CCTK_REAL m_half_x_1 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, im_half_0, j0, k0, x - 0.5 * dx[0], y, z, dx, plo, 1);
  const CCTK_REAL p_half_y_1 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jp_half_0, k0, x, y + 0.5 * dx[1], z, dx, plo, 1);
  const CCTK_REAL m_half_y_1 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jm_half_0, k0, x, y - 0.5 * dx[1], z, dx, plo, 1);
  const CCTK_REAL p_half_z_1 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, kp_half_0, x, y, z + 0.5 * dx[2], dx, plo, 1);
  const CCTK_REAL m_half_z_1 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, km_half_0, x, y, z - 0.5 * dx[2], dx, plo, 1);

  const CCTK_REAL p_half_x_2 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, ip_half_0, j0, k0, x + 0.5 * dx[0], y, z, dx, plo, 2);
  const CCTK_REAL m_half_x_2 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, im_half_0, j0, k0, x - 0.5 * dx[0], y, z, dx, plo, 2);
  const CCTK_REAL p_half_y_2 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jp_half_0, k0, x, y + 0.5 * dx[1], z, dx, plo, 2);
  const CCTK_REAL m_half_y_2 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jm_half_0, k0, x, y - 0.5 * dx[1], z, dx, plo, 2);
  const CCTK_REAL p_half_z_2 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, kp_half_0, x, y, z + 0.5 * dx[2], dx, plo, 2);
  const CCTK_REAL m_half_z_2 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, km_half_0, x, y, z - 0.5 * dx[2], dx, plo, 2);

  df[0][0] = (p_half_x_0 - m_half_x_0) / dx[0];
  df[0][1] = (p_half_x_1 - m_half_x_1) / dx[0];
  df[0][2] = (p_half_x_2 - m_half_x_2) / dx[0];

  df[1][0] = (p_half_y_0 - m_half_y_0) / dx[1];
  df[1][1] = (p_half_y_1 - m_half_y_1) / dx[1];
  df[1][2] = (p_half_y_2 - m_half_y_2) / dx[1];

  df[2][0] = (p_half_z_0 - m_half_z_0) / dx[2];
  df[2][1] = (p_half_z_1 - m_half_z_1) / dx[2];
  df[2][2] = (p_half_z_2 - m_half_z_2) / dx[2];
} // vector_barycentric_derivative

/**
 * Computes the first derivative of a symmetric matrix function.
 *
 * For compute the first derivative of a symmetric matrix function on a
 * discretized grid we can compute the values of the function in the following
 * way:
 *
 * \f[
 *   \frac{\partial \bar{f}}{\partial x_i} \approx \frac{\bar{f}(x+\Delta x) -
 * \bar{f}(x + \Delta x)}{2\Delta x_i}
 * \f]
 *
 * This function fills a size 3 vector with the derivatives of each component of
 * the symmetric matrix function on each direction.
 *
 * @param df Empty vecto of size 3 that contains the derivative on each
 * direction of the components of the \f$\bar{f}\f$.
 * @param f Array with the values of the symmetric matrix function on each grid
 * point.
 * @param f_xyz Interpolation of the value \f$\vec{f}(x, y, z)\f$.
 * @param i0 Basis cell's index on the x direction.
 * @param j0 Basis cell's index on the x direction.
 * @param k0 Basis cell's index on the x direction.
 * @param x The x coordinate to evaluate.
 * @param y The y coordinate to evaluate.
 * @param z The z coordinate to evaluate.
 * @param dx The grid's cell \f$\Delta x\f$ vector.
 * @param plo The lower bound of the entire domain.
 */
template <int DERIVATIVE_ORDER>
AMREX_GPU_HOST_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
smatrix_barycentric_derivative(
    amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 6>, 3> &df,
    amrex::Array4<CCTK_REAL const> const &f, int const &i0, int const &j0,
    int const &k0, CCTK_REAL const &x, CCTK_REAL const &y, CCTK_REAL const &z,
    const amrex::GpuArray<double, 3> &dx,
    const amrex::GpuArray<double, 3> &plo) {

  // Computing the f(x-dx, y, z) value
  const int ip_half_0 = amrex::Math::floor((x + 0.5 * dx[0] - plo[0]) / dx[0]);
  const int jp_half_0 = amrex::Math::floor((y + 0.5 * dx[1] - plo[1]) / dx[1]);
  const int kp_half_0 = amrex::Math::floor((z + 0.5 * dx[2] - plo[2]) / dx[2]);
  const int im_half_0 = amrex::Math::floor((x - 0.5 * dx[0] - plo[0]) / dx[0]);
  const int jm_half_0 = amrex::Math::floor((y - 0.5 * dx[1] - plo[1]) / dx[1]);
  const int km_half_0 = amrex::Math::floor((z - 0.5 * dx[2] - plo[2]) / dx[2]);

  const CCTK_REAL p_half_x_0 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, ip_half_0, j0, k0, x + 0.5 * dx[0], y, z, dx, plo, 0);
  const CCTK_REAL m_half_x_0 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, im_half_0, j0, k0, x - 0.5 * dx[0], y, z, dx, plo, 0);
  const CCTK_REAL p_half_y_0 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jp_half_0, k0, x, y + 0.5 * dx[1], z, dx, plo, 0);
  const CCTK_REAL m_half_y_0 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jm_half_0, k0, x, y - 0.5 * dx[1], z, dx, plo, 0);
  const CCTK_REAL p_half_z_0 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, kp_half_0, x, y, z + 0.5 * dx[2], dx, plo, 0);
  const CCTK_REAL m_half_z_0 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, km_half_0, x, y, z - 0.5 * dx[2], dx, plo, 0);

  const CCTK_REAL p_half_x_1 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, ip_half_0, j0, k0, x + 0.5 * dx[0], y, z, dx, plo, 1);
  const CCTK_REAL m_half_x_1 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, im_half_0, j0, k0, x - 0.5 * dx[0], y, z, dx, plo, 1);
  const CCTK_REAL p_half_y_1 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jp_half_0, k0, x, y + 0.5 * dx[1], z, dx, plo, 1);
  const CCTK_REAL m_half_y_1 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jm_half_0, k0, x, y - 0.5 * dx[1], z, dx, plo, 1);
  const CCTK_REAL p_half_z_1 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, kp_half_0, x, y, z + 0.5 * dx[2], dx, plo, 1);
  const CCTK_REAL m_half_z_1 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, km_half_0, x, y, z - 0.5 * dx[2], dx, plo, 1);

  const CCTK_REAL p_half_x_2 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, ip_half_0, j0, k0, x + 0.5 * dx[0], y, z, dx, plo, 2);
  const CCTK_REAL m_half_x_2 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, im_half_0, j0, k0, x - 0.5 * dx[0], y, z, dx, plo, 2);
  const CCTK_REAL p_half_y_2 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jp_half_0, k0, x, y + 0.5 * dx[1], z, dx, plo, 2);
  const CCTK_REAL m_half_y_2 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jm_half_0, k0, x, y - 0.5 * dx[1], z, dx, plo, 2);
  const CCTK_REAL p_half_z_2 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, kp_half_0, x, y, z + 0.5 * dx[2], dx, plo, 2);
  const CCTK_REAL m_half_z_2 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, km_half_0, x, y, z - 0.5 * dx[2], dx, plo, 2);

  const CCTK_REAL p_half_x_3 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, ip_half_0, j0, k0, x + 0.5 * dx[0], y, z, dx, plo, 3);
  const CCTK_REAL m_half_x_3 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, im_half_0, j0, k0, x - 0.5 * dx[0], y, z, dx, plo, 3);
  const CCTK_REAL p_half_y_3 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jp_half_0, k0, x, y + 0.5 * dx[1], z, dx, plo, 3);
  const CCTK_REAL m_half_y_3 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jm_half_0, k0, x, y - 0.5 * dx[1], z, dx, plo, 3);
  const CCTK_REAL p_half_z_3 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, kp_half_0, x, y, z + 0.5 * dx[2], dx, plo, 3);
  const CCTK_REAL m_half_z_3 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, km_half_0, x, y, z - 0.5 * dx[2], dx, plo, 3);

  const CCTK_REAL p_half_x_4 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, ip_half_0, j0, k0, x + 0.5 * dx[0], y, z, dx, plo, 4);
  const CCTK_REAL m_half_x_4 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, im_half_0, j0, k0, x - 0.5 * dx[0], y, z, dx, plo, 4);
  const CCTK_REAL p_half_y_4 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jp_half_0, k0, x, y + 0.5 * dx[1], z, dx, plo, 4);
  const CCTK_REAL m_half_y_4 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jm_half_0, k0, x, y - 0.5 * dx[1], z, dx, plo, 4);
  const CCTK_REAL p_half_z_4 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, kp_half_0, x, y, z + 0.5 * dx[2], dx, plo, 4);
  const CCTK_REAL m_half_z_4 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, km_half_0, x, y, z - 0.5 * dx[2], dx, plo, 4);

  const CCTK_REAL p_half_x_5 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, ip_half_0, j0, k0, x + 0.5 * dx[0], y, z, dx, plo, 5);
  const CCTK_REAL m_half_x_5 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, im_half_0, j0, k0, x - 0.5 * dx[0], y, z, dx, plo, 5);
  const CCTK_REAL p_half_y_5 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jp_half_0, k0, x, y + 0.5 * dx[1], z, dx, plo, 5);
  const CCTK_REAL m_half_y_5 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, jm_half_0, k0, x, y - 0.5 * dx[1], z, dx, plo, 5);
  const CCTK_REAL p_half_z_5 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, kp_half_0, x, y, z + 0.5 * dx[2], dx, plo, 5);
  const CCTK_REAL m_half_z_5 = barycentric_cubic_3d<DERIVATIVE_ORDER>(
      f, i0, j0, km_half_0, x, y, z - 0.5 * dx[2], dx, plo, 5);

  df[0][0] = (p_half_x_0 - m_half_x_0) / dx[0];
  df[0][1] = (p_half_x_1 - m_half_x_1) / dx[0];
  df[0][2] = (p_half_x_2 - m_half_x_2) / dx[0];
  df[0][3] = (p_half_x_3 - m_half_x_3) / dx[0];
  df[0][4] = (p_half_x_4 - m_half_x_4) / dx[0];
  df[0][5] = (p_half_x_5 - m_half_x_5) / dx[0];

  df[1][0] = (p_half_y_0 - m_half_y_0) / dx[1];
  df[1][1] = (p_half_y_1 - m_half_y_1) / dx[1];
  df[1][2] = (p_half_y_2 - m_half_y_2) / dx[1];
  df[1][3] = (p_half_y_3 - m_half_y_3) / dx[1];
  df[1][4] = (p_half_y_4 - m_half_y_4) / dx[1];
  df[1][5] = (p_half_y_5 - m_half_y_5) / dx[1];

  df[2][0] = (p_half_z_0 - m_half_z_0) / dx[2];
  df[2][1] = (p_half_z_1 - m_half_z_1) / dx[2];
  df[2][2] = (p_half_z_2 - m_half_z_2) / dx[2];
  df[2][3] = (p_half_z_3 - m_half_z_3) / dx[2];
  df[2][4] = (p_half_z_4 - m_half_z_4) / dx[2];
  df[2][5] = (p_half_z_5 - m_half_z_5) / dx[2];
} // smatrix_barycentric_derivative

} // namespace Discretize

#endif // !DISCRETIZER_HXX
