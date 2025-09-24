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
 * @param i0 Basis cell's index on the x direction.
 * @param j0 Basis cell's index on the x direction.
 * @param k0 Basis cell's index on the x direction.
 * @param x The x coordinate to evaluate.
 * @param y The y coordinate to evaluate.
 * @param z The z coordinate to evaluate.
 * @param dx The grid's cell \f$\Delta x\f$ vector.
 * @param plo The lower bound of the entire domain.
 */
template <int INTERPOLATION_ORDER, int DERIVATIVE_ORDER>
AMREX_GPU_HOST_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
scalar_barycentric_derivative(amrex::GpuArray<CCTK_REAL, 3> &df,
                              amrex::Array4<CCTK_REAL const> const &f,
                              int const &i0, int const &j0, int const &k0,
                              CCTK_REAL const &x, CCTK_REAL const &y,
                              CCTK_REAL const &z,
                              const amrex::GpuArray<double, 3> &dx,
                              const amrex::GpuArray<double, 3> &plo) {
  // Computing the f(x-dx, y, z) value
  const CCTK_REAL m1_x = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo);
  // Computing the f(x+dx, y, z) value
  const CCTK_REAL p1_x = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo);
  // Computing the f(x, y-dy, z) value
  const CCTK_REAL m1_y = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo);
  // Computing the f(x, y+dy, z) value
  const CCTK_REAL p1_y = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo);
  // Computing the f(x, y, z-dz) value
  const CCTK_REAL m1_z = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo);
  // Computing the f(x, y, z+dz) value
  const CCTK_REAL p1_z = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo);

  // fill the gradient
  df[0] = (p1_x - m1_x) / (2.0 * dx[0]);
  df[1] = (p1_y - m1_y) / (2.0 * dx[1]);
  df[2] = (p1_z - m1_z) / (2.0 * dx[2]);
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
 * @param i0 Basis cell's index on the x direction.
 * @param j0 Basis cell's index on the x direction.
 * @param k0 Basis cell's index on the x direction.
 * @param x The x coordinate to evaluate.
 * @param y The y coordinate to evaluate.
 * @param z The z coordinate to evaluate.
 * @param dx The grid's cell \f$\Delta x\f$ vector.
 * @param plo The lower bound of the entire domain.
 */
template <int INTERPOLATION_ORDER, int DERIVATIVE_ORDER>
AMREX_GPU_HOST_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
vector_barycentric_derivative(
    amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 3>, 3> &df,
    amrex::Array4<CCTK_REAL const> const &f, int const &i0, int const &j0,
    int const &k0, CCTK_REAL const &x, CCTK_REAL const &y, CCTK_REAL const &z,
    const amrex::GpuArray<double, 3> &dx,
    const amrex::GpuArray<double, 3> &plo) {

  const CCTK_REAL m1_x_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 0);
  const CCTK_REAL p1_x_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 0);
  const CCTK_REAL m1_y_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 0);
  const CCTK_REAL p1_y_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 0);
  const CCTK_REAL m1_z_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 0);
  const CCTK_REAL p1_z_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 0);

  const CCTK_REAL m1_x_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 1);
  const CCTK_REAL p1_x_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 1);
  const CCTK_REAL m1_y_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 1);
  const CCTK_REAL p1_y_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 1);
  const CCTK_REAL m1_z_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 1);
  const CCTK_REAL p1_z_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 1);

  const CCTK_REAL m1_x_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 2);
  const CCTK_REAL p1_x_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 2);
  const CCTK_REAL m1_y_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 2);
  const CCTK_REAL p1_y_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 2);
  const CCTK_REAL m1_z_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 2);
  const CCTK_REAL p1_z_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 2);

  df[0][0] = (p1_x_0 - m1_x_0) / (2.0 * dx[0]);
  df[0][1] = (p1_x_1 - m1_x_1) / (2.0 * dx[0]);
  df[0][2] = (p1_x_2 - m1_x_2) / (2.0 * dx[0]);

  df[1][0] = (p1_y_0 - m1_y_0) / (2.0 * dx[1]);
  df[1][1] = (p1_y_1 - m1_y_1) / (2.0 * dx[1]);
  df[1][2] = (p1_y_2 - m1_y_2) / (2.0 * dx[1]);

  df[2][0] = (p1_z_0 - m1_z_0) / (2.0 * dx[2]);
  df[2][1] = (p1_z_1 - m1_z_1) / (2.0 * dx[2]);
  df[2][2] = (p1_z_2 - m1_z_2) / (2.0 * dx[2]);
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
 * @param i0 Basis cell's index on the x direction.
 * @param j0 Basis cell's index on the x direction.
 * @param k0 Basis cell's index on the x direction.
 * @param x The x coordinate to evaluate.
 * @param y The y coordinate to evaluate.
 * @param z The z coordinate to evaluate.
 * @param dx The grid's cell \f$\Delta x\f$ vector.
 * @param plo The lower bound of the entire domain.
 */
template <int INTERPOLATION_ORDER, int DERIVATIVE_ORDER>
AMREX_GPU_HOST_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
smatrix_barycentric_derivative(
    amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 6>, 3> &df,
    amrex::Array4<CCTK_REAL const> const &f, int const &i0, int const &j0,
    int const &k0, CCTK_REAL const &x, CCTK_REAL const &y, CCTK_REAL const &z,
    const amrex::GpuArray<double, 3> &dx,
    const amrex::GpuArray<double, 3> &plo) {

  const CCTK_REAL m1_x_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 0);
  const CCTK_REAL p1_x_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 0);
  const CCTK_REAL m1_y_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 0);
  const CCTK_REAL p1_y_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 0);
  const CCTK_REAL m1_z_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 0);
  const CCTK_REAL p1_z_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 0);

  const CCTK_REAL m1_x_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 1);
  const CCTK_REAL p1_x_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 1);
  const CCTK_REAL m1_y_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 1);
  const CCTK_REAL p1_y_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 1);
  const CCTK_REAL m1_z_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 1);
  const CCTK_REAL p1_z_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 1);

  const CCTK_REAL m1_x_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 2);
  const CCTK_REAL p1_x_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 2);
  const CCTK_REAL m1_y_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 2);
  const CCTK_REAL p1_y_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 2);
  const CCTK_REAL m1_z_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 2);
  const CCTK_REAL p1_z_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 2);

  const CCTK_REAL m1_x_3 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 3);
  const CCTK_REAL p1_x_3 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 3);
  const CCTK_REAL m1_y_3 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 - 1, k0, x, y + dx[1], z, dx, plo, 3);
  const CCTK_REAL p1_y_3 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 3);
  const CCTK_REAL m1_z_3 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 3);
  const CCTK_REAL p1_z_3 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 3);

  const CCTK_REAL m1_x_4 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 4);
  const CCTK_REAL p1_x_4 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 4);
  const CCTK_REAL m1_y_4 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 4);
  const CCTK_REAL p1_y_4 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 4);
  const CCTK_REAL m1_z_4 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 4);
  const CCTK_REAL p1_z_4 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 4);

  const CCTK_REAL m1_x_5 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 5);
  const CCTK_REAL p1_x_5 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 5);
  const CCTK_REAL m1_y_5 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 5);
  const CCTK_REAL p1_y_5 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 5);
  const CCTK_REAL m1_z_5 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 5);
  const CCTK_REAL p1_z_5 = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 5);

  df[0][0] = (p1_x_0 - m1_x_0) / (2.0 * dx[0]);
  df[0][1] = (p1_x_1 - m1_x_1) / (2.0 * dx[0]);
  df[0][2] = (p1_x_2 - m1_x_2) / (2.0 * dx[0]);
  df[0][3] = (p1_x_3 - m1_x_3) / (2.0 * dx[0]);
  df[0][4] = (p1_x_4 - m1_x_4) / (2.0 * dx[0]);
  df[0][5] = (p1_x_5 - m1_x_5) / (2.0 * dx[0]);

  df[1][0] = (p1_y_0 - m1_y_0) / (2.0 * dx[1]);
  df[1][1] = (p1_y_1 - m1_y_1) / (2.0 * dx[1]);
  df[1][2] = (p1_y_2 - m1_y_2) / (2.0 * dx[1]);
  df[1][3] = (p1_y_3 - m1_y_3) / (2.0 * dx[1]);
  df[1][4] = (p1_y_4 - m1_y_4) / (2.0 * dx[1]);
  df[1][5] = (p1_y_5 - m1_y_5) / (2.0 * dx[1]);

  df[2][0] = (p1_z_0 - m1_z_0) / (2.0 * dx[2]);
  df[2][1] = (p1_z_1 - m1_z_1) / (2.0 * dx[2]);
  df[2][2] = (p1_z_2 - m1_z_2) / (2.0 * dx[2]);
  df[2][3] = (p1_z_3 - m1_z_3) / (2.0 * dx[2]);
  df[2][4] = (p1_z_4 - m1_z_4) / (2.0 * dx[2]);
  df[2][5] = (p1_z_5 - m1_z_5) / (2.0 * dx[2]);
} // smatrix_barycentric_derivative

} // namespace Discretize

#endif // !DISCRETIZER_HXX
