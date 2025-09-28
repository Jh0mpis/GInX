/**
 * \file Interpolator.hxx
 * \brief Interpolators definitions.
 *
 * This file define the Barycentric interpolation in 1 and 3 dimensions.
 */
#ifndef INTERPOLATOR_HXX
#define INTERPOLATOR_HXX

#include "AMReX_Box.H"
#include "AMReX_Config.H"
#include "AMReX_MFIter.H"
#include "AMReX_REAL.H"
#include "AMReX_Random.H"
#include "AMReX_RandomEngine.H"
#include "AMReX_Scan.H"
#include "cctk_Types.h"
#include <array>
#include <cctk.h>
#include <iostream>

/**
 * \brief Interpolators namespace.
 */
namespace Interpolator {

// #############################################################################
//                   Barycentric Lagrange Interpolator
// #############################################################################

/**
 * \brief Do a Barycentric interpolation in one direction.
 *
 * This function computes the interpolation of a function on just one direction
 * using a barycentric Lagrange interpolation by doing:
 *
 * \f[
 * f(x) = \frac{\sum\limits_{i = 0}^N \frac{w_i}{x - x_i}f(x_i)}{\sum\limits_{i
 * = 0}^N \frac{w_i}{x - x_i}}
 * \f]
 *
 * where \f$N\f$ is the order of interpolation.
 *
 * @param points Vector containing the coordinates of each point.
 * @param weights The weights of each datapoint.
 * @param x The value where we are interpolating.
 * @param dx Vector \f$\Delta x\f$  with the space steps value.
 * @param plo Lower values of the entire domain.
 *
 * @return The interpolated value.
 */
template <int N>
AMREX_GPU_DEVICE AMREX_GPU_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
barycentric_cubic_1d(const int (&points)[N + 1],
                     const CCTK_REAL (&weights)[N + 1],
                     const CCTK_REAL (&values)[N + 1], const CCTK_REAL &x,
                     const CCTK_REAL &plo, const CCTK_REAL &dx) {
  CCTK_REAL num{0.0};
  CCTK_REAL den{0.0};

  for (int i = 0; i <= N; i++) {
    // Check if the point belongs to the consulted points.
    if (x == plo + points[i] * dx) {
      return values[i];
    }
    // Compute the weights and values
    CCTK_REAL term = weights[i] / (x - (plo + points[i] * dx));
    num += term * values[i];
    den += term;
  }

  // Return the interpolation
  return num / den;
}

/**
 * \brief Do a Barycentric interpolation in three direction.
 *
 * This function computes the interpolation of a function on three directions
 * using a barycentric Lagrange interpolation by doing:
 *
 * \f[
 * f(x, y, z) = \frac{\sum\limits_{i,j,k = 0}^N \frac{u_i}{z - z_i}\frac{v_j}{y
 * - y_j}\frac{w_k}{x - x_k}f(x_k, y_j, x_i)}{\sum\limits_{i,j,k = 0}^N
 * \frac{u_i}{z - z_i}\frac{v_j}{y - y_j}\frac{w_k}{x - x_k}} =
 * \frac{\sum\limits_{i=0}^N\frac{u_i}{z -
 * z_i}\left(\frac{\sum\limits_{j=0}^N\frac{v_j}{y-y_j}\left(\frac{\sum\limits_{k=0}^N
 * \frac{w_k}{x - x_k}f(x_k, y_j, x_i)}{\sum\limits_{k= 0}^N \frac{w_k}{x -
 * x_k}}\right)}{\sum\limits_{j= 0}^N \frac{v_j}{y -
 * y_j}}\right)}{\sum\limits_{i= 0}^N \frac{u_i}{z - z_i}}
 * \f]
 *
 * where \f$N\f$ is the order of interpolation.
 *
 * @see barycentric_cubic_1d
 *
 * @param f Array containing the function values.
 * @param i0 Basis cell index accordingly to x.
 * @param j0 Basis cell index accordingly to y.
 * @param k0 Basis cell index accordingly to z.
 * @param x Coordinate x value to interpolate.
 * @param y Coordinate y value to interpolate.
 * @param z Coordinate z value to interpolate.
 * @param dx Vector \f$\Delta x\f$  with the space steps value.
 * @param plo Lower values of the entire domain.
 * @param comp Function's component to compute.
 *
 * @return The interpolated value.
 */
template <int INTERPOLATION_ORDER>
AMREX_GPU_HOST_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
barycentric_cubic_3d(amrex::Array4<CCTK_REAL const> const &f, int const &i0,
                     int const &j0, int const &k0, CCTK_REAL const &x,
                     CCTK_REAL const &y, CCTK_REAL const &z,
                     const amrex::GpuArray<double, 3> &dx,
                     const amrex::GpuArray<double, 3> &plo, const int &comp) {

  // Setting the pre-computed weights.
  int nodes[INTERPOLATION_ORDER + 1];
  CCTK_REAL w[INTERPOLATION_ORDER + 1];

  if constexpr (INTERPOLATION_ORDER == 1) {
    nodes[0] = 0;
    nodes[1] = 1;
    w[0] = -1.;
    w[1] = 1.;
  } else if constexpr (INTERPOLATION_ORDER == 2) {
    nodes[0] = -1;
    nodes[1] = 0;
    nodes[2] = 1;
    w[0] = 0.5;
    w[1] = 1.;
    w[2] = 0.5;
  } else if constexpr (INTERPOLATION_ORDER == 3) {
    nodes[0] = -1;
    nodes[1] = 0;
    nodes[2] = 1;
    nodes[3] = 2;
    w[0] = (-1.0 / 6.0);
    w[1] = 0.5;
    w[2] = -0.5;
    w[3] = (1.0 / 6.0);
  } else if constexpr (INTERPOLATION_ORDER == 4) {
    nodes[0] = -2;
    nodes[1] = -1;
    nodes[2] = 0;
    nodes[3] = 1;
    nodes[4] = 2;
    w[0] = (1.0 / 24.0);
    w[1] = (-1.0 / 6.0);
    w[2] = 0.25;
    w[3] = (-1.0 / 6.0);
    w[4] = (1.0 / 24.0);
  } else {
    CCTK_INFO("Barycentric Lagrange interpolation of desired order is not yet "
              "implemented. Available orders: 1, 2, 3.");
    throw std::invalid_argument(
        "Wrong order of barycentric Lagrange interpolation");
  }

  // Do the interpolation on x
  CCTK_REAL G[INTERPOLATION_ORDER + 1][INTERPOLATION_ORDER + 1];
  for (int j = 0; j <= INTERPOLATION_ORDER; j++) {
    for (int k = 0; k <= INTERPOLATION_ORDER; k++) {
      CCTK_REAL values[INTERPOLATION_ORDER + 1];
      CCTK_INT points[INTERPOLATION_ORDER + 1];
      for (int i = 0; i <= INTERPOLATION_ORDER; i++) {
        values[i] = f(i0 + nodes[i], j0 + nodes[j], k0 + nodes[k], comp);
        points[i] = i0 + nodes[i];
      }
      G[j][k] = barycentric_cubic_1d<INTERPOLATION_ORDER>(points, w, values, x,
                                                          plo[0], dx[0]);
    }
  }

  // Do the interpolation on y
  CCTK_REAL H[INTERPOLATION_ORDER + 1];
  for (int k = 0; k <= INTERPOLATION_ORDER; k++) {
    CCTK_REAL values[INTERPOLATION_ORDER + 1];
    CCTK_INT points[INTERPOLATION_ORDER + 1];
    for (int j = 0; j <= INTERPOLATION_ORDER; j++) {
      values[j] = G[j][k];
      points[j] = j0 + nodes[j];
    }
    H[k] = barycentric_cubic_1d<INTERPOLATION_ORDER>(points, w, values, y,
                                                     plo[1], dx[1]);
  }

  // Do the interpolation on z
  CCTK_INT points[INTERPOLATION_ORDER + 1];
  for (int k = 0; k <= INTERPOLATION_ORDER; k++) {
    points[k] = k0 + nodes[k];
  }
  return barycentric_cubic_1d<INTERPOLATION_ORDER>(points, w, H, z, plo[2],
                                                   dx[2]);
} // barycentric_cubic_3d with component

/**
 * \brief Do a Barycentric interpolation in three direction for a scalar
 * function.
 *
 * This function computes the interpolation of a function on three directions
 * using a barycentric Lagrange interpolation by doing:
 *
 * \f[
 * f(x, y, z) = \frac{\sum\limits_{i,j,k = 0}^N \frac{u_i}{z - z_i}\frac{v_j}{y
 * - y_j}\frac{w_k}{x - x_k}f(x_k, y_j, x_i)}{\sum\limits_{i,j,k = 0}^N
 * \frac{u_i}{z - z_i}\frac{v_j}{y - y_j}\frac{w_k}{x - x_k}} =
 * \frac{\sum\limits_{i=0}^N\frac{u_i}{z -
 * z_i}\left(\frac{\sum\limits_{j=0}^N\frac{v_j}{y-y_j}\left(\frac{\sum\limits_{k=0}^N
 * \frac{w_k}{x - x_k}f(x_k, y_j, x_i)}{\sum\limits_{k= 0}^N \frac{w_k}{x -
 * x_k}}\right)}{\sum\limits_{j= 0}^N \frac{v_j}{y -
 * y_j}}\right)}{\sum\limits_{i= 0}^N \frac{u_i}{z - z_i}}
 * \f]
 *
 * where \f$N\f$ is the order of interpolation.
 *
 * @see barycentric_cubic_1d
 *
 * @param f Array containing the function values.
 * @param i0 Basis cell index accordingly to x.
 * @param j0 Basis cell index accordingly to y.
 * @param k0 Basis cell index accordingly to z.
 * @param x Coordinate x value to interpolate.
 * @param y Coordinate y value to interpolate.
 * @param z Coordinate z value to interpolate.
 * @param dx Vector \f$\Delta x\f$  with the space steps value.
 * @param plo Lower values of the entire domain.
 *
 * @return The interpolated value.
 */
template <int INTERPOLATION_ORDER>
AMREX_GPU_HOST_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
barycentric_cubic_3d(amrex::Array4<CCTK_REAL const> const &f, int const &i0,
                     int const &j0, int const &k0, CCTK_REAL const &x,
                     CCTK_REAL const &y, CCTK_REAL const &z,
                     const amrex::GpuArray<double, 3> &dx,
                     const amrex::GpuArray<double, 3> &plo) {

  return barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0, x, y, z, dx,
                                                   plo, 0);
} // barycentric_cubic_3d No component

/**
 * \brief Do a Barycentric interpolation and its derivative in one direction.
 *
 * This function computes the interpolation of a function on just one direction
 * using a barycentric Lagrange interpolation by doing:
 *
 * \f[
 * f(x) = \frac{\sum\limits_{i = 0}^N \frac{w_i}{x - x_i}f(x_i)}{\sum\limits_{i
 * = 0}^N \frac{w_i}{x - x_i}}
 * \f]
 *
 * where \f$N\f$ is the order of interpolation.
 *
 * And at the same time computes the derivative on the desired direction by
 * doing:
 *
 * \f[
 * f'(x) = \sum\limits_{i = 0}^N f(x_i)\frac{d}{dx}\left(\frac{\frac{w_i}{x -
 * x_i}}{\sum\limits_{i = 0}^N \frac{w_i}{x - x_i}}\right) = \sum\limits_{i =
 * 0}^N f(x_i)\frac{\frac{w_j}{(x-x_j)}\sum\limits_{m =
 * 0}^N\frac{w_m}{(x-x_m)^2} - \frac{w_j}{(x-x_j)^2}\sum\limits_{i = 0}^N
 * \frac{w_i}{x - x_i}}{\left(\sum\limits_{i = 0}^N \frac{w_i}{x -
 * x_i}\right)^2}
 * \f]
 *
 * @see barycentric_cubic_3d
 * @see barycentric_cubic_1d
 *
 * @param f_x Reference to the final interpolated function value.
 * @param d_f_x Reference to the final interpolated derivative function value.
 * @param points Vector containing the coordinates of each point.
 * @param weights The weights of each datapoint.
 * @param x The value where we are interpolating.
 * @param plo Lower values of the entire domain.
 * @param dx Vector \f$\Delta x\f$  with the space steps value.
 */
template <int N>
AMREX_GPU_DEVICE AMREX_GPU_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
der_barycentric_cubic_1d(CCTK_REAL &f_x, CCTK_REAL &d_f_x,
                         const int (&points)[N + 1],
                         const CCTK_REAL (&weights)[N + 1],
                         const CCTK_REAL (&values)[N + 1], const CCTK_REAL &x,
                         const CCTK_REAL &plo, const CCTK_REAL &dx) {
  CCTK_REAL num{0.0};
  CCTK_REAL den{0.0};
  CCTK_REAL den_sqr{0.0};
  CCTK_REAL der_num{0.0};

  // Compute the interpolation
  for (int i = 0; i <= N; i++) {
    if (x == plo + points[i] * dx) {
      // Check if the point makes part of the points used on the interpolation.
      for (int j = 0; j <= N; j++) {
        if (i == j) {
          continue;
          d_f_x +=
              weights[j] * (values[i] - values[j]) / (x - plo + points[j] * dx);
        }
      }
      d_f_x /= weights[i];
      f_x = values[i];
      d_f_x = 0.0;
      return;
    }

    // Compute the weights for the interpolation and the derivative.
    CCTK_REAL term = weights[i] / (x - (plo + points[i] * dx));
    num += term * values[i];
    den += term;
    den_sqr += term / (x - (plo + points[i] * dx));
  }

  // Compute the weights for the derivative.
  for (int i = 0; i <= N; i++) {
    CCTK_REAL term = weights[i] / (x - (plo + points[i] * dx));
    der_num += (-term * den / (x - (plo + points[i] * dx)) + term * den_sqr) *
               values[i];
  }

  // Fill the values
  f_x = num / den;
  d_f_x = der_num / (den * den);
} // der_barycentric_cubic_1d

/**
 * \brief Do a Barycentric interpolation and its derivatives in three
 * directions.
 *
 * This function computes the interpolation of a function on three directions
 * using a barycentric Lagrange interpolation by doing:
 *
 * \f[
 * f(x, y, z) = \frac{\sum\limits_{i,j,k = 0}^N \frac{u_i}{z - z_i}\frac{v_j}{y
 * - y_j}\frac{w_k}{x - x_k}f(x_k, y_j, x_i)}{\sum\limits_{i,j,k = 0}^N
 * \frac{u_i}{z - z_i}\frac{v_j}{y - y_j}\frac{w_k}{x - x_k}} =
 * \frac{\sum\limits_{i=0}^N\frac{u_i}{z -
 * z_i}\left(\frac{\sum\limits_{j=0}^N\frac{v_j}{y-y_j}\left(\frac{\sum\limits_{k=0}^N
 * \frac{w_k}{x - x_k}f(x_k, y_j, x_i)}{\sum\limits_{k= 0}^N \frac{w_k}{x -
 * x_k}}\right)}{\sum\limits_{j= 0}^N \frac{v_j}{y -
 * y_j}}\right)}{\sum\limits_{i= 0}^N \frac{u_i}{z - z_i}}
 * \f]
 *
 * where \f$N\f$ is the order of interpolation.
 *
 * But at the same time computes the gradient of the same function by doing:
 *
 * \f[
 * \frac{\partial}{\partial x}f(x, y, z) = \frac{\sum\limits_{i,j,k = 0}^N
 * \frac{u_i}{z - z_i}\frac{v_j}{y
 * - y_j}\frac{d}{dx}\left(\frac{\frac{w_k}{x - x_k}}{\sum\limits_{l =
 * 0}^N\frac{w_l}{x - x_l}}\right)f(x_k, y_j, x_i)}{\sum\limits_{i,j = 0}^N
 * \frac{u_i}{z - z_i}\frac{v_j}{y - y_j}},
 * \f]
 *
 * \f[
 * \frac{\partial}{\partial y}f(x, y, z) = \frac{\sum\limits_{i,j,k = 0}^N
 * \frac{u_i}{z - z_i}\frac{d}{dy}\left(\frac{\frac{v_j}{y -
 * y_j}}{\sum\limits_{l = 0}^N\frac{v_l}{y - y_l}}\right)\frac{w_k}{x -
 * x_k}f(x_k, y_j, x_i)}{\sum\limits_{i,k = 0}^N
 * \frac{u_i}{z - z_i}\frac{w_k}{x - x_k}},
 * \f]
 *
 * \f[
 * \frac{\partial}{\partial z}f(x, y, z) = \frac{\sum\limits_{i,j,k = 0}^N
 * \frac{d}{dz}\left(\frac{\frac{u_i}{z - z_i}}{\sum\limits_{l =
 * 0}^N\frac{u_l}{z - z_l}}\right)\frac{v_j}{y
 * - y_j}\frac{w_k}{x - x_k}f(x_k, y_j, x_i)}{\sum\limits_{i,k = 0}^N
 * \frac{v_j}{y - y_j}\frac{w_k}{x - x_k}}.
 * \f]
 *
 * by calling the function der_barycentric_cubic_1d().
 *
 * @see der_barycentric_cubic_1d
 *
 * @param f_xyz Reference to the interpolated value, i.e., \f$f(x,y,z)\f$.
 * @param df_xyz_0 Reference to the interpolated derivative value on the
 * direction 0, i.e., \f$\partial_xf(x,y,z)\f$.
 * @param df_xyz_1 Reference to the interpolated derivative value on the
 * direction 1, i.e., \f$\partial_yf(x,y,z)\f$.
 * @param df_xyz_2 Reference to the interpolated derivative value on the
 * direction 2, i.e., \f$\partial_zf(x,y,z)\f$.
 * @param f Array containing the function values.
 * @param i0 Basis cell index accordingly to x.
 * @param j0 Basis cell index accordingly to y.
 * @param k0 Basis cell index accordingly to z.
 * @param x Coordinate x value to interpolate.
 * @param y Coordinate y value to interpolate.
 * @param z Coordinate z value to interpolate.
 * @param dx Vector \f$\Delta x\f$  with the space steps value.
 * @param plo Lower values of the entire domain.
 * @param comp Function's component to compute.
 */
template <int INTERPOLATION_ORDER>
AMREX_GPU_HOST_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
barycentric_derivative_and_interpolate(
    CCTK_REAL &f_xyz, CCTK_REAL &df_xyz_0, CCTK_REAL &df_xyz_1,
    CCTK_REAL &df_xyz_2, amrex::Array4<CCTK_REAL const> const &f, int const &i0,
    int const &j0, int const &k0, CCTK_REAL const &x, CCTK_REAL const &y,
    CCTK_REAL const &z, const amrex::GpuArray<double, 3> &dx,
    const amrex::GpuArray<double, 3> &plo, const int &comp) {

  // Setting the pre-computed weights.
  int nodes[INTERPOLATION_ORDER + 1];
  CCTK_REAL w[INTERPOLATION_ORDER + 1];

  if constexpr (INTERPOLATION_ORDER == 1) {
    nodes[0] = 0;
    nodes[1] = 1;
    w[0] = -1.;
    w[1] = 1.;
  } else if constexpr (INTERPOLATION_ORDER == 2) {
    nodes[0] = -1;
    nodes[1] = 0;
    nodes[2] = 1;
    w[0] = 0.5;
    w[1] = 1.;
    w[2] = 0.5;
  } else if constexpr (INTERPOLATION_ORDER == 3) {
    nodes[0] = -1;
    nodes[1] = 0;
    nodes[2] = 1;
    nodes[3] = 2;
    w[0] = (-1.0 / 6.0);
    w[1] = 0.5;
    w[2] = -0.5;
    w[3] = (1.0 / 6.0);
  } else if constexpr (INTERPOLATION_ORDER == 4) {
    nodes[0] = -2;
    nodes[1] = -1;
    nodes[2] = 0;
    nodes[3] = 1;
    nodes[4] = 2;
    w[0] = (1.0 / 24.0);
    w[1] = (-1.0 / 6.0);
    w[2] = 0.25;
    w[3] = (-1.0 / 6.0);
    w[4] = (1.0 / 24.0);
  } else {
    CCTK_INFO("Barycentric Lagrange interpolation of desired order is not yet "
              "implemented. Available orders: 1, 2, 3.");
    throw std::invalid_argument(
        "Wrong order of barycentric Lagrange interpolation");
  }

  // Computing f(x, y_i, z_i)
  CCTK_REAL G_xyz[INTERPOLATION_ORDER + 1][INTERPOLATION_ORDER + 1];
  // Computing d/dx f(x, y_i, z_i)
  CCTK_REAL G_dxyz[INTERPOLATION_ORDER + 1][INTERPOLATION_ORDER + 1];
  for (int j = 0; j <= INTERPOLATION_ORDER; j++) {
    for (int k = 0; k <= INTERPOLATION_ORDER; k++) {
      CCTK_REAL values[INTERPOLATION_ORDER + 1];
      CCTK_INT points[INTERPOLATION_ORDER + 1];
      for (int i = 0; i <= INTERPOLATION_ORDER; i++) {
        values[i] = f(i0 + nodes[i], j0 + nodes[j], k0 + nodes[k], comp);
        points[i] = i0 + nodes[i];
      }
      der_barycentric_cubic_1d<INTERPOLATION_ORDER>(
          G_xyz[j][k], G_dxyz[j][k], points, w, values, x, plo[0], dx[0]);
    }
  }

  // Computing f(x, y, z_i)
  CCTK_REAL H_xyz[INTERPOLATION_ORDER + 1];
  // Computing d/dx f(x, y, z_i)
  CCTK_REAL H_dxyz[INTERPOLATION_ORDER + 1];
  // Computing d/dy f(x, y, z_i)
  CCTK_REAL H_xdyz[INTERPOLATION_ORDER + 1];
  for (int k = 0; k <= INTERPOLATION_ORDER; k++) {
    CCTK_REAL values[INTERPOLATION_ORDER + 1];
    CCTK_REAL d_values[INTERPOLATION_ORDER + 1];
    CCTK_INT points[INTERPOLATION_ORDER + 1];
    for (int j = 0; j <= INTERPOLATION_ORDER; j++) {
      values[j] = G_xyz[j][k];
      d_values[j] = G_dxyz[j][k];
      points[j] = j0 + nodes[j];
    }
    der_barycentric_cubic_1d<INTERPOLATION_ORDER>(H_xyz[k], H_xdyz[k], points,
                                                  w, values, y, plo[1], dx[1]);
    H_dxyz[k] = barycentric_cubic_1d<INTERPOLATION_ORDER>(points, w, d_values,
                                                          y, plo[1], dx[1]);
  }

  CCTK_INT points[INTERPOLATION_ORDER + 1];
  for (int k = 0; k <= INTERPOLATION_ORDER; k++) {
    points[k] = k0 + nodes[k];
  }
  // Computing f(x, y, z)
  // Computing d/dz f(x, y, z)
  der_barycentric_cubic_1d<INTERPOLATION_ORDER>(f_xyz, df_xyz_2, points, w,
                                                H_xyz, z, plo[2], dx[2]);
  // Computing d/dx f(x, y, z)
  df_xyz_0 = barycentric_cubic_1d<INTERPOLATION_ORDER>(points, w, H_dxyz, z,
                                                       plo[2], dx[2]);
  // Computing d/dy f(x, y, z)
  df_xyz_1 = barycentric_cubic_1d<INTERPOLATION_ORDER>(points, w, H_xdyz, z,
                                                       plo[2], dx[2]);
} // barycentric_derivative_and_interpolate

} // namespace Interpolator

#endif // !INTERPOLATOR_HXX
