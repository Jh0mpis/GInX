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
 * Do a Barycentric interpolation in one direction.
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
    if (x == plo + points[i] * dx) {
      return values[i];
    }
    CCTK_REAL term = weights[i] / (x - (plo + points[i] * dx));
    num += term * values[i];
    den += term;
  }

  return num / den;
}

/**
 * Do a Barycentric interpolation in three direction.
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

  int nodes[INTERPOLATION_ORDER + 1];
  CCTK_REAL w[INTERPOLATION_ORDER + 1];
  CCTK_REAL v[INTERPOLATION_ORDER + 1];
  CCTK_REAL u[INTERPOLATION_ORDER + 1];

  if constexpr (INTERPOLATION_ORDER == 1) {
    nodes[0] = 0;
    nodes[1] = 1;
    w[0] = -1.;
    w[1] = 1.;
    v[0] = -1.;
    v[1] = 1.;
    u[0] = -1.;
    u[1] = 1.;
  } else if constexpr (INTERPOLATION_ORDER == 2) {
    nodes[0] = -1;
    nodes[1] = 0;
    nodes[2] = 1;
    w[0] = 0.5;
    w[1] = 1.;
    w[2] = 0.5;
    v[0] = 0.5;
    v[1] = 1.;
    v[2] = 0.5;
    u[0] = 0.5;
    u[1] = 1.;
    u[2] = 0.5;
  } else if constexpr (INTERPOLATION_ORDER == 3) {
    nodes[0] = -1;
    nodes[1] = 0;
    nodes[2] = 1;
    nodes[3] = 2;
    w[0] = (-1.0 / 6.0);
    w[1] = 0.5;
    w[2] = -0.5;
    w[3] = (1.0 / 6.0);
    v[0] = (-1.0 / 6.0);
    v[1] = 0.5;
    v[2] = -0.5;
    v[3] = (1.0 / 6.0);
    u[0] = (-1.0 / 6.0);
    u[1] = 0.5;
    u[2] = -0.5;
    u[3] = (1.0 / 6.0);
  } else {
    CCTK_INFO("Barycentric Lagrange interpolation of desired order is not yet "
              "implemented. Available orders: 1, 2, 3.");
    throw std::invalid_argument(
        "Wrong order of barycentric Lagrange interpolation");
  }

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

  CCTK_REAL H[INTERPOLATION_ORDER + 1];
  for (int k = 0; k <= INTERPOLATION_ORDER; k++) {
    CCTK_REAL values[INTERPOLATION_ORDER + 1];
    CCTK_INT points[INTERPOLATION_ORDER + 1];
    for (int j = 0; j <= INTERPOLATION_ORDER; j++) {
      values[j] = G[j][k];
      points[j] = j0 + nodes[j];
    }
    H[k] = barycentric_cubic_1d<INTERPOLATION_ORDER>(points, v, values, y,
                                                     plo[1], dx[1]);
  }

  CCTK_INT points[INTERPOLATION_ORDER + 1];
  for (int k = 0; k <= INTERPOLATION_ORDER; k++) {
    points[k] = k0 + nodes[k];
  }
  return barycentric_cubic_1d<INTERPOLATION_ORDER>(points, u, H, z, plo[2],
                                                   dx[2]);
} // barycentric_cubic_3d with component

/**
 * Do a Barycentric interpolation in three direction for a scalar function.
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

} // namespace Interpolator

#endif // !INTERPOLATOR_HXX
