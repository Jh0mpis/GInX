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
 * This function computes the interpolation of a function on just one
 * direction using a barycentric Lagrange interpolation by doing:
 *
 * \f[
 * f(x) = \frac{\sum\limits_{i = 0}^N \frac{w_i}{x -
 * x_i}f(x_i)}{\sum\limits_{i = 0}^N \frac{w_i}{x - x_i}}
 * \f]
 *
 * where \f$N\f$ is the order of interpolation.
 *
 * @param value The interpolated value reference.
 * @param points Vector containing the coordinates of each point.
 * @param weights The weights of each datapoint.
 * @param x The value where we are interpolating.
 * @param dx Vector \f$\Delta x\f$  with the space steps value.
 * @param plo Lower values of the entire domain.
 */
template <int N>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE void
barycentric_cubic_1d(CCTK_REAL &value, const int (&points)[N],
                     const CCTK_REAL *weights, const CCTK_REAL (&values)[N],
                     const CCTK_REAL &x, const CCTK_REAL &plo,
                     const CCTK_REAL &dx) {
  CCTK_REAL num{0.0};
  CCTK_REAL den{0.0};

  for (int i = 0; i < N; i++) {
    // Check if the point belongs to the consulted points.
    CCTK_REAL diff = x - (plo + points[i] * dx);
    const CCTK_REAL tolerance = 1e-12;
    if (diff < tolerance && diff > -tolerance) {
      value = values[i];
      return;
    }
    // Compute the weights and values
    CCTK_REAL term = weights[i] / diff;
    num += term * values[i];
    den += term;
  }

  // Return the interpolation
  value = num / den;
}

/**
 * \brief Do a Barycentric interpolation in three direction.
 *
 * This function computes the interpolation of a function on three directions
 * using a barycentric Lagrange interpolation by doing:
 *
 * \f[
 * f(x, y, z) = \frac{\sum\limits_{i,j,k = 0}^N \frac{u_i}{z -
 * z_i}\frac{v_j}{y
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
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_REAL
barycentric_cubic_3d(const amrex::Array4<CCTK_REAL const> &f,
                     const long int &i0, const long int &j0, const long int &k0,
                     const CCTK_REAL &x, const CCTK_REAL &y, const CCTK_REAL &z,
                     const amrex::GpuArray<double, 3> &dx,
                     const amrex::GpuArray<double, 3> &plo, const int &comp) {
  const int order =
      (((INTERPOLATION_ORDER - 1) * INTERPOLATION_ORDER) >> 1) - 1;
  /**
   * \brief Nodes interpolator's array.
   *
   * Contains the nodes that we are going to use for the orders from 2 to 5.
   */
  const CCTK_INT all_nodes[14] = {0, 1, -1, 0, 1, -1, 0, 1, 2, -2, -1, 0, 1, 2};

  /**
   * \brief Weights interpolator's array.
   *
   * Contains the weights that we are going to use for the orders from 2 to 5.
   */
  const CCTK_REAL all_weights[14] = {
      -1.,  1.,        0.5,        -1.,        0.5,  -1.0 / 6.0, 0.5,
      -0.5, 1.0 / 6.0, 1.0 / 24.0, -1.0 / 6.0, 0.25, -1.0 / 6.0, 1.0 / 24.0};
  const CCTK_INT *nodes = &all_nodes[order];
  const CCTK_REAL *w = &all_weights[order];

  // Do the interpolation on x
  CCTK_REAL G[INTERPOLATION_ORDER][INTERPOLATION_ORDER];
  for (int j = 0; j < INTERPOLATION_ORDER; j++) {
    for (int k = 0; k < INTERPOLATION_ORDER; k++) {
      CCTK_REAL values[INTERPOLATION_ORDER];
      CCTK_INT points[INTERPOLATION_ORDER];
      for (int i = 0; i < INTERPOLATION_ORDER; i++) {
        values[i] = f(i0 + nodes[i], j0 + nodes[j], k0 + nodes[k], comp);
        points[i] = i0 + nodes[i];
      }
      barycentric_cubic_1d<INTERPOLATION_ORDER>(G[j][k], points, w, values, x,
                                                plo[0], dx[0]);
    }
  }

  // Do the interpolation on y
  CCTK_REAL H[INTERPOLATION_ORDER];
  for (int k = 0; k < INTERPOLATION_ORDER; k++) {
    CCTK_REAL values[INTERPOLATION_ORDER];
    CCTK_INT points[INTERPOLATION_ORDER];
    for (int j = 0; j < INTERPOLATION_ORDER; j++) {
      values[j] = G[j][k];
      points[j] = j0 + nodes[j];
    }
    barycentric_cubic_1d<INTERPOLATION_ORDER>(H[k], points, w, values, y,
                                              plo[1], dx[1]);
  }

  // Do the interpolation on z
  CCTK_INT points[INTERPOLATION_ORDER];
  for (int k = 0; k < INTERPOLATION_ORDER; k++) {
    points[k] = k0 + nodes[k];
  }
  CCTK_REAL value;
  barycentric_cubic_1d<INTERPOLATION_ORDER>(value, points, w, H, z, plo[2],
                                            dx[2]);
  return value;
} // barycentric_cubic_3d with component

/**
 * \brief Do a Barycentric interpolation in three direction for a scalar
 * function.
 *
 * This function computes the interpolation of a function on three directions
 * using a barycentric Lagrange interpolation by doing:
 *
 * \f[
 * f(x, y, z) = \frac{\sum\limits_{i,j,k = 0}^N \frac{u_i}{z -
 * z_i}\frac{v_j}{y
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
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_REAL
barycentric_cubic_3d(const amrex::Array4<CCTK_REAL const> &f,
                     const long int &i0, const long int &j0, const long int &k0,
                     const CCTK_REAL &x, const CCTK_REAL &y, const CCTK_REAL &z,
                     const amrex::GpuArray<double, 3> &dx,
                     const amrex::GpuArray<double, 3> &plo) {

  return barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0, x, y, z, dx,
                                                   plo, 0);
} // barycentric_cubic_3d No component

/**
 * \brief Do a Barycentric interpolation and its derivative in one direction.
 *
 * This function computes the interpolation of a function on just one
 * direction using a barycentric Lagrange interpolation by doing:
 *
 * \f[
 * f(x) = \frac{\sum\limits_{i = 0}^N \frac{w_i}{x -
 * x_i}f(x_i)}{\sum\limits_{i = 0}^N \frac{w_i}{x - x_i}}
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
AMREX_GPU_DEVICE
    AMREX_GPU_HOST AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE void
    der_barycentric_cubic_1d(CCTK_REAL &f_x, CCTK_REAL &d_f_x,
                             const int (&points)[N], const CCTK_REAL *weights,
                             const CCTK_REAL (&values)[N], const CCTK_REAL &x,
                             const CCTK_REAL &plo, const CCTK_REAL &dx) {
  CCTK_REAL num{0.0};
  CCTK_REAL den{0.0};
  CCTK_REAL den_sqr{0.0};
  CCTK_REAL der_num{0.0};
  d_f_x = 0.0;

  // Compute the interpolation
  for (int i = 0; i < N; i++) {
    CCTK_REAL diff = x - (plo + points[i] * dx);
    const CCTK_REAL tolerance = 1e-12;
    if (diff < tolerance && diff > -tolerance) {
      // Check if the point makes part of the points used on the
      // interpolation.
      for (int j = 0; j < N; j++) {
        if (i == j) {
          continue;
        }
        const auto x_j = points[j] * dx;
        const auto x_i = points[i] * dx;
        d_f_x += weights[j] * (values[i] - values[j]) / (x_j - x_i);
      }
      d_f_x /= weights[i];
      f_x = values[i];
      return;
    }

    // Compute the weights for the interpolation and the derivative.
    CCTK_REAL term = weights[i] / diff;
    num += term * values[i];
    den += term;
    den_sqr += term / diff;
  }

  // Compute the weights for the derivative.
  for (int i = 0; i < N; i++) {
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
 * f(x, y, z) = \frac{\sum\limits_{i,j,k = 0}^N \frac{u_i}{z -
 * z_i}\frac{v_j}{y
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
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE void
barycentric_derivative_and_interpolate(CCTK_REAL &f_xyz, CCTK_REAL &df_xyz_0,
                                       CCTK_REAL &df_xyz_1, CCTK_REAL &df_xyz_2,
                                       amrex::Array4<CCTK_REAL const> const &f,
                                       const long int &i0, const long int &j0,
                                       const long int &k0, const CCTK_REAL &x,
                                       const CCTK_REAL &y, const CCTK_REAL &z,
                                       const amrex::GpuArray<double, 3> &dx,
                                       const amrex::GpuArray<double, 3> &plo,
                                       const int &comp) {

  const int order =
      (((INTERPOLATION_ORDER - 1) * INTERPOLATION_ORDER) >> 1) - 1;
  /**
   * \brief Nodes interpolator's array.
   *
   * Contains the nodes that we are going to use for the orders from 2 to 5.
   */
  const CCTK_INT all_nodes[14] = {0, 1, -1, 0, 1, -1, 0, 1, 2, -2, -1, 0, 1, 2};

  /**
   * \brief Weights interpolator's array.
   *
   * Contains the weights that we are going to use for the orders from 2 to 5.
   */
  const CCTK_REAL all_weights[14] = {
      -1.,  1.,        0.5,        -1.,        0.5,  -1.0 / 6.0, 0.5,
      -0.5, 1.0 / 6.0, 1.0 / 24.0, -1.0 / 6.0, 0.25, -1.0 / 6.0, 1.0 / 24.0};
  const CCTK_INT *nodes = &all_nodes[order];
  const CCTK_REAL *w = &all_weights[order];

  // Computing f(x, y_i, z_i)
  CCTK_REAL G_xyz[INTERPOLATION_ORDER][INTERPOLATION_ORDER];
  // Computing d/dx f(x, y_i, z_i)
  CCTK_REAL G_dxyz[INTERPOLATION_ORDER][INTERPOLATION_ORDER];
  for (int j = 0; j < INTERPOLATION_ORDER; j++) {
    for (int k = 0; k < INTERPOLATION_ORDER; k++) {
      CCTK_REAL values[INTERPOLATION_ORDER];
      CCTK_INT points[INTERPOLATION_ORDER];
      for (int i = 0; i < INTERPOLATION_ORDER; i++) {
        values[i] = f(i0 + nodes[i], j0 + nodes[j], k0 + nodes[k], comp);
        points[i] = i0 + nodes[i];
      }
      der_barycentric_cubic_1d<INTERPOLATION_ORDER>(
          G_xyz[j][k], G_dxyz[j][k], points, w, values, x, plo[0], dx[0]);
    }
  }

  // Computing f(x, y, z_i)
  CCTK_REAL H_xyz[INTERPOLATION_ORDER];
  // Computing d/dx f(x, y, z_i)
  CCTK_REAL H_dxyz[INTERPOLATION_ORDER];
  // Computing d/dy f(x, y, z_i)
  CCTK_REAL H_xdyz[INTERPOLATION_ORDER];
  for (int k = 0; k < INTERPOLATION_ORDER; k++) {
    CCTK_REAL values[INTERPOLATION_ORDER];
    CCTK_REAL d_values[INTERPOLATION_ORDER];
    CCTK_INT points[INTERPOLATION_ORDER];
    for (int j = 0; j < INTERPOLATION_ORDER; j++) {
      values[j] = G_xyz[j][k];
      d_values[j] = G_dxyz[j][k];
      points[j] = j0 + nodes[j];
    }
    der_barycentric_cubic_1d<INTERPOLATION_ORDER>(H_xyz[k], H_xdyz[k], points,
                                                  w, values, y, plo[1], dx[1]);
    barycentric_cubic_1d<INTERPOLATION_ORDER>(H_dxyz[k], points, w, d_values, y,
                                              plo[1], dx[1]);
  }

  CCTK_INT points[INTERPOLATION_ORDER];
  for (int k = 0; k < INTERPOLATION_ORDER; k++) {
    points[k] = k0 + nodes[k];
  }
  // Computing f(x, y, z)
  // Computing d/dz f(x, y, z)
  der_barycentric_cubic_1d<INTERPOLATION_ORDER>(f_xyz, df_xyz_2, points, w,
                                                H_xyz, z, plo[2], dx[2]);
  // Computing d/dx f(x, y, z)
  barycentric_cubic_1d<INTERPOLATION_ORDER>(df_xyz_0, points, w, H_dxyz, z,
                                            plo[2], dx[2]);
  // Computing d/dy f(x, y, z)
  barycentric_cubic_1d<INTERPOLATION_ORDER>(df_xyz_1, points, w, H_xdyz, z,
                                            plo[2], dx[2]);
} // barycentric_derivative_and_interpolate

/**
 * \brief Do the barycentric interpolation for a 3 dimensional and symmetric
 * matrix.
 *
 * This function computes the interpolation for a 3 dimensional and symmetric
 * matrix, such as the induced metric and the extrinsic curvature tensor in just
 * one call using the barycentric_cubic_1d function similar to the
 * barycentric_cubic_3d function.
 *
 * @see barycentric_cubic_3d
 * @see barycentric_cubic_1d
 *
 * @param array6 Reference to GpuArray of size 6 that is going to store the
 * interpolated values.
 * @param f Array containing the function values.
 * @param i0 Basis cell index accordingly to x.
 * @param j0 Basis cell index accordingly to y.
 * @param k0 Basis cell index accordingly to z.
 * @param x Coordinate x value to interpolate.
 * @param y Coordinate y value to interpolate.
 * @param z Coordinate z value to interpolate.
 * @param dx Vector \f$\Delta x\f$  with the space steps value.
 * @param plo Lower values of the entire domain.
 */
template <int INTERPOLATION_ORDER>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE void
interpolate_array(amrex::GpuArray<CCTK_REAL, 6> &array6,
                  const amrex::Array4<CCTK_REAL const> &f, const long int &i0,
                  const long int &j0, const long int &k0, const CCTK_REAL &x,
                  const CCTK_REAL &y, const CCTK_REAL &z,
                  const amrex::GpuArray<double, 3> &dx,
                  const amrex::GpuArray<double, 3> &plo) {
  const int order =
      (((INTERPOLATION_ORDER - 1) * INTERPOLATION_ORDER) >> 1) - 1;
  /**
   * \brief Nodes interpolator's array.
   *
   * Contains the nodes that we are going to use for the orders from 2 to 5.
   */
  const CCTK_INT all_nodes[14] = {0, 1, -1, 0, 1, -1, 0, 1, 2, -2, -1, 0, 1, 2};

  /**
   * \brief Weights interpolator's array.
   *
   * Contains the weights that we are going to use for the orders from 2 to 5.
   */
  const CCTK_REAL all_weights[14] = {
      -1.,  1.,        0.5,        -1.,        0.5,  -1.0 / 6.0, 0.5,
      -0.5, 1.0 / 6.0, 1.0 / 24.0, -1.0 / 6.0, 0.25, -1.0 / 6.0, 1.0 / 24.0};
  const CCTK_INT *nodes = &all_nodes[order];
  const CCTK_REAL *w = &all_weights[order];

  for (int comp = 0; comp < 6; comp++) {

    // Do the interpolation on x
    CCTK_REAL G[INTERPOLATION_ORDER][INTERPOLATION_ORDER];
    for (int j = 0; j < INTERPOLATION_ORDER; j++) {
      for (int k = 0; k < INTERPOLATION_ORDER; k++) {
        CCTK_REAL values[INTERPOLATION_ORDER];
        CCTK_INT points[INTERPOLATION_ORDER];
        for (int i = 0; i < INTERPOLATION_ORDER; i++) {
          values[i] = f(i0 + nodes[i], j0 + nodes[j], k0 + nodes[k], comp);
          points[i] = i0 + nodes[i];
        }
        barycentric_cubic_1d<INTERPOLATION_ORDER>(G[j][k], points, w, values, x,
                                                  plo[0], dx[0]);
      }
    }

    // Do the interpolation on y
    CCTK_REAL H[INTERPOLATION_ORDER];
    for (int k = 0; k < INTERPOLATION_ORDER; k++) {
      CCTK_REAL values[INTERPOLATION_ORDER];
      CCTK_INT points[INTERPOLATION_ORDER];
      for (int j = 0; j < INTERPOLATION_ORDER; j++) {
        values[j] = G[j][k];
        points[j] = j0 + nodes[j];
      }
      barycentric_cubic_1d<INTERPOLATION_ORDER>(H[k], points, w, values, y,
                                                plo[1], dx[1]);
    }

    // Do the interpolation on z
    CCTK_INT points[INTERPOLATION_ORDER];
    for (int k = 0; k < INTERPOLATION_ORDER; k++) {
      points[k] = k0 + nodes[k];
    }
    barycentric_cubic_1d<INTERPOLATION_ORDER>(array6[comp], points, w, H, z,
                                              plo[2], dx[2]);
  }
} // interpolate_array

/**
 * \brief Do the barycentric interpolation for a 3 dimensional and symmetric
 * matrix and its derivatives.
 *
 * This function computes the interpolation for a 3 dimensional and symmetric
 * matrix and its derivatives, such as the induced metric and the extrinsic
 * curvature tensor in just one call using the der_barycentric_cubic_1d function
 * similarly to the barycentric_derivative_and_interpolate function.
 *
 * @see barycentric_derivative_and_interpolate
 * @see der_barycentric_cubic_1d
 *
 * @param array6 Reference to the GpuArray of size 6 that is going to store the
 * interpolated values.
 * @param d_array6 Reference to the GpuArray of size 6 x 3 that stores all the
 * derivatives values.
 * @param f Array containing the function values.
 * @param i0 Basis cell index accordingly to x.
 * @param j0 Basis cell index accordingly to y.
 * @param k0 Basis cell index accordingly to z.
 * @param x Coordinate x value to interpolate.
 * @param y Coordinate y value to interpolate.
 * @param z Coordinate z value to interpolate.
 * @param dx Vector \f$\Delta x\f$  with the space steps value.
 * @param plo Lower values of the entire domain.
 */
template <int INTERPOLATION_ORDER>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE void
d_interpolate_array(amrex::GpuArray<CCTK_REAL, 6> &array6,
                    amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 6>, 3> &d_array6,
                    const amrex::Array4<CCTK_REAL const> &f, const long int &i0,
                    const long int &j0, const long int &k0, const CCTK_REAL &x,
                    const CCTK_REAL &y, const CCTK_REAL &z,
                    const amrex::GpuArray<double, 3> &dx,
                    const amrex::GpuArray<double, 3> &plo) {
  const int order =
      (((INTERPOLATION_ORDER - 1) * INTERPOLATION_ORDER) >> 1) - 1;
  /**
   * \brief Nodes interpolator's array.
   *
   * Contains the nodes that we are going to use for the orders from 2 to 5.
   */
  const CCTK_INT all_nodes[14] = {0, 1, -1, 0, 1, -1, 0, 1, 2, -2, -1, 0, 1, 2};

  /**
   * \brief Weights interpolator's array.
   *
   * Contains the weights that we are going to use for the orders from 2 to 5.
   */
  const CCTK_REAL all_weights[14] = {
      -1.,  1.,        0.5,        -1.,        0.5,  -1.0 / 6.0, 0.5,
      -0.5, 1.0 / 6.0, 1.0 / 24.0, -1.0 / 6.0, 0.25, -1.0 / 6.0, 1.0 / 24.0};
  const CCTK_INT *nodes = &all_nodes[order];
  const CCTK_REAL *w = &all_weights[order];

  for (int comp = 0; comp < 6; comp++) {

    // Computing f(x, y_i, z_i)
    CCTK_REAL G_xyz[INTERPOLATION_ORDER][INTERPOLATION_ORDER];
    // Computing d/dx f(x, y_i, z_i)
    CCTK_REAL G_dxyz[INTERPOLATION_ORDER][INTERPOLATION_ORDER];
    for (int j = 0; j < INTERPOLATION_ORDER; j++) {
      for (int k = 0; k < INTERPOLATION_ORDER; k++) {
        CCTK_REAL values[INTERPOLATION_ORDER];
        CCTK_INT points[INTERPOLATION_ORDER];
        for (int i = 0; i < INTERPOLATION_ORDER; i++) {
          values[i] = f(i0 + nodes[i], j0 + nodes[j], k0 + nodes[k], comp);
          points[i] = i0 + nodes[i];
        }
        der_barycentric_cubic_1d<INTERPOLATION_ORDER>(
            G_xyz[j][k], G_dxyz[j][k], points, w, values, x, plo[0], dx[0]);
      }
    }

    // Computing f(x, y, z_i)
    CCTK_REAL H_xyz[INTERPOLATION_ORDER];
    // Computing d/dx f(x, y, z_i)
    CCTK_REAL H_dxyz[INTERPOLATION_ORDER];
    // Computing d/dy f(x, y, z_i)
    CCTK_REAL H_xdyz[INTERPOLATION_ORDER];
    for (int k = 0; k < INTERPOLATION_ORDER; k++) {
      CCTK_REAL values[INTERPOLATION_ORDER];
      CCTK_REAL d_values[INTERPOLATION_ORDER];
      CCTK_INT points[INTERPOLATION_ORDER];
      for (int j = 0; j < INTERPOLATION_ORDER; j++) {
        values[j] = G_xyz[j][k];
        d_values[j] = G_dxyz[j][k];
        points[j] = j0 + nodes[j];
      }
      der_barycentric_cubic_1d<INTERPOLATION_ORDER>(
          H_xyz[k], H_xdyz[k], points, w, values, y, plo[1], dx[1]);
      barycentric_cubic_1d<INTERPOLATION_ORDER>(H_dxyz[k], points, w, d_values,
                                                y, plo[1], dx[1]);
    }

    CCTK_INT points[INTERPOLATION_ORDER];
    for (int k = 0; k < INTERPOLATION_ORDER; k++) {
      points[k] = k0 + nodes[k];
    }
    // Computing f(x, y, z)
    // Computing d/dz f(x, y, z)
    der_barycentric_cubic_1d<INTERPOLATION_ORDER>(
        array6[comp], d_array6[2][comp], points, w, H_xyz, z, plo[2], dx[2]);
    // Computing d/dx f(x, y, z)
    barycentric_cubic_1d<INTERPOLATION_ORDER>(d_array6[0][comp], points, w,
                                              H_dxyz, z, plo[2], dx[2]);
    // Computing d/dy f(x, y, z)
    barycentric_cubic_1d<INTERPOLATION_ORDER>(d_array6[1][comp], points, w,
                                              H_xdyz, z, plo[2], dx[2]);
  }
} // d_interpolate_array

/**
 * \brief Do the barycentric interpolation for a 3 dimensional and symmetric
 * vector and its derivatives.
 *
 * This function computes the interpolation for a 3 dimensional and symmetric
 * vector and its derivatives, such as the induced shift vector
 * in just one call using the der_barycentric_cubic_1d function
 * similarly to the barycentric_derivative_and_interpolate function.
 *
 * @see barycentric_derivative_and_interpolate
 * @see der_barycentric_cubic_1d
 *
 * @param array6 Reference to the GpuArray of size 3 that is going to store the
 * interpolated values.
 * @param d_array6 Reference to GpuArray of size 3 x 3 that stores all the
 * derivatives values.
 * @param f Array containing the function values.
 * @param i0 Basis cell index accordingly to x.
 * @param j0 Basis cell index accordingly to y.
 * @param k0 Basis cell index accordingly to z.
 * @param x Coordinate x value to interpolate.
 * @param y Coordinate y value to interpolate.
 * @param z Coordinate z value to interpolate.
 * @param dx Vector \f$\Delta x\f$  with the space steps value.
 * @param plo Lower values of the entire domain.
 */
template <int INTERPOLATION_ORDER>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE void
d_interpolate_array(amrex::GpuArray<CCTK_REAL, 3> &array3,
                    amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 3>, 3> &d_array3,
                    const amrex::Array4<CCTK_REAL const> &f, const long int &i0,
                    const long int &j0, const long int &k0, const CCTK_REAL &x,
                    const CCTK_REAL &y, const CCTK_REAL &z,
                    const amrex::GpuArray<double, 3> &dx,
                    const amrex::GpuArray<double, 3> &plo) {
  const int order =
      (((INTERPOLATION_ORDER - 1) * INTERPOLATION_ORDER) >> 1) - 1;
  /**
   * \brief Nodes interpolator's array.
   *
   * Contains the nodes that we are going to use for the orders from 2 to 5.
   */
  const CCTK_INT all_nodes[14] = {0, 1, -1, 0, 1, -1, 0, 1, 2, -2, -1, 0, 1, 2};

  /**
   * \brief Weights interpolator's array.
   *
   * Contains the weights that we are going to use for the orders from 2 to 5.
   */
  const CCTK_REAL all_weights[14] = {
      -1.,  1.,        0.5,        -1.,        0.5,  -1.0 / 6.0, 0.5,
      -0.5, 1.0 / 6.0, 1.0 / 24.0, -1.0 / 6.0, 0.25, -1.0 / 6.0, 1.0 / 24.0};
  const CCTK_INT *nodes = &all_nodes[order];
  const CCTK_REAL *w = &all_weights[order];

  for (int comp = 0; comp < 3; comp++) {

    // Computing f(x, y_i, z_i)
    CCTK_REAL G_xyz[INTERPOLATION_ORDER][INTERPOLATION_ORDER];
    // Computing d/dx f(x, y_i, z_i)
    CCTK_REAL G_dxyz[INTERPOLATION_ORDER][INTERPOLATION_ORDER];
    for (int j = 0; j < INTERPOLATION_ORDER; j++) {
      for (int k = 0; k < INTERPOLATION_ORDER; k++) {
        CCTK_REAL values[INTERPOLATION_ORDER];
        CCTK_INT points[INTERPOLATION_ORDER];
        for (int i = 0; i < INTERPOLATION_ORDER; i++) {
          values[i] = f(i0 + nodes[i], j0 + nodes[j], k0 + nodes[k], comp);
          points[i] = i0 + nodes[i];
        }
        der_barycentric_cubic_1d<INTERPOLATION_ORDER>(
            G_xyz[j][k], G_dxyz[j][k], points, w, values, x, plo[0], dx[0]);
      }
    }

    // Computing f(x, y, z_i)
    CCTK_REAL H_xyz[INTERPOLATION_ORDER];
    // Computing d/dx f(x, y, z_i)
    CCTK_REAL H_dxyz[INTERPOLATION_ORDER];
    // Computing d/dy f(x, y, z_i)
    CCTK_REAL H_xdyz[INTERPOLATION_ORDER];
    for (int k = 0; k < INTERPOLATION_ORDER; k++) {
      CCTK_REAL values[INTERPOLATION_ORDER];
      CCTK_REAL d_values[INTERPOLATION_ORDER];
      CCTK_INT points[INTERPOLATION_ORDER];
      for (int j = 0; j < INTERPOLATION_ORDER; j++) {
        values[j] = G_xyz[j][k];
        d_values[j] = G_dxyz[j][k];
        points[j] = j0 + nodes[j];
      }
      der_barycentric_cubic_1d<INTERPOLATION_ORDER>(
          H_xyz[k], H_xdyz[k], points, w, values, y, plo[1], dx[1]);
      barycentric_cubic_1d<INTERPOLATION_ORDER>(H_dxyz[k], points, w, d_values,
                                                y, plo[1], dx[1]);
    }

    CCTK_INT points[INTERPOLATION_ORDER];
    for (int k = 0; k < INTERPOLATION_ORDER; k++) {
      points[k] = k0 + nodes[k];
    }
    // Computing f(x, y, z)
    // Computing d/dz f(x, y, z)
    der_barycentric_cubic_1d<INTERPOLATION_ORDER>(
        array3[comp], d_array3[2][comp], points, w, H_xyz, z, plo[2], dx[2]);
    // Computing d/dx f(x, y, z)
    barycentric_cubic_1d<INTERPOLATION_ORDER>(d_array3[0][comp], points, w,
                                              H_dxyz, z, plo[2], dx[2]);
    // Computing d/dy f(x, y, z)
    barycentric_cubic_1d<INTERPOLATION_ORDER>(d_array3[1][comp], points, w,
                                              H_xdyz, z, plo[2], dx[2]);
  }
} // d_interpolate_array with size 3 arrays

/**
 * \brief Do the barycentric interpolation for a scalar function
 *  its derivatives.
 *
 * This function computes the interpolation for scalar function and its
 * derivatives, such as the lapse function in just one call using the
 * der_barycentric_cubic_1d function similar to the
 * barycentric_derivative_and_interpolate function.
 *
 * @see barycentric_derivative_and_interpolate
 * @see der_barycentric_cubic_1d
 *
 * @param array6 reference to the CCTK_REAL variable that is going to store the
 * interpolated value.
 * @param d_array6 Reference to the GpuArray of size 3 that stores all the
 * derivatives values.
 * @param f Array containing the function values.
 * @param i0 Basis cell index accordingly to x.
 * @param j0 Basis cell index accordingly to y.
 * @param k0 Basis cell index accordingly to z.
 * @param x Coordinate x value to interpolate.
 * @param y Coordinate y value to interpolate.
 * @param z Coordinate z value to interpolate.
 * @param dx Vector \f$\Delta x\f$  with the space steps value.
 * @param plo Lower values of the entire domain.
 */
template <int INTERPOLATION_ORDER>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE void
d_interpolate_array(CCTK_REAL &array, amrex::GpuArray<CCTK_REAL, 3> &d_array,
                    const amrex::Array4<CCTK_REAL const> &f, const long int &i0,
                    const long int &j0, const long int &k0, const CCTK_REAL &x,
                    const CCTK_REAL &y, const CCTK_REAL &z,
                    const amrex::GpuArray<double, 3> &dx,
                    const amrex::GpuArray<double, 3> &plo) {
  const int order =
      (((INTERPOLATION_ORDER - 1) * INTERPOLATION_ORDER) >> 1) - 1;
  /**
   * \brief Nodes interpolator's array.
   *
   * Contains the nodes that we are going to use for the orders from 2 to 5.
   */
  const CCTK_INT all_nodes[14] = {0, 1, -1, 0, 1, -1, 0, 1, 2, -2, -1, 0, 1, 2};

  /**
   * \brief Weights interpolator's array.
   *
   * Contains the weights that we are going to use for the orders from 2 to 5.
   */
  const CCTK_REAL all_weights[14] = {
      -1.,  1.,        0.5,        -1.,        0.5,  -1.0 / 6.0, 0.5,
      -0.5, 1.0 / 6.0, 1.0 / 24.0, -1.0 / 6.0, 0.25, -1.0 / 6.0, 1.0 / 24.0};
  const CCTK_INT *nodes = &all_nodes[order];
  const CCTK_REAL *w = &all_weights[order];

  // Computing f(x, y_i, z_i)
  CCTK_REAL G_xyz[INTERPOLATION_ORDER][INTERPOLATION_ORDER];
  // Computing d/dx f(x, y_i, z_i)
  CCTK_REAL G_dxyz[INTERPOLATION_ORDER][INTERPOLATION_ORDER];
  for (int j = 0; j < INTERPOLATION_ORDER; j++) {
    for (int k = 0; k < INTERPOLATION_ORDER; k++) {
      CCTK_REAL values[INTERPOLATION_ORDER];
      CCTK_INT points[INTERPOLATION_ORDER];
      for (int i = 0; i < INTERPOLATION_ORDER; i++) {
        values[i] = f(i0 + nodes[i], j0 + nodes[j], k0 + nodes[k], 0);
        points[i] = i0 + nodes[i];
      }
      der_barycentric_cubic_1d<INTERPOLATION_ORDER>(
          G_xyz[j][k], G_dxyz[j][k], points, w, values, x, plo[0], dx[0]);
    }
  }

  // Computing f(x, y, z_i)
  CCTK_REAL H_xyz[INTERPOLATION_ORDER];
  // Computing d/dx f(x, y, z_i)
  CCTK_REAL H_dxyz[INTERPOLATION_ORDER];
  // Computing d/dy f(x, y, z_i)
  CCTK_REAL H_xdyz[INTERPOLATION_ORDER];
  for (int k = 0; k < INTERPOLATION_ORDER; k++) {
    CCTK_REAL values[INTERPOLATION_ORDER];
    CCTK_REAL d_values[INTERPOLATION_ORDER];
    CCTK_INT points[INTERPOLATION_ORDER];
    for (int j = 0; j < INTERPOLATION_ORDER; j++) {
      values[j] = G_xyz[j][k];
      d_values[j] = G_dxyz[j][k];
      points[j] = j0 + nodes[j];
    }
    der_barycentric_cubic_1d<INTERPOLATION_ORDER>(H_xyz[k], H_xdyz[k], points,
                                                  w, values, y, plo[1], dx[1]);
    barycentric_cubic_1d<INTERPOLATION_ORDER>(H_dxyz[k], points, w, d_values, y,
                                              plo[1], dx[1]);
  }

  CCTK_INT points[INTERPOLATION_ORDER];
  for (int k = 0; k < INTERPOLATION_ORDER; k++) {
    points[k] = k0 + nodes[k];
  }
  // Computing f(x, y, z)
  // Computing d/dz f(x, y, z)
  der_barycentric_cubic_1d<INTERPOLATION_ORDER>(array, d_array[2], points, w,
                                                H_xyz, z, plo[2], dx[2]);
  // Computing d/dx f(x, y, z)
  barycentric_cubic_1d<INTERPOLATION_ORDER>(d_array[0], points, w, H_dxyz, z,
                                            plo[2], dx[2]);
  // Computing d/dy f(x, y, z)
  barycentric_cubic_1d<INTERPOLATION_ORDER>(d_array[1], points, w, H_xdyz, z,
                                            plo[2], dx[2]);
} // d_interpolate_array scalar

} // namespace Interpolator

#endif // !INTERPOLATOR_HXX
