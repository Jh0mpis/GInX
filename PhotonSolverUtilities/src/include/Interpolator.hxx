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

namespace Interpolator {

// #############################################################################
//                   Trilinear interpolator
// #############################################################################
template <typename T>
AMREX_GPU_DEVICE AMREX_GPU_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
trilinear_interpolator(amrex::Array4<T const> const &f,
                       amrex::GpuArray<CCTK_REAL, AMREX_SPACEDIM> const &x,
                       amrex::GpuArray<CCTK_REAL, AMREX_SPACEDIM> const &plo,
                       amrex::GpuArray<CCTK_REAL, AMREX_SPACEDIM> const &dx,
                       T &fx) {
  // Trilinear interpolator

  // Get the coordinates in terms of the grid
  CCTK_REAL xp = (x[0] - plo[0]) / dx[0];
  CCTK_REAL yp = (x[1] - plo[1]) / dx[1];
  CCTK_REAL zp = (x[2] - plo[2]) / dx[2];

  // Compute the closest points to the position x
  // cell indexes
  int i = amrex::Math::floor(xp);
  int j = amrex::Math::floor(yp);
  int k = amrex::Math::floor(zp);

  // Compute the distances to the points
  CCTK_REAL dist_x = xp - i;
  CCTK_REAL dist_y = yp - j;
  CCTK_REAL dist_z = zp - k;
  CCTK_REAL sx[] = {CCTK_REAL(1) - dist_x, dist_x};
  CCTK_REAL sy[] = {CCTK_REAL(1) - dist_y, dist_y};
  CCTK_REAL sz[] = {CCTK_REAL(1) - dist_z, dist_z};

  // Compute the result
  fx = sx[0] * sy[0] * sz[0] * f(i, j, k) +
       sx[0] * sy[0] * sz[1] * f(i, j, k + 1) +
       sx[0] * sy[1] * sz[0] * f(i, j + 1, k) +
       sx[0] * sy[1] * sz[1] * f(i, j + 1, k + 1) +
       sx[1] * sy[0] * sz[0] * f(i + 1, j, k) +
       sx[1] * sy[0] * sz[1] * f(i + 1, j, k + 1) +
       sx[1] * sy[1] * sz[0] * f(i + 1, j + 1, k) +
       sx[1] * sy[1] * sz[1] * f(i + 1, j + 1, k + 1);
}

// #############################################################################
//                   Barycentric Lagrange Interpolator
// #############################################################################
template <int N>
AMREX_GPU_DEVICE AMREX_GPU_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
barycentric_cubic_1d(const int (&points)[N + 1],
                     const CCTK_REAL (&weights)[N + 1],
                     const CCTK_REAL (&values)[N + 1], const CCTK_REAL &x,
                     const CCTK_REAL &plo, const CCTK_REAL &dx) {

  for (int i = 0; i <= N; i++) {
    if (x == plo + points[i] * dx) {
      return values[i];
    }
  }

  CCTK_REAL num{0.0};
  CCTK_REAL den{0.0};

  for (int i = 0; i <= N; i++) {
    CCTK_REAL term = weights[i] / (x - (plo + points[i] * dx));
    num += term * values[i];
    den += term;
  }

  return num / den;
}

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
