#ifndef DISCRETIZER_HXX
#define DISCRETIZER_HXX

#include "AMReX_Array.H"
#include "AMReX_Box.H"
#include "AMReX_Config.H"
#include "Interpolator.hxx"
#include "cctk_Types.h"
#include <array>
#include <cctk.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace Discretize {

using namespace Interpolator;

template <int INTERPOLATION_ORDER, int DERIVATIVE_ORDER>
AMREX_GPU_HOST_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
scalar_barycentric_derivative(amrex::GpuArray<CCTK_REAL, 3> &df,
                              amrex::Array4<CCTK_REAL const> const &f,
                              int const &i0, int const &j0, int const &k0,
                              CCTK_REAL const &x, CCTK_REAL const &y,
                              CCTK_REAL const &z,
                              const amrex::GpuArray<double, 3> &dx,
                              const amrex::GpuArray<double, 3> &plo) {
  const CCTK_REAL m1_x = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo);
  const CCTK_REAL p1_x = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo);
  const CCTK_REAL m1_y = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo);
  const CCTK_REAL p1_y = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo);
  const CCTK_REAL m1_z = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo);
  const CCTK_REAL p1_z = barycentric_cubic_3d<INTERPOLATION_ORDER>(
      f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo);

  df[0] = (p1_x - m1_x) / (2.0 * dx[0]);
  df[1] = (p1_y - m1_y) / (2.0 * dx[1]);
  df[2] = (p1_z - m1_z) / (2.0 * dx[2]);
}

template <int INTERPOLATION_ORDER, int DERIVATIVE_ORDER>
AMREX_GPU_HOST_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
vector_barycentric_derivative(amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 3>, 3> &df,
                              amrex::Array4<CCTK_REAL const> const &f,
                              int const &i0, int const &j0, int const &k0,
                              CCTK_REAL const &x, CCTK_REAL const &y,
                              CCTK_REAL const &z,
                              const amrex::GpuArray<double, 3> &dx,
                              const amrex::GpuArray<double, 3> &plo) {

  const CCTK_REAL m1_x_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 0);
  const CCTK_REAL p1_x_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 0);
  const CCTK_REAL m1_y_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 0);
  const CCTK_REAL p1_y_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 0);
  const CCTK_REAL m1_z_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 0);
  const CCTK_REAL p1_z_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 0);

  const CCTK_REAL m1_x_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 1);
  const CCTK_REAL p1_x_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 1);
  const CCTK_REAL m1_y_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 1);
  const CCTK_REAL p1_y_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 1);
  const CCTK_REAL m1_z_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 1);
  const CCTK_REAL p1_z_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 1);

  const CCTK_REAL m1_x_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 2);
  const CCTK_REAL p1_x_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 2);
  const CCTK_REAL m1_y_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 2);
  const CCTK_REAL p1_y_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 2);
  const CCTK_REAL m1_z_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 2);
  const CCTK_REAL p1_z_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 2);

  df[0][0] = (p1_x_0 - m1_x_0) / (2.0 * dx[0]);
  df[0][1] = (p1_x_1 - m1_x_1) / (2.0 * dx[0]);
  df[0][2] = (p1_x_2 - m1_x_2) / (2.0 * dx[0]);

  df[1][0] = (p1_y_0 - m1_y_0) / (2.0 * dx[1]);
  df[1][1] = (p1_y_1 - m1_y_1) / (2.0 * dx[1]);
  df[1][2] = (p1_y_2 - m1_y_2) / (2.0 * dx[1]);

  df[2][0] = (p1_z_0 - m1_z_0) / (2.0 * dx[2]);
  df[2][1] = (p1_z_1 - m1_z_1) / (2.0 * dx[2]);
  df[2][2] = (p1_z_2 - m1_z_2) / (2.0 * dx[2]);
}

template <int INTERPOLATION_ORDER, int DERIVATIVE_ORDER>
AMREX_GPU_HOST_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
smatrix_barycentric_derivative(amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 6>, 3> &df,
                              amrex::Array4<CCTK_REAL const> const &f,
                              int const &i0, int const &j0, int const &k0,
                              CCTK_REAL const &x, CCTK_REAL const &y,
                              CCTK_REAL const &z,
                              const amrex::GpuArray<double, 3> &dx,
                              const amrex::GpuArray<double, 3> &plo) {

  const CCTK_REAL m1_x_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 0);
  const CCTK_REAL p1_x_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 0);
  const CCTK_REAL m1_y_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 0);
  const CCTK_REAL p1_y_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 0);
  const CCTK_REAL m1_z_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 0);
  const CCTK_REAL p1_z_0 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 0);

  const CCTK_REAL m1_x_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 1);
  const CCTK_REAL p1_x_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 1);
  const CCTK_REAL m1_y_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 1);
  const CCTK_REAL p1_y_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 1);
  const CCTK_REAL m1_z_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 1);
  const CCTK_REAL p1_z_1 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 1);

  const CCTK_REAL m1_x_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 2);
  const CCTK_REAL p1_x_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 2);
  const CCTK_REAL m1_y_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 2);
  const CCTK_REAL p1_y_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 2);
  const CCTK_REAL m1_z_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 2);
  const CCTK_REAL p1_z_2 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 2);

  const CCTK_REAL m1_x_3 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 3);
  const CCTK_REAL p1_x_3 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 3);
  const CCTK_REAL m1_y_3 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 - 1, k0, x, y + dx[1], z, dx, plo, 3);
  const CCTK_REAL p1_y_3 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 3);
  const CCTK_REAL m1_z_3 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 3);
  const CCTK_REAL p1_z_3 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 3);

  const CCTK_REAL m1_x_4 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 4);
  const CCTK_REAL p1_x_4 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 4);
  const CCTK_REAL m1_y_4 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 4);
  const CCTK_REAL p1_y_4 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 4);
  const CCTK_REAL m1_z_4 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 4);
  const CCTK_REAL p1_z_4 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 4);

  const CCTK_REAL m1_x_5 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 - 1, j0, k0, x - dx[0], y, z, dx, plo, 5);
  const CCTK_REAL p1_x_5 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0 + 1, j0, k0, x + dx[0], y, z, dx, plo, 5);
  const CCTK_REAL m1_y_5 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 - 1, k0, x, y - dx[1], z, dx, plo, 5);
  const CCTK_REAL p1_y_5 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0 + 1, k0, x, y + dx[1], z, dx, plo, 5);
  const CCTK_REAL m1_z_5 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 - 1, x, y, z - dx[2], dx, plo, 5);
  const CCTK_REAL p1_z_5 = barycentric_cubic_3d<INTERPOLATION_ORDER>(f, i0, j0, k0 + 1, x, y, z + dx[2], dx, plo, 5);

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
}

template <int Dim, typename T, int Order>
AMREX_GPU_DEVICE AMREX_GPU_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
first_derivative(amrex::Array4<T const> const &gf, const int &i, const int &j,
                 const int &k, const int &comp,
                 amrex::GpuArray<T, 3> const &dx) {
  const unsigned int is_x = Dim == 0 ? 1 : 0;
  const unsigned int is_y = Dim == 1 ? 1 : 0;
  const unsigned int is_z = Dim == 2 ? 1 : 0;

  const T m1 = gf(i - is_x, j - is_y, k - is_z, comp);
  const T p1 = gf(i + is_x, j + is_y, k + is_z, comp);

  if constexpr (Order == 2) {

    return (-m1 + p1) / (2. * dx[Dim]);

  } else if constexpr (Order == 4) {

    const T m2 = gf(i - 2 * is_x, j - 2 * is_y, k - 2 * is_z, comp);
    const T p2 = gf(i + 2 * is_x, j + 2 * is_y, k + 2 * is_z, comp);

    return (-8. * m1 + m2 + 8. * p1 - p2) / (12. * dx[Dim]);

  } else if constexpr (Order == 6) {

    const T m2 = gf(i - 2 * is_x, j - 2 * is_y, k - 2 * is_z, comp);
    const T p2 = gf(i + 2 * is_x, j + 2 * is_y, k + 2 * is_z, comp);
    const T m3 = gf(i - 3 * is_x, j - 3 * is_y, k - 3 * is_z, comp);
    const T p3 = gf(i + 3 * is_x, j + 3 * is_y, k + 3 * is_z, comp);

    return (-45. * m1 + 9. * m2 - m3 + 45. * p1 - 9. * p2 + p3) /
           (60. * dx[Dim]);

  } else if constexpr (Order == 8) {

    const T m2 = gf(i - 2 * is_x, j - 2 * is_y, k - 2 * is_z, comp);
    const T p2 = gf(i + 2 * is_x, j + 2 * is_y, k + 2 * is_z, comp);
    const T m3 = gf(i - 3 * is_x, j - 3 * is_y, k - 3 * is_z, comp);
    const T p3 = gf(i + 3 * is_x, j + 3 * is_y, k + 3 * is_z, comp);
    const T m4 = gf(i - 4 * is_x, j - 4 * is_y, k - 4 * is_z, comp);
    const T p4 = gf(i + 4 * is_x, j + 4 * is_y, k + 4 * is_z, comp);

    return (-672. * m1 + 168. * m2 - 32. * m3 + 3. * m4 + 672. * p1 -
            168. * p2 + 32. * p3 - 3. * p4) /
           (840. * dx[Dim]);

  } else {
    CCTK_INFO("Discretization of the first derivative at specified order is "
              "not yet implemented. Available orders: 2, 4, 6, 8.");
    throw std::invalid_argument(
        "Wrong order of discretization for the first derivative");
  }
}

} // namespace Discretize

#endif // !DISCRETIZER_HXX
