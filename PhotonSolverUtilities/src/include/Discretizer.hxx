#ifndef DISCRETIZER_HXX
#define DISCRETIZER_HXX

#include "AMReX_Array.H"
#include "AMReX_Box.H"
#include "AMReX_Config.H"
#include "cctk_Types.h"
#include <array>
#include <cctk.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace Discretize {

template <int Order>
AMREX_GPU_DEVICE AMREX_GPU_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
gradient_scalar(amrex::GpuArray<CCTK_REAL, 3> &df,
                amrex::Array4<CCTK_REAL const> const &gf, const int &i,
                const int &j, const int &k,
                amrex::GpuArray<CCTK_REAL, 3> const &dx) {

  const CCTK_REAL m1_x = gf(i - 1, j, k);
  const CCTK_REAL p1_x = gf(i + 1, j, k);
  const CCTK_REAL m1_y = gf(i, j - 1, k);
  const CCTK_REAL p1_y = gf(i, j + 1, k);
  const CCTK_REAL m1_z = gf(i, j, k - 1);
  const CCTK_REAL p1_z = gf(i, j, k + 1);

  if constexpr (Order == 2) {

    df[0] = (-m1_x + p1_x) / (2. * dx[0]);
    df[1] = (-m1_y + p1_y) / (2. * dx[1]);
    df[2] = (-m1_z + p1_z) / (2. * dx[2]);

  } else if constexpr (Order == 4) {

    const CCTK_REAL m2_x = gf(i - 2, j, k);
    const CCTK_REAL p2_x = gf(i + 2, j, k);
    const CCTK_REAL m2_y = gf(i, j - 2, k);
    const CCTK_REAL p2_y = gf(i, j + 2, k);
    const CCTK_REAL m2_z = gf(i, j, k - 2);
    const CCTK_REAL p2_z = gf(i, j, k + 2);

    df[0] = (-8. * m1_x + m2_x + 8. * p1_x - p2_x) / (12. * dx[0]);
    df[1] = (-8. * m1_y + m2_y + 8. * p1_y - p2_y) / (12. * dx[1]);
    df[2] = (-8. * m1_z + m2_z + 8. * p1_z - p2_z) / (12. * dx[2]);

  } else if constexpr (Order == 6) {

    const CCTK_REAL m2_x = gf(i - 2, j, k);
    const CCTK_REAL p2_x = gf(i + 2, j, k);
    const CCTK_REAL m3_x = gf(i - 3, j, k);
    const CCTK_REAL p3_x = gf(i + 3, j, k);
    const CCTK_REAL m2_y = gf(i, j - 2, k);
    const CCTK_REAL p2_y = gf(i, j + 2, k);
    const CCTK_REAL m3_y = gf(i, j - 3, k);
    const CCTK_REAL p3_y = gf(i, j + 3, k);
    const CCTK_REAL m2_z = gf(i, j, k - 2);
    const CCTK_REAL p2_z = gf(i, j, k + 2);
    const CCTK_REAL m3_z = gf(i, j, k - 3);
    const CCTK_REAL p3_z = gf(i, j, k + 3);

    df[0] = (-45. * m1_x + 9. * m2_x - m3_x + 45. * p1_x - 9. * p2_x + p3_x) /
            (60. * dx[0]);
    df[1] = (-45. * m1_y + 9. * m2_y - m3_y + 45. * p1_y - 9. * p2_y + p3_y) /
            (60. * dx[1]);
    df[2] = (-45. * m1_z + 9. * m2_z - m3_z + 45. * p1_z - 9. * p2_z + p3_z) /
            (60. * dx[2]);

  } else if constexpr (Order == 8) {

    const CCTK_REAL m2_x = gf(i - 2, j, k);
    const CCTK_REAL p2_x = gf(i + 2, j, k);
    const CCTK_REAL m3_x = gf(i - 3, j, k);
    const CCTK_REAL p3_x = gf(i + 3, j, k);
    const CCTK_REAL m4_x = gf(i - 4, j, k);
    const CCTK_REAL p4_x = gf(i + 4, j, k);
    const CCTK_REAL m2_y = gf(i, j - 2, k);
    const CCTK_REAL p2_y = gf(i, j + 2, k);
    const CCTK_REAL m3_y = gf(i, j - 3, k);
    const CCTK_REAL p3_y = gf(i, j + 3, k);
    const CCTK_REAL m4_y = gf(i, j - 4, k);
    const CCTK_REAL p4_y = gf(i, j + 4, k);
    const CCTK_REAL m2_z = gf(i, j, k - 2);
    const CCTK_REAL p2_z = gf(i, j, k + 2);
    const CCTK_REAL m3_z = gf(i, j, k - 3);
    const CCTK_REAL p3_z = gf(i, j, k + 3);
    const CCTK_REAL m4_z = gf(i, j, k - 4);
    const CCTK_REAL p4_z = gf(i, j, k + 4);

    df[0] = (-672. * m1_x + 168. * m2_x - 32. * m3_x + 3. * m4_x + 672. * p1_x -
             168. * p2_x + 32. * p3_x - 3. * p4_x) /
            (840. * dx[0]);
    df[1] = (-672. * m1_y + 168. * m2_y - 32. * m3_y + 3. * m4_y + 672. * p1_y -
             168. * p2_y + 32. * p3_y - 3. * p4_y) /
            (840. * dx[1]);
    df[2] = (-672. * m1_z + 168. * m2_z - 32. * m3_z + 3. * m4_z + 672. * p1_z -
             168. * p2_z + 32. * p3_z - 3. * p4_z) /
            (840. * dx[2]);

  } else {
    CCTK_INFO("Discretization of the first derivative at specified order is "
              "not yet implemented. Available orders: 2, 4, 6, 8.");
    throw std::invalid_argument(
        "Wrong order of discretization for the gradient");
  }
}

template <int Order>
AMREX_GPU_DEVICE AMREX_GPU_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
gradient_vector(amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 3>, 3> &df,
                amrex::Array4<CCTK_REAL const> const &gf, const int &i,
                const int &j, const int &k,
                amrex::GpuArray<CCTK_REAL, 3> const &dx) {

  const CCTK_REAL m1_x_0 = gf(i - 1, j, k, 0);
  const CCTK_REAL p1_x_0 = gf(i + 1, j, k, 0);
  const CCTK_REAL m1_y_0 = gf(i, j - 1, k, 0);
  const CCTK_REAL p1_y_0 = gf(i, j + 1, k, 0);
  const CCTK_REAL m1_z_0 = gf(i, j, k - 1, 0);
  const CCTK_REAL p1_z_0 = gf(i, j, k + 1, 0);
  const CCTK_REAL m1_x_1 = gf(i - 1, j, k, 1);
  const CCTK_REAL p1_x_1 = gf(i + 1, j, k, 1);
  const CCTK_REAL m1_y_1 = gf(i, j - 1, k, 1);
  const CCTK_REAL p1_y_1 = gf(i, j + 1, k, 1);
  const CCTK_REAL m1_z_1 = gf(i, j, k - 1, 1);
  const CCTK_REAL p1_z_1 = gf(i, j, k + 1, 1);
  const CCTK_REAL m1_x_2 = gf(i - 1, j, k, 2);
  const CCTK_REAL p1_x_2 = gf(i + 1, j, k, 2);
  const CCTK_REAL m1_y_2 = gf(i, j - 1, k, 2);
  const CCTK_REAL p1_y_2 = gf(i, j + 1, k, 2);
  const CCTK_REAL m1_z_2 = gf(i, j, k - 1, 2);
  const CCTK_REAL p1_z_2 = gf(i, j, k + 1, 2);

  if constexpr (Order == 2) {

    df[0][0] = (-m1_x_0 + p1_x_0) / (2. * dx[0]);
    df[0][1] = (-m1_x_1 + p1_x_1) / (2. * dx[0]);
    df[0][2] = (-m1_x_2 + p1_x_2) / (2. * dx[0]);

    df[1][0] = (-m1_y_0 + p1_y_0) / (2. * dx[1]);
    df[1][1] = (-m1_y_1 + p1_y_1) / (2. * dx[1]);
    df[1][2] = (-m1_y_2 + p1_y_2) / (2. * dx[1]);

    df[2][0] = (-m1_z_0 + p1_z_0) / (2. * dx[2]);
    df[2][1] = (-m1_z_1 + p1_z_1) / (2. * dx[2]);
    df[2][2] = (-m1_z_2 + p1_z_2) / (2. * dx[2]);

  } else if constexpr (Order == 4) {

    const CCTK_REAL m2_x_0 = gf(i - 2, j, k, 0);
    const CCTK_REAL p2_x_0 = gf(i + 2, j, k, 0);
    const CCTK_REAL m2_y_0 = gf(i, j - 2, k, 0);
    const CCTK_REAL p2_y_0 = gf(i, j + 2, k, 0);
    const CCTK_REAL m2_z_0 = gf(i, j, k - 2, 0);
    const CCTK_REAL p2_z_0 = gf(i, j, k + 2, 0);
    const CCTK_REAL m2_x_1 = gf(i - 2, j, k, 1);
    const CCTK_REAL p2_x_1 = gf(i + 2, j, k, 1);
    const CCTK_REAL m2_y_1 = gf(i, j - 2, k, 1);
    const CCTK_REAL p2_y_1 = gf(i, j + 2, k, 1);
    const CCTK_REAL m2_z_1 = gf(i, j, k - 2, 1);
    const CCTK_REAL p2_z_1 = gf(i, j, k + 2, 1);
    const CCTK_REAL m2_x_2 = gf(i - 2, j, k, 2);
    const CCTK_REAL p2_x_2 = gf(i + 2, j, k, 2);
    const CCTK_REAL m2_y_2 = gf(i, j - 2, k, 2);
    const CCTK_REAL p2_y_2 = gf(i, j + 2, k, 2);
    const CCTK_REAL m2_z_2 = gf(i, j, k - 2, 2);
    const CCTK_REAL p2_z_2 = gf(i, j, k + 2, 2);

    df[0][0] = (-8. * m1_x_0 + m2_x_0 + 8. * p1_x_0 - p2_x_0) / (12. * dx[0]);
    df[0][1] = (-8. * m1_x_1 + m2_x_1 + 8. * p1_x_1 - p2_x_1) / (12. * dx[0]);
    df[0][2] = (-8. * m1_x_2 + m2_x_2 + 8. * p1_x_2 - p2_x_2) / (12. * dx[0]);

    df[1][0] = (-8. * m1_y_0 + m2_y_0 + 8. * p1_y_0 - p2_y_0) / (12. * dx[1]);
    df[1][1] = (-8. * m1_y_1 + m2_y_1 + 8. * p1_y_1 - p2_y_1) / (12. * dx[1]);
    df[1][2] = (-8. * m1_y_2 + m2_y_2 + 8. * p1_y_2 - p2_y_2) / (12. * dx[1]);

    df[2][0] = (-8. * m1_z_0 + m2_z_0 + 8. * p1_z_0 - p2_z_0) / (12. * dx[2]);
    df[2][1] = (-8. * m1_z_1 + m2_z_1 + 8. * p1_z_1 - p2_z_1) / (12. * dx[2]);
    df[2][2] = (-8. * m1_z_2 + m2_z_2 + 8. * p1_z_2 - p2_z_2) / (12. * dx[2]);

  } else if constexpr (Order == 6) {

    const CCTK_REAL m2_x_0 = gf(i - 2, j, k, 0);
    const CCTK_REAL p2_x_0 = gf(i + 2, j, k, 0);
    const CCTK_REAL m3_x_0 = gf(i - 3, j, k, 0);
    const CCTK_REAL p3_x_0 = gf(i + 3, j, k, 0);
    const CCTK_REAL m2_y_0 = gf(i, j - 2, k, 0);
    const CCTK_REAL p2_y_0 = gf(i, j + 2, k, 0);
    const CCTK_REAL m3_y_0 = gf(i, j - 3, k, 0);
    const CCTK_REAL p3_y_0 = gf(i, j + 3, k, 0);
    const CCTK_REAL m2_z_0 = gf(i, j, k - 2, 0);
    const CCTK_REAL p2_z_0 = gf(i, j, k + 2, 0);
    const CCTK_REAL m3_z_0 = gf(i, j, k - 3, 0);
    const CCTK_REAL p3_z_0 = gf(i, j, k + 3, 0);
    const CCTK_REAL m2_x_1 = gf(i - 2, j, k, 1);
    const CCTK_REAL p2_x_1 = gf(i + 2, j, k, 1);
    const CCTK_REAL m3_x_1 = gf(i - 3, j, k, 1);
    const CCTK_REAL p3_x_1 = gf(i + 3, j, k, 1);
    const CCTK_REAL m2_y_1 = gf(i, j - 2, k, 1);
    const CCTK_REAL p2_y_1 = gf(i, j + 2, k, 1);
    const CCTK_REAL m3_y_1 = gf(i, j - 3, k, 1);
    const CCTK_REAL p3_y_1 = gf(i, j + 3, k, 1);
    const CCTK_REAL m2_z_1 = gf(i, j, k - 2, 1);
    const CCTK_REAL p2_z_1 = gf(i, j, k + 2, 1);
    const CCTK_REAL m3_z_1 = gf(i, j, k - 3, 1);
    const CCTK_REAL p3_z_1 = gf(i, j, k + 3, 1);
    const CCTK_REAL m2_x_2 = gf(i - 2, j, k, 2);
    const CCTK_REAL p2_x_2 = gf(i + 2, j, k, 2);
    const CCTK_REAL m3_x_2 = gf(i - 3, j, k, 2);
    const CCTK_REAL p3_x_2 = gf(i + 3, j, k, 2);
    const CCTK_REAL m2_y_2 = gf(i, j - 2, k, 2);
    const CCTK_REAL p2_y_2 = gf(i, j + 2, k, 2);
    const CCTK_REAL m3_y_2 = gf(i, j - 3, k, 2);
    const CCTK_REAL p3_y_2 = gf(i, j + 3, k, 2);
    const CCTK_REAL m2_z_2 = gf(i, j, k - 2, 2);
    const CCTK_REAL p2_z_2 = gf(i, j, k + 2, 2);
    const CCTK_REAL m3_z_2 = gf(i, j, k - 3, 2);
    const CCTK_REAL p3_z_2 = gf(i, j, k + 3, 2);

    df[0][0] = (-45. * m1_x_0 + 9. * m2_x_0 - m3_x_0 + 45. * p1_x_0 -
                9. * p2_x_0 + p3_x_0) /
               (60. * dx[0]);
    df[0][1] = (-45. * m1_x_1 + 9. * m2_x_1 - m3_x_1 + 45. * p1_x_1 -
                9. * p2_x_1 + p3_x_1) /
               (60. * dx[0]);
    df[0][2] = (-45. * m1_x_2 + 9. * m2_x_2 - m3_x_2 + 45. * p1_x_2 -
                9. * p2_x_2 + p3_x_2) /
               (60. * dx[0]);

    df[1][0] = (-45. * m1_y_0 + 9. * m2_y_0 - m3_y_0 + 45. * p1_y_0 -
                9. * p2_y_0 + p3_y_0) /
               (60. * dx[1]);
    df[1][1] = (-45. * m1_y_1 + 9. * m2_y_1 - m3_y_1 + 45. * p1_y_1 -
                9. * p2_y_1 + p3_y_1) /
               (60. * dx[1]);
    df[1][2] = (-45. * m1_y_2 + 9. * m2_y_2 - m3_y_2 + 45. * p1_y_2 -
                9. * p2_y_2 + p3_y_2) /
               (60. * dx[1]);

    df[2][0] = (-45. * m1_z_0 + 9. * m2_z_0 - m3_z_0 + 45. * p1_z_0 -
                9. * p2_z_0 + p3_z_0) /
               (60. * dx[2]);
    df[2][1] = (-45. * m1_z_1 + 9. * m2_z_1 - m3_z_1 + 45. * p1_z_1 -
                9. * p2_z_1 + p3_z_1) /
               (60. * dx[2]);
    df[2][2] = (-45. * m1_z_2 + 9. * m2_z_2 - m3_z_2 + 45. * p1_z_2 -
                9. * p2_z_2 + p3_z_2) /
               (60. * dx[2]);

  } else if constexpr (Order == 8) {

    const CCTK_REAL m2_x_0 = gf(i - 2, j, k, 0);
    const CCTK_REAL p2_x_0 = gf(i + 2, j, k, 0);
    const CCTK_REAL m3_x_0 = gf(i - 3, j, k, 0);
    const CCTK_REAL p3_x_0 = gf(i + 3, j, k, 0);
    const CCTK_REAL m4_x_0 = gf(i - 4, j, k, 0);
    const CCTK_REAL p4_x_0 = gf(i + 4, j, k, 0);
    const CCTK_REAL m2_y_0 = gf(i, j - 2, k, 0);
    const CCTK_REAL p2_y_0 = gf(i, j + 2, k, 0);
    const CCTK_REAL m3_y_0 = gf(i, j - 3, k, 0);
    const CCTK_REAL p3_y_0 = gf(i, j + 3, k, 0);
    const CCTK_REAL m4_y_0 = gf(i, j - 4, k, 0);
    const CCTK_REAL p4_y_0 = gf(i, j + 4, k, 0);
    const CCTK_REAL m2_z_0 = gf(i, j, k - 2, 0);
    const CCTK_REAL p2_z_0 = gf(i, j, k + 2, 0);
    const CCTK_REAL m3_z_0 = gf(i, j, k - 3, 0);
    const CCTK_REAL p3_z_0 = gf(i, j, k + 3, 0);
    const CCTK_REAL m4_z_0 = gf(i, j, k - 4, 0);
    const CCTK_REAL p4_z_0 = gf(i, j, k + 4, 0);
    const CCTK_REAL m2_x_1 = gf(i - 2, j, k, 1);
    const CCTK_REAL p2_x_1 = gf(i + 2, j, k, 1);
    const CCTK_REAL m3_x_1 = gf(i - 3, j, k, 1);
    const CCTK_REAL p3_x_1 = gf(i + 3, j, k, 1);
    const CCTK_REAL m4_x_1 = gf(i - 4, j, k, 1);
    const CCTK_REAL p4_x_1 = gf(i + 4, j, k, 1);
    const CCTK_REAL m2_y_1 = gf(i, j - 2, k, 1);
    const CCTK_REAL p2_y_1 = gf(i, j + 2, k, 1);
    const CCTK_REAL m3_y_1 = gf(i, j - 3, k, 1);
    const CCTK_REAL p3_y_1 = gf(i, j + 3, k, 1);
    const CCTK_REAL m4_y_1 = gf(i, j - 4, k, 1);
    const CCTK_REAL p4_y_1 = gf(i, j + 4, k, 1);
    const CCTK_REAL m2_z_1 = gf(i, j, k - 2, 1);
    const CCTK_REAL p2_z_1 = gf(i, j, k + 2, 1);
    const CCTK_REAL m3_z_1 = gf(i, j, k - 3, 1);
    const CCTK_REAL p3_z_1 = gf(i, j, k + 3, 1);
    const CCTK_REAL m4_z_1 = gf(i, j, k - 4, 1);
    const CCTK_REAL p4_z_1 = gf(i, j, k + 4, 1);
    const CCTK_REAL m2_x_2 = gf(i - 2, j, k, 1);
    const CCTK_REAL p2_x_2 = gf(i + 2, j, k, 2);
    const CCTK_REAL m3_x_2 = gf(i - 3, j, k, 2);
    const CCTK_REAL p3_x_2 = gf(i + 3, j, k, 2);
    const CCTK_REAL m4_x_2 = gf(i - 4, j, k, 2);
    const CCTK_REAL p4_x_2 = gf(i + 4, j, k, 2);
    const CCTK_REAL m2_y_2 = gf(i, j - 2, k, 2);
    const CCTK_REAL p2_y_2 = gf(i, j + 2, k, 2);
    const CCTK_REAL m3_y_2 = gf(i, j - 3, k, 2);
    const CCTK_REAL p3_y_2 = gf(i, j + 3, k, 2);
    const CCTK_REAL m4_y_2 = gf(i, j - 4, k, 2);
    const CCTK_REAL p4_y_2 = gf(i, j + 4, k, 2);
    const CCTK_REAL m2_z_2 = gf(i, j, k - 2, 2);
    const CCTK_REAL p2_z_2 = gf(i, j, k + 2, 2);
    const CCTK_REAL m3_z_2 = gf(i, j, k - 3, 2);
    const CCTK_REAL p3_z_2 = gf(i, j, k + 3, 2);
    const CCTK_REAL m4_z_2 = gf(i, j, k - 4, 2);
    const CCTK_REAL p4_z_2 = gf(i, j, k + 4, 2);

    df[0][0] = (-672. * m1_x_0 + 168. * m2_x_0 - 32. * m3_x_0 + 3. * m4_x_0 +
                672. * p1_x_0 - 168. * p2_x_0 + 32. * p3_x_0 - 3. * p4_x_0) /
               (840. * dx[0]);
    df[0][1] = (-672. * m1_x_1 + 168. * m2_x_1 - 32. * m3_x_1 + 3. * m4_x_1 +
                672. * p1_x_1 - 168. * p2_x_1 + 32. * p3_x_1 - 3. * p4_x_1) /
               (840. * dx[0]);
    df[0][2] = (-672. * m1_x_2 + 168. * m2_x_2 - 32. * m3_x_2 + 3. * m4_x_2 +
                672. * p1_x_2 - 168. * p2_x_2 + 32. * p3_x_2 - 3. * p4_x_2) /
               (840. * dx[0]);

    df[1][0] = (-672. * m1_y_0 + 168. * m2_y_0 - 32. * m3_y_0 + 3. * m4_y_0 +
                672. * p1_y_0 - 168. * p2_y_0 + 32. * p3_y_0 - 3. * p4_y_0) /
               (840. * dx[1]);
    df[1][1] = (-672. * m1_y_1 + 168. * m2_y_1 - 32. * m3_y_1 + 3. * m4_y_1 +
                672. * p1_y_1 - 168. * p2_y_1 + 32. * p3_y_1 - 3. * p4_y_1) /
               (840. * dx[1]);
    df[1][2] = (-672. * m1_y_2 + 168. * m2_y_2 - 32. * m3_y_2 + 3. * m4_y_2 +
                672. * p1_y_2 - 168. * p2_y_2 + 32. * p3_y_2 - 3. * p4_y_2) /
               (840. * dx[1]);

    df[2][0] = (-672. * m1_z_0 + 168. * m2_z_0 - 32. * m3_z_0 + 3. * m4_z_0 +
                672. * p1_z_0 - 168. * p2_z_0 + 32. * p3_z_0 - 3. * p4_z_0) /
               (840. * dx[2]);
    df[2][1] = (-672. * m1_z_1 + 168. * m2_z_1 - 32. * m3_z_1 + 3. * m4_z_1 +
                672. * p1_z_1 - 168. * p2_z_1 + 32. * p3_z_1 - 3. * p4_z_1) /
               (840. * dx[2]);
    df[2][2] = (-672. * m1_z_2 + 168. * m2_z_2 - 32. * m3_z_2 + 3. * m4_z_2 +
                672. * p1_z_2 - 168. * p2_z_2 + 32. * p3_z_2 - 3. * p4_z_2) /
               (840. * dx[2]);

  } else {
    CCTK_INFO("Discretization of the first derivative at specified order is "
              "not yet implemented. Available orders: 2, 4, 6, 8.");
    throw std::invalid_argument(
        "Wrong order of discretization for the gradient.");
  }
}

template <int Order>
AMREX_GPU_DEVICE AMREX_GPU_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
gradient_simmetric_matrix(amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 6>, 3> &df,
                          amrex::Array4<CCTK_REAL const> const &gf,
                          const int &i, const int &j, const int &k,
                          amrex::GpuArray<CCTK_REAL, 3> const &dx) {

  const CCTK_REAL m1_x_0 = gf(i - 1, j, k, 0);
  const CCTK_REAL p1_x_0 = gf(i + 1, j, k, 0);
  const CCTK_REAL m1_y_0 = gf(i, j - 1, k, 0);
  const CCTK_REAL p1_y_0 = gf(i, j + 1, k, 0);
  const CCTK_REAL m1_z_0 = gf(i, j, k - 1, 0);
  const CCTK_REAL p1_z_0 = gf(i, j, k + 1, 0);
  const CCTK_REAL m1_x_1 = gf(i - 1, j, k, 1);
  const CCTK_REAL p1_x_1 = gf(i + 1, j, k, 1);
  const CCTK_REAL m1_y_1 = gf(i, j - 1, k, 1);
  const CCTK_REAL p1_y_1 = gf(i, j + 1, k, 1);
  const CCTK_REAL m1_z_1 = gf(i, j, k - 1, 1);
  const CCTK_REAL p1_z_1 = gf(i, j, k + 1, 1);
  const CCTK_REAL m1_x_2 = gf(i - 1, j, k, 2);
  const CCTK_REAL p1_x_2 = gf(i + 1, j, k, 2);
  const CCTK_REAL m1_y_2 = gf(i, j - 1, k, 2);
  const CCTK_REAL p1_y_2 = gf(i, j + 1, k, 2);
  const CCTK_REAL m1_z_2 = gf(i, j, k - 1, 2);
  const CCTK_REAL p1_z_2 = gf(i, j, k + 1, 2);
  const CCTK_REAL m1_x_3 = gf(i - 1, j, k, 3);
  const CCTK_REAL p1_x_3 = gf(i + 1, j, k, 3);
  const CCTK_REAL m1_y_3 = gf(i, j - 1, k, 3);
  const CCTK_REAL p1_y_3 = gf(i, j + 1, k, 3);
  const CCTK_REAL m1_z_3 = gf(i, j, k - 1, 3);
  const CCTK_REAL p1_z_3 = gf(i, j, k + 1, 3);
  const CCTK_REAL m1_x_4 = gf(i - 1, j, k, 4);
  const CCTK_REAL p1_x_4 = gf(i + 1, j, k, 4);
  const CCTK_REAL m1_y_4 = gf(i, j - 1, k, 4);
  const CCTK_REAL p1_y_4 = gf(i, j + 1, k, 4);
  const CCTK_REAL m1_z_4 = gf(i, j, k - 1, 4);
  const CCTK_REAL p1_z_4 = gf(i, j, k + 1, 4);
  const CCTK_REAL m1_x_5 = gf(i - 1, j, k, 5);
  const CCTK_REAL p1_x_5 = gf(i + 1, j, k, 5);
  const CCTK_REAL m1_y_5 = gf(i, j - 1, k, 5);
  const CCTK_REAL p1_y_5 = gf(i, j + 1, k, 5);
  const CCTK_REAL m1_z_5 = gf(i, j, k - 1, 5);
  const CCTK_REAL p1_z_5 = gf(i, j, k + 1, 5);

  if constexpr (Order == 2) {

    df[0][0] = (-m1_x_0 + p1_x_0) / (2. * dx[0]);
    df[0][1] = (-m1_x_1 + p1_x_1) / (2. * dx[0]);
    df[0][2] = (-m1_x_2 + p1_x_2) / (2. * dx[0]);
    df[0][3] = (-m1_x_3 + p1_x_3) / (2. * dx[0]);
    df[0][4] = (-m1_x_4 + p1_x_4) / (2. * dx[0]);
    df[0][5] = (-m1_x_5 + p1_x_5) / (2. * dx[0]);

    df[1][0] = (-m1_y_0 + p1_y_0) / (2. * dx[1]);
    df[1][1] = (-m1_y_1 + p1_y_1) / (2. * dx[1]);
    df[1][2] = (-m1_y_2 + p1_y_2) / (2. * dx[1]);
    df[1][3] = (-m1_y_3 + p1_y_3) / (2. * dx[1]);
    df[1][4] = (-m1_y_4 + p1_y_4) / (2. * dx[1]);
    df[1][5] = (-m1_y_5 + p1_y_5) / (2. * dx[1]);

    df[2][0] = (-m1_z_0 + p1_z_0) / (2. * dx[2]);
    df[2][1] = (-m1_z_1 + p1_z_1) / (2. * dx[2]);
    df[2][2] = (-m1_z_2 + p1_z_2) / (2. * dx[2]);
    df[2][3] = (-m1_z_3 + p1_z_3) / (2. * dx[2]);
    df[2][4] = (-m1_z_4 + p1_z_4) / (2. * dx[2]);
    df[2][5] = (-m1_z_5 + p1_z_5) / (2. * dx[2]);

  } else if constexpr (Order == 4) {

    const CCTK_REAL m2_x_0 = gf(i - 2, j, k, 0);
    const CCTK_REAL p2_x_0 = gf(i + 2, j, k, 0);
    const CCTK_REAL m2_y_0 = gf(i, j - 2, k, 0);
    const CCTK_REAL p2_y_0 = gf(i, j + 2, k, 0);
    const CCTK_REAL m2_z_0 = gf(i, j, k - 2, 0);
    const CCTK_REAL p2_z_0 = gf(i, j, k + 2, 0);
    const CCTK_REAL m2_x_1 = gf(i - 2, j, k, 1);
    const CCTK_REAL p2_x_1 = gf(i + 2, j, k, 1);
    const CCTK_REAL m2_y_1 = gf(i, j - 2, k, 1);
    const CCTK_REAL p2_y_1 = gf(i, j + 2, k, 1);
    const CCTK_REAL m2_z_1 = gf(i, j, k - 2, 1);
    const CCTK_REAL p2_z_1 = gf(i, j, k + 2, 1);
    const CCTK_REAL m2_x_2 = gf(i - 2, j, k, 2);
    const CCTK_REAL p2_x_2 = gf(i + 2, j, k, 2);
    const CCTK_REAL m2_y_2 = gf(i, j - 2, k, 2);
    const CCTK_REAL p2_y_2 = gf(i, j + 2, k, 2);
    const CCTK_REAL m2_z_2 = gf(i, j, k - 2, 2);
    const CCTK_REAL p2_z_2 = gf(i, j, k + 2, 2);
    const CCTK_REAL m2_x_3 = gf(i - 2, j, k, 3);
    const CCTK_REAL p2_x_3 = gf(i + 2, j, k, 3);
    const CCTK_REAL m2_y_3 = gf(i, j - 2, k, 3);
    const CCTK_REAL p2_y_3 = gf(i, j + 2, k, 3);
    const CCTK_REAL m2_z_3 = gf(i, j, k - 2, 3);
    const CCTK_REAL p2_z_3 = gf(i, j, k + 2, 3);
    const CCTK_REAL m2_x_4 = gf(i - 2, j, k, 4);
    const CCTK_REAL p2_x_4 = gf(i + 2, j, k, 4);
    const CCTK_REAL m2_y_4 = gf(i, j - 2, k, 4);
    const CCTK_REAL p2_y_4 = gf(i, j + 2, k, 4);
    const CCTK_REAL m2_z_4 = gf(i, j, k - 2, 4);
    const CCTK_REAL p2_z_4 = gf(i, j, k + 2, 4);
    const CCTK_REAL m2_x_5 = gf(i - 2, j, k, 5);
    const CCTK_REAL p2_x_5 = gf(i + 2, j, k, 5);
    const CCTK_REAL m2_y_5 = gf(i, j - 2, k, 5);
    const CCTK_REAL p2_y_5 = gf(i, j + 2, k, 5);
    const CCTK_REAL m2_z_5 = gf(i, j, k - 2, 5);
    const CCTK_REAL p2_z_5 = gf(i, j, k + 2, 5);

    df[0][0] = (-8. * m1_x_0 + m2_x_0 + 8. * p1_x_0 - p2_x_0) / (12. * dx[0]);
    df[0][1] = (-8. * m1_x_1 + m2_x_1 + 8. * p1_x_1 - p2_x_1) / (12. * dx[0]);
    df[0][2] = (-8. * m1_x_2 + m2_x_2 + 8. * p1_x_2 - p2_x_2) / (12. * dx[0]);
    df[0][3] = (-8. * m1_x_3 + m2_x_3 + 8. * p1_x_3 - p2_x_3) / (12. * dx[0]);
    df[0][4] = (-8. * m1_x_4 + m2_x_4 + 8. * p1_x_4 - p2_x_4) / (12. * dx[0]);
    df[0][5] = (-8. * m1_x_5 + m2_x_5 + 8. * p1_x_5 - p2_x_5) / (12. * dx[0]);

    df[1][0] = (-8. * m1_y_0 + m2_y_0 + 8. * p1_y_0 - p2_y_0) / (12. * dx[1]);
    df[1][1] = (-8. * m1_y_1 + m2_y_1 + 8. * p1_y_1 - p2_y_1) / (12. * dx[1]);
    df[1][2] = (-8. * m1_y_2 + m2_y_2 + 8. * p1_y_2 - p2_y_2) / (12. * dx[1]);
    df[1][3] = (-8. * m1_y_3 + m2_y_3 + 8. * p1_y_3 - p2_y_3) / (12. * dx[1]);
    df[1][4] = (-8. * m1_y_4 + m2_y_4 + 8. * p1_y_4 - p2_y_4) / (12. * dx[1]);
    df[1][5] = (-8. * m1_y_5 + m2_y_5 + 8. * p1_y_5 - p2_y_5) / (12. * dx[1]);

    df[2][0] = (-8. * m1_z_0 + m2_z_0 + 8. * p1_z_0 - p2_z_0) / (12. * dx[2]);
    df[2][1] = (-8. * m1_z_1 + m2_z_1 + 8. * p1_z_1 - p2_z_1) / (12. * dx[2]);
    df[2][2] = (-8. * m1_z_2 + m2_z_2 + 8. * p1_z_2 - p2_z_2) / (12. * dx[2]);
    df[2][3] = (-8. * m1_z_3 + m2_z_3 + 8. * p1_z_3 - p2_z_3) / (12. * dx[2]);
    df[2][4] = (-8. * m1_z_4 + m2_z_4 + 8. * p1_z_4 - p2_z_4) / (12. * dx[2]);
    df[2][5] = (-8. * m1_z_5 + m2_z_5 + 8. * p1_z_5 - p2_z_5) / (12. * dx[2]);

  } else if constexpr (Order == 6) {

    const CCTK_REAL m2_x_0 = gf(i - 2, j, k, 0);
    const CCTK_REAL p2_x_0 = gf(i + 2, j, k, 0);
    const CCTK_REAL m3_x_0 = gf(i - 3, j, k, 0);
    const CCTK_REAL p3_x_0 = gf(i + 3, j, k, 0);
    const CCTK_REAL m2_y_0 = gf(i, j - 2, k, 0);
    const CCTK_REAL p2_y_0 = gf(i, j + 2, k, 0);
    const CCTK_REAL m3_y_0 = gf(i, j - 3, k, 0);
    const CCTK_REAL p3_y_0 = gf(i, j + 3, k, 0);
    const CCTK_REAL m2_z_0 = gf(i, j, k - 2, 0);
    const CCTK_REAL p2_z_0 = gf(i, j, k + 2, 0);
    const CCTK_REAL m3_z_0 = gf(i, j, k - 3, 0);
    const CCTK_REAL p3_z_0 = gf(i, j, k + 3, 0);
    const CCTK_REAL m2_x_1 = gf(i - 2, j, k, 1);
    const CCTK_REAL p2_x_1 = gf(i + 2, j, k, 1);
    const CCTK_REAL m3_x_1 = gf(i - 3, j, k, 1);
    const CCTK_REAL p3_x_1 = gf(i + 3, j, k, 1);
    const CCTK_REAL m2_y_1 = gf(i, j - 2, k, 1);
    const CCTK_REAL p2_y_1 = gf(i, j + 2, k, 1);
    const CCTK_REAL m3_y_1 = gf(i, j - 3, k, 1);
    const CCTK_REAL p3_y_1 = gf(i, j + 3, k, 1);
    const CCTK_REAL m2_z_1 = gf(i, j, k - 2, 1);
    const CCTK_REAL p2_z_1 = gf(i, j, k + 2, 1);
    const CCTK_REAL m3_z_1 = gf(i, j, k - 3, 1);
    const CCTK_REAL p3_z_1 = gf(i, j, k + 3, 1);
    const CCTK_REAL m2_x_2 = gf(i - 2, j, k, 2);
    const CCTK_REAL p2_x_2 = gf(i + 2, j, k, 2);
    const CCTK_REAL m3_x_2 = gf(i - 3, j, k, 2);
    const CCTK_REAL p3_x_2 = gf(i + 3, j, k, 2);
    const CCTK_REAL m2_y_2 = gf(i, j - 2, k, 2);
    const CCTK_REAL p2_y_2 = gf(i, j + 2, k, 2);
    const CCTK_REAL m3_y_2 = gf(i, j - 3, k, 2);
    const CCTK_REAL p3_y_2 = gf(i, j + 3, k, 2);
    const CCTK_REAL m2_z_2 = gf(i, j, k - 2, 2);
    const CCTK_REAL p2_z_2 = gf(i, j, k + 2, 2);
    const CCTK_REAL m3_z_2 = gf(i, j, k - 3, 2);
    const CCTK_REAL p3_z_2 = gf(i, j, k + 3, 2);
    const CCTK_REAL m2_x_3 = gf(i - 2, j, k, 3);
    const CCTK_REAL p2_x_3 = gf(i + 2, j, k, 3);
    const CCTK_REAL m3_x_3 = gf(i - 3, j, k, 3);
    const CCTK_REAL p3_x_3 = gf(i + 3, j, k, 3);
    const CCTK_REAL m2_y_3 = gf(i, j - 2, k, 3);
    const CCTK_REAL p2_y_3 = gf(i, j + 2, k, 3);
    const CCTK_REAL m3_y_3 = gf(i, j - 3, k, 3);
    const CCTK_REAL p3_y_3 = gf(i, j + 3, k, 3);
    const CCTK_REAL m2_z_3 = gf(i, j, k - 2, 3);
    const CCTK_REAL p2_z_3 = gf(i, j, k + 2, 3);
    const CCTK_REAL m3_z_3 = gf(i, j, k - 3, 3);
    const CCTK_REAL p3_z_3 = gf(i, j, k + 3, 3);
    const CCTK_REAL m2_x_4 = gf(i - 2, j, k, 4);
    const CCTK_REAL p2_x_4 = gf(i + 2, j, k, 4);
    const CCTK_REAL m3_x_4 = gf(i - 3, j, k, 4);
    const CCTK_REAL p3_x_4 = gf(i + 3, j, k, 4);
    const CCTK_REAL m2_y_4 = gf(i, j - 2, k, 4);
    const CCTK_REAL p2_y_4 = gf(i, j + 2, k, 4);
    const CCTK_REAL m3_y_4 = gf(i, j - 3, k, 4);
    const CCTK_REAL p3_y_4 = gf(i, j + 3, k, 4);
    const CCTK_REAL m2_z_4 = gf(i, j, k - 2, 4);
    const CCTK_REAL p2_z_4 = gf(i, j, k + 2, 4);
    const CCTK_REAL m3_z_4 = gf(i, j, k - 3, 4);
    const CCTK_REAL p3_z_4 = gf(i, j, k + 3, 4);
    const CCTK_REAL m2_x_5 = gf(i - 2, j, k, 5);
    const CCTK_REAL p2_x_5 = gf(i + 2, j, k, 5);
    const CCTK_REAL m3_x_5 = gf(i - 3, j, k, 5);
    const CCTK_REAL p3_x_5 = gf(i + 3, j, k, 5);
    const CCTK_REAL m2_y_5 = gf(i, j - 2, k, 5);
    const CCTK_REAL p2_y_5 = gf(i, j + 2, k, 5);
    const CCTK_REAL m3_y_5 = gf(i, j - 3, k, 5);
    const CCTK_REAL p3_y_5 = gf(i, j + 3, k, 5);
    const CCTK_REAL m2_z_5 = gf(i, j, k - 2, 5);
    const CCTK_REAL p2_z_5 = gf(i, j, k + 2, 5);
    const CCTK_REAL m3_z_5 = gf(i, j, k - 3, 5);
    const CCTK_REAL p3_z_5 = gf(i, j, k + 3, 5);

    df[0][0] = (-45. * m1_x_0 + 9. * m2_x_0 - m3_x_0 + 45. * p1_x_0 -
                9. * p2_x_0 + p3_x_0) /
               (60. * dx[0]);
    df[0][1] = (-45. * m1_x_1 + 9. * m2_x_1 - m3_x_1 + 45. * p1_x_1 -
                9. * p2_x_1 + p3_x_1) /
               (60. * dx[0]);
    df[0][2] = (-45. * m1_x_2 + 9. * m2_x_2 - m3_x_2 + 45. * p1_x_2 -
                9. * p2_x_2 + p3_x_2) /
               (60. * dx[0]);
    df[0][3] = (-45. * m1_x_3 + 9. * m2_x_3 - m3_x_3 + 45. * p1_x_3 -
                9. * p2_x_3 + p3_x_3) /
               (60. * dx[0]);
    df[0][4] = (-45. * m1_x_4 + 9. * m2_x_4 - m3_x_4 + 45. * p1_x_4 -
                9. * p2_x_4 + p3_x_4) /
               (60. * dx[0]);
    df[0][5] = (-45. * m1_x_5 + 9. * m2_x_5 - m3_x_5 + 45. * p1_x_5 -
                9. * p2_x_5 + p3_x_5) /
               (60. * dx[0]);

    df[1][0] = (-45. * m1_y_0 + 9. * m2_y_0 - m3_y_0 + 45. * p1_y_0 -
                9. * p2_y_0 + p3_y_0) /
               (60. * dx[1]);
    df[1][1] = (-45. * m1_y_1 + 9. * m2_y_1 - m3_y_1 + 45. * p1_y_1 -
                9. * p2_y_1 + p3_y_1) /
               (60. * dx[1]);
    df[1][2] = (-45. * m1_y_2 + 9. * m2_y_2 - m3_y_2 + 45. * p1_y_2 -
                9. * p2_y_2 + p3_y_2) /
               (60. * dx[1]);
    df[1][3] = (-45. * m1_y_3 + 9. * m2_y_3 - m3_y_3 + 45. * p1_y_3 -
                9. * p2_y_3 + p3_y_3) /
               (60. * dx[1]);
    df[1][4] = (-45. * m1_y_4 + 9. * m2_y_4 - m3_y_4 + 45. * p1_y_4 -
                9. * p2_y_4 + p3_y_4) /
               (60. * dx[1]);
    df[1][5] = (-45. * m1_y_5 + 9. * m2_y_5 - m3_y_5 + 45. * p1_y_5 -
                9. * p2_y_5 + p3_y_5) /
               (60. * dx[1]);

    df[2][0] = (-45. * m1_z_0 + 9. * m2_z_0 - m3_z_0 + 45. * p1_z_0 -
                9. * p2_z_0 + p3_z_0) /
               (60. * dx[2]);
    df[2][1] = (-45. * m1_z_1 + 9. * m2_z_1 - m3_z_1 + 45. * p1_z_1 -
                9. * p2_z_1 + p3_z_1) /
               (60. * dx[2]);
    df[2][2] = (-45. * m1_z_2 + 9. * m2_z_2 - m3_z_2 + 45. * p1_z_2 -
                9. * p2_z_2 + p3_z_2) /
               (60. * dx[2]);
    df[2][3] = (-45. * m1_z_3 + 9. * m2_z_3 - m3_z_3 + 45. * p1_z_3 -
                9. * p2_z_3 + p3_z_3) /
               (60. * dx[2]);
    df[2][4] = (-45. * m1_z_4 + 9. * m2_z_4 - m3_z_4 + 45. * p1_z_4 -
                9. * p2_z_4 + p3_z_4) /
               (60. * dx[2]);
    df[2][5] = (-45. * m1_z_5 + 9. * m2_z_5 - m3_z_5 + 45. * p1_z_5 -
                9. * p2_z_5 + p3_z_5) /
               (60. * dx[2]);

  } else if constexpr (Order == 8) {

    const CCTK_REAL m2_x_0 = gf(i - 2, j, k, 0);
    const CCTK_REAL p2_x_0 = gf(i + 2, j, k, 0);
    const CCTK_REAL m3_x_0 = gf(i - 3, j, k, 0);
    const CCTK_REAL p3_x_0 = gf(i + 3, j, k, 0);
    const CCTK_REAL m4_x_0 = gf(i - 4, j, k, 0);
    const CCTK_REAL p4_x_0 = gf(i + 4, j, k, 0);
    const CCTK_REAL m2_y_0 = gf(i, j - 2, k, 0);
    const CCTK_REAL p2_y_0 = gf(i, j + 2, k, 0);
    const CCTK_REAL m3_y_0 = gf(i, j - 3, k, 0);
    const CCTK_REAL p3_y_0 = gf(i, j + 3, k, 0);
    const CCTK_REAL m4_y_0 = gf(i, j - 4, k, 0);
    const CCTK_REAL p4_y_0 = gf(i, j + 4, k, 0);
    const CCTK_REAL m2_z_0 = gf(i, j, k - 2, 0);
    const CCTK_REAL p2_z_0 = gf(i, j, k + 2, 0);
    const CCTK_REAL m3_z_0 = gf(i, j, k - 3, 0);
    const CCTK_REAL p3_z_0 = gf(i, j, k + 3, 0);
    const CCTK_REAL m4_z_0 = gf(i, j, k - 4, 0);
    const CCTK_REAL p4_z_0 = gf(i, j, k + 4, 0);
    const CCTK_REAL m2_x_1 = gf(i - 2, j, k, 1);
    const CCTK_REAL p2_x_1 = gf(i + 2, j, k, 1);
    const CCTK_REAL m3_x_1 = gf(i - 3, j, k, 1);
    const CCTK_REAL p3_x_1 = gf(i + 3, j, k, 1);
    const CCTK_REAL m4_x_1 = gf(i - 4, j, k, 1);
    const CCTK_REAL p4_x_1 = gf(i + 4, j, k, 1);
    const CCTK_REAL m2_y_1 = gf(i, j - 2, k, 1);
    const CCTK_REAL p2_y_1 = gf(i, j + 2, k, 1);
    const CCTK_REAL m3_y_1 = gf(i, j - 3, k, 1);
    const CCTK_REAL p3_y_1 = gf(i, j + 3, k, 1);
    const CCTK_REAL m4_y_1 = gf(i, j - 4, k, 1);
    const CCTK_REAL p4_y_1 = gf(i, j + 4, k, 1);
    const CCTK_REAL m2_z_1 = gf(i, j, k - 2, 1);
    const CCTK_REAL p2_z_1 = gf(i, j, k + 2, 1);
    const CCTK_REAL m3_z_1 = gf(i, j, k - 3, 1);
    const CCTK_REAL p3_z_1 = gf(i, j, k + 3, 1);
    const CCTK_REAL m4_z_1 = gf(i, j, k - 4, 1);
    const CCTK_REAL p4_z_1 = gf(i, j, k + 4, 1);
    const CCTK_REAL m2_x_2 = gf(i - 2, j, k, 1);
    const CCTK_REAL p2_x_2 = gf(i + 2, j, k, 2);
    const CCTK_REAL m3_x_2 = gf(i - 3, j, k, 2);
    const CCTK_REAL p3_x_2 = gf(i + 3, j, k, 2);
    const CCTK_REAL m4_x_2 = gf(i - 4, j, k, 2);
    const CCTK_REAL p4_x_2 = gf(i + 4, j, k, 2);
    const CCTK_REAL m2_y_2 = gf(i, j - 2, k, 2);
    const CCTK_REAL p2_y_2 = gf(i, j + 2, k, 2);
    const CCTK_REAL m3_y_2 = gf(i, j - 3, k, 2);
    const CCTK_REAL p3_y_2 = gf(i, j + 3, k, 2);
    const CCTK_REAL m4_y_2 = gf(i, j - 4, k, 2);
    const CCTK_REAL p4_y_2 = gf(i, j + 4, k, 2);
    const CCTK_REAL m2_z_2 = gf(i, j, k - 2, 2);
    const CCTK_REAL p2_z_2 = gf(i, j, k + 2, 2);
    const CCTK_REAL m3_z_2 = gf(i, j, k - 3, 2);
    const CCTK_REAL p3_z_2 = gf(i, j, k + 3, 2);
    const CCTK_REAL m4_z_2 = gf(i, j, k - 4, 2);
    const CCTK_REAL p4_z_2 = gf(i, j, k + 4, 2);
    const CCTK_REAL m2_x_3 = gf(i - 2, j, k, 3);
    const CCTK_REAL p2_x_3 = gf(i + 2, j, k, 3);
    const CCTK_REAL m3_x_3 = gf(i - 3, j, k, 3);
    const CCTK_REAL p3_x_3 = gf(i + 3, j, k, 3);
    const CCTK_REAL m4_x_3 = gf(i - 4, j, k, 3);
    const CCTK_REAL p4_x_3 = gf(i + 4, j, k, 3);
    const CCTK_REAL m2_y_3 = gf(i, j - 2, k, 3);
    const CCTK_REAL p2_y_3 = gf(i, j + 2, k, 3);
    const CCTK_REAL m3_y_3 = gf(i, j - 3, k, 3);
    const CCTK_REAL p3_y_3 = gf(i, j + 3, k, 3);
    const CCTK_REAL m4_y_3 = gf(i, j - 4, k, 3);
    const CCTK_REAL p4_y_3 = gf(i, j + 4, k, 3);
    const CCTK_REAL m2_z_3 = gf(i, j, k - 2, 3);
    const CCTK_REAL p2_z_3 = gf(i, j, k + 2, 3);
    const CCTK_REAL m3_z_3 = gf(i, j, k - 3, 3);
    const CCTK_REAL p3_z_3 = gf(i, j, k + 3, 3);
    const CCTK_REAL m4_z_3 = gf(i, j, k - 4, 3);
    const CCTK_REAL p4_z_3 = gf(i, j, k + 4, 3);
    const CCTK_REAL m2_x_4 = gf(i - 2, j, k, 4);
    const CCTK_REAL p2_x_4 = gf(i + 2, j, k, 4);
    const CCTK_REAL m3_x_4 = gf(i - 3, j, k, 4);
    const CCTK_REAL p3_x_4 = gf(i + 3, j, k, 4);
    const CCTK_REAL m4_x_4 = gf(i - 4, j, k, 4);
    const CCTK_REAL p4_x_4 = gf(i + 4, j, k, 4);
    const CCTK_REAL m2_y_4 = gf(i, j - 2, k, 4);
    const CCTK_REAL p2_y_4 = gf(i, j + 2, k, 4);
    const CCTK_REAL m3_y_4 = gf(i, j - 3, k, 4);
    const CCTK_REAL p3_y_4 = gf(i, j + 3, k, 4);
    const CCTK_REAL m4_y_4 = gf(i, j - 4, k, 4);
    const CCTK_REAL p4_y_4 = gf(i, j + 4, k, 4);
    const CCTK_REAL m2_z_4 = gf(i, j, k - 2, 4);
    const CCTK_REAL p2_z_4 = gf(i, j, k + 2, 4);
    const CCTK_REAL m3_z_4 = gf(i, j, k - 3, 4);
    const CCTK_REAL p3_z_4 = gf(i, j, k + 3, 4);
    const CCTK_REAL m4_z_4 = gf(i, j, k - 4, 4);
    const CCTK_REAL p4_z_4 = gf(i, j, k + 4, 4);
    const CCTK_REAL m2_x_5 = gf(i - 2, j, k, 4);
    const CCTK_REAL p2_x_5 = gf(i + 2, j, k, 5);
    const CCTK_REAL m3_x_5 = gf(i - 3, j, k, 5);
    const CCTK_REAL p3_x_5 = gf(i + 3, j, k, 5);
    const CCTK_REAL m4_x_5 = gf(i - 4, j, k, 5);
    const CCTK_REAL p4_x_5 = gf(i + 4, j, k, 5);
    const CCTK_REAL m2_y_5 = gf(i, j - 2, k, 5);
    const CCTK_REAL p2_y_5 = gf(i, j + 2, k, 5);
    const CCTK_REAL m3_y_5 = gf(i, j - 3, k, 5);
    const CCTK_REAL p3_y_5 = gf(i, j + 3, k, 5);
    const CCTK_REAL m4_y_5 = gf(i, j - 4, k, 5);
    const CCTK_REAL p4_y_5 = gf(i, j + 4, k, 5);
    const CCTK_REAL m2_z_5 = gf(i, j, k - 2, 5);
    const CCTK_REAL p2_z_5 = gf(i, j, k + 2, 5);
    const CCTK_REAL m3_z_5 = gf(i, j, k - 3, 5);
    const CCTK_REAL p3_z_5 = gf(i, j, k + 3, 5);
    const CCTK_REAL m4_z_5 = gf(i, j, k - 4, 5);
    const CCTK_REAL p4_z_5 = gf(i, j, k + 4, 5);

    df[0][0] = (-672. * m1_x_0 + 168. * m2_x_0 - 32. * m3_x_0 + 3. * m4_x_0 +
                672. * p1_x_0 - 168. * p2_x_0 + 32. * p3_x_0 - 3. * p4_x_0) /
               (840. * dx[0]);
    df[0][1] = (-672. * m1_x_1 + 168. * m2_x_1 - 32. * m3_x_1 + 3. * m4_x_1 +
                672. * p1_x_1 - 168. * p2_x_1 + 32. * p3_x_1 - 3. * p4_x_1) /
               (840. * dx[0]);
    df[0][2] = (-672. * m1_x_2 + 168. * m2_x_2 - 32. * m3_x_2 + 3. * m4_x_2 +
                672. * p1_x_2 - 168. * p2_x_2 + 32. * p3_x_2 - 3. * p4_x_2) /
               (840. * dx[0]);
    df[0][3] = (-672. * m1_x_3 + 168. * m2_x_3 - 32. * m3_x_3 + 3. * m4_x_3 +
                672. * p1_x_3 - 168. * p2_x_3 + 32. * p3_x_3 - 3. * p4_x_3) /
               (840. * dx[0]);
    df[0][4] = (-672. * m1_x_4 + 168. * m2_x_4 - 32. * m3_x_4 + 3. * m4_x_4 +
                672. * p1_x_4 - 168. * p2_x_4 + 32. * p3_x_4 - 3. * p4_x_4) /
               (840. * dx[0]);
    df[0][5] = (-672. * m1_x_5 + 168. * m2_x_5 - 32. * m3_x_5 + 3. * m4_x_5 +
                672. * p1_x_5 - 168. * p2_x_5 + 32. * p3_x_5 - 3. * p4_x_5) /
               (840. * dx[0]);

    df[1][0] = (-672. * m1_y_0 + 168. * m2_y_0 - 32. * m3_y_0 + 3. * m4_y_0 +
                672. * p1_y_0 - 168. * p2_y_0 + 32. * p3_y_0 - 3. * p4_y_0) /
               (840. * dx[1]);
    df[1][1] = (-672. * m1_y_1 + 168. * m2_y_1 - 32. * m3_y_1 + 3. * m4_y_1 +
                672. * p1_y_1 - 168. * p2_y_1 + 32. * p3_y_1 - 3. * p4_y_1) /
               (840. * dx[1]);
    df[1][2] = (-672. * m1_y_2 + 168. * m2_y_2 - 32. * m3_y_2 + 3. * m4_y_2 +
                672. * p1_y_2 - 168. * p2_y_2 + 32. * p3_y_2 - 3. * p4_y_2) /
               (840. * dx[1]);
    df[1][3] = (-672. * m1_y_3 + 168. * m2_y_3 - 32. * m3_y_3 + 3. * m4_y_3 +
                672. * p1_y_3 - 168. * p2_y_3 + 32. * p3_y_3 - 3. * p4_y_3) /
               (840. * dx[1]);
    df[1][4] = (-672. * m1_y_4 + 168. * m2_y_4 - 32. * m3_y_4 + 3. * m4_y_4 +
                672. * p1_y_4 - 168. * p2_y_4 + 32. * p3_y_4 - 3. * p4_y_4) /
               (840. * dx[1]);
    df[1][5] = (-672. * m1_y_5 + 168. * m2_y_5 - 32. * m3_y_5 + 3. * m4_y_5 +
                672. * p1_y_5 - 168. * p2_y_5 + 32. * p3_y_5 - 3. * p4_y_5) /
               (840. * dx[1]);

    df[2][0] = (-672. * m1_z_0 + 168. * m2_z_0 - 32. * m3_z_0 + 3. * m4_z_0 +
                672. * p1_z_0 - 168. * p2_z_0 + 32. * p3_z_0 - 3. * p4_z_0) /
               (840. * dx[2]);
    df[2][1] = (-672. * m1_z_1 + 168. * m2_z_1 - 32. * m3_z_1 + 3. * m4_z_1 +
                672. * p1_z_1 - 168. * p2_z_1 + 32. * p3_z_1 - 3. * p4_z_1) /
               (840. * dx[2]);
    df[2][2] = (-672. * m1_z_2 + 168. * m2_z_2 - 32. * m3_z_2 + 3. * m4_z_2 +
                672. * p1_z_2 - 168. * p2_z_2 + 32. * p3_z_2 - 3. * p4_z_2) /
               (840. * dx[2]);
    df[2][3] = (-672. * m1_z_3 + 168. * m2_z_3 - 32. * m3_z_3 + 3. * m4_z_3 +
                672. * p1_z_3 - 168. * p2_z_3 + 32. * p3_z_3 - 3. * p4_z_3) /
               (840. * dx[2]);
    df[2][4] = (-672. * m1_z_4 + 168. * m2_z_4 - 32. * m3_z_4 + 3. * m4_z_4 +
                672. * p1_z_4 - 168. * p2_z_4 + 32. * p3_z_4 - 3. * p4_z_4) /
               (840. * dx[2]);
    df[2][5] = (-672. * m1_z_5 + 168. * m2_z_5 - 32. * m3_z_5 + 3. * m4_z_5 +
                672. * p1_z_5 - 168. * p2_z_5 + 32. * p3_z_5 - 3. * p4_z_5) /
               (840. * dx[2]);

  } else {
    CCTK_INFO("Discretization of the first derivative at specified order is "
              "not yet implemented. Available orders: 2, 4, 6, 8.");
    throw std::invalid_argument(
        "Wrong order of discretization for the gradient.");
  }
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
