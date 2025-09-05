#ifndef DISCRETIZER_HXX
#define DISCRETIZER_HXX

#include "AMReX_Box.H"
#include "AMReX_Config.H"
#include "cctk_Types.h"
#include <array>
#include <cctk.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace Discretize {

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
