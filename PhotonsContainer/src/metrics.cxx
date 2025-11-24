#include <cctk.h>

#include "Photons.hxx"
#include "PhotonsContainer.hxx"
#include "PhotonsInitializers.hxx"
#include <AMReX_ParallelDescriptor.H>
#include <CParameters.h>

#include <cstring>
#include <driver.hxx>

#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <cctk_Types.h>
#include <cctk_core.h>

#include <iostream>
#include <loop_device.hxx>
#include <ostream>

extern "C" void PhotonsContainer_init_metric(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_PhotonsContainer_init_metric;

  if (!std::strcmp(metric, "minkowski")) {
    CCTK_INFO("Initializing Minkowski coordinates");

    // Initialize the metric, lapse, beta and K
    grid.loop_all_device<0, 0, 0>(grid.nghostzones,
                                  [=] CCTK_DEVICE(const Loop::PointDesc &p)
                                      CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                        alp(p.I) = 1.0;
                                        betax(p.I) = 0.0;
                                        betay(p.I) = 0.0;
                                        betaz(p.I) = 0.0;
                                        gxx(p.I) = 1.0;
                                        gxy(p.I) = 0.0;
                                        gxz(p.I) = 0.0;
                                        gyy(p.I) = 1.0;
                                        gyz(p.I) = 0.0;
                                        gzz(p.I) = 1.0;
                                        kxx(p.I) = 0.0;
                                        kxy(p.I) = 0.0;
                                        kxz(p.I) = 0.0;
                                        kyy(p.I) = 0.0;
                                        kyz(p.I) = 0.0;
                                        kzz(p.I) = 0.0;
                                      });

    CCTK_INFO("FIELDS INITIALIZED");
  } else if (!std::strcmp(metric, "iso")) {
    CCTK_INFO("Initializing Schwarzschild using Isotropic coordinates");
    const double black_hole_mass = metric_params_d[0];

    // Initialize the metric, lapse, beta and K
    grid.loop_all_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          const double R = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);

          const double psi = 1.0 + black_hole_mass / (2.0 * R);
          const double psi_2 = psi * psi;
          const double psi_4 = psi_2 * psi_2;

          if (R >= 0.5) {
            alp(p.I) = (1.0 - black_hole_mass / (2.0 * R)) /
                       (1.0 + black_hole_mass / (2.0 * R));
            gxx(p.I) = psi_4;
            gyy(p.I) = psi_4;
            gzz(p.I) = psi_4;
          } else {
            alp(p.I) = 0.;
            gxx(p.I) = 10e-12;
            gyy(p.I) = 10e-12;
            gzz(p.I) = 10e-12;
          }

          betax(p.I) = 0.0;
          betay(p.I) = 0.0;
          betaz(p.I) = 0.0;

          gxy(p.I) = 0.0;
          gxz(p.I) = 0.0;
          gyz(p.I) = 0.0;

          kxx(p.I) = 0.0;
          kxy(p.I) = 0.0;
          kxz(p.I) = 0.0;
          kyy(p.I) = 0.0;
          kyz(p.I) = 0.0;
          kzz(p.I) = 0.0;
        });
  } else if (!std::strcmp(metric, "schwarzschild")) {
    CCTK_INFO("Initializing Schwarzschild using Schwarzschild coordinates");
    // Initialize the metric, lapse, beta and K
    const double black_hole_mass = metric_params_d[0];
    grid.loop_all_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          auto R = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
          auto r2 = p.x * p.x + p.y * p.y;

          if (R < 10e-10) {
            R = 10e-10;
          }

          if (r2 < 10e-10) {
            r2 = 10e-10;
          }

          auto f = 1.0 - 2 * black_hole_mass / R;
          auto f_inv = 1.0 / f;

          const auto den = R * R * r2;

          if (R > 0.5) {
            alp(p.I) = std::sqrt(f);
            gxx(p.I) = (p.z * p.z * p.x * p.x + R * R * p.y * p.y +
                        p.x * p.x * r2 * f_inv) /
                       den;
            gyy(p.I) = (p.z * p.z * p.y * p.y + R * R * p.x * p.x +
                        p.y * p.y * r2 * f_inv) /
                       den;
            gzz(p.I) = (p.z * p.z * p.z * p.z + R * R * R * R -
                        2. * p.z * p.z * R * R + p.z * p.z * r2 * f_inv) /
                       den;
            gxy(p.I) = (p.z * p.z * p.x * p.y - R * R * p.x * p.y +
                        p.x * p.y * r2 * f_inv) /
                       den;
            gxz(p.I) = (p.z * p.z * p.z * p.x - p.z * p.x * R * R +
                        p.x * p.z * r2 * f_inv) /
                       den;
            gyz(p.I) = (p.z * p.z * p.z * p.y - p.z * p.y * R * R +
                        p.z * p.y * r2 * f_inv) /
                       den;
          } else {
            alp(p.I) = 0.0;
            gxx(p.I) = 10e-12;
            gyy(p.I) = 10e-12;
            gzz(p.I) = 10e-12;
            gxy(p.I) = 0.0;
            gxz(p.I) = 0.0;
            gyz(p.I) = 0.0;
          }

          betax(p.I) = 0.0;
          betay(p.I) = 0.0;
          betaz(p.I) = 0.0;

          kxx(p.I) = 0.0;
          kxy(p.I) = 0.0;
          kxz(p.I) = 0.0;
          kyy(p.I) = 0.0;
          kyz(p.I) = 0.0;
          kzz(p.I) = 0.0;
        });
  }
}
