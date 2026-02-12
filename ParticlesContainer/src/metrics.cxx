/**
 * \file metrics.cxx
 * \brief Available metrics functions to initialize.
 *
 *  This file contains the function that initialize all the available metrics
 * depending on the user's parameter input.
 */
#include <cctk.h>

#include <AMReX_ParallelDescriptor.H>
#include <CParameters.h>

#include <cstring>
#include <driver.hxx>

#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <cctk_Types.h>
#include <cctk_core.h>

#include <loop_device.hxx>

/**
 * \brief initialize the fields data using the chosen metric.
 *
 * This function initializes the \f$\alpha\f$, \f$\vec{\beta}\f$,
 * \f$\gamma_{\mu\nu}\f$ and \f$K_{\mu\nu}\f$ ADM elements needed for the
 * geodesics evolution.
 *
 * - The chose of the metric that are going to be initialized is defined by the
 * KEYWORD parameter 'metric'.
 * - Any extra parameter can be passed by using the size 10 double
 * 'metric_params_d' and integer 'metric_params_i' arrays.
 */
extern "C" void ParticlesContainer_init_metric(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_ParticlesContainer_init_metric;

  if (!std::strcmp(metric, "Minkowski")) {

    CCTK_INFO("Initializing Minkowski coordinates");

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

  } else if (!std::strcmp(metric, "Isotropic Schwarzschild")) {

    CCTK_INFO("Initializing Schwarzschild using Isotropic coordinates");

    const double black_hole_mass = metric_params_d[0];

    grid.loop_all_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          const double R = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);

          const double psi = 1.0 + black_hole_mass / (2.0 * R);
          const double psi_2 = psi * psi;
          const double psi_4 = psi_2 * psi_2;

          if (R >= 0.5 * black_hole_mass) {
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

  } else if (!std::strcmp(metric, "Schwarzschild")) {

    CCTK_INFO("Initializing Schwarzschild using Schwarzschild coordinates");

    const double black_hole_mass = metric_params_d[0];

    grid.loop_all_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          auto R = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
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
            alp(p.I) = sqrt(f);
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

  } else if (!std::strcmp(metric, "Kerr-Schild")) {

    CCTK_INFO("Initializing Kerr-Shild coordinates");

    const double m = metric_params_d[0];
    const double a = metric_params_d[1];

    grid.loop_all_device<
        0, 0,
        0>(grid.nghostzones,
           [=] CCTK_DEVICE(
               const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
             const CCTK_REAL R2 = p.x * p.x + p.y * p.y + p.z * p.z;
             CCTK_REAL r2 = (R2 - a * a +
                             std::sqrt((R2 - a * a) * (R2 - a * a) +
                                       4. * p.z * p.z * a * a)) /
                            2.0;

             if (r2 < 10e-12) {
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
               return;
             }

             const CCTK_REAL r = std::sqrt(r2);

             const CCTK_REAL f =
                 2. * m * r2 * r / (r2 * r2 + a * a * p.z * p.z);
             const CCTK_REAL l[3] = {(r * p.x + a * p.y) / (r2 + a * a),
                                     (r * p.y - a * p.x) / (r2 + a * a),
                                     p.z / r};

             alp(p.I) = std::sqrt(1 / (1 + f));

             betax(p.I) = f * l[0] / (f + 1);
             betay(p.I) = f * l[1] / (f + 1);
             betaz(p.I) = f * l[2] / (f + 1);

             gxx(p.I) = 1. + f * l[0] * l[0];
             gxy(p.I) = 0. + f * l[0] * l[1];
             gxz(p.I) = 0. + f * l[0] * l[2];
             gyy(p.I) = 1. + f * l[1] * l[1];
             gyz(p.I) = 0. + f * l[1] * l[2];
             gzz(p.I) = 1. + f * l[2] * l[2];


             const CCTK_REAL sqrt_r = std::sqrt((R2 - a * a) * (R2 - a * a) +
                                                4. * a * a * p.z * p.z);
             const CCTK_REAL dr[3] = {
std::sqrt(2.)*p.x*std::sqrt(-a*a + p.x*p.x + p.y*p.y + p.z*p.z + std::sqrt(4.*a*a*p.z*p.z + ((-a*a + p.x*p.x + p.y*p.y + p.z*p.z)*(-a*a + p.x*p.x + p.y*p.y + p.z*p.z))))/(2.*std::sqrt(4.*a*a*p.z*p.z + ((-a*a + p.x*p.x + p.y*p.y + p.z*p.z)*(-a*a + p.x*p.x + p.y*p.y + p.z*p.z)))),
 std::sqrt(2.)*p.y*std::sqrt(-a*a + p.x*p.x + p.y*p.y + p.z*p.z + std::sqrt(4.*a*a*p.z*p.z + ((-a*a + p.x*p.x + p.y*p.y + p.z*p.z)*(-a*a + p.x*p.x + p.y*p.y + p.z*p.z))))/(2.*std::sqrt(4.*a*a*p.z*p.z + ((-a*a + p.x*p.x + p.y*p.y + p.z*p.z)*(-a*a + p.x*p.x + p.y*p.y + p.z*p.z)))),
 std::sqrt(2.)*p.z*(a*a + p.x*p.x + p.y*p.y + p.z*p.z + std::sqrt(4.*a*a*p.z*p.z + ((-a*a + p.x*p.x + p.y*p.y + p.z*p.z)*(-a*a + p.x*p.x + p.y*p.y + p.z*p.z))))/(2.*std::sqrt(4.*a*a*p.z*p.z + ((-a*a + p.x*p.x + p.y*p.y + p.z*p.z)*(-a*a + p.x*p.x + p.y*p.y + p.z*p.z)))*std::sqrt(-a*a + p.x*p.x + p.y*p.y + p.z*p.z + std::sqrt(4.*a*a*p.z*p.z + ((-a*a + p.x*p.x + p.y*p.y + p.z*p.z)*(-a*a + p.x*p.x + p.y*p.y + p.z*p.z)))))
           };

            
             const CCTK_REAL dbeta[3][3] =
                 {
                     {2. * m *
                          (-4. * (a * a + r2) * (a * p.y + p.x * r) * r2 * r2 *
                               dr[0] +
                           (a * a + r2) * (a * a * p.z * p.z + r2 * r2) *
                               (3. * (a * p.y + p.x * r) * dr[0] +
                                (p.x * dr[0] + r) * r) -
                           2. * (a * p.y + p.x * r) *
                               (a * a * p.z * p.z + r2 * r2) * r2 * dr[0]) *
                          r2 /
                          ((a * a + r2) * (a * a + r2) *
                           (a * a * p.z * p.z + r2 * r2) *
                           (a * a * p.z * p.z + r2 * r2)),
                      2. * m *
                          (4. * (a * a + r2) * (a * p.x - p.y * r) * r2 * r2 *
                               dr[0] -
                           (a * a + r2) * (a * a * p.z * p.z + r2 * r2) *
                               ((a - p.y * dr[0]) * r +
                                3. * (a * p.x - p.y * r) * dr[0]) +
                           2. * (a * p.x - p.y * r) *
                               (a * a * p.z * p.z + r2 * r2) * r2 * dr[0]) *
                          r2 /
                          ((a * a + r2) * (a * a + r2) *
                           (a * a * p.z * p.z + r2 * r2) *
                           (a * a * p.z * p.z + r2 * r2)),
                      4. * m * p.z * (a * a * p.z * p.z - r2 * r2) * r * dr[0] /
                          ((a * a * p.z * p.z + r2 * r2) *
                           (a * a * p.z * p.z + r2 * r2))},
                     {2. * m *
                          (-4. * (a * a + r2) * (a * p.y + p.x * r) * r2 * r2 *
                               dr[1] +
                           (a * a + r2) * (a * a * p.z * p.z + r2 * r2) *
                               ((a + p.x * dr[1]) * r +
                                3. * (a * p.y + p.x * r) * dr[1]) -
                           2. * (a * p.y + p.x * r) *
                               (a * a * p.z * p.z + r2 * r2) * r2 * dr[1]) *
                          r2 /
                          ((a * a + r2) * (a * a + r2) *
                           (a * a * p.z * p.z + r2 * r2) *
                           (a * a * p.z * p.z + r2 * r2)),
                      2. * m *
                          (4. * (a * a + r2) * (a * p.x - p.y * r) * r2 * r2 *
                               dr[1] +
                           (a * a + r2) * (a * a * p.z * p.z + r2 * r2) *
                               (-3. * (a * p.x - p.y * r) * dr[1] +
                                (p.y * dr[1] + r) * r) +
                           2. * (a * p.x - p.y * r) *
                               (a * a * p.z * p.z + r2 * r2) * r2 * dr[1]) *
                          r2 /
                          ((a * a + r2) * (a * a + r2) *
                           (a * a * p.z * p.z + r2 * r2) *
                           (a * a * p.z * p.z + r2 * r2)),
                      4. * m * p.z * (a * a * p.z * p.z - r2 * r2) * r * dr[1] /
                          ((a * a * p.z * p.z + r2 * r2) *
                           (a * a * p.z * p.z + r2 * r2))},
                 {2.*m*(-2.*(a*a + r2)*(a*p.y + p.x*r)*(a*a*p.z + 2.*r2*r*dr[2])*r + (a*a + r2)*(3.*a*p.y + 4.*p.x*r)*(a*a*p.z*p.z + r2*r2)*dr[2] - 2.*(a*p.y + p.x*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[2])*r2/((a*a+r2)*(a*a+r2)*(a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2)),
 2.*m*((a*a + r2)*(-3.*a*p.x + 4.*p.y*r)*(a*a*p.z*p.z + r2*r2)*dr[2] + 2.*(a*a + r2)*(a*p.x - p.y*r)*(a*a*p.z + 2.*r2*r*dr[2])*r + 2.*(a*p.x - p.y*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[2])*r2/((a*a+r2)*(a*a+r2)*(a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2)),
 2.*m*(-2.*p.z*(a*a*p.z + 2.*r2*r*dr[2])*r + (a*a*p.z*p.z + r2*r2)*(2.*p.z*dr[2] + r))*r/((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))}
                 };

          const CCTK_REAL dgamma[3][6] = {{2.*m*(a*p.y + p.x*r)*(-4.*(a*a + r2)*(a*p.y + p.x*r)*r2*r2*dr[0] + (a*a + r2)*(a*a*p.z*p.z + r2*r2)*(3.*(a*p.y + p.x*r)*dr[0] + 2.*(p.x*dr[0] + r)*r) - 4.*(a*p.y + p.x*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[0])*r2/((a*a + r2)*(a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*(4.*(a*a + r2)*(a*p.x - p.y*r)*(a*p.y + p.x*r)*r2*r2*dr[0] - (a*a + r2)*(a*a*p.z*p.z + r2*r2)*((a - p.y*dr[0])*(a*p.y + p.x*r)*r + 3.*(a*p.x - p.y*r)*(a*p.y + p.x*r)*dr[0] + (a*p.x - p.y*r)*(p.x*dr[0] + r)*r) + 4.*(a*p.x - p.y*r)*(a*p.y + p.x*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[0])*r2/((a*a + r2)*(a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*p.z*(-4.*(a*a + r2)*(a*p.y + p.x*r)*r2*r2*dr[0] + (a*a + r2)*(a*a*p.z*p.z + r2*r2)*(2.*(a*p.y + p.x*r)*dr[0] + (p.x*dr[0] + r)*r) - 2.*(a*p.y + p.x*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[0])*r/((a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*(a*p.x - p.y*r)*(-4.*(a*a + r2)*(a*p.x - p.y*r)*r2*r2*dr[0] + (a*a + r2)*(a*a*p.z*p.z + r2*r2)*(2.*(a - p.y*dr[0])*r + 3.*(a*p.x - p.y*r)*dr[0]) - 4.*(a*p.x - p.y*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[0])*r2/((a*a + r2)*(a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*p.z*(4.*(a*a + r2)*(a*p.x - p.y*r)*r2*r2*dr[0] - (a*a + r2)*(a*a*p.z*p.z + r2*r2)*((a - p.y*dr[0])*r + 2.*(a*p.x - p.y*r)*dr[0]) + 2.*(a*p.x - p.y*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[0])*r/((a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*p.z*p.z*(a*a*p.z*p.z - 3.*r2*r2)*dr[0]/((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))},
                                          {2.*m*(a*p.y + p.x*r)*(-4.*(a*a + r2)*(a*p.y + p.x*r)*r2*r2*dr[1] + (a*a + r2)*(a*a*p.z*p.z + r2*r2)*(2.*(a + p.x*dr[1])*r + 3.*(a*p.y + p.x*r)*dr[1]) - 4.*(a*p.y + p.x*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[1])*r2/((a*a + r2)*(a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*(4.*(a*a + r2)*(a*p.x - p.y*r)*(a*p.y + p.x*r)*r2*r2*dr[1] + (a*a + r2)*(a*a*p.z*p.z + r2*r2)*(-(a + p.x*dr[1])*(a*p.x - p.y*r)*r - 3.*(a*p.x - p.y*r)*(a*p.y + p.x*r)*dr[1] + (a*p.y + p.x*r)*(p.y*dr[1] + r)*r) + 4.*(a*p.x - p.y*r)*(a*p.y + p.x*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[1])*r2/((a*a + r2)*(a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*p.z*(-4.*(a*a + r2)*(a*p.y + p.x*r)*r2*r2*dr[1] + (a*a + r2)*(a*a*p.z*p.z + r2*r2)*((a + p.x*dr[1])*r + 2.*(a*p.y + p.x*r)*dr[1]) - 2.*(a*p.y + p.x*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[1])*r/((a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*(a*p.x - p.y*r)*(-4.*(a*a + r2)*(a*p.x - p.y*r)*r2*r2*dr[1] + (a*a + r2)*(a*a*p.z*p.z + r2*r2)*(3.*(a*p.x - p.y*r)*dr[1] - 2.*(p.y*dr[1] + r)*r) - 4.*(a*p.x - p.y*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[1])*r2/((a*a + r2)*(a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*p.z*(4.*(a*a + r2)*(a*p.x - p.y*r)*r2*r2*dr[1] + (a*a + r2)*(a*a*p.z*p.z + r2*r2)*(-2.*(a*p.x - p.y*r)*dr[1] + (p.y*dr[1] + r)*r) + 2.*(a*p.x - p.y*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[1])*r/((a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*p.z*p.z*(a*a*p.z*p.z - 3.*r2*r2)*dr[1]/((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))},
                                          {2.*m*(a*p.y + p.x*r)*(-2.*(a*a + r2)*(a*p.y + p.x*r)*(a*a*p.z + 2.*r2*r*dr[2])*r + (a*a + r2)*(3.*a*p.y + 5.*p.x*r)*(a*a*p.z*p.z + r2*r2)*dr[2] - 4.*(a*p.y + p.x*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[2])*r2/((a*a + r2)*(a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*(2.*(a*a + r2)*(a*p.x - p.y*r)*(a*p.y + p.x*r)*(a*a*p.z + 2.*r2*r*dr[2])*r + (a*a + r2)*(a*a*p.z*p.z + r2*r2)*(-p.x*(a*p.x - p.y*r)*r + p.y*(a*p.y + p.x*r)*r - 3.*(a*p.x - p.y*r)*(a*p.y + p.x*r))*dr[2] + 4.*(a*p.x - p.y*r)*(a*p.y + p.x*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[2])*r2/((a*a + r2)*(a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*(-2.*p.z*(a*a + r2)*(a*p.y + p.x*r)*(a*a*p.z + 2.*r2*r*dr[2])*r - 2.*p.z*(a*p.y + p.x*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[2] + (a*a + r2)*(a*a*p.z*p.z + r2*r2)*(p.x*p.z*r*dr[2] + 2.*p.z*(a*p.y + p.x*r)*dr[2] + (a*p.y + p.x*r)*r))*r/((a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*(a*p.x - p.y*r)*(-2.*(a*a + r2)*(a*p.x - p.y*r)*(a*a*p.z + 2.*r2*r*dr[2])*r + (a*a + r2)*(3.*a*p.x - 5.*p.y*r)*(a*a*p.z*p.z + r2*r2)*dr[2] - 4.*(a*p.x - p.y*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[2])*r2/((a*a + r2)*(a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*(2.*p.z*(a*a + r2)*(a*p.x - p.y*r)*(a*a*p.z + 2.*r2*r*dr[2])*r + 2.*p.z*(a*p.x - p.y*r)*(a*a*p.z*p.z + r2*r2)*r2*dr[2] + (a*a + r2)*(a*a*p.z*p.z + r2*r2)*(p.y*p.z*r*dr[2] - 2.*p.z*(a*p.x - p.y*r)*dr[2] - (a*p.x - p.y*r)*r))*r/((a*a + r2)*(a*a + r2)*((a*a*p.z*p.z + r2*r2)*(a*a*p.z*p.z + r2*r2))),
 2.*m*p.z*(a*a*p.z*p.z*p.z*dr[2] - 3.*p.z*r2*r2*dr[2] + 2.*r2*r2*r)/(a*a*a*a*p.z*p.z*p.z*p.z + 2.*a*a*p.z*p.z*r2*r2 + r2*r2*r2*r2)}};

             kxx(p.I) = dbeta[0][0] + dbeta[0][0];
             kxy(p.I) = dbeta[0][1] + dbeta[1][0];
             kxz(p.I) = dbeta[0][2] + dbeta[2][0];
             kyy(p.I) = dbeta[1][1] + dbeta[1][1];
             kyz(p.I) = dbeta[1][2] + dbeta[2][1];
             kzz(p.I) = dbeta[2][2] + dbeta[2][2];

             const CCTK_REAL beta[3] = {betax(p.I), betay(p.I), betaz(p.I)};

             kxx(p.I) -= beta[0] * (dgamma[0][0] + dgamma[0][0] - dgamma[0][0]);
             kxx(p.I) -= beta[1] * (dgamma[0][1] + dgamma[0][1] - dgamma[1][0]);
             kxx(p.I) -= beta[2] * (dgamma[0][2] + dgamma[0][2] - dgamma[2][0]);

             kxy(p.I) -= beta[0] * (dgamma[0][1] + dgamma[1][0] - dgamma[0][1]);
             kxy(p.I) -= beta[1] * (dgamma[0][3] + dgamma[1][1] - dgamma[1][1]);
             kxy(p.I) -= beta[2] * (dgamma[0][4] + dgamma[1][2] - dgamma[2][1]);

             kxz(p.I) -= beta[0] * (dgamma[0][2] + dgamma[2][0] - dgamma[0][2]);
             kxz(p.I) -= beta[1] * (dgamma[0][4] + dgamma[2][1] - dgamma[1][2]);
             kxz(p.I) -= beta[2] * (dgamma[0][5] + dgamma[2][2] - dgamma[2][2]);

             kyy(p.I) -= beta[0] * (dgamma[1][1] + dgamma[1][1] - dgamma[0][3]);
             kyy(p.I) -= beta[1] * (dgamma[1][3] + dgamma[1][3] - dgamma[1][3]);
             kyy(p.I) -= beta[2] * (dgamma[1][4] + dgamma[1][4] - dgamma[2][3]);

             kyz(p.I) -= beta[0] * (dgamma[1][2] + dgamma[2][1] - dgamma[0][4]);
             kyz(p.I) -= beta[1] * (dgamma[1][4] + dgamma[2][3] - dgamma[1][4]);
             kyz(p.I) -= beta[2] * (dgamma[1][5] + dgamma[2][4] - dgamma[2][4]);

             kzz(p.I) -= beta[0] * (dgamma[2][2] + dgamma[2][2] - dgamma[0][5]);
             kzz(p.I) -= beta[1] * (dgamma[2][4] + dgamma[2][4] - dgamma[1][5]);
             kzz(p.I) -= beta[2] * (dgamma[2][5] + dgamma[2][5] - dgamma[2][5]);

             kxx(p.I) /= (2. * alp(p.I));
             kxy(p.I) /= (2. * alp(p.I));
             kxz(p.I) /= (2. * alp(p.I));
             kyy(p.I) /= (2. * alp(p.I));
             kyz(p.I) /= (2. * alp(p.I));
             kzz(p.I) /= (2. * alp(p.I));

           });
  } else if (!std::strcmp(metric, "Painleve")) {

    CCTK_INFO("Initializing Painlevé coordinates");

    const double m = metric_params_d[0];

    grid.loop_all_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          const CCTK_REAL r2 = p.x * p.x + p.y * p.y + p.z * p.z;
          CCTK_REAL r = sqrt(r2);

          if (r2 < 10e-10) {
            gxx(p.I) = 1.0;
            gyy(p.I) = 1.0;
            gzz(p.I) = 1.0;
            gxy(p.I) = 0.0;
            gxz(p.I) = 0.0;
            gyz(p.I) = 0.0;

            betax(p.I) = 0.0;
            betay(p.I) = 0.0;
            betaz(p.I) = 0.0;

            alp(p.I) = 0.0;

            kxx(p.I) = 0.0;
            kxy(p.I) = 0.0;
            kxz(p.I) = 0.0;
            kyy(p.I) = 0.0;
            kyz(p.I) = 0.0;
            kzz(p.I) = 0.0;
            return;
          }

          const CCTK_REAL f = sqrt(2. * m / r);

          gxx(p.I) = 1.0;
          gyy(p.I) = 1.0;
          gzz(p.I) = 1.0;
          gxy(p.I) = 0.0;
          gxz(p.I) = 0.0;
          gyz(p.I) = 0.0;

          betax(p.I) = p.x * f / r;
          betay(p.I) = p.y * f / r;
          betaz(p.I) = p.z * f / r;

          alp(p.I) = 1.0;

          const CCTK_REAL dr[3] = {p.x / r, p.y / r, p.z / r};

          const CCTK_REAL dbeta[3][3] = {
              {f / r - p.x * 3. * m * dr[0] / (r2 * r * f),
               -p.y * 3. * m * dr[0] / (r2 * r * f),
               -p.z * 3. * m * dr[0] / (r2 * r * f)},
              {-p.x * 3. * m * dr[1] / (r2 * r * f),
               f / r - p.y * 3 * m * dr[1] / (r2 * r * f),
               -p.z * 3. * m * dr[1] / (r2 * r * f)},
              {-p.x * 3. * m * dr[2] / (r2 * r * f),
               -p.y * 3 * m * dr[2] / (r2 * r * f),
               f / r - p.z * 3. * m * dr[2] / (r2 * r * f)}};

          const CCTK_REAL dgamma[3][6] = {{0., 0., 0., 0., 0., 0.},
                                          {0., 0., 0., 0., 0., 0.},
                                          {0., 0., 0., 0., 0., 0.}};

          kxx(p.I) = dbeta[0][0] + dbeta[0][0];
          kxy(p.I) = dbeta[0][1] + dbeta[1][0];
          kxz(p.I) = dbeta[0][2] + dbeta[2][0];
          kyy(p.I) = dbeta[1][1] + dbeta[1][1];
          kyz(p.I) = dbeta[1][2] + dbeta[2][1];
          kzz(p.I) = dbeta[2][2] + dbeta[2][2];

          const CCTK_REAL beta[3] = {betax(p.I), betay(p.I), betaz(p.I)};

          kxx(p.I) -= beta[0] * (dgamma[0][0] + dgamma[0][0] - dgamma[0][0]);
          kxx(p.I) -= beta[1] * (dgamma[0][1] + dgamma[0][1] - dgamma[1][0]);
          kxx(p.I) -= beta[2] * (dgamma[0][2] + dgamma[0][2] - dgamma[2][0]);

          kxy(p.I) -= beta[0] * (dgamma[0][1] + dgamma[1][0] - dgamma[0][1]);
          kxy(p.I) -= beta[1] * (dgamma[0][3] + dgamma[1][1] - dgamma[1][1]);
          kxy(p.I) -= beta[2] * (dgamma[0][4] + dgamma[1][2] - dgamma[2][1]);

          kxz(p.I) -= beta[0] * (dgamma[0][2] + dgamma[2][0] - dgamma[0][2]);
          kxz(p.I) -= beta[1] * (dgamma[0][4] + dgamma[2][1] - dgamma[1][2]);
          kxz(p.I) -= beta[2] * (dgamma[0][5] + dgamma[2][2] - dgamma[2][2]);

          kyy(p.I) -= beta[0] * (dgamma[1][1] + dgamma[1][1] - dgamma[0][3]);
          kyy(p.I) -= beta[1] * (dgamma[1][3] + dgamma[1][3] - dgamma[1][3]);
          kyy(p.I) -= beta[2] * (dgamma[1][4] + dgamma[1][4] - dgamma[2][3]);

          kyz(p.I) -= beta[0] * (dgamma[1][2] + dgamma[2][1] - dgamma[0][4]);
          kyz(p.I) -= beta[1] * (dgamma[1][4] + dgamma[2][3] - dgamma[1][4]);
          kyz(p.I) -= beta[2] * (dgamma[1][5] + dgamma[2][4] - dgamma[2][4]);

          kzz(p.I) -= beta[0] * (dgamma[2][2] + dgamma[2][2] - dgamma[0][5]);
          kzz(p.I) -= beta[1] * (dgamma[2][4] + dgamma[2][4] - dgamma[1][5]);
          kzz(p.I) -= beta[2] * (dgamma[2][5] + dgamma[2][5] - dgamma[2][5]);

          kxx(p.I) /= (2. * alp(p.I));
          kxy(p.I) /= (2. * alp(p.I));
          kxz(p.I) /= (2. * alp(p.I));
          kyy(p.I) /= (2. * alp(p.I));
          kyz(p.I) /= (2. * alp(p.I));
          kzz(p.I) /= (2. * alp(p.I));
        });
  }
}
