#include <cctk.h>

#include "AMReX_GpuDevice.H"
#include "Photons.hxx"
#include "PhotonsContainer.hxx"
#include "PhotonsInitializers.hxx"
#include <AMReX_ParallelDescriptor.H>
#include <CParameters.h>

#include <driver.hxx>

#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <cctk_Types.h>
#include <cctk_core.h>

#include <iostream>
#include <loop_device.hxx>
#include <ostream>

using ParticleData = Photons::PhotonsData;

using PC = Containers::PhotonsContainer<ParticleData>;
std::vector<std::unique_ptr<PC>> g_nupcs;

extern "C" void setup(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  const int tl = 0;
  const int gi_metric = CCTK_GroupIndex("ADMBaseX::metric");
  assert(gi_metric >= 0 && "Failed to get the metric group index");

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
    const auto &patchdata = CarpetX::ghext->patchdata.at(patch);
    g_nupcs.push_back(std::make_unique<PC>(patchdata.amrcore.get()));

    auto &pc = g_nupcs.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);

    for (int lev = 0; lev < pd.leveldata.size(); ++lev) {
      const auto &ld = pd.leveldata.at(lev);
      const auto &gd_metric = *ld.groupdata.at(gi_metric);
      const amrex::MultiFab &metric = *gd_metric.mfab[tl];

      pc->initialize(
          photons_init::random_parallel_photons_per_container_initializer<
              ParticleData, PC>,
          total_photons, metric);
    }
  }
}

extern "C" void init_minkowski(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_init_minkowski;

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
}

extern "C" void init_iso_schwarzschild(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_init_iso_schwarzschild;

  CCTK_INFO("Initializing Schwarzschild using Isotropic coordinates");

  // Initialize the metric, lapse, beta and K
  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        auto R = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);

        const auto psi = 1.0 + black_hole_mass / (2.0 * R);
        const auto psi_2 = psi * psi;
        const auto psi_4 = psi_2 * psi_2;

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
}

extern "C" void init_schwarzschild(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_init_schwarzschild;

  CCTK_INFO("Initializing Schwarzschild using Schwarzschild coordinates");

  // Initialize the metric, lapse, beta and K
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

        auto f = 1.0 - 0.5 / R;
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

extern "C" void test(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  {
    auto t0 = amrex::ParallelDescriptor::second();
    const CCTK_REAL dt = CCTK_DELTA_TIME;

    const int tl = 0;
    const int gi_lapse = CCTK_GroupIndex("ADMBaseX::lapse");
    const int gi_shift = CCTK_GroupIndex("ADMBaseX::shift");
    const int gi_metric = CCTK_GroupIndex("ADMBaseX::metric");
    const int gi_curv = CCTK_GroupIndex("ADMBaseX::curv");
    assert(gi_lapse >= 0 && "Failed to get the lapse group index");
    assert(gi_shift >= 0 && "Failed to get the shift group index");
    assert(gi_metric >= 0 && "Failed to get the metric group index");
    assert(gi_curv >= 0 && "Failed to get the curvature group index");

    for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
      auto &pc = g_nupcs.at(patch);
      auto &pd = CarpetX::ghext->patchdata.at(patch);
      for (int lev = 0; lev < pd.leveldata.size(); ++lev) {
        const auto &ld = pd.leveldata.at(lev);
        const auto &gd_lapse = *ld.groupdata.at(gi_lapse);
        const auto &gd_shift = *ld.groupdata.at(gi_shift);
        const auto &gd_metric = *ld.groupdata.at(gi_metric);
        const auto &gd_curv = *ld.groupdata.at(gi_curv);
        const amrex::MultiFab &lapse = *gd_lapse.mfab[tl];
        const amrex::MultiFab &shift = *gd_shift.mfab[tl];
        const amrex::MultiFab &metric = *gd_metric.mfab[tl];
        const amrex::MultiFab &curv = *gd_curv.mfab[tl];

        pc->evolve(lapse, shift, metric, curv, CCTK_DELTA_TIME, lev);
      }
    }
    amrex::ParallelDescriptor::Barrier();
    amrex::Gpu::streamSynchronize();
    auto t1 = amrex::ParallelDescriptor::second();
    amrex::Print() << "\n\t[PROFILING] evolve method    \t" << t1 - t0
                   << " seconds \n";
  }

  {
    auto t0 = amrex::ParallelDescriptor::second();
    for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
      auto &pc = g_nupcs.at(patch);
      pc->Redistribute();
    }
    amrex::ParallelDescriptor::Barrier();
    amrex::Gpu::streamSynchronize();
    auto t1 = amrex::ParallelDescriptor::second();
    amrex::Print() << "\t[PROFILING] Redistribute method\t" << t1 - t0
                   << " seconds\n\n";
  }
}

extern "C" void print(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  CCTK_INFO("Printing particles to files");

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
    auto &pc = g_nupcs.at(patch);
    pc->outputParticlesPlot(CCTK_PASS_CTOC, particle_plot_every,
                            std::string(out_dir));
    pc->outputParticlesAscii(CCTK_PASS_CTOC, particle_tsv_every,
                             std::string(out_dir));
  }
}

extern "C" void check_velocities(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  const int it = cctkGH->cctk_iteration;
  if (print_photons_constants > 0 && it % print_photons_constants == 0) {
    const int tl = 0;
    const int gi_metric = CCTK_GroupIndex("ADMBaseX::metric");
    const int gi_lapse = CCTK_GroupIndex("ADMBaseX::lapse");
    const int gi_shift = CCTK_GroupIndex("ADMBaseX::shift");
    assert(gi_lapse >= 0 && "Failed to get the lapse group index");
    assert(gi_metric >= 0 && "Failed to get the metric group index");
    assert(gi_shift >= 0 && "Failed to get the shift group index");

    for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
      auto &pc = g_nupcs.at(patch);
      auto &pd = CarpetX::ghext->patchdata.at(patch);
      for (int lev = 0; lev < pd.leveldata.size(); ++lev) {
        const auto &ld = pd.leveldata.at(lev);
        const auto &gd_metric = *ld.groupdata.at(gi_metric);
        const auto &gd_lapse = *ld.groupdata.at(gi_lapse);
        const auto &gd_shift = *ld.groupdata.at(gi_shift);
        const amrex::MultiFab &lapse = *gd_lapse.mfab[tl];
        const amrex::MultiFab &metric = *gd_metric.mfab[tl];
        const amrex::MultiFab &shift = *gd_shift.mfab[tl];
        pc->check_constants(CCTK_PASS_CTOC, metric, lapse, shift, lev);
      }
    }
  }
}

extern "C" int PhotonsContainer_final_cleanup() {
  amrex::Gpu::Device::synchronize();
  g_nupcs.clear();
  return 0;
}
