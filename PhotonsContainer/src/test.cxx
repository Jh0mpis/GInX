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

using ParticleData = Photons::PhotonsData;
using PC = Containers::PhotonsContainer<ParticleData>;
std::vector<std::unique_ptr<PC>> photons;

extern "C" void PhotonsContainer_setup(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  const int tl = 0;
  const int gi_metric = CCTK_GroupIndex("ADMBaseX::metric");
  assert(gi_metric >= 0 && "Failed to get the metric group index");

  const CCTK_INT int_parameters[] = {
      num_photons * (photons.size() < CarpetX::ghext->num_patches())};

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
    const auto &patchdata = CarpetX::ghext->patchdata.at(patch);
    if (photons.size() < CarpetX::ghext->num_patches()) {
      photons.push_back(std::make_unique<PC>(patchdata.amrcore.get()));
    }

    auto &pc = photons.at(patch);
    pc->initialize(photons_init::random_uniform_initializer<ParticleData, PC>,
                   init_params_d, int_parameters);
  }

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
    auto &pc = photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);
    for (int lev = 0; lev < pd.leveldata.size(); ++lev) {

      const auto &ld = pd.leveldata.at(lev);
      const auto &gd_metric = *ld.groupdata.at(gi_metric);
      const amrex::MultiFab &metric = *gd_metric.mfab[tl];

      pc->normalize_velocity(metric, lev);
    }
  }
}

extern "C" void PhotonsContainer_evolve(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  const CCTK_REAL dt = CCTK_DELTA_TIME;

  CCTK_VWarn(1, __LINE__, __FILE__, CCTK_THORNSTRING, "Patches: %d size: %d",
             CarpetX::ghext->num_patches(), photons.size());
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
    auto &pc = photons.at(patch);
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

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
    auto &pc = photons.at(patch);
    pc->Redistribute();
  }
}

extern "C" void PhotonsContainer_print(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  CCTK_INFO("Printing particles to files");

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
    auto &pc = photons.at(patch);
    pc->outputParticlesPlot(CCTK_PASS_CTOC, particle_plot_every,
                            std::string(out_dir));
    pc->outputParticlesAscii(CCTK_PASS_CTOC, particle_tsv_every,
                             std::string(out_dir));
  }
}

extern "C" int PhotonsContainer_final_cleanup() {
  amrex::Gpu::Device::synchronize();
  photons.clear();
  return 0;
}
