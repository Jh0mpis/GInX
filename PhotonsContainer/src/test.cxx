/**
 * \file test.cxx
 * \brief Available particle container implementations.
 *
 *  This file includes the definitions of the thorn's scheduled functions. This
 * includes the set up, evolution, output particle data and the data clean up.
 */
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

using ParticleData = GInX::PhotonsData;
using PC = GInX::PhotonsContainer<ParticleData>;
std::vector<std::unique_ptr<PC>> photons;

/**
 * \brief Initialize particles' data
 *
 * This function initializes particles' position, velocity and energy
 * distributing the particle using the methods defined inside of the
 * PhotonsInitializers.hxx file and allwed in the param.ccl file
 */
extern "C" void PhotonsContainer_setup(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (!run_test) {
    return;
  }

  const int tl = 0;
  const int gi_metric = CCTK_GroupIndex("ADMBaseX::metric");
  assert(gi_metric >= 0 && "Failed to get the metric group index");

  const CCTK_INT int_parameters[] = {
      num_photons * (photons.size() < CarpetX::ghext->num_patches())};

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
    const auto &patchdata = CarpetX::ghext->patchdata.at(patch);
    if (photons.size() < CarpetX::ghext->num_patches()) {
      photons.push_back(
          std::make_unique<PC>(patchdata.amrcore.get(), particles_mass));

      auto &pc = photons.at(patch);
      if (!std::strcmp(initializer, "box")) {
        pc->initialize(random_uniform_initializer<ParticleData, PC>,
                       init_params_d, int_parameters);
      } else if (!std::strcmp(initializer, "spherical")) {
        pc->initialize(random_spherical_initializer<ParticleData, PC>,
                       init_params_d, int_parameters);
      }
    }
  }

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
    auto &pc = photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);
    for (int lev = 0; lev < pd.leveldata.size(); ++lev) {

      const auto &ld = pd.leveldata.at(lev);
      const auto &gd_metric = *ld.groupdata.at(gi_metric);
      const amrex::MultiFab &metric = *gd_metric.mfab[tl];

      pc->Redistribute();
      pc->normalize_velocity(metric, lev);
    }
  }
}

/**
 * \brief Evolve the geodesics
 *
 * This function evolves the particles position by numerically solve the
 * geodesic equations.
 */
extern "C" void PhotonsContainer_evolve(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (!run_test) {
    return;
  }

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

  // Bounds check
  const CCTK_REAL regions_x[10] = {region_1_position[0], region_2_position[0],
                                   region_3_position[0], region_4_position[0],
                                   region_5_position[0], region_6_position[0],
                                   region_7_position[0], region_8_position[0],
                                   region_9_position[0], region_10_position[0]};
  const CCTK_REAL regions_y[10] = {region_1_position[1], region_2_position[1],
                                   region_3_position[1], region_4_position[1],
                                   region_5_position[1], region_6_position[1],
                                   region_7_position[1], region_8_position[1],
                                   region_9_position[1], region_10_position[1]};
  const CCTK_REAL regions_z[10] = {region_1_position[2], region_2_position[2],
                                   region_3_position[2], region_4_position[2],
                                   region_5_position[2], region_6_position[2],
                                   region_7_position[2], region_8_position[2],
                                   region_9_position[2], region_10_position[2]};
  const CCTK_REAL regions_radius[10] = {
      region_1_radius, region_2_radius, region_3_radius, region_4_radius,
      region_5_radius, region_6_radius, region_7_radius, region_8_radius,
      region_9_radius, region_10_radius};

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
    auto &pc = photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);
    for (int lev = 0; (lev < pd.leveldata.size()) & banned_regions; ++lev) {
      pc->check_banned_zones(lev, banned_regions, regions_x, regions_y,
                             regions_z, regions_radius);
    }
    pc->Redistribute();
  }
}

/**
 * \brief Print out particle data.
 */
extern "C" void PhotonsContainer_print(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  if (!run_test) {
    return;
  }

  CCTK_INFO("Printing particles to files");
  const int it = cctkGH->cctk_iteration;

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
    auto &pc = photons.at(patch);
    pc->outputParticlesPlot(it, particle_plot_every, std::string(out_dir));
    pc->outputParticlesAscii(it, particle_tsv_every, std::string(out_dir));
  }
}

/**
 * \brief Clean the memory and data structures.
 */
extern "C" int PhotonsContainer_final_cleanup() {
  amrex::Gpu::Device::synchronize();
  photons.clear();
  return 0;
}
