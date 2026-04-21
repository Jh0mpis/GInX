/**
 * \file RaytracingInitializers.hxx
 * \brief RaytracingContainer initialization functions.
 *
 * This file contains initial conditions for photons, imposing the initial conditions on the velocity
 * based on camera parameters.
 */
#include "RaytracingX.h"

#include <cctk.h>

#include <AMReX_MFIter.H>
#include "AMReX_GpuDevice.H"
#include "AMReX_ParallelDescriptor.H"
#include "BaseParticleContainer.hxx"

#define DEG2RAD 0.01745329251

#ifndef RAYTRACING_INITIALIZERS
#define RAYTRACING_INITIALIZERS

/**
 * \brief This function random initializes particles according to camera parameters (see RaytracingX/README.md).
 *
 * @param pc The particle container that is going to be initialized.
 *
 * @param real_params Double type array that contains the real parameters needed
 * to initialize the particles. Assumed to be defined by setup_camera_initializer_reals().
 * @param int_params Integer type array that contains the integer parameters
 * needed to initialize the particles. Assumed to be defined by setup_camera_initializer_ints().
 */
template <typename StructType, typename ParticleContainerClass>
void camera_initializer(ParticleContainerClass &pc, const CCTK_REAL *real_params, const CCTK_INT *int_params)
{
  CCTK_INFO("Initializing particles using the RaytracingX::camera_initializer");

  /**
   * These variable definitions are only to make it more clear what each part of real_params
   * actually is.
   * 
   * e0, e1, e2, and e3 are the orthonormal basis vectors defined by the camera parameters. 
   * e0 is parallel to the camera's worldline, usually (1, 0, 0, 0).
   * e1 is parallel to the camera's facing direction.
   * e2 is parallel to the direction that defines 'up' in the camera's reference frame.
   * e3 is parallel to the 'right' direction in the camera's reference frame.
   * 
   * alpha_h is the horizontal field of view of the camera in radians.
   * alpha_v is the verical field of view of the camera in radians.
   * lapse is the lapse from the 3+1 spacetime decomposition interpolated at the camera's position.
   * num_pixels_width and num_pixels_height are the number of pixels of the image horizontally and vertically respectfully.
   * camera_pos are the x, y, and z coordinates of the camera.
   */
  CCTK_REAL e0[4], e1[4], e2[4], e3[4];
  e0[0] = real_params[10];
  e0[1] = real_params[11];
  e0[2] = real_params[12];
  e0[3] = real_params[13];
  e1[0] = real_params[14];
  e1[1] = real_params[15];
  e1[2] = real_params[16];
  e1[3] = real_params[17];
  e2[0] = real_params[18];
  e2[1] = real_params[19];
  e2[2] = real_params[20];
  e2[3] = real_params[21];
  e3[0] = real_params[22];
  e3[1] = real_params[23];
  e3[2] = real_params[24];
  e3[3] = real_params[25];
  CCTK_REAL alpha_h = real_params[26];
  CCTK_REAL alpha_v = real_params[27];
  CCTK_REAL lapse = real_params[28];
  CCTK_INT num_pixels_width = int_params[0];
  CCTK_INT num_pixels_height = int_params[1];
  CCTK_REAL camera_pos[3];
  camera_pos[0] = real_params[29];
  camera_pos[1] = real_params[30];
  camera_pos[2] = real_params[31];

  const CCTK_INT level = 0;
  const CCTK_INT num_pixels = num_pixels_width * num_pixels_height;
  
  //Assert that the number of pixels is valid. This is due to a hack storing the pixel id as a real, and 16,777,217 is the first integer to not be exact in IEEE 32-bit floats. Reasoning for hack explained below.
  if (num_pixels > 16777216) {
    CCTK_ERROR("Number of pixels exceeds the maximum of 16,777,216.");
  }

  const int n_procs = amrex::ParallelDescriptor::NProcs();
  const int proc_id = amrex::ParallelDescriptor::MyProc();

  //Get number of particles that need to be created by this processor.
  const int local_particles_size = num_pixels / n_procs + (proc_id < num_pixels % n_procs);
  //Since each pixel is different and the id of each pixel is important, this calculates the offset in indices for this processor.
  const int local_offset = proc_id * (num_pixels / n_procs) + std::min(proc_id, num_pixels % n_procs);

  //Find the total number of tiles for this processsor.
  int total_tiles = 0;
  for (amrex::MFIter mfi = pc.MakeMFIter(level); mfi.isValid(); ++mfi)
  {
    total_tiles++;
  }

  int current_tile = 0;
  int total_particles_local = 0; //Number of particles already initialized on this processor.

  // Iterating over all the tiles for this processor
  for (amrex::MFIter mfi = pc.MakeMFIter(level); mfi.isValid(); ++mfi)
  { 
    //Get the corresponding number of particles for this tile.
    const unsigned int particles_per_tile = local_particles_size / total_tiles + (current_tile < local_particles_size % total_tiles);

    auto &particles = pc.GetParticles(level);
    auto &particle_tile = pc.DefineAndReturnParticleTile(level, mfi);
    assert(particle_tile.numParticles() == 0);
    auto old_size = particle_tile.numParticles();
    auto new_size = old_size + particles_per_tile;
    particle_tile.resize(new_size); //Increase memory allocated to particles.
    auto arrdata = particle_tile.GetStructOfArrays().realarray();
    auto ptd = particle_tile.getParticleTileData();

#pragma omp parallel for
    for (int local_particle_id = 0; local_particle_id < particles_per_tile; ++local_particle_id)
    { 
      //Create 4-vector \chi parallel to geodesic and fill geodesic initial conditions for each pixel (see https://arxiv.org/pdf/1410.777).
      
      //Get the index for the current pixel, defined by the offset for this processer plus the number of particles created on this tile already plus the number of particles created on previous tiles.
      int pidx = local_offset + local_particle_id + total_particles_local;

      /**
       * Convert pixel index into 2D index with:
       * i=0,                  j=0                   -> upper-left corner
       * i=num_pixels_width-1, j=0                   -> upper-right corner
       * i=0,                  j=num_pixels_height-1 -> lower-left corner
       * i=num_pixels_width-1, j=num_pixels_height-1 -> lower-right corner
       * 
       * ex: for a 4x3 pixel image:
       * 0 1 2  3       [0, 0] [1, 0], [2, 0], [3, 0]
       * 4 5 6  7   <-> [0, 1] [1, 1], [2, 1], [3, 1]
       * 8 9 10 11      [0, 2] [1, 2], [2, 2], [3, 2]
       */
      int i = pidx % num_pixels_width;
      int j = pidx / num_pixels_width;

      //Calculate offset per pixel. The offset can be thought of as $/Delta\theta$ and $\Delta\phi$ with respect to camera facing direction,
      //but is calculated by finding a vector in the equivalent direction. This also gives the direction corresponding to the center of the pixels.
      CCTK_REAL a_adj = ((2.0 * i) / num_pixels_width - 1.0) * tan(alpha_h / 2.0);  // a_{adj} = (2a-1)tan(\alpha_h/2)
      CCTK_REAL b_adj = ((2.0 * j) / num_pixels_height - 1.0) * tan(alpha_v / 2.0); // b_{adj} = (2b-1)tan(\alpha_v/2)

      CCTK_REAL C = sqrt(1 + pow(b_adj, 2) + pow(a_adj, 2));

      CCTK_REAL chi[4];
      chi[0] = C * e0[0] + e1[0] - b_adj * e2[0] + a_adj * e3[0];
      chi[1] = C * e0[1] + e1[1] - b_adj * e2[1] + a_adj * e3[1];
      chi[2] = C * e0[2] + e1[2] - b_adj * e2[2] + a_adj * e3[2];
      chi[3] = C * e0[3] + e1[3] - b_adj * e2[3] + a_adj * e3[3];

      CCTK_REAL chi_lower[4];
      vectorToOneFormArr(chi_lower, chi, real_params);

      ptd.id(local_particle_id) = ParticleContainerClass::ParticleType::NextID();
      ptd.cpu(local_particle_id) = amrex::ParallelDescriptor::MyProc();

      ptd.pos(0, local_particle_id) = camera_pos[0];
      ptd.pos(1, local_particle_id) = camera_pos[1];
      ptd.pos(2, local_particle_id) = camera_pos[2];
      CCTK_REAL A = 1 / lapse * chi[0];

      //The direction of the geodesic needs to be reversed, because the geodesics are evolved backwards in time, but the evolution routine doesn't "know" that.
      arrdata[StructType::vx][local_particle_id] = -chi_lower[1] * A; 
      arrdata[StructType::vy][local_particle_id] = -chi_lower[2] * A;
      arrdata[StructType::vz][local_particle_id] = -chi_lower[3] * A;
      arrdata[StructType::ln_E][local_particle_id] = 0;
      arrdata[StructType::tau][local_particle_id] = 0;

      //The pixel indices *should* be ints, but the data structure GInX::PhotonsData does not easily extend with integer parameters, so it is stored as a real instead. While this is hacky, it is done for clarity and ease of output. It is possible to add an integer variable at runtime, but this wouldn't be written to disk in the standard amrex particle output routine.
      arrdata[StructType::pixel_number][local_particle_id] = (CCTK_REAL) pidx;
    }

    total_particles_local += particles_per_tile;
    current_tile++;
  }
  pc.Redistribute();
  pc.SortParticlesByCell();
  CCTK_VINFO("%d particles created", pc.TotalNumberOfParticles());
}

/**
 * \brief This function sets up the necessary data for the camera initializer.
 * 
 * @param real_params An empty array to store the data.
 */
void setup_camera_initializer_reals(CCTK_ARGUMENTS, CCTK_REAL* real_params) {
    DECLARE_CCTK_ARGUMENTS
    DECLARE_CCTK_PARAMETERS

    Metric metric;
    interpolateMetricAtPoint(CCTK_PASS_CTOC, &metric); //interpolate metric values and store in Metric struct

    CCTK_REAL e0[4];
    CCTK_REAL e1[4];
    CCTK_REAL e2[4];
    CCTK_REAL e3[4];
    gramSchmidtProcess(CCTK_PASS_CTOC, e0, e1, e2, e3, &metric); //create orthonormal basis for camera POV

    CCTK_REAL alpha_h = DEG2RAD * horizontal_fov; //convert FOV to radians
    CCTK_REAL alpha_v = DEG2RAD * vertical_fov;

    real_params[0] = metric.g_tt;
    real_params[1] = metric.beta_x;
    real_params[2] = metric.beta_y;
    real_params[3] = metric.beta_z;
    real_params[4] = metric.g_xx;
    real_params[5] = metric.g_xy;
    real_params[6] = metric.g_xz;
    real_params[7] = metric.g_yy;
    real_params[8] = metric.g_yz;
    real_params[9] = metric.g_zz;
    real_params[10] = e0[0];
    real_params[11] = e0[1];
    real_params[12] = e0[2];
    real_params[13] = e0[3];
    real_params[14] = e1[0];
    real_params[15] = e1[1];
    real_params[16] = e1[2];
    real_params[17] = e1[3];
    real_params[18] = e2[0];
    real_params[19] = e2[1];
    real_params[20] = e2[2];
    real_params[21] = e2[3];
    real_params[22] = e3[0];
    real_params[23] = e3[1];
    real_params[24] = e3[2];
    real_params[25] = e3[3];
    real_params[26] = alpha_h;
    real_params[27] = alpha_v;
    real_params[28] = metric.alpha;
    real_params[29] = camera_pos[0];
    real_params[30] = camera_pos[1];
    real_params[31] = camera_pos[2];
}

/**
 * \brief This function sets up the necessary data for the camera initializer.
 * 
 * @param int_params An empty array to store the data.
 */
void setup_camera_initializer_ints(CCTK_ARGUMENTS, CCTK_INT* int_params) {
    DECLARE_CCTK_ARGUMENTS
    DECLARE_CCTK_PARAMETERS

    int_params[0] = num_pixels_width;
    int_params[1] = num_pixels_height;
}

#endif // !PHOTON_INITIALIZERS