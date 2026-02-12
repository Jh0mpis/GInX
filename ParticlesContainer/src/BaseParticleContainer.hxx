/**
 * \file BaseParticleContainer.hxx
 *
 * \brief Contains the BaseParticlesContainer and ParticleIterator classes
 * along with the BaseContainer and Iterator namespaces definition.
 *
 * This file contains the BaseParticlesContainer abstract class. This
 * class extends from amrex::AmrParticleContainer and contains the particles
 * data using an array of structs (AoS).
 *
 * Also, includes the definition of the ParticleIterator class, that could be
 * helpful for the manipulation of the particles.
 */
#ifndef BASEPARTICLESCONTAINER_HXX
#define BASEPARTICLESCONTAINER_HXX

// Include libraries
#include "AMReX_CTOParallelForImpl.H"
#include <cctk.h>

#include <AMReX_AmrParticles.H>
#include <AMReX_Particles.H>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <cctk_core.h>
#include <string>

namespace Iterator {

template <typename StructType>
class ParticleIterator
    : public amrex::ParIter<0, 0, StructType::n_attributes, 0> {
public:
  using amrex::ParIter<0, 0, StructType::n_attributes, 0>::ParIter;
  using RealVector = typename amrex::ParIter<
      0, 0, StructType::n_attributes>::ContainerType::RealVector;

  const std::array<RealVector, StructType::n_attributes> &GetAttribs() const {
    return this->GetStructOfArrays().GetRealData();
  }

  std::array<RealVector, StructType::n_attributes> &GetAttributes() {
    return this->GetStructOfArrays().GetRealData();
  }

  const RealVector &GetAttribs(int comp) const {
    return this->GetStructOfArrays().GetRealData(comp);
  }

  RealVector &GetAttributes(int comp) {
    return this->GetStructOfArrays().GetRealData(comp);
  }
}; // class ParicleIterator

} // namespace Iterator

namespace BaseContainer {

/**
 * \brief BaseParticleContainer abstract class definition.
 *
 * The BaseParticleContainer class is an abstract class that defines the methods
 * that have to be defined for the other <Particle>Containers derived classes.
 * This is templated on the new Container and the struct that defines the
 * parameters of the particles.
 */
template <typename OtherContainer, typename StructType>
class BaseParticleContainer
    : public amrex::AmrParticleContainer<0, 0, StructType::n_attributes, 0> {
public:
  // Name of the particles
  const std::string name = StructType::name;
  // Number of attributes per each particle
  static constexpr int n_attributes = StructType::n_attributes;

  /**
   * The BaseParticleContainer constructor initialize the AmrCore superclass
   * receiving an external AmrCore instance.
   * @param amr_core A pointer to an amrex::AmrCore instance.
   */
  BaseParticleContainer(amrex::AmrCore *amr_core)
      : amrex::AmrParticleContainer<0, 0, StructType::n_attributes, 0>(
            amr_core) {}

  virtual ~BaseParticleContainer() = default;

  /**
   * \brief Initialize the particles given a custom initialization function.
   *
   * The initialize method receives a function and use it to assign the initial
   * conditions over the particles. This function can be implemented by other
   * users or use the ones defined in the file Initializers.hxx.
   *
   * @param initializer_function Function that receives a BaseParticleContainer
   * instance, one array full of doubles and another full of integers as
   * parameters, how to use this parameters have to be defined inside of the
   * custom function.
   * @param real_params Double type array that contains the real parameters
   * needed to initialize the particles.
   * @param int_params Integer type array that contains the integer parameters
   * needed to initialize the particles.
   */
  template <typename Function>
  void initialize(Function initializer_function, const CCTK_REAL *real_params,
                  const CCTK_INT *int_params) {
    initializer_function(static_cast<OtherContainer &>(*this), real_params,
                         int_params);
  };

  /**
   * \brief Initialize the particles given a custom initialization function.
   *
   * The initialize method receives a function and use it to assign the initial
   * conditions over the particles. This function can be implemented by other
   * users or use the ones defined in the file Initializers.hxx.
   *
   * @param initializer_function Function that receives a BaseParticleContainer
   * instance, one array full of doubles and another full of integers as
   * parameters, how to use this parameters have to be defined inside of the
   * custom function.
   * @param metric 3D ADM Metric.
   * @param level Current refinement level.
   * @param real_params Double type array that contains the real parameters
   * needed to initialize the particles.
   * @param int_params Integer type array that contains the integer parameters
   * needed to initialize the particles.
   */
  template <typename Function>
  void initialize(Function initializer_function, const amrex::MultiFab &metric,
                  const int &level, const CCTK_REAL *real_params,
                  const CCTK_INT *int_params) {
    initializer_function(static_cast<OtherContainer &>(*this), metric, level,
                         real_params, int_params);
  };

  /**
   * The evolve abstract method evolve the system given the differential
   * equations and the computed rhs. Has to be override for each different type
   * particle.
   *
   * @param lapse ADM lapse function
   * @param shift ADM shift vector
   * @param metric ADM induced metric
   * @param curv ADM extrinsic curvature
   * @param dt Time step
   * @param lev AMR level
   */
  virtual void evolve(const amrex::MultiFab &lapse,
                      const amrex::MultiFab &shift,
                      const amrex::MultiFab &metric,
                      const amrex::MultiFab &curv, const CCTK_REAL &dt,
                      const int &lev) = 0;

  /**
   * The check banned zones function check for user defined invalid particles
   * zones.
   *
   * @param level Adaptive Mesh Refinement level
   * @param zones Number of banned zones
   * @param x x-coordinates array for each region
   * @param y y-coordinates array for each region
   * @param z z-coordinates array for each region
   * @param radius Radius array for each region
   */
  void check_banned_zones(const int &level, const CCTK_INT4 &zones,
                          const CCTK_REAL (&x)[10], const CCTK_REAL (&y)[10],
                          const CCTK_REAL (&z)[10],
                          const CCTK_REAL (&radius)[10]) {

    if (!zones) {
      return;
    }

    for (Iterator::ParticleIterator<StructType> pti(*this, level);
         pti.isValid(); ++pti) {
      const int np = pti.numParticles();
      auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

      auto self = this;
      amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
        bool out = false;
        for (int check = 0; check < zones; check++) {
          const CCTK_REAL dx = particles[i].pos(0) - x[check];
          const CCTK_REAL dy = particles[i].pos(1) - y[check];
          const CCTK_REAL dz = particles[i].pos(2) - z[check];
          out |= (dx * dx + dy * dy + dz * dz <= radius[check] * radius[check]);
        }

        if (out) {
          particles[i].id() = -1;
          return;
        }
      });
    }
  }

  void outputParticlesAscii(const int &it, const int &plot_every,
                            const std::string &out_dir) {
    if (plot_every > 0 && it % plot_every == 0) {
      const std::string &file_name =
          out_dir + "/" + amrex::Concatenate(this->name, it);
      CCTK_VINFO(" Writing ascii file %s", file_name.c_str());

      this->WriteAsciiFile(file_name);
    }
  };

  void outputParticlesPlot(const int &it, const int &plot_every,
                           const std::string &out_dir) {
    if (plot_every > 0 && it % plot_every == 0) {
      const std::string file_name =
          out_dir + "/" + amrex::Concatenate("plt", it);
      CCTK_VINFO("Writing plot file %s", file_name.c_str());

      this->WritePlotFile(file_name, this->name);
    }
  };

}; // class BaseParticlesContainer

} // namespace BaseContainer

#endif // !BASEPARTICLESCONTAINER_HXX
