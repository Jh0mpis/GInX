/**
 * @file Particles.hxx
 *
 * \brief Particles Data structs examples.
 *
 * This file contains some predefined structs that defines the parameters
 * needed for a particle in order to be used on a <Particles>Container instance.
 */

#ifndef PARTICLES_HXX
#define PARTICLES_HXX

namespace Particles {
/** \brief Struct to managing null particles
 *
 * The following struct contains the example of a base particle that only
 * contains the velocities.
 */
struct BaseParticleData {
  /** Name of the particle type. */
  static constexpr const char *name = "Basic Particles";
  /** \brief Enum type that contains the information of the particles.
   *
   *  The BaseParticleData struct contains the lower index velocity on each
   * directions of the domain and the number of
   * attributes of the particles.
   */
  enum {
    // Particle Velocity
    vx = 0, /**< Velocity's lower index on the x direction.*/
    vy,     /**< Velocity's lower index on the y direction.*/
    vz,     /**< Velocity's lower index on the z direction.*/
    // Number of attributes on the struct
    n_attributes, /**< Total number of attributes*/
  }; // enum
}; // struct PhotonsData

/**
 * \brief Struct to managing null geodesic particles.
 */
struct PhotonsData {
  /** Name of the particle type. */
  static constexpr const char *name = "Photons";
  /** \brief Enum type that contains the information of the particles.
   *
   *  The PhotonsData struct contains the lower index velocity on each
   * directions of the domain, the energy of the photons and the number of
   * attributes of the particles.
   */
  enum {
    vx = 0,       /**< Velocity's lower index on the x direction.*/
    vy,           /**< Velocity's lower index on the y direction.*/
    vz,           /**< Velocity's lower index on the z direction.*/
    E,            /**< Energy value.*/
    n_attributes, /**< Total number of attributes*/
  }; // enum
}; // struct PhotonsData

// Struct to managing massive particles
/**
 * \brief Struct to managing time-like geodesic particles.
 */
struct MassiveParticleData {
  /** Name of the particle type. */
  static constexpr const char *name = "Massive Particles";
  /** \brief Enum type that contains the information of the particles.
   *
   *  The MassiveParticleData struct contains the lower index velocity on each
   * directions of the domain, the energy, the mass of the massive particles and
   * the number of attributes of the particles.
   */
  enum {
    // Massive particle velocity
    vx = 0, /**< Velocity's lower index on the x direction.*/
    vy,     /**< Velocity's lower index on the y direction.*/
    vz,     /**< Velocity's lower index on the z direction.*/
    // Massive particle Energy
    E, /**< Energy value.*/
    // Massive particle mass
    m, /**< Particle's mass value.*/
    // Number of attributes on the struct
    n_attributes, /**< Total number of attributes*/
  }; // enum
}; // struct MassiveParticleData

} // namespace Particles
#endif // !PARTICLES_HXX
