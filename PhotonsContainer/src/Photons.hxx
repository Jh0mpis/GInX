/**
 * @file Particles.hxx
 *
 * This file contains some predefined structs that defines the parameters
 * needed for a particle in order to be used on a <Particles>Container instance.
 */

#ifndef PHOTONS_HXX
#define PHOTONS_HXX

namespace Photons {
// Struct to managing null geodesic particles
struct PhotonsData {
  /** Name of the particle type. */
  static constexpr const char *name = "Photons";
  /** Enum type that contains the information of the particles.
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

} // namespace Photons
#endif // !PHOTONS_HXX
