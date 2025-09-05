/**
 * @file Particles.hxx
 * 
 * This file contains some predefined structs that defines the parameters
 * needed for a particle in order to be used on a <Particles>Container instance.
 */

#ifndef PHOTONS_HXX
#define PHOTONS_HXX

namespace Photons {
// Struct to managing null particles
struct PhotonsData {
  static constexpr const char *name = "Photons";
  enum {
    // Photon Velocity
    vx = 0, vy, vz,
    // Photon Energy
    E,
    // Number of attributes on the struct
    n_attributes,
  }; // enum
}; // struct PhotonsData

} // namespace Particles
#endif // !PHOTONS_HXX
