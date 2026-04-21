#ifndef RAYTRACINGCONTAINER
#define RAYTRACINGCONTAINER

#include <cctk.h>

#include "Photons.hxx"
#include "BaseParticlesContainer.hxx"
#include "ParticlesContainer.hxx"
#include "RaytracingX.h"
#include <AMReX_ParallelDescriptor.H>
#include <CParameters.h>
#include <fstream>
#include "Interpolator.hxx"
#include <AMReX_Particles.H>
#include <AMReX_AmrParticles.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MFIter.H>

//Macros that check if the position is out of bounds, and each one sets the optical depth to a specific value for debugging purposes.
#define CHECK_OUT_OF_BOUNDS_X(X) \
    if (X > boundarie_hx)        \
    {                            \
        out_of_bounds = true;    \
        deletion_reason = -1;    \
    }                            \
    if (X < boundarie_lx)        \
    {                            \
        out_of_bounds = true;    \
        deletion_reason = -2;    \
    }
#define CHECK_OUT_OF_BOUNDS_Y(Y) \
    if (Y > boundarie_hy)        \
    {                            \
        out_of_bounds = true;    \
        deletion_reason = -3;    \
    }                            \
    if (Y < boundarie_ly)        \
    {                            \
        out_of_bounds = true;    \
        deletion_reason = -4;    \
    }
#define CHECK_OUT_OF_BOUNDS_Z(Z) \
    if (Z > boundarie_hz)        \
    {                            \
        out_of_bounds = true;    \
        deletion_reason = -5;    \
    }                            \
    if (Z < boundarie_lz)        \
    {                            \
        out_of_bounds = true;    \
        deletion_reason = -6;    \
    }

namespace RaytracingX
{

    struct RaytracingPhotonsData : public GInX::PhotonsData
    {
        enum
        {
            vx = 0,       /**< Velocity's lower index on the x direction.*/
            vy,           /**< Velocity's lower index on the y direction.*/
            vz,           /**< Velocity's lower index on the z direction.*/
            ln_alphaE,         /**< Ln Energy value.*/
            tau,          /**< Optical depth value. Also used as particle deletion code, see above macros, also iterates across banned regions*/
            pixel_number, /**< Number used to match particle to corresponding pixel in the image. Defined as a real since BaseParticleContainer does not have options for int parameters, unless defined at runtime, in which case they will not print with WriteAsciiFile*/
            n_attributes, /**< Total number of attributes*/
        }; // enum
    };

    template <typename StructType>
    class RaytracingParticlesContainer : public GInX::BaseParticleContainer<RaytracingParticlesContainer<StructType>,
                                                                                   StructType>
    {

    /**
     * \brief RaytracingParticlesContainer class definition.
     *
     * The following class defines the needed functions to evolve the position and
     * velocity of the photons in the simulation for raytracing purposes. Many of the methods
     * are similar to those found in ParticlesContainer/ParticlesContainer.hxx, so changes will be prefaced with 
     * 'RaytraingX:'.
     */
    protected:
        CCTK_REAL mass = 0.;

    public:
        /**
         * \brief Using BaseParticlesContainer constructor
         */

        using Base =
            GInX::BaseParticleContainer<RaytracingParticlesContainer<StructType>,StructType>;
        using Base::Base;

        RaytracingParticlesContainer(amrex::AmrCore *amr_core, const CCTK_REAL m)
            : Base(amr_core), mass{m} { };

        ~RaytracingParticlesContainer() = default;

        //RaytracingX: Add method that writes particle information when the particle is deleted.
        void write_deleted_particle_data(const CCTK_REAL particle_id, const CCTK_REAL x, const CCTK_REAL y, const CCTK_REAL z, const CCTK_REAL vx, const CCTK_REAL vy, const CCTK_REAL vz, const CCTK_REAL ln_alphaE, const CCTK_REAL tau, int deletion_reason, bool output_final_data, std::string final_data_file_name) {
            if (!output_final_data) {return; }
            
            amrex::AllPrintToFile(final_data_file_name) << (int) particle_id << "\t" << x << "\t" << y << "\t" << z << "\t" << vx << "\t" << vy << "\t" << vz << "\t" << ln_alphaE << "\t" << tau << "\t" << deletion_reason << std::endl;
        }

        /**
         * \brief Computes the right hand side of the geodesic differential equation.
         *
         * Given differential equation \f[\frac{d}{dt}U = f\left(U, \frac{dU}{dx};
         * t\right)\f] computes
         * \f[f\left(U, \frac{dU}{dx}; t\right)\f]
         *
         * where \f$U\f$ is a vector which contains \f$(x, y, z, v_x, v_y, v_z,
         * \ln E)\f$. The differential equation for the particles' position is
         *
         * \f[\frac{d}{dt} U[i] = \alpha \gamma^{ij} U[3 + j] - \beta^i\f]
         *
         * Where \f$i,j = 0, 1, 2\f$, \f$\gamma\f$ is the induced metric, \f$\alpha\f$
         * is the lapse function and \f$\beta\f$ is the shift vector.
         *
         * For the Velocity_d the differential equation is:
         *
         * \f{eqnarray*}{
         * \frac{d}{dt}U[3 + i] &= -\partial_i\alpha + \left(\gamma^{kj} U[3 + k]
         * \partial_j\alpha - \alpha K_{jk}\gamma^{jl}\gamma^{km}U[3+l]U[3+m]\right) U[3
         * + i]\\ & +
         * \frac{1}{2}\alpha\gamma^{jl}\gamma^{km}U[3+l]U[3+m]\partial_i\gamma{jk} + U[3
         * + j] \partial_i\beta^j
         * \f}
         *
         * and finally, for the \f$ \ln \alphaE \f$ the differential equation is:
         *
         * \f[ \dfrac{d}{dt} U[6] = \alpha K_{jk}U[3 + l]U[3 + m]\gamma^{lj}\gamma^{mk}
         * - U[3+l]\gamma^{lj}\partial_j\alpha\f]
         *
         *  Where \f$i, j, k, l, m = 0, 1, 2\f$ and \f$K_{ij}\f$ is the extrinsic
         * curvature. We have been using Einstein notation.
         * 
         *  RaytracingX: optical depth iteration as \f \frac{d\tau}{ds}=\kappa\rho \f so \frac{d}{dt}\tau = \kappa\rho\frac{ds}{dt}
         *
         *  @param u A GpuArray of size n_attributes + the coordinates that contains the
         * varaibles needed to evolve.
         *  @param t Current time t.
         *  @param lapse ADM lapse function.
         *  @param shift Shift vector \beta^i
         *  @param metric 3 dimensional ADM metric.
         *  @param curv Extrinsic curvature.
         *  @param rho RaytracingX: Gas density.
         *  @param dt Timestep.
         *  @param dx Spacestep
         *  @param lev AMR Level of discretization.
         *  @param plo Physical lower bounds of the whole domain.
         *  @return The right hind side of the differential equation.
         */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
            amrex::GpuArray<CCTK_REAL, 8>
            compute_rhs(
                const amrex::GpuArray<CCTK_REAL, 8> &u, const CCTK_REAL &t,
                amrex::Array4<CCTK_REAL const> const &lapse,
                const amrex::Array4<CCTK_REAL const> &shift,
                const amrex::Array4<CCTK_REAL const> &metric,
                const amrex::Array4<CCTK_REAL const> &curv,
                const amrex::Array4<CCTK_REAL const> &rho, const CCTK_REAL dt,
                const amrex::GpuArray<double, 3> &dx, const int lev,
                const amrex::GpuArray<double, 3> &plo)
        {

            //RaytracingX: Add space for optical depth variable.
            amrex::GpuArray<CCTK_REAL, 8> rhs = {0., 0., 0., 0., 0., 0., 0., 0.};

            const long int i0 = amrex::Math::floor((u[0] - plo[0]) / dx[0]);
            const long int j0 = amrex::Math::floor((u[1] - plo[1]) / dx[1]);
            const long int k0 = amrex::Math::floor((u[2] - plo[2]) / dx[2]);

            // Interpolate lapse & partial lapse at \vect{x}
            CCTK_REAL lapse_x;
            amrex::GpuArray<CCTK_REAL, 3> d_lapse_x;
            GInX::d_interpolate_array<5>(lapse_x, d_lapse_x, lapse, i0, j0, k0, u[0], u[1],
                                         u[2], dx, plo);

            // Interpolate shift & partial shift at \vect{x}
            amrex::GpuArray<CCTK_REAL, 3> shift_x;
            amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 3>, 3> d_shift_x;
            GInX::d_interpolate_array<5>(shift_x, d_shift_x, shift, i0, j0, k0, u[0], u[1],
                                         u[2], dx, plo);

            // Interpolate metric & partial metric at \vect{x}
            amrex::GpuArray<CCTK_REAL, 6> gamma_x;
            amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 6>, 3> d_gamma_x;
            GInX::d_interpolate_array<5>(gamma_x, d_gamma_x, metric, i0, j0, k0, u[0], u[1],
                                         u[2], dx, plo);

            // Interpolate Curvature at \vect{x}
            amrex::GpuArray<CCTK_REAL, 6> curv_x;
            GInX::interpolate_array<5>(curv_x, curv, i0, j0, k0, u[0], u[1], u[2], dx, plo);

            //RaytracingX: Interpolate density for optical depth calculation.
            // Interpolate rho at \vect{x}
            CCTK_REAL rho_x;
            amrex::GpuArray<CCTK_REAL, 3> d_rho_x;
            GInX::d_interpolate_array<5>(rho_x, d_rho_x, rho, i0, j0, k0, u[0], u[1],
                                         u[2], dx, plo);

            // Compute the inverse of the metric.
            const CCTK_REAL inv_det_gamma =
                1.0 / (gamma_x[0] * gamma_x[3] * gamma_x[5] +
                       2. * gamma_x[1] * gamma_x[2] * gamma_x[4] -
                       gamma_x[2] * gamma_x[2] * gamma_x[3] -
                       gamma_x[4] * gamma_x[4] * gamma_x[0] -
                       gamma_x[1] * gamma_x[1] * gamma_x[5]);

            const amrex::GpuArray<CCTK_REAL, 6> gamma_inv_x = {
                (gamma_x[3] * gamma_x[5] - gamma_x[4] * gamma_x[4]) * inv_det_gamma,
                (gamma_x[4] * gamma_x[2] - gamma_x[1] * gamma_x[5]) * inv_det_gamma,
                (gamma_x[1] * gamma_x[4] - gamma_x[2] * gamma_x[3]) * inv_det_gamma,
                (gamma_x[0] * gamma_x[5] - gamma_x[2] * gamma_x[2]) * inv_det_gamma,
                (gamma_x[2] * gamma_x[1] - gamma_x[0] * gamma_x[4]) * inv_det_gamma,
                (gamma_x[0] * gamma_x[3] - gamma_x[1] * gamma_x[1]) * inv_det_gamma};

            const amrex::GpuArray<CCTK_REAL, 3> V_down = {u[3], u[4], u[5]};

            // Compute the upper index velocity terms.
            const amrex::GpuArray<CCTK_REAL, 3> V_up = {
                gamma_inv_x[0] * u[3] + gamma_inv_x[1] * u[4] + gamma_inv_x[2] * u[5],
                gamma_inv_x[1] * u[3] + gamma_inv_x[3] * u[4] + gamma_inv_x[4] * u[5],
                gamma_inv_x[2] * u[3] + gamma_inv_x[4] * u[4] + gamma_inv_x[5] * u[5]};

            // Compute the rhs for position
            rhs[0] = lapse_x * V_up[0] - shift_x[0];
            rhs[1] = lapse_x * V_up[1] - shift_x[1];
            rhs[2] = lapse_x * V_up[2] - shift_x[2];

            // Compute the rhs for velocity
            for (int i = 0; i < 3; i++)
            {
                rhs[3 + i] =
                    -d_lapse_x[i] +
                    (VecVecMul(d_lapse_x, V_up) -
                     lapse_x * VecVecMul(SMatVecMul(curv_x, V_up), V_up)) *
                        V_down[i] +
                    0.5 * lapse_x * VecVecMul(SMatVecMul(d_gamma_x[i], V_up), V_up) +
                    VecVecMul(V_down, d_shift_x[i]);
            }

            // Compute the rhs for energy
            rhs[3 + StructType::ln_alphaE] =
                lapse_x * VecVecMul(SMatVecMul(curv_x, V_up), V_up) -
                VecVecMul(V_up, d_lapse_x);

            //RaytracingX: Add evolution for optical depth calculation.
            // Compute the rhs for optical depth
            const CCTK_REAL ds = dx[0] * dx[0] * gamma_inv_x[0] +
                                 dx[1] * dx[1] * gamma_inv_x[3] +
                                 dx[2] * dx[2] * gamma_inv_x[5] +
                                 2.0 * dx[0] * dx[1] * gamma_inv_x[1] +
                                 2.0 * dx[0] * dx[2] * gamma_inv_x[2] +
                                 2.0 * dx[1] * dx[2] * gamma_inv_x[4];
            rhs[3 + StructType::tau] = (0.4 * cgs2cactusOpacity) * (rho_x * cgs2cactusDensity) * (ds / dt);

            return rhs;

        } // RaytracingParticlesContainer::compute_rhs

        /**
         *  \brief Evolving using Runge-Kutta 4.
         *
         * We are solving the differential equation
         * \f$\frac{dU}{dt} = f\left(U, \frac{dU}{dx}, t\right)\f$ using:
         *
         *  \f[
         *  U_{n+1} = U_n + \frac{1}{6}\Delta t \left(f_1 + 2f_2 + 2f_3 + f_4\right)
         *  \f]
         *
         *  where:
         *
         *  * \f$f_1 = f(U_n, t),\f$
         *  * \f$f_2 = f\left(U_n + \frac{\Delta t}{2} f_1, t + \frac{\Delta
         * t}{2}\right),\f$
         *  * \f$f_3 = f\left(U_n + \frac{\Delta t}{2} f_2, t + \frac{\Delta
         * t}{2}\right),\f$
         *  * \f$f_4 = f(U_n + \Delta t f_3, t + \Delta t),\f$
         *
         *  While computing we are checking if the particles still in the physical domain.
         *
         *  @see compute_rhs()
         *  @param lapse ADM lapse function.
         *  @param shift ADM shift vector.
         *  @param metric ADM induced metric.
         *  @param curv Extrinsic curvature.
         *  @param rho RaytracingX: gas density
         *  @param dt Timestep.
         *  @param lev Refinement level.
         *  @param max_energy RaytracingX: Maximum energy threshold for event horizon detection.
         */
        void evolve(const amrex::MultiFab &lapse,
                    const amrex::MultiFab &shift,
                    const amrex::MultiFab &metric,
                    const amrex::MultiFab &curv,
                    const amrex::MultiFab &rho,
                    const CCTK_REAL &dt, const int &lev,
                    const CCTK_REAL max_energy, bool output_final_data, std::string final_data_file_name) //RaytracingX: Add information for maximum energy for photons defining event horizon and output information.
        {

            const auto plo0 = this->Geom(0).ProbLoArray();
            const auto phi0 = this->Geom(0).ProbHiArray();

            const auto dx = this->Geom(lev).CellSizeArray();
            const auto plo = this->Geom(lev).ProbLoArray();

            const CCTK_REAL boundarie_hx = phi0[0] - 0.0 * dx[0];
            const CCTK_REAL boundarie_lx = plo0[0] + 0.0 * dx[0];
            const CCTK_REAL boundarie_hy = phi0[1] - 0.0 * dx[1];
            const CCTK_REAL boundarie_ly = plo0[1] + 0.0 * dx[1];
            const CCTK_REAL boundarie_hz = phi0[2] - 0.0 * dx[2];
            const CCTK_REAL boundarie_lz = plo0[2] + 0.0 * dx[2];

            for (GInX::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
                 ++pti)
            {

                const int np = pti.numParticles();

                // Get the information relate to the velocities and energy.
                auto &attribs = pti.GetAttributes();
                CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
                CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
                CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
                CCTK_REAL *AMREX_RESTRICT ln_alphaenergy = attribs[StructType::ln_alphaE].data();
                CCTK_REAL *AMREX_RESTRICT tau = attribs[StructType::tau].data(); //RaytracingX: Add optical depth.
                CCTK_REAL *AMREX_RESTRICT index = attribs[StructType::pixel_number].data(); //RaytracingX: Add pixel index.
                auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

                // Get the array of each parameter.
                auto const lapse_array = lapse.array(pti);
                auto const shift_array = shift.array(pti);
                auto const metric_array = metric.array(pti);
                auto const curv_array = curv.array(pti);
                auto const rho_array = rho.array(pti);//RaytracingX: Add optical depth.

                // Needed for GPU
                auto self = this;

                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept
                                   {
      const amrex::GpuArray<CCTK_REAL, 8> U = {
          particles[i].pos(0), particles[i].pos(1), particles[i].pos(2),
          vels_x[i],           vels_y[i],           vels_z[i],
          ln_alphaenergy[i], tau[i]}; //RaytracingX: Add density for optical depth.

      bool out_of_bounds = false;
      int deletion_reason = -999; //RaytracingX: Keep track of each reason a particle is deleted.

      //RaytracingX: Add optical depth.
      amrex::GpuArray<CCTK_REAL, 8> U_tmp = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      // f1 = rhs(u , t) for the runge kutta 4 step
      auto k_odd =
          self->compute_rhs(U, 0.0, lapse_array, shift_array, metric_array,
                            curv_array, rho_array, dt, dx, lev, plo0); //RaytracingX: Add density for optical depth.

      U_tmp[0] = U[0] + 0.5 * dt * k_odd[0];
      U_tmp[1] = U[1] + 0.5 * dt * k_odd[1];
      U_tmp[2] = U[2] + 0.5 * dt * k_odd[2];
      U_tmp[3] = U[3] + 0.5 * dt * k_odd[3];
      U_tmp[4] = U[4] + 0.5 * dt * k_odd[4];
      U_tmp[5] = U[5] + 0.5 * dt * k_odd[5];
      U_tmp[6] = U[6] + 0.5 * dt * k_odd[6];
      U_tmp[7] = U[7] + 0.5 * dt * k_odd[7]; //RaytracingX: Add optical depth.
      
      //RaytracingX: Change bounds check for debug information.
      CHECK_OUT_OF_BOUNDS_X(U_tmp[0])
      CHECK_OUT_OF_BOUNDS_Y(U_tmp[1])
      CHECK_OUT_OF_BOUNDS_Z(U_tmp[2])

      if (out_of_bounds) {
        //RaytracingX: Write particle information on deletion.
        write_deleted_particle_data(index[i], particles[i].pos(0), particles[i].pos(1), particles[i].pos(2), vels_x[i], vels_y[i], vels_z[i], ln_alphaenergy[i], tau[i], deletion_reason, output_final_data, final_data_file_name);
        particles[i].id() = -1;
        return;
      }

      // f2 = rhs(u + 0.5 * dt * f1, t) for the runge kutta 4 step
      auto k_even =
          self->compute_rhs(U_tmp, 0.5 * dt, lapse_array, shift_array,
                            metric_array, curv_array, rho_array, dt, dx, lev, plo0);

      // Update particles with the f1 and f2 from RK4
      U_tmp[0] = U[0] + 0.5 * dt * k_even[0];
      U_tmp[1] = U[1] + 0.5 * dt * k_even[1];
      U_tmp[2] = U[2] + 0.5 * dt * k_even[2];
      U_tmp[3] = U[3] + 0.5 * dt * k_even[3];
      U_tmp[4] = U[4] + 0.5 * dt * k_even[4];
      U_tmp[5] = U[5] + 0.5 * dt * k_even[5];
      U_tmp[6] = U[6] + 0.5 * dt * k_even[6];
      U_tmp[7] = U[7] + 0.5 * dt * k_even[7]; //RaytracingX: Add optical depth.

      particles[i].pos(0) += (1. / 6.) * dt * (k_odd[0] + 2. * k_even[0]);
      particles[i].pos(1) += (1. / 6.) * dt * (k_odd[1] + 2. * k_even[1]);
      particles[i].pos(2) += (1. / 6.) * dt * (k_odd[2] + 2. * k_even[2]);
      vels_x[i] += (1. / 6.) * dt * (k_odd[3] + 2. * k_even[3]);
      vels_y[i] += (1. / 6.) * dt * (k_odd[4] + 2. * k_even[4]);
      vels_z[i] += (1. / 6.) * dt * (k_odd[5] + 2. * k_even[5]);
      ln_alphaenergy[i] += (1. / 6.) * dt * (k_odd[6] + 2. * k_even[6]);
      tau[i] += (1. / 6.) * dt * (k_odd[7] + 2. * k_even[7]); //RaytracingX: Add optical depth.
      
      //RaytracingX: Change bounds check for debug information.
      CHECK_OUT_OF_BOUNDS_X(U_tmp[0])
      CHECK_OUT_OF_BOUNDS_Y(U_tmp[1])
      CHECK_OUT_OF_BOUNDS_Z(U_tmp[2])

      if (out_of_bounds) {
        //RaytracingX: Write particle information on deletion.
        write_deleted_particle_data(index[i], particles[i].pos(0), particles[i].pos(1), particles[i].pos(2), vels_x[i], vels_y[i], vels_z[i], ln_alphaenergy[i], tau[i], deletion_reason, output_final_data, final_data_file_name);
        particles[i].id() = -1;
        return;
      }

      // f3 = rhs(u + 0.5 * dt * f2, t) for the runge kutta 4 step
      k_odd = self->compute_rhs(U_tmp, 0.5 * dt, lapse_array, shift_array,
                                metric_array, curv_array, rho_array, dt, dx, lev, plo0); //RaytracingX: Add optical depth.

      U_tmp[0] = U[0] + dt * k_odd[0];
      U_tmp[1] = U[1] + dt * k_odd[1];
      U_tmp[2] = U[2] + dt * k_odd[2];
      U_tmp[3] = U[3] + dt * k_odd[3];
      U_tmp[4] = U[4] + dt * k_odd[4];
      U_tmp[5] = U[5] + dt * k_odd[5];
      U_tmp[6] = U[6] + dt * k_odd[6];
      U_tmp[7] = U[7] + dt * k_odd[7]; //RaytracingX: Add optical depth.
      
      //RaytracingX: Change bounds check for debug information.
      CHECK_OUT_OF_BOUNDS_X(U_tmp[0])
      CHECK_OUT_OF_BOUNDS_Y(U_tmp[1])
      CHECK_OUT_OF_BOUNDS_Z(U_tmp[2])

      if (out_of_bounds) {
        //RaytracingX: Write particle information on deletion.
        write_deleted_particle_data(index[i], particles[i].pos(0), particles[i].pos(1), particles[i].pos(2), vels_x[i], vels_y[i], vels_z[i], ln_alphaenergy[i], tau[i], deletion_reason, output_final_data, final_data_file_name);
        particles[i].id() = -1;
        return;
      }

      // f4 = rhs(u + dt * f3, t) for the runge kutta 4 step
      k_even = self->compute_rhs(U_tmp, dt, lapse_array, shift_array,
                                 metric_array, curv_array, rho_array, dt, dx, lev, plo0); //RaytracingX: Add optical depth.

      // Update particles with the f3 and f4 from RK4
      particles[i].pos(0) += (1. / 6.) * dt * (2. * k_odd[0] + k_even[0]);
      particles[i].pos(1) += (1. / 6.) * dt * (2. * k_odd[1] + k_even[1]);
      particles[i].pos(2) += (1. / 6.) * dt * (2. * k_odd[2] + k_even[2]);
      vels_x[i] += (1. / 6.) * dt * (2. * k_odd[3] + k_even[3]);
      vels_y[i] += (1. / 6.) * dt * (2. * k_odd[4] + k_even[4]);
      vels_z[i] += (1. / 6.) * dt * (2. * k_odd[5] + k_even[5]);
      ln_alphaenergy[i] += (1. / 6.) * dt * (2. * k_odd[6] + k_even[6]);
      tau[i] += (1. / 6.) * dt * (2. * k_odd[7] + k_even[7]); //RaytracingX: Add optical depth.

      //RaytracingX: Change bounds check for debug information.
      CHECK_OUT_OF_BOUNDS_X(particles[i].pos(0))
      CHECK_OUT_OF_BOUNDS_Y(particles[i].pos(1))
      CHECK_OUT_OF_BOUNDS_Z(particles[i].pos(2))

      //RaytracingX: Delete particle (i.e. stop evolving geodesic) when geodesic hits photosphere (tau=1).
      out_of_bounds |= (tau[i] > 1.);
      
      //RaytracingX: Delete particle when geodesic reaches event horizon.
      const long int i0 = amrex::Math::floor((particles[i].pos(0) - plo[0]) / dx[0]);
      const long int j0 = amrex::Math::floor((particles[i].pos(1) - plo[1]) / dx[1]);
      const long int k0 = amrex::Math::floor((particles[i].pos(2) - plo[2]) / dx[2]);
      // Interpolate lapse & partial lapse at \vect{x}
      CCTK_REAL lapse_x;
      amrex::GpuArray<CCTK_REAL, 3> d_lapse_x;
      GInX::d_interpolate_array<5>(lapse_x, d_lapse_x, lapse_array, i0, j0, k0, particles[i].pos(0), particles[i].pos(1),
                                   particles[i].pos(2), dx, plo);
      if (exp(ln_alphaenergy[i]) / lapse_x > max_energy) {
        out_of_bounds = true;
        deletion_reason = -7;
      }

      if (out_of_bounds) {
        //RaytracingX: Write particle information on deletion.
        write_deleted_particle_data(index[i], particles[i].pos(0), particles[i].pos(1), particles[i].pos(2), vels_x[i], vels_y[i], vels_z[i], ln_alphaenergy[i], tau[i], deletion_reason, output_final_data, final_data_file_name);
        particles[i].id() = -1;
        return;
      } });
            }
        } // RaytracingParticlesContainer::evolve

        /**
         * The check banned zones function check for user defined invalid particles
         * zones. 
         * RaytracingX: Changed to work with spinning BHs and output data.
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
                                const CCTK_REAL (&radius)[10],
                                const CCTK_REAL (&a)[10],
                                bool output_final_data, std::string final_data_file_name) //RaytracingX: Add output data.
        {

            if (!zones)
            {
                return;
            }

            for (GInX::ParticleIterator<StructType> pti(*this, level);
                 pti.isValid(); ++pti)
            {
                const int np = pti.numParticles();
                auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

                auto &attribs = pti.GetAttributes();
                CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
                CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
                CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
                CCTK_REAL *AMREX_RESTRICT ln_alphaenergy = attribs[StructType::ln_alphaE].data(); //RaytracingX: Add ln_alphaE.
                CCTK_REAL *AMREX_RESTRICT tau = attribs[StructType::tau].data(); //RaytracingX: Add optical depth.
                CCTK_REAL *AMREX_RESTRICT index = attribs[StructType::pixel_number].data(); //RaytracingX: Add pixel number.

                auto self = this;
                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept
                                   {
        for (int check = 0; check < zones; check++) {
          const CCTK_REAL dx = particles[i].pos(0) - x[check];
          const CCTK_REAL dy = particles[i].pos(1) - y[check];
          const CCTK_REAL dz = particles[i].pos(2) - z[check];

          //RaytracingX: Change to work for spinning BHs.
          const CCTK_REAL R2minusa2 = dx*dx + dy*dy + dz*dz - a[check]*a[check];
          const CCTK_REAL r = sqrt(R2minusa2 + sqrt(R2minusa2*R2minusa2+4*a[check]*a[check]*z[check]*z[check])) / 2;

          if (!(r > 0)) { CCTK_ERROR("Issue with calculating distance to banned zone."); }
          
          if (r <= (radius[check] + sqrt(radius[check]*radius[check]-4*a[check]*a[check])) / 2.0) {
            particles[i].id() = -1;
            int deletion_reason = -check - 8;
            //RaytracingX: Write particle information on deletion.
            write_deleted_particle_data(index[i], particles[i].pos(0), particles[i].pos(1), particles[i].pos(2), vels_x[i], vels_y[i], vels_z[i], ln_alphaenergy[i], tau[i], deletion_reason, output_final_data, final_data_file_name);
          }
        } });
            }
        }

        /**
         * \brief Normalize the velocity accordingly to the metric on each particle
         * position.
         *
         * This function is made to normalize the velocity given a random initial data
         * using the Photons positions by using that
         *
         *  \f[
         *  P^\mu P_\mu = 0.
         *  \f]
         *
         *  or equivalently
         *
         *  \f[
         *  V^\alpha V_\alpha = V_\alpha V_\beta \gamma^{\alpha\beta} = 1.
         *  \f]
         *
         * @param metric ADM 3 dimension metric.
         * @param Current refinement level.
         */
        void normalize_velocity(
            const amrex::MultiFab &metric, const int level)
        {

            // Get the with of the discretization on each direction.
            const auto dx = this->Geom(level).CellSizeArray();
            // Get the lower and higher value over the ParticleContainer Geometry
            const auto p_lo = this->Geom(level).ProbLoArray();
            const auto p_hi = this->Geom(level).ProbHiArray();

            for (amrex::MFIter mfi = this->MakeMFIter(level); mfi.isValid(); ++mfi)
            {

                // Get a reference to the particles
                auto &particle_tile = this->DefineAndReturnParticleTile(level, mfi);

                // Determines the current size and the required new size
                const auto current_size = particle_tile.GetArrayOfStructs().size();

                // Gets raw pointers to the two different ways particle data is stored for
                // performance reasons: Array of Struct (AoS) and Struct of Arrays (SoA)
                auto *p_struct = particle_tile.GetArrayOfStructs()().data();
                auto arrdata = particle_tile.GetStructOfArrays().realarray();

                // get the current process id
                const auto metric_array = metric.array(mfi);
                const CCTK_REAL m = this->mass;

                amrex::ParallelFor(current_size, [=] AMREX_GPU_DEVICE(int i) noexcept
                                   {
                // Start a for loop with Random Number evolution for the velocity
                const CCTK_REAL ratio[3] = {arrdata[StructType::vx][i],
                                            arrdata[StructType::vy][i],
                                            arrdata[StructType::vz][i]};
                const CCTK_REAL E = std::exp(arrdata[StructType::ln_alphaE][i]);
                
                // Generate a random position
                const auto &p = p_struct[i];
                
                const int i0 = amrex::Math::floor((p.pos(0) - p_lo[0]) / dx[0]);
                const int j0 = amrex::Math::floor((p.pos(1) - p_lo[1]) / dx[1]);
                const int k0 = amrex::Math::floor((p.pos(2) - p_lo[2]) / dx[2]);
                
                // Interpolate metric
                amrex::GpuArray<CCTK_REAL, 6> gamma_x;
                GInX::interpolate_array<5>(gamma_x, metric_array, i0, j0, k0, p.pos(0),
                                     p.pos(1), p.pos(2), dx, p_lo);
                
                const CCTK_REAL inv_det_gamma =
                    1.0 / (gamma_x[0] * gamma_x[3] * gamma_x[5] +
                           2. * gamma_x[1] * gamma_x[2] * gamma_x[4] -
                           gamma_x[2] * gamma_x[2] * gamma_x[3] -
                           gamma_x[4] * gamma_x[4] * gamma_x[0] -
                           gamma_x[1] * gamma_x[1] * gamma_x[5]);
                    
                const amrex::GpuArray<CCTK_REAL, 6> gamma_inv_x = {
                    (gamma_x[3] * gamma_x[5] - gamma_x[4] * gamma_x[4]) * inv_det_gamma,
                    (gamma_x[4] * gamma_x[2] - gamma_x[1] * gamma_x[5]) * inv_det_gamma,
                    (gamma_x[1] * gamma_x[4] - gamma_x[2] * gamma_x[3]) * inv_det_gamma,
                    (gamma_x[0] * gamma_x[5] - gamma_x[2] * gamma_x[2]) * inv_det_gamma,
                    (gamma_x[2] * gamma_x[1] - gamma_x[0] * gamma_x[4]) * inv_det_gamma,
                    (gamma_x[0] * gamma_x[3] - gamma_x[1] * gamma_x[1]) * inv_det_gamma};
                
                // Normalizing the velocity.
                const CCTK_REAL v_squared = ratio[0] * ratio[0] * gamma_inv_x[0] +
                                            ratio[1] * ratio[1] * gamma_inv_x[3] +
                                            ratio[2] * ratio[2] * gamma_inv_x[5] +
                                            2.0 * ratio[0] * ratio[1] * gamma_inv_x[1] +
                                            2.0 * ratio[0] * ratio[2] * gamma_inv_x[2] +
                                            2.0 * ratio[1] * ratio[2] * gamma_inv_x[4];
                
                const CCTK_REAL v = std::sqrt(v_squared);
                const CCTK_REAL alpha = std::sqrt(1. - m * m / (E * E));

                arrdata[StructType::vx][i] = ratio[0] * alpha / v;
                arrdata[StructType::vy][i] = ratio[1] * alpha / v;
                arrdata[StructType::vz][i] = ratio[2] * alpha / v; });
            }
        } // RaytracingParticlesContainer::normalize_velocity

        void redistribute_particles()
        {
            CCTK_INFO("Redistributing particles");
        } // RaytracingParticlesContainer::redistribute_particles

        //RaytracingX: evolve method override necessary because expanding virtual class from BaseParticleContainer
        void evolve(const amrex::MultiFab &lapse,
                    const amrex::MultiFab &shift,
                    const amrex::MultiFab &metric,
                    const amrex::MultiFab &curv, const CCTK_REAL &dt,
                    const int &lev)
        {
            CCTK_ERROR("This evolve method should not be used! Use the other one.");
            return;
        }
    };
}

#endif