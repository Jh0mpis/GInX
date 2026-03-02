/**
 * \file Utilities.hxx
 * \brief Code utilities such as matrix and vector multiplications.
 *
 * This file contains the definition of some functions that could be helpful at
 * the developing stage.
 */
#ifndef UTILITIES_HXX
#define UTILITIES_HXX

#include "AMReX_REAL.H"
#include "AMReX_Scan.H"
#include "cctk_Types.h"
#include <cctk.h>

/**
 * Symmetric matrix and vector multiplication.
 *
 * This function computes the Symmetric matrix and vector multiplication using
 * that the size 3x3 matrices can be stored as a vector of size 6 we can compute
 * its multiplication with any vector.
 *
 * @param A Vector of size 6 Symmetric matrix.
 * @param V Vector of size 3.
 *
 * @return A vector of size 3 result.
 */
AMREX_GPU_DEVICE AMREX_GPU_HOST
    AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE amrex::GpuArray<CCTK_REAL, 3>
    SMatVecMul(amrex::GpuArray<CCTK_REAL, 6> A,
               amrex::GpuArray<CCTK_REAL, 3> V) {

  amrex::GpuArray<CCTK_REAL, 3> result = {
      A[0] * V[0] + A[1] * V[1] + A[2] * V[2],
      A[1] * V[0] + A[3] * V[1] + A[4] * V[2],
      A[2] * V[0] + A[4] * V[1] + A[5] * V[2]};

  return result;
} // SMatVecMul

/**
 * Vector Vector cartesian product.
 *
 * @param U Vector of size 3.
 * @param V Vector of size 3.
 *
 * @return A Real number result.
 */
AMREX_GPU_DEVICE
AMREX_GPU_HOST AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_REAL
VecVecMul(amrex::GpuArray<CCTK_REAL, 3> U, amrex::GpuArray<CCTK_REAL, 3> V) {

  return U[0] * V[0] + U[1] * V[1] + U[2] * V[2];
} // SMatVecMul

#endif // !UTILITIES_HXX
