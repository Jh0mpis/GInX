#ifndef UTILITIES_HXX
#define UTILITIES_HXX

#include "AMReX_Box.H"
#include "AMReX_Config.H"
#include "AMReX_MFIter.H"
#include "AMReX_REAL.H"
#include "AMReX_Random.H"
#include "AMReX_RandomEngine.H"
#include "AMReX_Scan.H"
#include "Discretizer.hxx"
#include "cctk_Types.h"
#include <array>
#include <cctk.h>

AMREX_GPU_DEVICE AMREX_GPU_HOST
    CCTK_ATTRIBUTE_ALWAYS_INLINE inline const amrex::GpuArray<CCTK_REAL, 3>
    SMatVecMul(amrex::GpuArray<CCTK_REAL, 6> A,
               amrex::GpuArray<CCTK_REAL, 3> V) {

  amrex::GpuArray<CCTK_REAL, 3> result =
  { A[0] * V[0] + A[1] * V[1] + A[2] * V[2],
    A[1] * V[0] + A[3] * V[1] + A[4] * V[2],
    A[2] * V[0] + A[4] * V[1] + A[5] * V[2]};

  return result;
} // SMatVecMul

AMREX_GPU_DEVICE
    AMREX_GPU_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline const CCTK_REAL
    VecVecMul(amrex::GpuArray<CCTK_REAL, 3> U,
               amrex::GpuArray<CCTK_REAL, 3> V) {

  return U[0] * V[0] + U[1] * V[1] + U[2] * V[2];
} // SMatVecMul

#endif // !UTILITIES_HXX
