#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <string.h>
#include "util_Table.h"
#include <math.h>

#ifndef RAYTRACINGX
#define RAYTRACINGX

struct Metric { //struct that contains information about the metric interpolated at a point
    CCTK_REAL alpha;
    CCTK_REAL beta_x, beta_y, beta_z; //g_{tx}, g_{ty}, g_{tz}
    CCTK_REAL g_xx, g_xy, g_xz, g_yy, g_yz, g_zz; //g_{ii} <-> \gamma_{ii}
    CCTK_REAL g_tt; //\sqrt{\alpha^2 + \beta_i\beta^i}
    CCTK_REAL beta_xup, beta_yup, beta_zup; //beta^i = \beta_j\gamma_{ij}
    CCTK_REAL g_upup_tt, g_upup_tx, g_upup_ty, g_upup_tz, g_upup_xx, g_upup_xy, g_upup_xz, g_upup_yy, g_upup_yz, g_upup_zz; //g^{\mu\nu}

    std::string to_string() {
        return "alpha: " + std::to_string(alpha) + "\n" +
               "beta: (" + std::to_string(beta_xup) + ", " + std::to_string(beta_yup) + ", " + std::to_string(beta_zup) + ")\n" +
               "metric: " + std::to_string(g_tt) + ", " + std::to_string(beta_x) + ", " + std::to_string(beta_y) + ", " + std::to_string(beta_z) + "\n" +
               "        " + std::to_string(beta_x) + ", " + std::to_string(g_xx) + ", " + std::to_string(g_xy) + ", " + std::to_string(g_xz) + "\n" +
               "        " + std::to_string(beta_y) + ", " + std::to_string(g_xy) + ", " + std::to_string(g_yy) + ", " + std::to_string(g_yz) + "\n" +
               "        " + std::to_string(beta_z) + ", " + std::to_string(g_xz) + ", " + std::to_string(g_yz) + ", " + std::to_string(g_zz) + "\n";
    }

    std::string to_string_inv() {
        return "metric inv: " + std::to_string(g_upup_tt) + ", " + std::to_string(g_upup_tx) + ", " + std::to_string(g_upup_ty) + ", " + std::to_string(g_upup_tz) + "\n" +
               "            " + std::to_string(g_upup_tx) + ", " + std::to_string(g_upup_xx) + ", " + std::to_string(g_upup_xy) + ", " + std::to_string(g_upup_xz) + "\n" +
               "            " + std::to_string(g_upup_ty) + ", " + std::to_string(g_upup_xy) + ", " + std::to_string(g_upup_yy) + ", " + std::to_string(g_upup_yz) + "\n" +
               "            " + std::to_string(g_upup_tz) + ", " + std::to_string(g_upup_xz) + ", " + std::to_string(g_upup_yz) + ", " + std::to_string(g_upup_zz) + "\n";
    }

    void fillInverseMetric() {
        //find inverse via normal matrix inverse (adjoint method)
        CCTK_REAL inv_det_g = 1/(g_xx*g_yy*g_zz+2.*g_xy*g_xz*g_yz-g_xz*g_xz*g_yy-g_yz*g_yz*g_xx-g_xy*g_xy*g_zz);
        CCTK_REAL gamma_upup_xx = (g_yy*g_zz-g_yz*g_yz)*inv_det_g;
        CCTK_REAL gamma_upup_xy = (g_xz*g_yz-g_xy*g_zz)*inv_det_g;
        CCTK_REAL gamma_upup_xz = (g_xy*g_yz-g_yy*g_xz)*inv_det_g;
        CCTK_REAL gamma_upup_yy = (g_xx*g_zz-g_xz*g_xz)*inv_det_g;
        CCTK_REAL gamma_upup_yz = (g_xz*g_xy-g_xx*g_yz)*inv_det_g;
        CCTK_REAL gamma_upup_zz = (g_xx*g_yy-g_xy*g_xy)*inv_det_g;

        g_upup_tt = -1/(pow(alpha,2));
        g_upup_tx = -g_upup_tt*beta_xup;
        g_upup_ty = -g_upup_tt*beta_yup;
        g_upup_tz = -g_upup_tt*beta_zup;
        g_upup_xx = gamma_upup_xx-g_upup_tx*beta_xup;
        g_upup_xy = gamma_upup_xy-g_upup_tx*beta_yup;
        g_upup_xz = gamma_upup_xz-g_upup_tx*beta_zup;
        g_upup_yy = gamma_upup_yy-g_upup_ty*beta_yup;
        g_upup_yz = gamma_upup_yz-g_upup_ty*beta_zup;
        g_upup_zz = gamma_upup_zz-g_upup_tz*beta_zup;
    }
};

void gramSchmidtProcess(CCTK_ARGUMENTS, CCTK_REAL* e0, CCTK_REAL* e1, CCTK_REAL* e2, CCTK_REAL* e3, Metric* metric); //Utilities.cc
void interpolateMetricAtPoint(CCTK_ARGUMENTS, Metric* metric_at_point); //InterpolateMetric.cc
CCTK_REAL innerProduct(const CCTK_REAL* U, const CCTK_REAL* V, const Metric* m); //utilities.cc
void generalizedCrossProduct(CCTK_REAL* X, const CCTK_REAL* U, const CCTK_REAL* V, const CCTK_REAL* W, const Metric* m); //Utilities.cc
void oneFormToVector(CCTK_REAL* X_vector, const CCTK_REAL* X_oneform, const Metric* m); //Utilities.cc
void vectorToOneForm(CCTK_REAL* X_oneform, const CCTK_REAL* X_vector, const Metric* m); //Utilities.cc
void vectorToOneFormArr(CCTK_REAL* X_oneform, const CCTK_REAL* X_vector, const CCTK_REAL* arr); //Utilities.cc
void projectUontoV(CCTK_REAL* X, const CCTK_REAL* U, const CCTK_REAL* V, const Metric* m); //Utilities.cc
void normalize(CCTK_REAL* X_norm, const CCTK_REAL* X, const Metric* m); //Utilities.cc
CCTK_REAL getTimeComponentOf4Velocity(const CCTK_REAL vx, const CCTK_REAL vy, const CCTK_REAL vz, Metric* m); //Utilities.cc

//constants for unit conversion to code units
static const CCTK_REAL M_cgs = 1.9884e33;      //g               -> M
static const CCTK_REAL c_cgs = 2.99792458e+10; //cm s^-1         -> 1
static const CCTK_REAL G_cgs = 6.67430e-8;     //cm^3 g^-1 s^-2  -> 1

//base conversion factors
static const CCTK_REAL cgs2cactusDensity          = 1.6192e-18; //calculated using wolfram alpha, POW3(G_cgs)*POW2(M_cgs)/POW6(c_cgs) g cm^-3              -> M^-2
static const CCTK_REAL cgs2cactusLength           = 6.7722e-6; //calculated using wolfram alpha, POW2(c_cgs)/(G_cgs * M_cgs)          cm                   -> M

//derived conversion factors
static const CCTK_REAL cactus2cgsOpacity       = 1.0; //1.09656e-23; //cgs2cactusDensity * cgs2cactusLength;                //used in GetLeakageCoolingRate
static const CCTK_REAL cgs2cactusOpacity       = 1.0; //9.119e22; //1.0 / cactus2cgsOpacity;                             //used in CalcOpacity

#endif