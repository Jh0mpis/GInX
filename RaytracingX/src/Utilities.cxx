#include "RaytracingX.h"

/**
 * \brief Returns \f$g_{\mu\nu} U^\nu V\nu\f$.
 * 
 * @param U Four-vector.
 * 
 * @param V Four-vector.
 * 
 * @param m Metric object containing the metric quantities at the point of interest.
 */
CCTK_REAL innerProduct(const CCTK_REAL* U, const CCTK_REAL* V, const Metric* m) { //return g_{\mu\nu} U^\nu V\nu
    return m->g_tt*U[0]*V[0] + m->beta_x*U[1]*V[0] + m->beta_y*U[2]*V[0] + m->beta_z*U[3]*V[0]
         + m->beta_x*U[0]*V[1] + m->g_xx*U[1]*V[1] + m->g_xy*U[2]*V[1] + m->g_xz*U[3]*V[1]
         + m->beta_y*U[0]*V[2] + m->g_xy*U[1]*V[2] + m->g_yy*U[2]*V[2] + m->g_yz*U[3]*V[2]
         + m->beta_z*U[0]*V[3] + m->g_xz*U[1]*V[3] + m->g_yz*U[2]*V[3] + m->g_zz*U[3]*V[3];
}

/**
 * \brief Gives a four-vector \f$X\f$ that is orthogonal to \f$U\f$, \f$V\f$, and \f$W\f$.
 * Given by \f$X_\rho = \vareps_{\lambda\mu\nu\rho} U^\lambda V^\mu W^\nu\f$, then raising to get \f$X^\rho\f$.
 * 
 * @param X Empty array to store result.
 * 
 * @param U Four-vector.
 * 
 * @param V Four-vector.
 * 
 * @param W Four-vector.
 * 
 * @param m Metric object containing the metric quantities at the point of interest.
 */
void generalizedCrossProduct(CCTK_REAL* X, const CCTK_REAL* U, const CCTK_REAL* V, const CCTK_REAL* W, const Metric* m) {//
    CCTK_REAL temp[4];
    temp[0] = -U[3]*V[1]*W[2] +  U[2]*V[1]*W[3] +  U[3]*V[2]*W[1] + -U[1]*V[2]*W[3] + -U[2]*V[3]*W[1] +  U[1]*V[3]*W[2];
    temp[1] =  U[3]*V[0]*W[2] + -U[2]*V[0]*W[3] + -U[3]*V[2]*W[0] +  U[0]*V[2]*W[3] +  U[2]*V[3]*W[0] + -U[0]*V[3]*W[2];
    temp[2] = -U[3]*V[0]*W[1] +  U[1]*V[0]*W[3] +  U[3]*V[1]*W[0] + -U[0]*V[1]*W[3] + -U[1]*V[3]*W[0] +  U[0]*V[3]*W[1];
    temp[3] =  U[2]*V[0]*W[1] + -U[1]*V[0]*W[2] + -U[2]*V[1]*W[0] +  U[0]*V[1]*W[2] +  U[1]*V[2]*W[0] + -U[0]*V[2]*W[1];
    oneFormToVector(X, temp, m);
}

/**
 * \brief Calculates \f$X^\alpha=X_\beta g^{\alpha\beta}\f$ 
 * 
 * @param X_vector Empty array to store result.
 * 
 * @param X_oneform Oneform with 4 components.
 * 
 * @param m Metric object containing the metric quantities at the point of interest.
 */
void oneFormToVector(CCTK_REAL* X_vector, const CCTK_REAL* X_oneform, const Metric* m) {
    X_vector[0] = X_oneform[0]*m->g_upup_tt + X_oneform[1]*m->g_upup_tx + X_oneform[2]*m->g_upup_ty + X_oneform[3]*m->g_upup_tz;
    X_vector[1] = X_oneform[0]*m->g_upup_tx + X_oneform[1]*m->g_upup_xx + X_oneform[2]*m->g_upup_xy + X_oneform[3]*m->g_upup_xz;
    X_vector[2] = X_oneform[0]*m->g_upup_ty + X_oneform[1]*m->g_upup_xy + X_oneform[2]*m->g_upup_yy + X_oneform[3]*m->g_upup_yz;
    X_vector[3] = X_oneform[0]*m->g_upup_tz + X_oneform[1]*m->g_upup_xz + X_oneform[2]*m->g_upup_yz + X_oneform[3]*m->g_upup_zz;
}

/**
 * \brief Calculates \f$X_\alpha=X^\beta g_{\alpha\beta}\f$ 
 * 
 * @param X_oneform Empty array to store result.
 * 
 * @param X_vector Four-vector.
 * 
 * @param m Metric object containing the metric quantities at the point of interest.
 */
void vectorToOneForm(CCTK_REAL* X_oneform, const CCTK_REAL* X_vector, const Metric* m) {
    X_oneform[0] = X_vector[0]*m->g_tt + X_vector[1]*m->beta_x + X_vector[2]*m->beta_y + X_vector[3]*m->beta_z;
    X_oneform[1] = X_vector[0]*m->beta_x + X_vector[1]*m->g_xx + X_vector[2]*m->g_xy + X_vector[3]*m->g_xz;
    X_oneform[2] = X_vector[0]*m->beta_y + X_vector[1]*m->g_xy + X_vector[2]*m->g_yy + X_vector[3]*m->g_yz;
    X_oneform[3] = X_vector[0]*m->beta_z + X_vector[1]*m->g_xz + X_vector[2]*m->g_yz + X_vector[3]*m->g_zz;
}

/**
 * \brief Calculates \f$X_\alpha=X^\beta g_{\alpha\beta}\f$ 
 * 
 * @param X_oneform Empty array to store result.
 * 
 * @param X_vector Four-vector.
 * 
 * @param metric_arr Array containing the metric quantities at the point of interest. Given by
 * \f g_{00} =  metric_arr[0] \f
 * \f \beta_1 = g_{01} =  metric_arr[1] \f
 * \f \beta_2 = g_{02} =  metric_arr[2] \f
 * \f \beta_3 = g_{03} =  metric_arr[3] \f
 * \f g_{11} =  metric_arr[4] \f
 * \f g_{12} =  metric_arr[5] \f
 * \f g_{13} =  metric_arr[6] \f
 * \f g_{22} =  metric_arr[7] \f
 * \f g_{23} =  metric_arr[8] \f
 * \f g_{33} =  metric_arr[9] \f
 */
void vectorToOneFormArr(CCTK_REAL* X_oneform, const CCTK_REAL* X_vector, const CCTK_REAL* metric_arr) { //X_\nu = g_{\mu\nu} X^\mu
    X_oneform[0] = X_vector[0]*metric_arr[0] + X_vector[1]*metric_arr[1] + X_vector[2]*metric_arr[2] + X_vector[3]*metric_arr[3];
    X_oneform[1] = X_vector[0]*metric_arr[1] + X_vector[1]*metric_arr[4] + X_vector[2]*metric_arr[5] + X_vector[3]*metric_arr[6];
    X_oneform[2] = X_vector[0]*metric_arr[2] + X_vector[1]*metric_arr[5] + X_vector[2]*metric_arr[7] + X_vector[3]*metric_arr[8];
    X_oneform[3] = X_vector[0]*metric_arr[3] + X_vector[1]*metric_arr[6] + X_vector[2]*metric_arr[8] + X_vector[3]*metric_arr[9];
}

/**
 * \brief Projects \f$U\f$ onto \f$V\f$. Given by \f$X^\mu = \frac{U \cdot V}{V \cdot V} V^\mu\f$
 * 
 * @param X Empty array to store result.
 * 
 * @param U Four-vector.
 * 
 * @param V Four-vector.
 * 
 * @param m Metric object containing the metric quantities at the point of interest.
 */
void projectUontoV(CCTK_REAL* X, const CCTK_REAL* U, const CCTK_REAL* V, const Metric* m) {
    CCTK_REAL r = innerProduct(U, V, m) / innerProduct(V, V, m);
    X[0] = r*V[0];
    X[1] = r*V[1];
    X[2] = r*V[2];
    X[3] = r*V[3];
}

/**
 * \brief Normalizes \f$X\f$. Given by \f$X_{norm}^\mu = \frac{X^\mu}{\sqrt{X \cdot X}}\f$
 * 
 * @param X_norm Empty array to store result.
 * 
 * @param X Four-vector.
 * 
 * @param m Metric object containing the metric quantities at the point of interest.
 */
void normalize(CCTK_REAL* X_norm, const CCTK_REAL* X, const Metric* m) { 
    CCTK_REAL mag2 = innerProduct(X, X, m);
    CCTK_REAL mag = (mag2 > 0) ? sqrt(mag2) : sqrt(-mag2);
    X_norm[0] = X[0] / mag;
    X_norm[1] = X[1] / mag;
    X_norm[2] = X[2] / mag;
    X_norm[3] = X[3] / mag;
}

/**
 * \brief Gets the time-component of the Four-velocity for a massive particle (i.e. the camera). This is done so the 
 * camera velocity can be defined by the x, y, and z components only, and thus matches the other camera quantities with
 * the number of components specified.
 * 
 * \f g_{\mu\nu} v^\mu v^\nu = -1 \f gives a quadratic equation for \f$v^0\f$ if \f$v^1\f$, \f$v^2\f$, and \f$v^3\f$ are known.
 * \f g_{00} v^0 v^0 + 2v^0 (g_{01} v^1 + g_{02} v^2 + g_{03} v^3) + (g_{ij}) + 1 = 0 \f
 * \f g_{00} v^0 v^0 + 2v^0 (g_{01} v^1 + g_{02} v^2 + g_{03} v^3) + (g_{ij}) + 1 = 0 \f
 * \f Ax^2 + Bx + C = 0 \f
 * \f A = g_{00} \f
 * \f B = 2(g_{01} v^1 + g_{02} v^2 + g_{03} v^3) \f
 * \f C = g_{11} (v^1)^2 + 2g_{12} v^1v^2 + 2g_{13} v^1v^3 + g_{22}(v^2)^2 + 2g_{23} v^2v^3 + g_{33}(v^3)^2 + 1\f
 * 
 * @param X_norm Empty array to store result.
 * 
 * @param X Four-vector.
 * 
 * @param m Metric object containing the metric quantities at the point of interest.
 */
CCTK_REAL getTimeComponentOf4Velocity(const CCTK_REAL vx, const CCTK_REAL vy, const CCTK_REAL vz, Metric* m) { //from knowing v^1, v^2, v^3 and having v \cdot v = -1, find v^0
    CCTK_REAL A = m->g_tt;
    CCTK_REAL B = 2*(m->beta_x*vx + m->beta_y*vy + m->beta_z*vz);
    CCTK_REAL C = m->g_xx*vx*vx + 2*m->g_xy*vx*vy + 2*m->g_xz*vx*vz +
                                   m->g_yy*vy*vy + 2*m->g_yz*vy*vz +
                                                    m->g_zz*vz*vz + 1;
    CCTK_REAL v0 = (-B - sqrt(B*B - 4*A*C))/(2*A);
    CCTK_REAL v[4] = {v0, vx, vy, vz};

    //The root with v0>0 is the physical solution
    if (innerProduct(v, v, m) + 1 < 0.00001 && v0 > 0) {
        return v0;
    }
    v0 = (-B + sqrt(B*B - 4*A*C))/(2*A);
    v[0] = v0;
    if (innerProduct(v, v, m) + 1 < 0.00001 && v0 > 0) {
        return v0;
    }
    
    CCTK_VERROR("RaytracingX: Problem encountered with calculating the time component of the camera's 4-velocity from the 3-velocity");
    return -1;
}

/**
 * \brief This routine performs the Gram-Schmidt Process on the camera velocity, facing direction, and up direction to construct
 * an orthonormal basis for the camera.
 * 
 * @param e0, e1, e2, e3 Empty arrays to store output in.
 * 
 * @param metric Metric quantities at camera in struct defined in RaytracingX.h.
 */
void gramSchmidtProcess(CCTK_ARGUMENTS, CCTK_REAL* e0, CCTK_REAL* e1, CCTK_REAL* e2, CCTK_REAL* e3, Metric* metric) { //use Gram-Schmidt Process to turn e0, e1, e2, and e3 into orthonormal vectors for the camera's POV (see https://arxiv.org/pdf/1410.7775)
    DECLARE_CCTK_ARGUMENTS
    DECLARE_CCTK_PARAMETERS

    //Notation borrowed from https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process.

    CCTK_REAL u0[4] = {getTimeComponentOf4Velocity(camera_vel[0], camera_vel[1], camera_vel[2], metric), camera_vel[0], camera_vel[1], camera_vel[2]}; // u0=v0 is 4-velocity of camera.
    CCTK_REAL u1[4];
    CCTK_REAL u2[4];
    CCTK_REAL u3[4];
    CCTK_REAL v1[4] = {0, camera_point[0], camera_point[1], camera_point[2]}; //v1 is camera pointing direciton.
    CCTK_REAL v2[4] = {0, camera_up[0], camera_up[1], camera_up[2]}; //v2 is camera up direction.
    

    CCTK_REAL projv1_onto_u0[4];
    projectUontoV(projv1_onto_u0, v1, u0, metric);
    u1[0] = v1[0] - projv1_onto_u0[0];
    u1[1] = v1[1] - projv1_onto_u0[1];
    u1[2] = v1[2] - projv1_onto_u0[2];
    u1[3] = v1[3] - projv1_onto_u0[3];

    CCTK_REAL projv2_onto_u0[4];
    projectUontoV(projv2_onto_u0, v2, u0, metric);
    CCTK_REAL projv2_onto_u1[4];
    projectUontoV(projv2_onto_u1, v2, u1, metric);
    u2[0] = v2[0] - projv2_onto_u0[0] - projv2_onto_u1[0];
    u2[1] = v2[1] - projv2_onto_u0[1] - projv2_onto_u1[1];
    u2[2] = v2[2] - projv2_onto_u0[2] - projv2_onto_u1[2];
    u2[3] = v2[3] - projv2_onto_u0[3] - projv2_onto_u1[3];

    normalize(e0, u0, metric); //e0 is the unit vector in the direction of camera 4-velocity.
    normalize(e1, u1, metric); //e1 is the unit vector in the camera facting direction.
    normalize(e2, u2, metric); //e2 is the unit vector from the center of the camera to the center of the top edge.

    generalizedCrossProduct(e3, e0, e1, e2, metric); //e3 is an normalized vector orthogonal to e0, e1, e2, e3, and gives a unit vector in the direction from the center of the camera to the center of the right edge.

    //Check that each vector is orthogonal to others.
    bool error = false;
    error |= !(abs(innerProduct(e0, e1, metric)) < 0.001);
    error |= !(abs(innerProduct(e0, e2, metric)) < 0.001);
    error |= !(abs(innerProduct(e0, e3, metric)) < 0.001);
    error |= !(abs(innerProduct(e1, e2, metric)) < 0.001);
    error |= !(abs(innerProduct(e1, e3, metric)) < 0.001);
    error |= !(abs(innerProduct(e2, e3, metric)) < 0.001);
    if (error) { CCTK_ERROR("Given camera coordinate system is invalid."); }
}