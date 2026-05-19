#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>

// ─────────────────────────────────────────────────────────────────────────────
// Coordinate conventions (written here once, referenced throughout)
//
//  VINS local frame
//    X_local = R_local_body * X_body + P_local_body
//    R_local  ≡  rotation from body  → local  (R_local_body)
//    P_local  ≡  position of body origin in local frame
//
//  COLMAP world frame
//    X_cam = R_c_w * X_world + t_c_w      (q_c_w, t_c_w in Image struct)
//    camera centre in world: o_j = -R_c_w^T * t_c_w
//
//  Optimisation variable per keyframe i
//    omega_i[3]  : axis-angle of  R_body_world[i]  (world → body)
//    t_i[3]      : position of body origin in world frame  (t_world_body[i])
//
//    Derived query camera pose (world → cam):
//      R_c_w_query = R_cam_body * R_body_world
//      t_c_w_query = R_cam_body * t_body_world + t_cam_body
//    where
//      R_cam_body = GLOC_CAM_MODULES[g].ric_[0]  (rotation body → cam)
//      t_cam_body = GLOC_CAM_MODULES[g].tic_[0]  (translation body → cam)
//
//  Virtual camera (same as FeatureTracker::rejectWithF)
//    fx = fy = FOCAL_LENGTH = 460
//    cx = image_width / 2,  cy = image_height / 2
//    Used for both undistorted query and train keypoints so both sides
//    live in a consistent pixel space regardless of actual intrinsics.
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// GlocReprojCost
//
//  Inverse-depth reprojection cost for one correspondence.
//
//  Variables : omega[3]  axis-angle of  R_body_world[i]  (world → body)
//              t_i[3]    position of body origin in world (t_world_body[i])
//              rho[1]    inverse depth anchored in the train camera frame
//
//  Constants stored in functor:
//    Rtm[3]   = R_j^T * m_t  where m_t = normalised train bearing (unit-less)
//               This is the world-frame bearing from the train camera centre
//               toward the 3D point. Precomputed to avoid redundant matmul.
//    oj[3]    = train camera centre in world:  -R_j^T * t_j
//    pq[2]    = observed query pixel in virtual camera (undistorted)
//    Rcr[9]   = R_cam_body  row-major  (body → query cam rotation)
//    tcr[3]   = t_cam_body             (body → query cam translation)
//    fx,fy,cx_,cy_ = virtual camera intrinsics
//
//  Model (stable at rho → 0):
//    X_world = oj + (1/rho) * Rtm            ← 3D point in world
//    R_bw    = AngleAxis(omega)               ← body ← world
//    t_bw    = R_bw * X_world when C_world = -R_bw^T * t_b  ... simplified:
//    t_b     = R_bw * (-t_i)                 ← body translation
//    X_body  = R_bw * X_world + t_b
//    X_cam   = Rcr * X_body   + tcr
//    h       = rho * (Rcr*(R_bw*oj + t_b) + tcr)   ← b (depth-scaled offset)
//            +       (Rcr * R_bw * Rtm)             ← f (bearing direction)
//    proj    = [fx * h[0]/h[2] + cx,  fy * h[1]/h[2] + cy]
//    res     = proj - pq                            ← 2D pixel residual
// ─────────────────────────────────────────────────────────────────────────────
struct GlocReprojCost
{
    double Rtm[3];           // R_j^T * normalised(m_t)  — world bearing from train centre
    double oj[3];            // train camera centre in world: -R_j^T * t_j
    double pq[2];            // observed query pixel (virtual camera, undistorted)
    double Rcr[9];           // R_cam_body  row-major
    double tcr[3];           // t_cam_body
    double fx, fy, cx_, cy_; // virtual camera intrinsics

    template <typename T>
    bool operator()(const T *__restrict__ omega, // [3] axis-angle R_body_world
                    const T *__restrict__ t_i,   // [3] t_world_body
                    const T *__restrict__ rho,   // [1] inverse depth
                    T *__restrict__ res) const
    {
        // ── Derive body pose in world ────────────────────────────────────────
        // t_body_world = R_body_world * (-t_world_body)
        const T neg_t[3] = {-t_i[0], -t_i[1], -t_i[2]};
        T t_bw[3];
        ceres::AngleAxisRotatePoint(omega, neg_t, t_bw);

        // ── Rotate world-frame vectors into body frame ───────────────────────
        const T oj_w[3] = {T(oj[0]), T(oj[1]), T(oj[2])};
        const T Rtm_w[3] = {T(Rtm[0]), T(Rtm[1]), T(Rtm[2])};

        T oj_b[3], Rtm_b[3];
        ceres::AngleAxisRotatePoint(omega, oj_w, oj_b);
        ceres::AngleAxisRotatePoint(omega, Rtm_w, Rtm_b);

        // oj_b now holds R_bw * oj; add t_bw to get X_body at the train centre
        oj_b[0] += t_bw[0];
        oj_b[1] += t_bw[1];
        oj_b[2] += t_bw[2];

        // ── Transform into query camera frame (body → cam) ───────────────────
        // b = Rcr * oj_b + tcr   (depth-scaled offset)
        // f = Rcr * Rtm_b        (bearing direction)
        T b[3], f[3];
        for (int i = 0; i < 3; ++i)
        {
            b[i] = T(Rcr[i * 3 + 0]) * oj_b[0] + T(Rcr[i * 3 + 1]) * oj_b[1] + T(Rcr[i * 3 + 2]) * oj_b[2] + T(tcr[i]);
            f[i] = T(Rcr[i * 3 + 0]) * Rtm_b[0] + T(Rcr[i * 3 + 1]) * Rtm_b[1] + T(Rcr[i * 3 + 2]) * Rtm_b[2];
        }

        // ── Homogeneous projection (stable at rho → 0) ───────────────────────
        const T h0 = rho[0] * b[0] + f[0];
        const T h1 = rho[0] * b[1] + f[1];
        const T h2 = rho[0] * b[2] + f[2];

        if (h2 < T(1e-7))
        {
            res[0] = T(1000.0);
            res[1] = T(1000.0);
            return true;
        }

        const T inv_h2 = T(1.0) / h2;
        res[0] = T(fx) * h0 * inv_h2 + T(cx_) - T(pq[0]);
        res[1] = T(fy) * h1 * inv_h2 + T(cy_) - T(pq[1]);
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GlocEpipolarCost
//
//  Sampson epipolar cost for one correspondence.
//  Scale-free — no depth/rho variable — constrains rotation robustly.
//
//  Variables : omega[3]  axis-angle of  R_body_world[i]
//              t_i[3]    t_world_body[i]
//
//  Constants:
//    x_q[3]  normalised query bearing  [(u-cx)/fx, (v-cy)/fy, 1]
//    x_t[3]  normalised train bearing
//    R_j[9]  train cam rotation  R_c_w  row-major
//    t_j[3]  train cam translation  t_c_w
//    Rcr[9]  R_cam_body  row-major
//    tcr[3]  t_cam_body
//    scale   sqrt(fx*fy) — converts Sampson error → ~pixels
// ─────────────────────────────────────────────────────────────────────────────
struct GlocEpipolarCost
{
    double x_q[3]; // normalised query bearing
    double x_t[3]; // normalised train bearing
    double R_j[9]; // train cam R_c_w  row-major
    double t_j[3]; // train cam t_c_w
    double Rcr[9]; // R_cam_body  row-major
    double tcr[3]; // t_cam_body
    double scale;  // sqrt(fx_q * fy_q) → pixel-scale output

    template <typename T>
    bool operator()(const T *__restrict__ omega,
                    const T *__restrict__ t_i,
                    T *__restrict__ res) const
    {
        // ── Derive query camera pose (world → query cam) ─────────────────────
        // t_body_world = R_body_world * (-t_world_body)
        const T neg_t[3] = {-t_i[0], -t_i[1], -t_i[2]};
        T t_bw[3];
        ceres::AngleAxisRotatePoint(omega, neg_t, t_bw);

        // t_c_w_query = Rcr * t_bw + tcr
        T t_qw[3];
        for (int i = 0; i < 3; ++i)
            t_qw[i] = T(Rcr[i * 3 + 0]) * t_bw[0] + T(Rcr[i * 3 + 1]) * t_bw[1] + T(Rcr[i * 3 + 2]) * t_bw[2] + T(tcr[i]);

        // R_qw = Rcr * R_bw   (Ceres gives R_bw col-major)
        T R_bw[9];
        ceres::AngleAxisToRotationMatrix(omega, R_bw);
        T R_qw[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                R_qw[i * 3 + j] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_qw[i * 3 + j] += T(Rcr[i * 3 + k]) * R_bw[k + j * 3]; // R_bw col-major: R_bw[k][j] = R_bw[k + j*3]
            }

        // ── Relative pose: query ← train ────────────────────────────────────
        // R_rel = R_qw * R_j^T
        T R_rel[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                R_rel[i * 3 + j] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_rel[i * 3 + j] += R_qw[i * 3 + k] * T(R_j[j * 3 + k]); // R_j^T[k][j] = R_j[j][k]
            }

        // t_rel = t_qw - R_rel * t_j
        T t_rel[3];
        for (int i = 0; i < 3; ++i)
        {
            t_rel[i] = t_qw[i];
            for (int j = 0; j < 3; ++j)
                t_rel[i] -= R_rel[i * 3 + j] * T(t_j[j]);
        }

        // ── Essential matrix E = [t_rel]_× * R_rel ──────────────────────────
        T E[9];
        for (int j = 0; j < 3; ++j)
        {
            E[0 * 3 + j] = -t_rel[2] * R_rel[1 * 3 + j] + t_rel[1] * R_rel[2 * 3 + j];
            E[1 * 3 + j] = t_rel[2] * R_rel[0 * 3 + j] - t_rel[0] * R_rel[2 * 3 + j];
            E[2 * 3 + j] = -t_rel[1] * R_rel[0 * 3 + j] + t_rel[0] * R_rel[1 * 3 + j];
        }

        // ── Sampson error ────────────────────────────────────────────────────
        T Ex_t[3] = {T(0), T(0), T(0)};
        T ETx_q[3] = {T(0), T(0), T(0)};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                Ex_t[i] += E[i * 3 + j] * T(x_t[j]);
                ETx_q[i] += E[j * 3 + i] * T(x_q[j]);
            }

        T f = T(0);
        for (int i = 0; i < 3; ++i)
            f += T(x_q[i]) * Ex_t[i];

        const T denom = Ex_t[0] * Ex_t[0] + Ex_t[1] * Ex_t[1] + ETx_q[0] * ETx_q[0] + ETx_q[1] * ETx_q[1];

        if (denom < T(1e-14))
        {
            res[0] = T(1000.0);
            return true;
        }

        res[0] = T(scale) * f / ceres::sqrt(denom);
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GlocRelPoseCost
//
//  Relative local pose soft constraint between keyframes i and j.
//
//  VINS gives us:
//    T_local_body[i]: (R_local[i], P_local[i])   X_local = R_local[i] * X_body + P_local[i]
//    T_local_body[j]: (R_local[j], P_local[j])
//
//  The relative pose in world (derived from optimisation variables) must match
//  the relative pose from VINS (which is accurate over short intervals):
//
//    R_world_body[i]^T * R_world_body[j]  ≈  R_local[i]^T * R_local[j]
//    R_world_body[i]^T * (t_world_body[j] - t_world_body[i])
//                                         ≈  R_local[i]^T * (P_local[j] - P_local[i])
//
//  Residual:
//    rot_res[3]  = log( R_rel_world^T * R_rel_local )   (axis-angle, 3D)
//    trans_res[3]= t_rel_world - t_rel_local             (3D)
//
//  Variables: omega_i[3], t_i[3], omega_j[3], t_j[3]
// ─────────────────────────────────────────────────────────────────────────────
struct GlocRelPoseCost
{
    // Relative pose from VINS local frame:
    //   R_rel_local = R_local[i]^T * R_local[j]
    //   t_rel_local = R_local[i]^T * (P_local[j] - P_local[i])
    double R_rel_local[9]; // row-major
    double t_rel_local[3];

    template <typename T>
    bool operator()(const T *__restrict__ omega_i,
                    const T *__restrict__ t_i,
                    const T *__restrict__ omega_j,
                    const T *__restrict__ t_j,
                    T *__restrict__ res) const
    {
        // ── R_world_body from axis-angle (col-major from Ceres) ───────────────
        T R_wi[9], R_wj[9];
        ceres::AngleAxisToRotationMatrix(omega_i, R_wi);
        ceres::AngleAxisToRotationMatrix(omega_j, R_wj);

        // ── R_rel_world = R_world_body[i]^T * R_world_body[j] ───────────────
        // R_world_body is stored col-major → R_wi[r + c*3] = R_wi(r,c)
        // R_wi^T * R_wj:  (R_wi^T)[i][k] = R_wi[k][i] = R_wi[k + i*3]  (col-major → R_wi[k + i*3])
        T R_rel_world[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                R_rel_world[i * 3 + j] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_rel_world[i * 3 + j] += R_wi[k + i * 3] * R_wj[k + j * 3]; // R_wi^T (row i) dot R_wj (col j)
            }

        // ── t_rel_world = R_world_body[i]^T * (t_j - t_i) ──────────────────
        T dt[3] = {t_j[0] - t_i[0], t_j[1] - t_i[1], t_j[2] - t_i[2]};
        T t_rel_world[3] = {T(0), T(0), T(0)};
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k)
                t_rel_world[i] += R_wi[k + i * 3] * dt[k]; // R_wi^T row i

        // ── Rotation residual: log(R_rel_world^T * R_rel_local) ─────────────
        // R_rel_local is row-major; R_rel_world^T * R_rel_local:
        T R_err[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                R_err[i * 3 + j] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_err[i * 3 + j] += R_rel_world[k * 3 + i] * T(R_rel_local[k * 3 + j]); // R_rel_world^T row-major
            }

        // Convert rotation error matrix to axis-angle (rotation vector)
        // Ceres expects col-major for RotationMatrixToAngleAxis.
        // R_err is currently row-major — convert to col-major:
        T R_err_cm[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R_err_cm[r + c * 3] = R_err[r * 3 + c];

        T rot_res[3];
        ceres::RotationMatrixToAngleAxis(R_err_cm, rot_res);

        // ── Translation residual ─────────────────────────────────────────────
        res[0] = rot_res[0];
        res[1] = rot_res[1];
        res[2] = rot_res[2];
        res[3] = t_rel_world[0] - T(t_rel_local[0]);
        res[4] = t_rel_world[1] - T(t_rel_local[1]);
        res[5] = t_rel_world[2] - T(t_rel_local[2]);
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GlocWorldPriorCost
//
//  World prior: when snapped, penalise T_map_body[i] deviating from the
//  prior derived from the last known T_map_local:
//
//    T_map_body_prior[i] = T_map_local_prior ⊕ T_local_body[i]
//      R_prior = R_map_local * R_local_body[i]
//      t_prior = R_map_local * P_local_body[i] + t_map_local
//
//  Residual (same form as GlocRelPoseCost rotation + translation):
//    rot_res[3]  = log( R_world_body[i]^T * R_prior )
//    trans_res[3]= t_i - t_prior
//
//  Variables: omega_i[3], t_i[3]
// ─────────────────────────────────────────────────────────────────────────────
struct GlocWorldPriorCost
{
    double R_prior[9]; // R_map_body_prior[i]  row-major
    double t_prior[3]; // t_world_body_prior[i]

    template <typename T>
    bool operator()(const T *__restrict__ omega_i,
                    const T *__restrict__ t_i,
                    T *__restrict__ res) const
    {
        // R_world_body from axis-angle (col-major)
        T R_wi[9];
        ceres::AngleAxisToRotationMatrix(omega_i, R_wi);

        // R_err = R_world_body[i]^T * R_prior  (row-major R_prior)
        T R_err[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                R_err[i * 3 + j] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_err[i * 3 + j] += R_wi[k + i * 3] * T(R_prior[k * 3 + j]);
            }

        // Row-major → col-major for Ceres
        T R_err_cm[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R_err_cm[r + c * 3] = R_err[r * 3 + c];

        T rot_res[3];
        ceres::RotationMatrixToAngleAxis(R_err_cm, rot_res);

        res[0] = rot_res[0];
        res[1] = rot_res[1];
        res[2] = rot_res[2];
        res[3] = t_i[0] - T(t_prior[0]);
        res[4] = t_i[1] - T(t_prior[1]);
        res[5] = t_i[2] - T(t_prior[2]);
        return true;
    }
};
// ─────────────────────────────────────────────────────────────────────────────
// Fixed-relative-pose functors
//
// When gloc_fix_rel_poses: true, relative keyframe poses are trusted from
// VINS and held fixed. The only optimization variables are T_map_local:
//   omega_map[3]  axis-angle of R_map_local^{-1} = R_local_map  (world→local, i.e. R_body_world when local≡world)
//
// Convention: we optimize T_map_local directly:
//   R_map_local  encoded as axis-angle of its INVERSE: omega_map = AA(R_local_map)
//                so that AngleAxisRotatePoint(omega_map, x_world) gives x_local.
//
// Actually simpler: store omega_map as AA(R_map_local) i.e. local→world,
// and derive each keyframe's world pose inside the functor:
//
//   R_world_body[i] = R_map_local * R_local_body[i]
//   t_world_body[i] = R_map_local * P_local[i] + t_map_local
//
// where R_local_body[i] and P_local[i] are baked in as constants.
//
// Variables : omega_map[3]  axis-angle of R_map_local  (local → world)
//             t_map[3]      t_map_local (translation part of T_map_local)
//             rho[1]        inverse depth (reprojection functor only)
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// GlocFixedRelReprojCost
//
// Reprojection cost when relative poses are fixed.
// Bakes in R_local_body[i] and P_local[i] for one keyframe.
//
// The body-world derivation inside:
//   R_map_local   = AngleAxis(omega_map)          (local → world)
//   R_world_body  = R_map_local * R_local_body    (body → world)
//   t_world_body  = R_map_local * P_local + t_map
//   omega_body_world = AA(R_world_body^T)         (world → body, as in GlocReprojCost)
//   t_i            = t_world_body
// Then the reprojection math is identical to GlocReprojCost.
// ─────────────────────────────────────────────────────────────────────────────
struct GlocFixedRelReprojCost
{
    // Per-keyframe VINS local pose (constants)
    double R_local_body[9]; // R_local_body  row-major
    double P_local[3];      // t_local_body (body position in local frame)

    // Same correspondence constants as GlocReprojCost
    double Rtm[3];
    double oj[3];
    double pq[2];
    double Rcr[9]; // R_cam_rig
    double tcr[3];
    double fx, fy, cx_, cy_;

    template <typename T>
    bool operator()(const T *__restrict__ omega_map, // [3] AA(R_map_local)
                    const T *__restrict__ t_map,     // [3] t_map_local
                    const T *__restrict__ rho,       // [1] inverse depth
                    T *__restrict__ res) const
    {
        // ── Derive R_world_body and t_world_body from T_map_local ─────────────
        // R_map_local from axis-angle (col-major output from Ceres)
        T R_ml[9]; // col-major R_map_local
        ceres::AngleAxisToRotationMatrix(omega_map, R_ml);

        // R_world_body = R_map_local * R_local_body  (both col-major)
        // R_local_body is stored row-major → convert on the fly
        T R_wb[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
            {
                R_wb[r + c * 3] = T(0); // col-major
                for (int k = 0; k < 3; ++k)
                    R_wb[r + c * 3] += R_ml[r + k * 3] * T(R_local_body[k * 3 + c]);
            }

        // t_world_body = R_map_local * P_local + t_map
        const T pl[3] = {T(P_local[0]), T(P_local[1]), T(P_local[2])};
        T t_i[3];
        ceres::AngleAxisRotatePoint(omega_map, pl, t_i);
        t_i[0] += t_map[0];
        t_i[1] += t_map[1];
        t_i[2] += t_map[2];

        // omega_body_world = AA(R_world_body^T) = AA(R_body_world)
        // R_wb is col-major R_world_body; R_body_world = R_wb^T col-major = R_wb row-major
        T R_bw_cm[9]; // col-major R_body_world
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R_bw_cm[r + c * 3] = R_wb[c + r * 3]; // transpose
        T omega_bw[3];
        ceres::RotationMatrixToAngleAxis(R_bw_cm, omega_bw);

        // ── Now identical to GlocReprojCost ───────────────────────────────────
        const T neg_t[3] = {-t_i[0], -t_i[1], -t_i[2]};
        T t_bw[3];
        ceres::AngleAxisRotatePoint(omega_bw, neg_t, t_bw);

        const T oj_w[3] = {T(oj[0]), T(oj[1]), T(oj[2])};
        const T Rtm_w[3] = {T(Rtm[0]), T(Rtm[1]), T(Rtm[2])};
        T oj_b[3], Rtm_b[3];
        ceres::AngleAxisRotatePoint(omega_bw, oj_w, oj_b);
        ceres::AngleAxisRotatePoint(omega_bw, Rtm_w, Rtm_b);
        oj_b[0] += t_bw[0];
        oj_b[1] += t_bw[1];
        oj_b[2] += t_bw[2];

        T b[3], f[3];
        for (int i = 0; i < 3; ++i)
        {
            b[i] = T(Rcr[i * 3 + 0]) * oj_b[0] + T(Rcr[i * 3 + 1]) * oj_b[1] + T(Rcr[i * 3 + 2]) * oj_b[2] + T(tcr[i]);
            f[i] = T(Rcr[i * 3 + 0]) * Rtm_b[0] + T(Rcr[i * 3 + 1]) * Rtm_b[1] + T(Rcr[i * 3 + 2]) * Rtm_b[2];
        }

        const T h0 = rho[0] * b[0] + f[0];
        const T h1 = rho[0] * b[1] + f[1];
        const T h2 = rho[0] * b[2] + f[2];
        if (h2 < T(1e-7))
        {
            res[0] = T(1000.0);
            res[1] = T(1000.0);
            return true;
        }
        const T inv_h2 = T(1.0) / h2;
        res[0] = T(fx) * h0 * inv_h2 + T(cx_) - T(pq[0]);
        res[1] = T(fy) * h1 * inv_h2 + T(cy_) - T(pq[1]);
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GlocFixedRelEpipolarCost
//
// Sampson epipolar cost when relative poses are fixed.
// ─────────────────────────────────────────────────────────────────────────────
struct GlocFixedRelEpipolarCost
{
    double R_local_body[9]; // row-major
    double P_local[3];

    double x_q[3];
    double x_t[3];
    double R_j[9];
    double t_j[3];
    double Rcr[9];
    double tcr[3];
    double scale;

    template <typename T>
    bool operator()(const T *__restrict__ omega_map,
                    const T *__restrict__ t_map,
                    T *__restrict__ res) const
    {
        // ── Derive omega_bw and t_i (same as GlocFixedRelReprojCost) ─────────
        T R_ml[9];
        ceres::AngleAxisToRotationMatrix(omega_map, R_ml);

        T R_wb[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
            {
                R_wb[r + c * 3] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_wb[r + c * 3] += R_ml[r + k * 3] * T(R_local_body[k * 3 + c]);
            }

        const T pl[3] = {T(P_local[0]), T(P_local[1]), T(P_local[2])};
        T t_i[3];
        ceres::AngleAxisRotatePoint(omega_map, pl, t_i);
        t_i[0] += t_map[0];
        t_i[1] += t_map[1];
        t_i[2] += t_map[2];

        T R_bw_cm[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R_bw_cm[r + c * 3] = R_wb[c + r * 3];
        T omega_bw[3];
        ceres::RotationMatrixToAngleAxis(R_bw_cm, omega_bw);

        // ── Now identical to GlocEpipolarCost ─────────────────────────────────
        const T neg_t[3] = {-t_i[0], -t_i[1], -t_i[2]};
        T t_bw[3];
        ceres::AngleAxisRotatePoint(omega_bw, neg_t, t_bw);

        T t_qw[3];
        for (int i = 0; i < 3; ++i)
            t_qw[i] = T(Rcr[i * 3 + 0]) * t_bw[0] + T(Rcr[i * 3 + 1]) * t_bw[1] + T(Rcr[i * 3 + 2]) * t_bw[2] + T(tcr[i]);

        T R_bw2[9];
        ceres::AngleAxisToRotationMatrix(omega_bw, R_bw2);
        T R_qw[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                R_qw[i * 3 + j] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_qw[i * 3 + j] += T(Rcr[i * 3 + k]) * R_bw2[k + j * 3];
            }

        T R_rel[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                R_rel[i * 3 + j] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_rel[i * 3 + j] += R_qw[i * 3 + k] * T(R_j[j * 3 + k]);
            }

        T t_rel[3];
        for (int i = 0; i < 3; ++i)
        {
            t_rel[i] = t_qw[i];
            for (int j = 0; j < 3; ++j)
                t_rel[i] -= R_rel[i * 3 + j] * T(t_j[j]);
        }

        T E[9];
        for (int j = 0; j < 3; ++j)
        {
            E[0 * 3 + j] = -t_rel[2] * R_rel[1 * 3 + j] + t_rel[1] * R_rel[2 * 3 + j];
            E[1 * 3 + j] = t_rel[2] * R_rel[0 * 3 + j] - t_rel[0] * R_rel[2 * 3 + j];
            E[2 * 3 + j] = -t_rel[1] * R_rel[0 * 3 + j] + t_rel[0] * R_rel[1 * 3 + j];
        }

        T Ex_t[3] = {T(0), T(0), T(0)};
        T ETx_q[3] = {T(0), T(0), T(0)};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                Ex_t[i] += E[i * 3 + j] * T(x_t[j]);
                ETx_q[i] += E[j * 3 + i] * T(x_q[j]);
            }

        T f_val = T(0);
        for (int i = 0; i < 3; ++i)
            f_val += T(x_q[i]) * Ex_t[i];

        const T denom = Ex_t[0] * Ex_t[0] + Ex_t[1] * Ex_t[1] + ETx_q[0] * ETx_q[0] + ETx_q[1] * ETx_q[1];
        if (denom < T(1e-14))
        {
            res[0] = T(1000.0);
            return true;
        }
        res[0] = T(scale) * f_val / ceres::sqrt(denom);
        return true;
    }
};