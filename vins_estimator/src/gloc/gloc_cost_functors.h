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
// GlocWorldPriorRotCost
//
//  Rotation part of world prior for 6DOF/4DOF optimizer.
//  One block per keyframe, weighted via ScaledLoss(GLOC_W_WORLD_PRIOR_ROT).
//
//  Residual (3D, pseudo-pixels):
//    res[0..2] = r_scale * log( R_body_world[i] * R_prior[i] )
//
//  where r_scale = FOCAL_LENGTH * ref_depth converts radians → pixels
//  so that GLOC_W_WORLD_PRIOR_ROT is on the same scale as GLOC_W_REPROJ.
//
//  Variables: omega_i[3]  axis-angle of R_body_world[i]  (world → body)
// ─────────────────────────────────────────────────────────────────────────────
struct GlocWorldPriorRotCost
{
    double R_prior[9]; // R_world_body_prior[i]  row-major
    double r_scale;    // FOCAL_LENGTH * ref_depth_m  (radians → pixels)

    template <typename T>
    bool operator()(const T *__restrict__ omega_i,
                    T *__restrict__ res) const
    {
        // R_body_world from axis-angle (col-major from Ceres)
        // omega_i encodes R_body_world (world → body)
        T R_bw[9];
        ceres::AngleAxisToRotationMatrix(omega_i, R_bw);

        // R_err = R_body_world * R_prior = R_world_body^T * R_prior
        // R_bw col-major: R_bw(i,k) = R_bw[i + k*3]
        // R_prior row-major: R_prior[r*3 + c] = element(r,c)
        T R_err[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                R_err[i * 3 + j] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_err[i * 3 + j] += R_bw[i + k * 3] * T(R_prior[k * 3 + j]);
            }

        // Row-major → col-major for Ceres RotationMatrixToAngleAxis
        T R_err_cm[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R_err_cm[r + c * 3] = R_err[r * 3 + c];

        T rot_res[3];
        ceres::RotationMatrixToAngleAxis(R_err_cm, rot_res);
        res[0] = T(r_scale) * rot_res[0];
        res[1] = T(r_scale) * rot_res[1];
        res[2] = T(r_scale) * rot_res[2];
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GlocWorldPriorTransCost
//
//  Translation part of world prior for 6DOF/4DOF optimizer.
//  One block per keyframe, weighted via ScaledLoss(GLOC_W_WORLD_PRIOR_TRANS).
//
//  Residual (3D, pseudo-pixels):
//    res[0..2] = t_scale * (t_i - t_prior[i])
//
//  where t_scale = FOCAL_LENGTH / ref_depth converts metres → pixels
//  so that GLOC_W_WORLD_PRIOR_TRANS is on the same scale as GLOC_W_REPROJ.
//
//  Variables: t_i[3]  t_world_body[i]
// ─────────────────────────────────────────────────────────────────────────────
struct GlocWorldPriorTransCost
{
    double t_prior[3]; // t_world_body_prior[i]
    double t_scale;    // FOCAL_LENGTH / ref_depth_m  (metres → pixels)

    template <typename T>
    bool operator()(const T *__restrict__ t_i,
                    T *__restrict__ res) const
    {
        res[0] = T(t_scale) * (t_i[0] - T(t_prior[0]));
        res[1] = T(t_scale) * (t_i[1] - T(t_prior[1]));
        res[2] = T(t_scale) * (t_i[2] - T(t_prior[2]));
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

// GlocFixedRelPriorCost
//
// Simple L2 prior: penalises a parameter block deviating from its seed.
//   residual[i] = param[i] - seed[i]
// ─────────────────────────────────────────────────────────────────────────────
template <int N>
struct GlocFixedRelPriorCost
{
    double seed[N];
    double scale; // multiply residual to match reprojection pixel units

    explicit GlocFixedRelPriorCost(const double *s, double scale_ = 1.0)
        : scale(scale_)
    {
        std::copy(s, s + N, seed);
    }

    template <typename T>
    bool operator()(const T *__restrict__ param, T *__restrict__ res) const
    {
        for (int i = 0; i < N; ++i)
            res[i] = T(scale) * (param[i] - T(seed[i]));
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GlocFixedRelWorldPriorRotCost
//
//  Rotation part of world prior for fixed-rel optimizer.
//  Weighted via ScaledLoss(GLOC_W_WORLD_PRIOR_ROT).
//
//  Residual (3D, pseudo-pixels):
//    res[0..2] = r_scale * log( R_map_local^T * R_seed )
//
//  where r_scale = FOCAL_LENGTH * ref_depth converts radians → pixels.
//
//  Variables: omega_map[3]  axis-angle of R_map_local  (local → world)
// ─────────────────────────────────────────────────────────────────────────────
struct GlocFixedRelWorldPriorRotCost
{
    double R_seed[9]; // R_map_local_seed  row-major
    double r_scale;   // FOCAL_LENGTH * ref_depth_m  (radians → pixels)

    template <typename T>
    bool operator()(const T *__restrict__ omega_map,
                    T *__restrict__ res) const
    {
        // R_map_local from axis-angle (col-major from Ceres)
        T R_ml[9];
        ceres::AngleAxisToRotationMatrix(omega_map, R_ml);

        // R_err = R_map_local^T * R_seed
        // R_ml col-major: R_ml(i,k) = R_ml[i + k*3]
        // R_seed row-major: R_seed[r*3 + c] = element(r,c)
        T R_err[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                R_err[i * 3 + j] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_err[i * 3 + j] += R_ml[i + k * 3] * T(R_seed[k * 3 + j]);
            }

        // Row-major → col-major for Ceres RotationMatrixToAngleAxis
        T R_err_cm[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R_err_cm[r + c * 3] = R_err[r * 3 + c];

        T rot_res[3];
        ceres::RotationMatrixToAngleAxis(R_err_cm, rot_res);
        res[0] = T(r_scale) * rot_res[0];
        res[1] = T(r_scale) * rot_res[1];
        res[2] = T(r_scale) * rot_res[2];
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GlocFixedRelWorldPriorTransCost
//
//  Translation part of world prior for fixed-rel optimizer.
//  Weighted via ScaledLoss(GLOC_W_WORLD_PRIOR_TRANS).
//
//  Residual (3D, pseudo-pixels):
//    res[0..2] = t_scale * (t_map - t_seed)
//
//  where t_scale = FOCAL_LENGTH / ref_depth converts metres → pixels.
//
//  Variables: t_map[3]  t_map_local
// ─────────────────────────────────────────────────────────────────────────────
struct GlocFixedRelWorldPriorTransCost
{
    double t_seed[3]; // t_map_local_seed
    double t_scale;   // FOCAL_LENGTH / ref_depth_m  (metres → pixels)

    template <typename T>
    bool operator()(const T *__restrict__ t_map,
                    T *__restrict__ res) const
    {
        res[0] = T(t_scale) * (t_map[0] - T(t_seed[0]));
        res[1] = T(t_scale) * (t_map[1] - T(t_seed[1]));
        res[2] = T(t_scale) * (t_map[2] - T(t_seed[2]));
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// MeshRayPriorCost
//
//   The 3-D point model is:  X_world = o_j + (1/rho) * Rtm
//   where  o_j  = train camera centre (world),
//          Rtm  = R_j^T * K_j^{-1} * q_train  (unnormalised world bearing).
//
//   We cast the ray (o_j, Rtm) against the mesh and obtain the intersection
//   depth  lambda_mesh  such that  X_mesh = o_j + lambda_mesh * Rtm.
//   The corresponding inverse depth is  rho_mesh = 1 / lambda_mesh.
//
//   Equivalently (avoids 1/rho inside AutoDiff, better conditioning):
//     res[0] = f_avg * (rho_mesh - rho) / (sigma_m * rho_mesh^2)
//   which is the first-order expansion of  (1/rho - 1/rho_mesh)  around rho_mesh.
//
//   This is rho-only (group-0 in Schur ordering) — it does NOT couple to
//   the pose blocks and therefore does not disturb the Schur complement.
// ─────────────────────────────────────────────────────────────────────────────
struct MeshRayPriorCost
{
    double rho_mesh; // 1 / lambda_mesh  (inverse of ray–mesh intersection depth)
    double f_avg;    // sqrt(fx_q * fy_q) — pixel-scale normalisation
    double sigma_m;  // expected depth uncertainty in metres

    template <typename T>
    bool operator()(const T *__restrict__ rho, T *__restrict__ res) const
    {
        // First-order linearisation of (1/rho - 1/rho_mesh):
        //   d(1/rho)/d(rho) = -1/rho^2  ≈ -1/rho_mesh^2  near rho_mesh
        // → residual in metres ≈ (rho_mesh - rho) / rho_mesh^2
        // → scaled to ~pixels:  f_avg * (rho_mesh - rho) / (sigma_m * rho_mesh^2)
        const T scale = T(f_avg / (sigma_m * rho_mesh * rho_mesh));
        res[0] = scale * (T(rho_mesh) - rho[0]);
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GlocFixedRelReprojCostYaw / GlocFixedRelEpipolarCostYaw
//
// 4-DOF variants of GlocFixedRelReprojCost / GlocFixedRelEpipolarCost where
// R_map_local is represented as a scalar yaw angle (radians) instead of a
// 3-vector axis-angle.
//
// Motivation: YawOnlyParameterization on omega_map[3] is numerically unstable
// when the yaw is near ±π/2 or larger — AngleAxisd(rotation_matrix) can flip
// sign across the π boundary as the optimizer accumulates steps, jumping to
// the antipodal solution and producing 0 inliers. Representing R_map_local as
// a single scalar yaw eliminates the singularity entirely since cos/sin are
// smooth and periodic.
//
// R_map_local = Rz(yaw_map):
//   [ cos  -sin  0 ]
//   [ sin   cos  0 ]
//   [  0     0   1 ]
//
// Variables: yaw_map[1]  yaw of R_map_local (radians, local → world)
//            t_map[3]    t_map_local
//            rho[1]      inverse depth  (ReprojCostYaw only)
// ─────────────────────────────────────────────────────────────────────────────

// Helper: build col-major R_ml from scalar yaw and rotate a point.
// Used inside both yaw functors — avoids repeating the same 9 lines.
template <typename T>
inline void yaw_to_R_ml(const T yaw, T R_ml[9])
{
    const T c = ceres::cos(yaw);
    const T s = ceres::sin(yaw);
    // col-major: col0=(c,s,0) col1=(-s,c,0) col2=(0,0,1)
    R_ml[0] = c;
    R_ml[3] = -s;
    R_ml[6] = T(0);
    R_ml[1] = s;
    R_ml[4] = c;
    R_ml[7] = T(0);
    R_ml[2] = T(0);
    R_ml[5] = T(0);
    R_ml[8] = T(1);
}

template <typename T>
inline void R_ml_rotate(const T R_ml[9], const double p[3], T out[3])
{
    // out = R_ml * p  (col-major R_ml, double p)
    out[0] = R_ml[0] * T(p[0]) + R_ml[3] * T(p[1]) + R_ml[6] * T(p[2]);
    out[1] = R_ml[1] * T(p[0]) + R_ml[4] * T(p[1]) + R_ml[7] * T(p[2]);
    out[2] = R_ml[2] * T(p[0]) + R_ml[5] * T(p[1]) + R_ml[8] * T(p[2]);
}

struct GlocFixedRelReprojCostYaw
{
    // Same fields as GlocFixedRelReprojCost
    double R_local_body[9];
    double P_local[3];
    double Rtm[3];
    double oj[3];
    double pq[2];
    double Rcr[9];
    double tcr[3];
    double fx, fy, cx_, cy_;

    template <typename T>
    bool operator()(const T *__restrict__ yaw_map,
                    const T *__restrict__ t_map,
                    const T *__restrict__ rho,
                    T *__restrict__ res) const
    {
        // Build R_ml from scalar yaw
        T R_ml[9];
        yaw_to_R_ml(yaw_map[0], R_ml);

        // R_wb = R_ml * R_local_body  (col-major, R_local_body row-major)
        T R_wb[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
            {
                R_wb[r + c * 3] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_wb[r + c * 3] += R_ml[r + k * 3] * T(R_local_body[k * 3 + c]);
            }

        // t_world_body = R_ml * P_local + t_map
        T t_i[3];
        R_ml_rotate(R_ml, P_local, t_i);
        t_i[0] += t_map[0];
        t_i[1] += t_map[1];
        t_i[2] += t_map[2];

        // omega_bw = AA(R_wb^T)
        T R_bw_cm[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R_bw_cm[r + c * 3] = R_wb[c + r * 3];
        T omega_bw[3];
        ceres::RotationMatrixToAngleAxis(R_bw_cm, omega_bw);

        // Identical to GlocFixedRelReprojCost from here ───────────────────────
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

struct GlocFixedRelEpipolarCostYaw
{
    // Same fields as GlocFixedRelEpipolarCost
    double R_local_body[9];
    double P_local[3];
    double x_q[3];
    double x_t[3];
    double R_j[9];
    double t_j[3];
    double Rcr[9];
    double tcr[3];
    double scale;

    template <typename T>
    bool operator()(const T *__restrict__ yaw_map,
                    const T *__restrict__ t_map,
                    T *__restrict__ res) const
    {
        // Build R_ml from scalar yaw
        T R_ml[9];
        yaw_to_R_ml(yaw_map[0], R_ml);

        // R_wb = R_ml * R_local_body
        T R_wb[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
            {
                R_wb[r + c * 3] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_wb[r + c * 3] += R_ml[r + k * 3] * T(R_local_body[k * 3 + c]);
            }

        // t_world_body = R_ml * P_local + t_map
        T t_i[3];
        R_ml_rotate(R_ml, P_local, t_i);
        t_i[0] += t_map[0];
        t_i[1] += t_map[1];
        t_i[2] += t_map[2];

        // omega_bw = AA(R_wb^T)
        T R_bw_cm[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R_bw_cm[r + c * 3] = R_wb[c + r * 3];
        T omega_bw[3];
        ceres::RotationMatrixToAngleAxis(R_bw_cm, omega_bw);

        // Identical to GlocFixedRelEpipolarCost from here ─────────────────────
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