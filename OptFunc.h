#ifndef OPTIMIZE_FUNC_H
#define OPTIMIZE_FUNC_H

#include <Eigen/Eigen>
#include <Eigen/Core>

namespace OPTIMIZE_LYJ
{
    namespace OPTIMIZE_BASE
    {
        Eigen::Matrix<double, 3, 4> relPose(const Eigen::Matrix<double, 3, 4> &_Tw1, const Eigen::Matrix<double, 3, 4> &_Tw2);
        Eigen::Matrix<double, 3, 4> relPose(const double *const _Tw1, const double *const _Tw2);
        Eigen::Matrix<double, 3, 4> invPose(const Eigen::Matrix<double, 3, 4> &_T);
        Eigen::Vector2d Point2Image(double *_K, const Eigen::Matrix<double, 3, 4> &_Tcw, double *_Pw);

        // using m3d = Eigen::Matrix3d;
        using m33 = Eigen::Matrix3d;
        using m34 = Eigen::Matrix<double, 3, 4>;
        using m44 = Eigen::Matrix<double, 4, 4>;
        using v2d = Eigen::Vector2d;
        using v3d = Eigen::Vector3d;
        using v4d = Eigen::Vector4d;
        using v6d = Eigen::Matrix<double, 6, 1>;
        // using m2d = Eigen::Matrix2d;
        using m22 = Eigen::Matrix2d;
        using m23 = Eigen::Matrix<double, 2, 3>;
        using m26 = Eigen::Matrix<double, 2, 6>;
        using m24 = Eigen::Matrix<double, 2, 4>;
        using m36 = Eigen::Matrix<double, 3, 6>;
        using m64 = Eigen::Matrix<double, 6, 4>;
        using m66 = Eigen::Matrix<double, 6, 6>;

        Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d &v);
        Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R);
        Eigen::Matrix3d ExpSO3(const double x, const double y, const double z);
        Eigen::Vector3d Lnso3(const Eigen::Matrix3d &R);
        v6d orth_to_plk(const v4d &orth);
        v6d plk_to_pose(const v6d &plk_w, const m33 &Rcw, const v3d &tcw);
        v6d orth_to_line(const v4d &orth);
        v4d line_to_orth(const v3d &p, const v3d &v);
        // 普吕克与正交转换
        v4d plk_to_orth(const v3d &n, const v3d &v);

        void cal_jac_errUV_Tcw_Pw(const Eigen::Matrix<double, 3, 4> &Tcw, const Eigen::Matrix3d &K,
                                  const Eigen::Vector3d &Pw, const Eigen::Vector2d &uv,
                                  Eigen::Vector2d &err, Eigen::Matrix<double, 2, 6> &jac, const double w, const double invalidErr);
        //---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - ;
        void cal_jac_errT_T(const m44 &priTcw, const m44 &Tcw, v6d &err, m66 &jac);
        //---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - ;
        void cal_jac_errL2D_Tcw_L3D(const m34 &Tcw, const v4d &lineOrth, v2d &err, m26 &jacT, m24 &jacL, const m33 &KK, const v4d &obs);
        //---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ----;
        void update_Tcw(m34 &Tcw, const v6d &detX, const double rate);
        //---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - ;
        void update_lineOrth(v4d &orthW, const v4d &detX, const double rate);

        // 3D点投影到像素坐标（带深度校验）
        Eigen::Vector2d project(const Eigen::Vector3d &P_c,
                                double fx, double fy,
                                double cx, double cy,
                                double min_z = 1e-3);
        // 计算雅可比矩阵（2x6，对应像素误差关于SE3的导数）
        Eigen::Matrix<double, 2, 6> compute_se3_jacobian(
            const Eigen::Matrix3d &R_cw, // 相机位姿（世界到相机的变换）
            const Eigen::Vector3d &t_cw, // 相机位姿（世界到相机的变换）
            const Eigen::Vector3d &P_w,  // 世界坐标系下的3D点
            double fx, double fy,        // 相机内参
            double cx, double cy,
            double min_z = 1e-3);

        // 定义 SE3 变换矩阵类型
        using SE3 = Eigen::Matrix4d;
        // 定义 se3 李代数向量类型 (ρ, ω)
        using se3 = Eigen::Matrix<double, 6, 1>;
        // 生成3x3反对称矩阵
        Eigen::Matrix3d hat(const Eigen::Vector3d &v);
        // 从反对称矩阵恢复向量
        Eigen::Vector3d vee(const Eigen::Matrix3d &M);
        // SE3 对数映射（将变换矩阵转换为李代数）
        se3 se3_log(const SE3 &T);
        // SE3 指数映射（将李代数转换为变换矩阵）
        SE3 se3_exp(const se3 &xi);
        // 构造伴随矩阵 ad(ξ)
        Eigen::Matrix<double, 6, 6> ad_se3(const se3 &xi);
        // 计算先验位姿误差的雅可比矩阵
        Eigen::Matrix<double, 6, 6> compute_prior_jacobian(
            const SE3 &T_current, // 当前位姿 (4x4)
            const SE3 &T_prior,   // 先验位姿 (4x4)
            se3 &_err);

        // 反对称矩阵生成
        Eigen::Matrix3d skew(const Eigen::Vector3d &v);
        // Plücker直线结构体
        struct PluckerLine
        {
            Eigen::Vector3d n; // 方向向量 (单位化)
            Eigen::Vector3d v; // 矩向量 v = p × n
        };
        // 计算SE3的伴随矩阵
        Eigen::Matrix<double, 6, 6> adjoint_SE3(const SE3 &T);
        // 三维线投影到图像直线
        Eigen::Vector3d projectLineToImage(
            const PluckerLine &L_camera,
            const Eigen::Matrix3d &K_invT);
        void computeJacobians(
            const SE3 &T,                 // 当前位姿
            const PluckerLine &L_world,   // 世界坐标系下的线
            const Eigen::Vector3d &l_obs, // 观测到的图像线
            Eigen::Matrix<double, 3, 6> &J_pose,
            Eigen::Matrix<double, 3, 6> &J_line);

        Eigen::Matrix3d EulerToRot(const double &r, const double &p, const double &w);
        void computeJacRelPose3d_Pose3d_Pose3d(
            const Eigen::Matrix3d &Rw1, const Eigen::Vector3d &tw1,
            const Eigen::Matrix3d &Rw2, const Eigen::Vector3d &tw2,
            const Eigen::Matrix3d &R12, const Eigen::Vector3d &t12,
            const Eigen::Vector3d &cw1, const Eigen::Vector3d &cw2,
            Eigen::Matrix<double, 6, 6> &jacT12_Tw1, Eigen::Matrix<double, 6, 6> &jacT12_Tw2,
            Eigen::Matrix<double, 6, 1> &err);
    }

    namespace OPTIMIZE_BASE_TWC
    {
        // using m3d = Eigen::Matrix3d;
        using m33 = Eigen::Matrix3d;
        using m34 = Eigen::Matrix<double, 3, 4>;
        using m44 = Eigen::Matrix<double, 4, 4>;
        using v2d = Eigen::Vector2d;
        using v3d = Eigen::Vector3d;
        using v4d = Eigen::Vector4d;
        using v6d = Eigen::Matrix<double, 6, 1>;
        // using m2d = Eigen::Matrix2d;
        using m22 = Eigen::Matrix2d;
        using m23 = Eigen::Matrix<double, 2, 3>;
        using m26 = Eigen::Matrix<double, 2, 6>;
        using m24 = Eigen::Matrix<double, 2, 4>;
        using m36 = Eigen::Matrix<double, 3, 6>;
        using m64 = Eigen::Matrix<double, 6, 4>;
        using m66 = Eigen::Matrix<double, 6, 6>;

        void cal_jac_errT_T(const m34 &priTwc, const m34 &Twc, v6d &err, m66 &jac);
        void cal_jac_errUV_Twc_Pw(const m34 &Twc, const m33 &K, const v3d &Pw, const v2d &uv,
                                  v2d &err, m26 &jacUV_Twc, m23 &jac_UV_Pw);
        void cal_jac_errPlane_Pw(const v4d &planew, const v3d &Pw, double &err, v3d &jac);
    }

    namespace OPTIMIZE_BASE_TCW
    {
        // using m3d = Eigen::Matrix3d;
        using m33 = Eigen::Matrix3d;
        using m34 = Eigen::Matrix<double, 3, 4>;
        using m44 = Eigen::Matrix<double, 4, 4>;
        using v2d = Eigen::Vector2d;
        using v3d = Eigen::Vector3d;
        using v4d = Eigen::Vector4d;
        using v6d = Eigen::Matrix<double, 6, 1>;
        // using m2d = Eigen::Matrix2d;
        using m22 = Eigen::Matrix2d;
        using m23 = Eigen::Matrix<double, 2, 3>;
        using m26 = Eigen::Matrix<double, 2, 6>;
        using m24 = Eigen::Matrix<double, 2, 4>;
        using m36 = Eigen::Matrix<double, 3, 6>;
        using m64 = Eigen::Matrix<double, 6, 4>;
        using m66 = Eigen::Matrix<double, 6, 6>;

        void cal_jac_errT_T(const m34& priTcw, const m34& Tcw, v6d& err, m66& jac);
        void cal_jac_errUV_Tcw_Pw(const m34& Tcw, const m33& K, const v3d& Pw, const v2d& uv,
            v2d& err, m26& jacUV_Tcw, m23& jac_UV_Pw);
    }
}

#endif