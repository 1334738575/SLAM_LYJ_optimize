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
        // ������������ת��
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

        // 3D��ͶӰ���������꣨�����У�飩
        Eigen::Vector2d project(const Eigen::Vector3d &P_c,
                                double fx, double fy,
                                double cx, double cy,
                                double min_z = 1e-3);
        // �����ſɱȾ���2x6����Ӧ����������SE3�ĵ�����
        Eigen::Matrix<double, 2, 6> compute_se3_jacobian(
            const Eigen::Matrix3d &R_cw, // ���λ�ˣ����絽����ı任��
            const Eigen::Vector3d &t_cw, // ���λ�ˣ����絽����ı任��
            const Eigen::Vector3d &P_w,  // ��������ϵ�µ�3D��
            double fx, double fy,        // ����ڲ�
            double cx, double cy,
            double min_z = 1e-3);

        // ���� SE3 �任��������
        using SE3 = Eigen::Matrix4d;
        // ���� se3 ������������� (��, ��)
        using se3 = Eigen::Matrix<double, 6, 1>;
        // ����3x3���Գƾ���
        Eigen::Matrix3d hat(const Eigen::Vector3d &v);
        // �ӷ��Գƾ���ָ�����
        Eigen::Vector3d vee(const Eigen::Matrix3d &M);
        // SE3 ����ӳ�䣨���任����ת��Ϊ�������
        se3 se3_log(const SE3 &T);
        // SE3 ָ��ӳ�䣨�������ת��Ϊ�任����
        SE3 se3_exp(const se3 &xi);
        // ���������� ad(��)
        Eigen::Matrix<double, 6, 6> ad_se3(const se3 &xi);
        // ��������λ�������ſɱȾ���
        Eigen::Matrix<double, 6, 6> compute_prior_jacobian(
            const SE3 &T_current, // ��ǰλ�� (4x4)
            const SE3 &T_prior,   // ����λ�� (4x4)
            se3 &_err);

        // ���Գƾ�������
        Eigen::Matrix3d skew(const Eigen::Vector3d &v);
        // Pl��ckerֱ�߽ṹ��
        struct PluckerLine
        {
            Eigen::Vector3d n; // �������� (��λ��)
            Eigen::Vector3d v; // ������ v = p �� n
        };
        // ����SE3�İ������
        Eigen::Matrix<double, 6, 6> adjoint_SE3(const SE3 &T);
        // ��ά��ͶӰ��ͼ��ֱ��
        Eigen::Vector3d projectLineToImage(
            const PluckerLine &L_camera,
            const Eigen::Matrix3d &K_invT);
        void computeJacobians(
            const SE3 &T,                 // ��ǰλ��
            const PluckerLine &L_world,   // ��������ϵ�µ���
            const Eigen::Vector3d &l_obs, // �۲⵽��ͼ����
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