#ifndef OPTIMIZE_VARIABLE_H
#define OPTIMIZE_VARIABLE_H

#include <iostream>
#include "VariableAbr.h"
#include <Eigen/Eigen>

namespace OPTIMIZE_LYJ
{
    template <typename T>
    static Eigen::Matrix<T, 3, 3> skewSymmetric(const Eigen::Matrix<T, 3, 1>& v)
    {
        Eigen::Matrix<T, 3, 3> S;
        S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
        return S;
    }

    // 3D��ͶӰ���������꣨�����У�飩
    static Eigen::Vector2d project(const Eigen::Vector3d& P_c,
        double fx, double fy,
        double cx, double cy,
        double min_z = 1e-3)
    {
        const double z = P_c.z();
        if (z < min_z) { // ������ȹ�С����Ч���
            std::cerr << "Warning: Invalid depth " << z << " during projection!" << std::endl;
            return Eigen::Vector2d(-1, -1); // ������Ч����
        }
        return Eigen::Vector2d(
            fx * P_c.x() / z + cx,
            fy * P_c.y() / z + cy
        );
    }
    // �����ſɱȾ���2x6����Ӧ����������SE3�ĵ�����
    static Eigen::Matrix<double, 2, 6> compute_se3_jacobian(
        const Eigen::Matrix3d& R_cw,         // ���λ�ˣ����絽����ı任��
        const Eigen::Vector3d& t_cw,         // ���λ�ˣ����絽����ı任��
        const Eigen::Vector3d& P_w,        // ��������ϵ�µ�3D��
        double fx, double fy,              // ����ڲ�
        double cx, double cy,
        double min_z = 1e-3)               // ��С�����ֵ
    {
        // Step 1: ����任���������ϵ
        const Eigen::Vector3d P_c = R_cw * P_w + t_cw;
        const double& x = P_c.x();
        const double& y = P_c.y();
        const double& z = P_c.z();

        // ��������Ч��
        if (z < min_z) {
            std::cerr << "Error: Negative depth! Jacobian is invalid." << std::endl;
            return Eigen::Matrix<double, 2, 6>::Zero();
        }

        // Step 2: ����ͶӰ���� de/dPc
        Eigen::Matrix<double, 2, 3> de_dpc;
        const double z_inv = 1.0 / z;
        const double z_inv2 = z_inv * z_inv;
        de_dpc << -fx * z_inv, 0, fx* x* z_inv2,
            0, -fy * z_inv, fy* y* z_inv2;

        // Step 3: ������������������ĵ��� dPc/d��
        Eigen::Matrix<double, 3, 6> dpc_dxi;
        dpc_dxi.leftCols<3>() = Eigen::Matrix3d::Identity(); // ƽ�Ʋ���
        dpc_dxi.rightCols<3>() = -skewSymmetric(P_c);    // ��ת���֣����Գƾ���

        // Step 4: ��ʽ��������ſɱȾ���
        return de_dpc * dpc_dxi;
    }


    // ���� SE3 �任��������
    using SE3 = Eigen::Matrix4d;
    // ���� se3 ������������� (��, ��)
    using se3 = Eigen::Matrix<double, 6, 1>;
    // ����3x3���Գƾ���
    Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
        Eigen::Matrix3d M;
        M << 0, -v.z(), v.y(),
            v.z(), 0, -v.x(),
            -v.y(), v.x(), 0;
        return M;
    }
    // �ӷ��Գƾ���ָ�����
    Eigen::Vector3d vee(const Eigen::Matrix3d& M) {
        return Eigen::Vector3d(M(2, 1), M(0, 2), M(1, 0));
    }
    // SE3 ����ӳ�䣨���任����ת��Ϊ�������
    se3 se3_log(const SE3& T) {
        Eigen::Matrix3d R = T.block<3, 3>(0, 0);
        Eigen::Vector3d t = T.block<3, 1>(0, 3);

        // ������ת���� �Ȧ�
        Eigen::AngleAxisd angle_axis(R);
        double theta = angle_axis.angle();
        Eigen::Vector3d omega = angle_axis.axis();

        // ����Ƚӽ�0�����
        constexpr double kEpsilon = 1e-8;
        if (theta < kEpsilon) {
            se3 ret;
            ret.setZero();
            ret.head<3>() = t;
            return ret;
        }

        // ����ƽ�Ʋ��� ��
        Eigen::Matrix3d J_inv = Eigen::Matrix3d::Identity()
            - 0.5 * hat(omega * theta)
            + (1.0 / (theta * theta) - (1 + cos(theta)) / (2 * theta * sin(theta))) * hat(omega) * hat(omega);
        Eigen::Vector3d rho = J_inv * t;

        // ��װ se3 ���� [��; �ئ�]
        se3 xi;
        xi << rho, omega* theta;
        return xi;
    }
    // ���������� ad(��)
    Eigen::Matrix<double, 6, 6> ad_se3(const se3& xi) {
        Eigen::Vector3d rho = xi.head<3>();
        Eigen::Vector3d omega = xi.tail<3>();

        Eigen::Matrix<double, 6, 6> ad;
        ad.block<3, 3>(0, 0) = hat(omega);
        ad.block<3, 3>(0, 3) = hat(rho);
        ad.block<3, 3>(3, 3) = hat(omega);
        ad.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
        return ad;
    }
    // ��������λ�������ſɱȾ���
    Eigen::Matrix<double, 6, 6> compute_prior_jacobian(
        const SE3& T_current,  // ��ǰλ�� (4x4)
        const SE3& T_prior      // ����λ�� (4x4)
    ) {
        // ������Ա任 T_rel = T_prior^{-1} * T_current
        SE3 T_rel = T_prior.inverse() * T_current;

        // ������������ �� = log(T_rel)
        se3 xi = se3_log(T_rel);

        // �������ſɱȾ���������: J_r^{-1} �� I + 0.5 * ad(��)
        Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Identity();
        if (xi.norm() > 1e-6) { // ����С�������������ֵ���ȶ�
            J += 0.5 * ad_se3(xi);
        }

        return J;
    }


    // ���Գƾ�������
    Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d S;
        S << 0, -v.z(), v.y(),
            v.z(), 0, -v.x(),
            -v.y(), v.x(), 0;
        return S;
    }
    // Pl��ckerֱ�߽ṹ��
    struct PluckerLine {
        Eigen::Vector3d n; // �������� (��λ��)
        Eigen::Vector3d v; // ������ v = p �� n
    };
    // ����SE3�İ������
    Eigen::Matrix<double, 6, 6> adjoint_SE3(const SE3& T) {
        Eigen::Matrix3d R = T.block<3, 3>(0, 0);
        Eigen::Vector3d t = T.block<3, 1>(0, 3);
        Eigen::Matrix<double, 6, 6> adj;
        adj.block<3, 3>(0, 0) = R;
        adj.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
        adj.block<3, 3>(3, 0) = skew(t) * R;
        adj.block<3, 3>(3, 3) = R;
        return adj;
    }
    // ��ά��ͶӰ��ͼ��ֱ��
    Eigen::Vector3d projectLineToImage(
        const PluckerLine& L_camera,
        const Eigen::Matrix3d& K_invT)
    {
        // ��ȡ����������ͶӰ
        Eigen::Vector3d l_homog = K_invT * L_camera.n;
        return l_homog.normalized(); // ��������һ��
    }
    void computeJacobians(
        const SE3& T,                    // ��ǰλ��
        const PluckerLine& L_world,      // ��������ϵ�µ���
        const Eigen::Vector3d& l_obs,    // �۲⵽��ͼ����
        Eigen::Matrix<double, 3, 6>& J_pose,
        Eigen::Matrix<double, 3, 6>& J_line)
    {
        // Step 1: �任���������ϵ
        Eigen::Matrix<double, 6, 6> adj_T = adjoint_SE3(T);
        PluckerLine L_camera;
        L_camera.n = adj_T.block<3, 3>(0, 0) * L_world.n + adj_T.block<3, 3>(0, 3) * L_world.v;
        L_camera.v = adj_T.block<3, 3>(3, 0) * L_world.n + adj_T.block<3, 3>(3, 3) * L_world.v;

        // Step 2: ͶӰ��ͼ��ƽ��
        Eigen::Matrix3d K_invT;// = /* ����ڲε���ת�� */;
        Eigen::Vector3d l_pred = projectLineToImage(L_camera, K_invT);

        // Step 3: ������ͶӰ�ߵĵ���
        Eigen::Matrix3d de_dl = -Eigen::Matrix3d::Identity();

        // Step 4: ͶӰ�߹���Pl��cker����ĵ���
        Eigen::Matrix<double, 3, 3> dl_dLc = K_invT.block<3, 3>(0, 0);

        // Step 5: Pl��cker�������λ�˵ĵ���
        Eigen::Matrix<double, 3, 6> dLc_dxi;
        dLc_dxi.block<3, 3>(0, 0) = -skew(L_camera.n);
        dLc_dxi.block<3, 3>(0, 3) = -skew(L_camera.v);

        // Step 6: Pl��cker��������߲����ĵ���
        Eigen::Matrix<double, 3, 6> dLc_dLw = adj_T.block<3, 6>(0, 0);

        // ����ſɱȾ���
        J_pose = de_dl * dl_dLc * dLc_dxi;
        J_line = de_dl * dl_dLc * dLc_dLw;
    }


    class OptVarPoint3d : public OptVar<double, 3, 3>
    {
    public:
        OptVarPoint3d(const uint16_t _id) : OptVar(_id, VAR_POINT3D) {}
        ~OptVarPoint3d() {}

        bool update(double *_detX) override
        {
            for (int i = 0; i < 3; ++i)
            {
                m_data[i] += _detX[i];
            }
            return true;
        }
        friend std::ostream &operator<<(std::ostream &os, const OptVarPoint3d &cls)
        {
            std::cout << "(" << cls.m_data[0] << ", " << cls.m_data[1] << ", " << cls.m_data[2] << ")";
            return os;
        }

    private:
    };

}

#endif // OPTIMIZE_VARIABLE_H