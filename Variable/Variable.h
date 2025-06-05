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

    // 3D点投影到像素坐标（带深度校验）
    static Eigen::Vector2d project(const Eigen::Vector3d& P_c,
        double fx, double fy,
        double cx, double cy,
        double min_z = 1e-3)
    {
        const double z = P_c.z();
        if (z < min_z) { // 处理深度过小的无效情况
            std::cerr << "Warning: Invalid depth " << z << " during projection!" << std::endl;
            return Eigen::Vector2d(-1, -1); // 返回无效坐标
        }
        return Eigen::Vector2d(
            fx * P_c.x() / z + cx,
            fy * P_c.y() / z + cy
        );
    }
    // 计算雅可比矩阵（2x6，对应像素误差关于SE3的导数）
    static Eigen::Matrix<double, 2, 6> compute_se3_jacobian(
        const Eigen::Matrix3d& R_cw,         // 相机位姿（世界到相机的变换）
        const Eigen::Vector3d& t_cw,         // 相机位姿（世界到相机的变换）
        const Eigen::Vector3d& P_w,        // 世界坐标系下的3D点
        double fx, double fy,              // 相机内参
        double cx, double cy,
        double min_z = 1e-3)               // 最小深度阈值
    {
        // Step 1: 将点变换到相机坐标系
        const Eigen::Vector3d P_c = R_cw * P_w + t_cw;
        const double& x = P_c.x();
        const double& y = P_c.y();
        const double& z = P_c.z();

        // 检查深度有效性
        if (z < min_z) {
            std::cerr << "Error: Negative depth! Jacobian is invalid." << std::endl;
            return Eigen::Matrix<double, 2, 6>::Zero();
        }

        // Step 2: 计算投影导数 de/dPc
        Eigen::Matrix<double, 2, 3> de_dpc;
        const double z_inv = 1.0 / z;
        const double z_inv2 = z_inv * z_inv;
        de_dpc << -fx * z_inv, 0, fx* x* z_inv2,
            0, -fy * z_inv, fy* y* z_inv2;

        // Step 3: 计算点坐标关于李代数的导数 dPc/dξ
        Eigen::Matrix<double, 3, 6> dpc_dxi;
        dpc_dxi.leftCols<3>() = Eigen::Matrix3d::Identity(); // 平移部分
        dpc_dxi.rightCols<3>() = -skewSymmetric(P_c);    // 旋转部分（反对称矩阵）

        // Step 4: 链式法则组合雅可比矩阵
        return de_dpc * dpc_dxi;
    }


    // 定义 SE3 变换矩阵类型
    using SE3 = Eigen::Matrix4d;
    // 定义 se3 李代数向量类型 (ρ, ω)
    using se3 = Eigen::Matrix<double, 6, 1>;
    // 生成3x3反对称矩阵
    Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
        Eigen::Matrix3d M;
        M << 0, -v.z(), v.y(),
            v.z(), 0, -v.x(),
            -v.y(), v.x(), 0;
        return M;
    }
    // 从反对称矩阵恢复向量
    Eigen::Vector3d vee(const Eigen::Matrix3d& M) {
        return Eigen::Vector3d(M(2, 1), M(0, 2), M(1, 0));
    }
    // SE3 对数映射（将变换矩阵转换为李代数）
    se3 se3_log(const SE3& T) {
        Eigen::Matrix3d R = T.block<3, 3>(0, 0);
        Eigen::Vector3d t = T.block<3, 1>(0, 3);

        // 计算旋转向量 θω
        Eigen::AngleAxisd angle_axis(R);
        double theta = angle_axis.angle();
        Eigen::Vector3d omega = angle_axis.axis();

        // 处理θ接近0的情况
        constexpr double kEpsilon = 1e-8;
        if (theta < kEpsilon) {
            se3 ret;
            ret.setZero();
            ret.head<3>() = t;
            return ret;
        }

        // 计算平移部分 ρ
        Eigen::Matrix3d J_inv = Eigen::Matrix3d::Identity()
            - 0.5 * hat(omega * theta)
            + (1.0 / (theta * theta) - (1 + cos(theta)) / (2 * theta * sin(theta))) * hat(omega) * hat(omega);
        Eigen::Vector3d rho = J_inv * t;

        // 组装 se3 向量 [ρ; ωθ]
        se3 xi;
        xi << rho, omega* theta;
        return xi;
    }
    // 构造伴随矩阵 ad(ξ)
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
    // 计算先验位姿误差的雅可比矩阵
    Eigen::Matrix<double, 6, 6> compute_prior_jacobian(
        const SE3& T_current,  // 当前位姿 (4x4)
        const SE3& T_prior      // 先验位姿 (4x4)
    ) {
        // 计算相对变换 T_rel = T_prior^{-1} * T_current
        SE3 T_rel = T_prior.inverse() * T_current;

        // 计算李代数误差 ξ = log(T_rel)
        se3 xi = se3_log(T_rel);

        // 计算右雅可比矩阵的逆近似: J_r^{-1} ≈ I + 0.5 * ad(ξ)
        Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Identity();
        if (xi.norm() > 1e-6) { // 避免小量计算带来的数值不稳定
            J += 0.5 * ad_se3(xi);
        }

        return J;
    }


    // 反对称矩阵生成
    Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d S;
        S << 0, -v.z(), v.y(),
            v.z(), 0, -v.x(),
            -v.y(), v.x(), 0;
        return S;
    }
    // Plücker直线结构体
    struct PluckerLine {
        Eigen::Vector3d n; // 方向向量 (单位化)
        Eigen::Vector3d v; // 矩向量 v = p × n
    };
    // 计算SE3的伴随矩阵
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
    // 三维线投影到图像直线
    Eigen::Vector3d projectLineToImage(
        const PluckerLine& L_camera,
        const Eigen::Matrix3d& K_invT)
    {
        // 提取方向向量并投影
        Eigen::Vector3d l_homog = K_invT * L_camera.n;
        return l_homog.normalized(); // 齐次坐标归一化
    }
    void computeJacobians(
        const SE3& T,                    // 当前位姿
        const PluckerLine& L_world,      // 世界坐标系下的线
        const Eigen::Vector3d& l_obs,    // 观测到的图像线
        Eigen::Matrix<double, 3, 6>& J_pose,
        Eigen::Matrix<double, 3, 6>& J_line)
    {
        // Step 1: 变换到相机坐标系
        Eigen::Matrix<double, 6, 6> adj_T = adjoint_SE3(T);
        PluckerLine L_camera;
        L_camera.n = adj_T.block<3, 3>(0, 0) * L_world.n + adj_T.block<3, 3>(0, 3) * L_world.v;
        L_camera.v = adj_T.block<3, 3>(3, 0) * L_world.n + adj_T.block<3, 3>(3, 3) * L_world.v;

        // Step 2: 投影到图像平面
        Eigen::Matrix3d K_invT;// = /* 相机内参的逆转置 */;
        Eigen::Vector3d l_pred = projectLineToImage(L_camera, K_invT);

        // Step 3: 误差关于投影线的导数
        Eigen::Matrix3d de_dl = -Eigen::Matrix3d::Identity();

        // Step 4: 投影线关于Plücker坐标的导数
        Eigen::Matrix<double, 3, 3> dl_dLc = K_invT.block<3, 3>(0, 0);

        // Step 5: Plücker坐标关于位姿的导数
        Eigen::Matrix<double, 3, 6> dLc_dxi;
        dLc_dxi.block<3, 3>(0, 0) = -skew(L_camera.n);
        dLc_dxi.block<3, 3>(0, 3) = -skew(L_camera.v);

        // Step 6: Plücker坐标关于线参数的导数
        Eigen::Matrix<double, 3, 6> dLc_dLw = adj_T.block<3, 6>(0, 0);

        // 组合雅可比矩阵
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