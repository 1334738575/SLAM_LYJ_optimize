#include <iostream>
#include <Optimize_LYJ.h>
#include <Eigen/Core>
#include <Eigen/Eigen>




#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions> // 用于矩阵指数运算

using namespace Eigen;

// 定义SE(3)相关操作
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 6, 6> Matrix6d;


Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;
    S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return S;
}

// 计算SO(3)右雅可比矩阵
Matrix3d computeJacobianSO3(const Vector3d& phi) {
    const double theta = phi.norm();
    if (theta < 1e-6) return Matrix3d::Identity();

    const Matrix3d phi_hat = skew(phi);
    const Matrix3d J = Matrix3d::Identity()
        + (1 - cos(theta)) / (theta * theta) * phi_hat
        + (theta - sin(theta)) / (theta * theta * theta) * phi_hat * phi_hat;

    return J;
}

// 将李代数向量转换为4x4变换矩阵
Isometry3d expSE3(const Vector6d& xi) {
    Isometry3d T = Isometry3d::Identity();
    const Vector3d rho = xi.head<3>();
    const Vector3d phi = xi.tail<3>();

    // 旋转部分
    const Matrix3d R = AngleAxisd(phi.norm(), phi.normalized()).toRotationMatrix();

    // 平移部分
    const Matrix3d J = computeJacobianSO3(phi);
    T.linear() = R;
    T.translation() = J * rho;

    return T;
}
Eigen::Matrix4d expSE3_2(const Vector6d& xi) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    const Vector3d rho = xi.head<3>();
    const Vector3d phi = xi.tail<3>();

    // 旋转部分
    const Matrix3d R = AngleAxisd(phi.norm(), phi.normalized()).toRotationMatrix();

    // 平移部分
    const Matrix3d J = computeJacobianSO3(phi);
    T.block(0, 0, 3, 3) = R;
    T.block(0, 3, 3, 1) = J * rho;

    return T;
}

// 计算SO(3)右雅可比逆矩阵
Matrix3d computeJacobianInvSO3(const Vector3d& phi) {
    const double theta = phi.norm();
    if (theta < 1e-6) return Matrix3d::Identity();

    const Matrix3d phi_hat = skew(phi);
    const Matrix3d J_inv = Matrix3d::Identity()
        - 0.5 * phi_hat
        + (1.0 / (theta * theta) - (1 + cos(theta)) / (2 * theta * sin(theta))) * phi_hat * phi_hat;

    return J_inv;
}

// 将变换矩阵转换为李代数向量
Vector6d logSE3(const Isometry3d& T) {
    Vector6d xi;
    const AngleAxisd aa(T.linear());

    // 旋转部分
    const Vector3d phi = aa.angle() * aa.axis();
    const Matrix3d J_inv = computeJacobianInvSO3(phi);

    // 平移部分
    const Vector3d rho = J_inv * T.translation();

    xi << rho, phi;
    return xi;
}
// 将变换矩阵转换为李代数向量
Vector6d logSE3(const Eigen::Matrix4d& T) {
    Vector6d xi;
    Eigen::Matrix3d R = T.block(0, 0, 3, 3);
    const AngleAxisd aa(R);

    // 旋转部分
    const Vector3d phi = aa.angle() * aa.axis();
    const Matrix3d J_inv = computeJacobianInvSO3(phi);

    // 平移部分
    const Vector3d rho = J_inv * T.block(0, 3, 3, 1);

    xi << rho, phi;
    return xi;
}

// 先验因子类
class PriorFactor {
public:
    PriorFactor(const Isometry3d& prior_pose)
        : prior_pose_(prior_pose) {
    }
    PriorFactor(const Eigen::Matrix4d& prior_pose_m)
        : prior_pose_m_(prior_pose_m) {
    }

    // 计算残差和雅可比
    void compute(const Isometry3d& current_pose,
        Vector6d& residual,
        Matrix6d& jacobian) const
    {
        // 计算残差：log(prior^{-1} * current)
        const Isometry3d error = prior_pose_.inverse() * current_pose;
        residual = logSE3(error);

        // 计算雅可比矩阵
        jacobian = computeJacobianInvSE3(residual);
    }
    // 计算残差和雅可比
    void compute(const Eigen::Matrix4d& current_pose,
        Vector6d& residual,
        Matrix6d& jacobian) const
    {
        // 计算残差：log(prior^{-1} * current)
        const Eigen::Matrix4d error = prior_pose_m_.inverse() * current_pose;
        residual = logSE3(error);

        // 计算雅可比矩阵
        jacobian = computeJacobianInvSE3(residual);
    }

private:
    // 计算SE(3)右雅可比逆矩阵
    Matrix6d computeJacobianInvSE3(const Vector6d& xi) const {
        const Vector3d rho = xi.head<3>();
        const Vector3d phi = xi.tail<3>();
        const Matrix3d J_rot_inv = computeJacobianInvSO3(phi);

        Matrix6d J_inv = Matrix6d::Zero();
        J_inv.block<3, 3>(0, 0) = J_rot_inv;
        J_inv.block<3, 3>(3, 3) = J_rot_inv;
        J_inv.block<3, 3>(0, 3) = skew(rho) * J_rot_inv;

        return J_inv;
    }

    Isometry3d prior_pose_;
    Eigen::Matrix4d prior_pose_m_;
};

// 高斯-牛顿优化器
void optimizePose(Isometry3d& current_pose,
    const PriorFactor& factor,
    int max_iterations = 10,
    double tolerance = 1e-6)
{
    for (int iter = 0; iter < max_iterations; ++iter) {
        // 计算残差和雅可比
        Vector6d residual;
        Matrix6d J;
        factor.compute(current_pose, residual, J);

        // 构建线性系统
        Matrix6d H = J.transpose() * J;
        Vector6d b = -J.transpose() * residual;

        // 求解增量
        Vector6d delta = H.ldlt().solve(b);

        // 更新位姿
        Isometry3d delta_pose = expSE3(delta);
        current_pose = current_pose * delta_pose;
        std::cout << "iter " << iter << " pose:\n" << current_pose.matrix() << std::endl;

        // 检查收敛
        if (delta.norm() < tolerance) break;
    }
}
void optimizePose(Eigen::Matrix4d& current_pose,
    const PriorFactor& factor,
    int max_iterations = 10,
    double tolerance = 1e-6)
{
    for (int iter = 0; iter < max_iterations; ++iter) {
        // 计算残差和雅可比
        Vector6d residual;
        Matrix6d J;
        factor.compute(current_pose, residual, J);

        // 构建线性系统
        Matrix6d H = J.transpose() * J;
        Vector6d b = -J.transpose() * residual;

        // 求解增量
        //Vector6d delta = H.ldlt().solve(b);
        for (int i = 0; i < 6; ++i)
            H(i, i) += 1e-6;
        // 创建求解器
        Eigen::LDLT<Eigen::MatrixXd> solver;
        solver.compute(H);
        // 求解 Ax = b
        Vector6d delta = solver.solve(b);

        // 更新位姿
        Eigen::Matrix4d delta_pose = expSE3_2(delta);
        current_pose = current_pose * delta_pose;
        std::cout << "iter " << iter << " pose:\n" << current_pose.matrix() << std::endl;

        // 检查收敛
        if (delta.norm() < tolerance) break;
    }
}

int main2() {
    // 初始化先验位姿和当前位姿
    Isometry3d prior_pose = Isometry3d::Identity();
    prior_pose.translation() << 1.0, 10, 222;
    prior_pose.linear() = AngleAxisd(3.14 / 4, Vector3d::UnitZ()).toRotationMatrix();

    Isometry3d current_pose = Isometry3d::Identity();

    std::cout << "init pose:\n" << current_pose.matrix() << std::endl;
    // 创建先验因子
    PriorFactor factor(prior_pose);

    // 执行优化
    optimizePose(current_pose, factor);

    // 输出结果
    std::cout << "Prior pose:\n" << prior_pose.matrix() << std::endl;
    std::cout << "Optimized pose:\n" << current_pose.matrix() << std::endl;

    return 0;
}
int main3() {
    // 初始化先验位姿和当前位姿
    Eigen::Matrix4d prior_pose = Eigen::Matrix4d::Identity();
    Eigen::Vector3d prior_t(1.0, 10, 222);
    Eigen::Matrix3d prior_R = AngleAxisd(3.14 / 4, Vector3d::UnitZ()).toRotationMatrix();
    prior_pose.block(0, 0, 3, 3) = prior_R;
    prior_pose.block(0, 3, 3, 1) = prior_t;

    Eigen::Matrix4d current_pose = Eigen::Matrix4d::Identity();

    std::cout << "init pose:\n" << current_pose.matrix() << std::endl;
    // 创建先验因子
    PriorFactor factor(prior_pose);

    // 执行优化
    optimizePose(current_pose, factor);

    // 输出结果
    std::cout << "Prior pose:\n" << prior_pose.matrix() << std::endl;
    std::cout << "Optimized pose:\n" << current_pose.matrix() << std::endl;

    return 0;
}





Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> testEigenMap(double* _data)
{
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> mat(_data, 2, 3);
    //std::cout << "eigen map: " << std::endl << mat << std::endl;
    return mat;
}

int main(int argc, char *argv[])
{
    //double data[6] = { 1, 2, 3, 4, 5, 6 };
    //auto mat = testEigenMap(data);
    //std::cout << "eigen map2: " << std::endl << mat << std::endl;
    //return 0;
    std::cout << "Optimize Version: " << OPTIMIZE_LYJ::optimize_version() << std::endl;
    //OPTIMIZE_LYJ::test_optimize_P3d_P3d();
    //OPTIMIZE_LYJ::test_optimize_Pose3d_Pose3d();
    OPTIMIZE_LYJ::test_optimize_RelPose3d_Pose3d_Pose3d();
    //main3();
    return 0;
}