#pragma once
#ifdef HAS_CERES

#include <ceres/ceres.h>

using ceres::EigenQuaternionManifold;
using ceres::QuaternionManifold;
using ceres::EuclideanManifold;
using ceres::ProductManifold;


using CeresRotationParameter = ceres::QuaternionManifold;
using CeresTransParameter = ceres::EuclideanManifold<3>;
using CeresPoint3DParameter = ceres::EuclideanManifold<3>;
using CeresPoint1DParameter = ceres::EuclideanManifold<1>;
using CeresGlobalScaleParameter = ceres::EuclideanManifold<1>;
using CeresPose3DParameter = ceres::ProductManifold<CeresRotationParameter, CeresTransParameter>;
// 一维角度流形（归一化到 [-pi, pi)，弧度）
class AngleManifold : public ceres::Manifold {
public:
    AngleManifold() = default;
    ~AngleManifold() override = default;

    // 原空间维度：1（角度）
    int AmbientSize() const override {
        return 1;
    }

    // 切空间维度：1
    int TangentSize() const override {
        return 1;
    }

    /**
     * Plus：切空间增量 → 角度空间（核心）
     * x: 原角度（弧度），delta: 切空间增量（弧度），x_plus_delta: 输出新角度
     */
    bool Plus(const double* x,
        const double* delta,
        double* x_plus_delta) const override {
        // 空指针校验（Ceres 2.2.0 严格要求）
        if (x == nullptr || delta == nullptr || x_plus_delta == nullptr) {
            return false;
        }
        // 角度 + 增量 → 归一化到 [-pi, pi)
        const double total = x[0] + delta[0];
        x_plus_delta[0] = ceres::atan2(ceres::sin(total), ceres::cos(total));
        return true;
    }

    /**
     * PlusJacobian：计算 Plus 对 delta 的雅克比矩阵
     * 一维角度的雅克比是 1x1 矩阵 [1.0]（切空间增量直接对应角度增量）
     */
    bool PlusJacobian(const double* x, double* jacobian) const override {
        if (x == nullptr || jacobian == nullptr) {
            return false;
        }
        // 雅克比矩阵是 row-major（行优先），1x1 矩阵直接赋值 1.0
        jacobian[0] = 1.0;
        return true;
    }

    /**
     * Minus：原空间差值 → 切空间增量（核心）
     * y: 目标角度，x: 原角度，y_minus_x: 输出切空间增量（最短路径）
     */
    bool Minus(const double* y,
        const double* x,
        double* y_minus_x) const override {
        if (y == nullptr || x == nullptr || y_minus_x == nullptr) {
            return false;
        }
        // 计算差值并归一化到 [-pi, pi)，保证是最短路径增量
        double diff = y[0] - x[0];
        y_minus_x[0] = ceres::atan2(ceres::sin(diff), ceres::cos(diff));
        return true;
    }

    /**
     * MinusJacobian：计算 Minus 对 y 的雅克比矩阵
     * 一维角度的雅克比是 1x1 矩阵 [1.0]
     */
    bool MinusJacobian(const double* x, double* jacobian) const override {
        if (x == nullptr || jacobian == nullptr) {
            return false;
        }
        // 1x1 行优先矩阵，赋值 1.0
        jacobian[0] = 1.0;
        return true;
    }

    // 可选：复用父类的 RightMultiplyByPlusJacobian 默认实现（无需修改）
    bool RightMultiplyByPlusJacobian(const double* x,
        const int num_rows,
        const double* ambient_matrix,
        double* tangent_matrix) const override {
        return ceres::Manifold::RightMultiplyByPlusJacobian(x, num_rows, ambient_matrix, tangent_matrix);
    }
};
using CeresLine3DParameter = ceres::ProductManifold<CeresRotationParameter, AngleManifold>;
using CeresLine2DParameter = ceres::ProductManifold<AngleManifold, CeresPoint1DParameter>;
using CeresRay3DParameter = ceres::ProductManifold<CeresRotationParameter, AngleManifold, CeresPoint1DParameter>;
using CeresRay3DParameter2 = ceres::ProductManifold<CeresPoint3DParameter, AngleManifold, AngleManifold>;


#endif // HAS_CERES
