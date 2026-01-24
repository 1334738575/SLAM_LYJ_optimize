#pragma once
#ifdef HAS_CERES


#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <Eigen/Dense>
#include <vector>
#include <iostream>


namespace OPTIMIZE_LYJ
{
	namespace OPTIMIZE_CERES
	{
        using namespace Eigen;
        using namespace std;

        template <typename T>
        void AngleAxisToRotationMatrix(const T* angle_axis, T* R) {
            using EigenVector = Eigen::Matrix<T, 3, 1>;
            using EigenMap = Eigen::Map<Eigen::Matrix<T, 3, 3>>;

            Eigen::AngleAxis<T> aa(EigenVector(angle_axis).norm(),
                EigenVector(angle_axis).normalized());
            Matrix<T, 3, 3> r = aa.toRotationMatrix();
            memcpy((void*)R, (void*)r.data(), sizeof(T) * 9);
            //EigenMap(R) = aa.toRotationMatrix();
            //Matrix3d rr = r.cast<double>();
            //std::cout << "R: " << r << std::endl;
            return;
        }

        // 重投影误差模型
        struct ReprojectionError {
            ReprojectionError(const Vector2d& observed_uv, const Matrix3d& K)
                : observed_uv(observed_uv), K(K) {
            }

            template <typename T>
            bool operator()(const T* const camera_pose, // 6D: [angle_axis(3), translation(3)]
                const T* const point3d,    // 3D点坐标
                T* residual) const {
                // 解析旋转向量和平移
                const T* angle_axis = camera_pose;
                const T* translation = camera_pose + 3;

                // 将旋转向量转换为旋转矩阵
                Matrix<T, 3, 3> R;
                AngleAxisToRotationMatrix<T>(angle_axis, R.data());

                // 将3D点转换到相机坐标系
                Matrix<T, 3, 1> p_cam = R * Matrix<T, 3, 1>(point3d[0], point3d[1], point3d[2])
                    + Matrix<T, 3, 1>(translation[0], translation[1], translation[2]);

                // 投影到图像平面
                p_cam /= p_cam[2];
                Matrix<T, 3, 1> uv_hom = K.cast<T>() * p_cam;

                // 计算残差
                residual[0] = uv_hom[0] - T(observed_uv.x());
                residual[1] = uv_hom[1] - T(observed_uv.y());
				//std::cout << "Residual: " << residual[0] << ", " << residual[1] << std::endl;
                return true;
            }

            const Vector2d observed_uv;
            const Matrix3d K;
        };

        // 生成测试数据（与之前代码保持一致）
        void generateTestData(vector<Vector3d>& points3d,
            vector<pair<Vector2d, Vector2d>>& matches,
            Matrix3d& K,
            Matrix4d& T1,
            Matrix4d& T2);

        int ceresCheckTcwUV();



        // 观测数据结构体（图像中的线段端点）
        struct LineObservation {
            double x1, y1;  // 端点1坐标
            double x2, y2;  // 端点2坐标
        };

        // 三维线参数结构：起点 + 方向向量（自动保持单位化）
        struct Line3D {
            Eigen::Vector3d origin;
            Eigen::Vector3d direction;
        };

        // 重投影误差模型
        struct LineReprojectionError {
            LineReprojectionError(double x1, double y1, double x2, double y2)
                : observed_x1(x1), observed_y1(y1),
                observed_x2(x2), observed_y2(y2) {
            }

            template <typename T>
            bool operator()(const T* const pose,
                const T* const line,
                T* residuals) const {
                // 解析位姿参数 (四元数 + 平移)
                const Eigen::Map<const Eigen::Quaternion<T>> q(pose);
                const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(pose + 4);

                // 解析三维线参数 (起点 + 方向)
                const Eigen::Map<const Eigen::Matrix<T, 3, 1>> line_origin(line);
                const Eigen::Map<const Eigen::Matrix<T, 3, 1>> line_dir(line + 3);

                // 计算旋转矩阵
                const Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();

                // 坐标系变换
                const Eigen::Matrix<T, 3, 1> p_cam = R * line_origin + t;
                const Eigen::Matrix<T, 3, 1> dir_cam = R * line_dir;

                // 计算投影线段端点
                const Eigen::Matrix<T, 3, 1> p1 = p_cam;
                const Eigen::Matrix<T, 3, 1> p2 = p_cam + dir_cam;

                // 投影到归一化平面
                const T x1 = p1.x() / p1.z();
                const T y1 = p1.y() / p1.z();
                const T x2 = p2.x() / p2.z();
                const T y2 = p2.y() / p2.z();

                // 计算直线方程 ax + by + c = 0
                const T a = y2 - y1;
                const T b = x1 - x2;
                const T c = x2 * y1 - x1 * y2;
                const T norm = ceres::sqrt(a * a + b * b);

                // 归一化直线参数
                const T a_norm = a / norm;
                const T b_norm = b / norm;
                const T c_norm = c / norm;

                // 端点重投影误差
                residuals[0] = a_norm * T(observed_x1) + b_norm * T(observed_y1) + c_norm;
                residuals[1] = a_norm * T(observed_x2) + b_norm * T(observed_y2) + c_norm;

                return true;
            }

        private:
            const double observed_x1, observed_y1;
            const double observed_x2, observed_y2;
        };
        
        int ceresCheckTcwLineUV(std::vector<Eigen::Matrix<double, 3, 4>>& _Tcws, std::vector<Eigen::Matrix<double, 6, 1>>& _line3Ds, const std::vector<std::vector<Eigen::Vector4d>>& _obs);
	}
}

#endif // HAS_CERES
