#include <ceres/ceres.h>
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
	}
}