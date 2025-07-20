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

        // ��ͶӰ���ģ��
        struct ReprojectionError {
            ReprojectionError(const Vector2d& observed_uv, const Matrix3d& K)
                : observed_uv(observed_uv), K(K) {
            }

            template <typename T>
            bool operator()(const T* const camera_pose, // 6D: [angle_axis(3), translation(3)]
                const T* const point3d,    // 3D������
                T* residual) const {
                // ������ת������ƽ��
                const T* angle_axis = camera_pose;
                const T* translation = camera_pose + 3;

                // ����ת����ת��Ϊ��ת����
                Matrix<T, 3, 3> R;
                AngleAxisToRotationMatrix<T>(angle_axis, R.data());

                // ��3D��ת�����������ϵ
                Matrix<T, 3, 1> p_cam = R * Matrix<T, 3, 1>(point3d[0], point3d[1], point3d[2])
                    + Matrix<T, 3, 1>(translation[0], translation[1], translation[2]);

                // ͶӰ��ͼ��ƽ��
                p_cam /= p_cam[2];
                Matrix<T, 3, 1> uv_hom = K.cast<T>() * p_cam;

                // ����в�
                residual[0] = uv_hom[0] - T(observed_uv.x());
                residual[1] = uv_hom[1] - T(observed_uv.y());
				//std::cout << "Residual: " << residual[0] << ", " << residual[1] << std::endl;
                return true;
            }

            const Vector2d observed_uv;
            const Matrix3d K;
        };

        // ���ɲ������ݣ���֮ǰ���뱣��һ�£�
        void generateTestData(vector<Vector3d>& points3d,
            vector<pair<Vector2d, Vector2d>>& matches,
            Matrix3d& K,
            Matrix4d& T1,
            Matrix4d& T2);

        int ceresCheckTcwUV();
	}
}