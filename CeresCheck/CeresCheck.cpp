#include "CeresCheck.h"


namespace OPTIMIZE_LYJ
{
	namespace OPTIMIZE_CERES
	{
		using namespace Eigen;
		using namespace std;

        // 生成测试数据
        void generateTestData(vector<Vector3d>& points3d,
            vector<pair<Vector2d, Vector2d>>& matches,
            Matrix3d& K,
            Matrix4d& T1,
            Matrix4d& T2) {
            // 相机内参
            K << 500, 0, 320,
                0, 500, 240,
                0, 0, 1;

            // 生成相机位姿
            double* T1_fixed = new double[6] {0.01, 0, 0, 1, 2, 3}; // [rx, ry, rz, tx, ty, tz] // 固定第一个相机位姿
            Matrix<double, 3, 3> R;
            AngleAxisToRotationMatrix<double>(T1_fixed, R.data());
            T1 = Matrix4d::Identity();
            T1.block(0, 0, 3, 3) = R;
            T1.block<3, 1>(0, 3) = Vector3d(1, 2, 3);

            T2 = Matrix4d::Identity();
            T2.block<3, 3>(0, 0) = AngleAxisd(0.2, Vector3d::UnitX()).toRotationMatrix()
                * AngleAxisd(0.1, Vector3d::UnitY()).toRotationMatrix();
            T2.block<3, 1>(0, 3) = Vector3d(0.5, -0.2, 0.3);

            // 生成3D点
            const int num_points = 1000;
            points3d.resize(num_points);
            for (int i = 0; i < num_points; ++i) {
                points3d[i] = Vector3d::Random()+ Vector3d(0, 0, 2);
            }

            // 生成匹配点对
            auto project = [&K](const Vector3d& p, const Matrix4d& T) {
                Matrix3d R = T.block<3, 3>(0, 0);
                Vector3d t = T.block<3, 1>(0, 3);
                Vector3d p_cam = R * p + t;
                Vector3d uv = K * p_cam;
                return Vector2d(uv[0] / uv[2], uv[1] / uv[2]);
                };

            matches.clear();
            for (const auto& p : points3d) {
                matches.emplace_back(project(p, T1), project(p, T2));
            }
        }

        int ceresCheckTcwUV() {
            // 生成测试数据
            vector<Vector3d> points3d_true;
            vector<pair<Vector2d, Vector2d>> matches;
            Matrix3d K;
            Matrix4d T1_true, T2_true;
            generateTestData(points3d_true, matches, K, T1_true, T2_true);
            std::cout << T1_true << std::endl;
            std::cout << T2_true << std::endl;
            for (auto& p : points3d_true) {
                std::cout << p(0) << " " << p(1) << " " << p(2) << std::endl;
            }

            // 添加噪声创建初始猜测
            vector<Vector3d> points3d_est = points3d_true;
            for (auto& p : points3d_est) {
                p += Eigen::Vector3d(1, 0.1, 0);
            }

            // 转换T2为旋转向量+平移的初始猜测
            Vector3d angle_axis_est =
                AngleAxisd(Matrix3d(T2_true.block<3, 3>(0, 0))).angle()
                * AngleAxisd(Matrix3d(T2_true.block<3, 3>(0, 0))).axis();
            Vector3d translation_est = T2_true.block<3, 1>(0, 3);// +Vector3d::Random() * 0.05;
            double T2_est[6] = { angle_axis_est[0], angle_axis_est[1], angle_axis_est[2],
                               translation_est[0], translation_est[1], translation_est[2] };

            // 构建优化问题
            ceres::Problem problem;

            // 添加所有残差项
            double* T1_fixed = new double[6] {0.01, 0, 0, 1, 2, 3}; // [rx, ry, rz, tx, ty, tz] // 固定第一个相机位姿
            problem.AddParameterBlock(T1_fixed, 6); // 显式添加参数块
            problem.SetParameterBlockConstant(T1_fixed); // 固定参数
            //problem.AddParameterBlock(T2_est, 6); // 显式添加参数块
            //problem.SetParameterBlockConstant(T2_est); // 固定参数
            //T1_fixed[0] = 0.001;
            for (size_t i = 0; i < matches.size(); ++i) {
                // 第一相机的观测（固定T1）
                {
                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
                            new ReprojectionError(matches[i].first, K));
                    problem.AddResidualBlock(cost_function, nullptr, T1_fixed,
                        points3d_est[i].data());
                    //problem.SetParameterBlockConstant(T1_fixed); // 固定第一个相机
                }

                // 第二相机的观测
                {
                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
                            new ReprojectionError(matches[i].second, K));
                    problem.AddResidualBlock(cost_function, nullptr, T2_est,
                        points3d_est[i].data());
                }
            }

            // 配置优化器
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_SCHUR;
            options.minimizer_progress_to_stdout = true;
            options.max_num_iterations = 20;

            // 运行优化
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.FullReport() << std::endl;

            // 结果验证
            auto evaluateError = [&]() {
                double total_error = 0;
                for (size_t i = 0; i < points3d_est.size(); ++i) {
                    Vector3d p = points3d_est[i];
                    auto fff = ReprojectionError(matches[i].second, K);
                    Vector2d uv;
                    fff(T2_est, p.data(), uv.data());
                    total_error += uv.norm();
                }
                return total_error / points3d_est.size();
                };


            Matrix<double, 3, 3> Rtmp;
            AngleAxisToRotationMatrix<double>(T1_fixed, Rtmp.data());
            Matrix4d T1 = Matrix4d::Identity();
            T1.block(0, 0, 3, 3) = Rtmp;
            T1(0, 3) = T1_fixed[3];
            T1(1, 3) = T1_fixed[4];
            T1(2, 3) = T1_fixed[5];
            AngleAxisToRotationMatrix<double>(T2_est, Rtmp.data());
            Matrix4d T2 = Matrix4d::Identity();
            T2.block(0, 0, 3, 3) = Rtmp;
            T2(0, 3) = T2_est[3];
            T2(1, 3) = T2_est[4];
            T2(2, 3) = T2_est[5];
            std::cout << T1 << std::endl;
            std::cout << T2 << std::endl;
            for (auto& p : points3d_est) {
                std::cout << p(0) << " " << p(1) << " " << p(2) << std::endl;
            }
            std::cout << "\n优化结果验证:"
                << "\n平均重投影误差: " << evaluateError() << " pixels"
                << "\n平移误差: " << (Map<Vector3d>(T2_est + 3) - T2_true.block<3, 1>(0, 3)).norm()
                << " meters"
                << "\n3D点平均误差: " << [&]() {
                double err = 0;
                for (size_t i = 0; i < points3d_est.size(); ++i)
                    err += (points3d_est[i] - points3d_true[i]).norm();
                return err / points3d_est.size();
                }() << " meters" << std::endl;

            return 0;
        }
	}
}