#include <iostream>
#include <Optimize_LYJ.h>
#include <Eigen/Core>
#include <Eigen/Eigen>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>

using namespace Eigen;
using namespace std;

// 生成3D点云 ([-1,1]x[-1,1]x[2,5]范围)
vector<Vector3d> generate3DPoints(int num_points) {
    vector<Vector3d> points;
    points.reserve(num_points);

    for (int i = 0; i < num_points; ++i) {
        Vector3d p = Vector3d::Random();
        p = p + Vector3d(0, 0, 2);
        points.push_back(p);
    }
    return points;
}

// 投影3D点到图像平面
Vector2d project(const Vector3d& p3d, const Matrix3d& K, const Matrix4d& T_cam) {
    Matrix3d R = T_cam.block<3, 3>(0, 0);
    Vector3d t = T_cam.block<3, 1>(0, 3);
    Vector3d p_cam = R * p3d + t;
    Vector3d uv_hom = K * p_cam;
    return uv_hom.hnormalized();
}

// 验证重投影误差
void verifyReprojection(const vector<Vector3d>& points3d,
    const vector<pair<Vector2d, Vector2d>>& matches,
    const Matrix3d& K,
    const Matrix4d& T1,
    const Matrix4d& T2)
{
    double max_error = 0;
    for (size_t i = 0; i < points3d.size(); ++i) {
        Vector2d p1 = project(points3d[i], K, T1);
        Vector2d p2 = project(points3d[i], K, T2);

        double e1 = (p1 - matches[i].first).norm();
        double e2 = (p2 - matches[i].second).norm();
        max_error = max({ max_error, e1, e2 });
    }
    cout << "Max reprojection error: " << max_error << " pixels" << endl;
}

int main2() {
    // 相机内参矩阵
    Matrix3d K;
    K << 500, 0, 320,
        0, 500, 240,
        0, 0, 1;

    // 生成相机位姿
    Matrix4d T1 = Matrix4d::Identity(); // 第一个相机位于原点

    Matrix4d T2 = Matrix4d::Identity();
    T2.block<3, 3>(0, 0) = AngleAxisd(0.2, Vector3d::UnitX()).toRotationMatrix() *
        AngleAxisd(0.1, Vector3d::UnitY()).toRotationMatrix();
    T2.block<3, 1>(0, 3) = Vector3d(0.5, -0.2, 0.3);

    // 生成3D点
    const int num_points = 50;
    auto points3d = generate3DPoints(num_points);

    // 生成匹配点对
    vector<pair<Vector2d, Vector2d>> matches;
    for (const auto& p3d : points3d) {
        matches.emplace_back(
            project(p3d, K, T1),
            project(p3d, K, T2)
        );
    }

    // 验证数据正确性
    verifyReprojection(points3d, matches, K, T1, T2);

    // 打印部分数据示例
    cout << "\nCamera Intrinsic Matrix:\n" << K << endl;
    cout << "\nCamera Pose 1:\n" << T1 << endl;
    cout << "\nCamera Pose 2:\n" << T2 << endl;
    for (size_t i = 0; i < points3d.size(); ++i) {
        cout << "\ncnt: " << i << endl;
        cout << "\nSample 3D Point: " << points3d[i].transpose() << endl;
        cout << "Projection in Camera1: " << matches[i].first.transpose() << endl;
        cout << "Projection in Camera2: " << matches[i].second.transpose() << endl;
    }

    return 0;
}

Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> testEigenMap(double *_data)
{
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> mat(_data, 2, 3);
    // std::cout << "eigen map: " << std::endl << mat << std::endl;
    return mat;
}

int main(int argc, char *argv[])
{
    //// double data[6] = { 1, 2, 3, 4, 5, 6 };
    //// auto mat = testEigenMap(data);
    //// std::cout << "eigen map2: " << std::endl << mat << std::endl;
    //main2();
    //return 0;
    std::cout << "Optimize Version: " << OPTIMIZE_LYJ::optimize_version() << std::endl;
    // OPTIMIZE_LYJ::test_optimize_P3d_P3d();
    //OPTIMIZE_LYJ::test_optimize_Pose3d_Pose3d();
     //OPTIMIZE_LYJ::test_optimize_RelPose3d_Pose3d_Pose3d();
    //OPTIMIZE_LYJ::test_optimize_Plane_P();
    OPTIMIZE_LYJ::test_optimize_UV_Pose3d_P3d2();
    //OPTIMIZE_LYJ::ceres_Check_UV_Pose3d_P3d2();
    // main3();
    return 0;
}