#include <iostream>
#include <vector>

#include <CeresCheck/CeresProblem/CeresProblem.h>
#include <Optimize_LYJ.h>

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <IO/ColmapIO.h>
#include <IO/MeshIO.h>
#include <base/Pose.h>
#include <base/CameraModule.h>


void testColmapOptimize()
{
    // std::string btmPath = "D:/SLAM_LYJ_Packages/SLAM_LYJ_qt/tmp/fuse_unbounded_post.ply";
    // SLAM_LYJ::SLAM_LYJ_MATH::BaseTriMesh btm;
    // SLAM_LYJ::readPLYMesh(btmPath, btm);
    std::string pth = "D:/gsWin/gaussian-splatting/gaussian-splatting/data/mask/sparse/0";
    // std::string imgDir = "D:/gsWin/gaussian-splatting/gaussian-splatting/data/mask/images/";
    COMMON_LYJ::ColmapData colmapData;
    colmapData.readFromColmap(pth);
    std::vector<COMMON_LYJ::ColmapImage>& colmapImages = colmapData.images_;
    std::vector<COMMON_LYJ::ColmapCamera>& colmapCameras = colmapData.cameras_;
    std::vector<COMMON_LYJ::ColmapPoint>& colmapPoints = colmapData.point3Ds_;
    int camSz = colmapData.num_cameras;
    int imgSz = colmapData.num_reg_images;
    int pointSz = colmapData.num_point3Ds;
    std::map<int, COMMON_LYJ::ColmapImage*> imagesPtr;
    std::map<int, COMMON_LYJ::ColmapCamera*> camsPtr;
    std::map<int, COMMON_LYJ::ColmapPoint*> pointsPtr;
    for(int i=0;i<imgSz;++i)
    {
        imagesPtr[colmapImages[i].image_id] = &colmapImages[i];
    }
    for(int i=0;i<camSz;++i)
    {
        camsPtr[colmapCameras[i].camera_id] = &colmapCameras[i];
    }
    for(int i=0;i<pointSz;++i)
    {
        pointsPtr[colmapPoints[i].point3D_id] = &colmapPoints[i];
    }

    std::map<int, Eigen::Matrix3d> ceresKs;
    for(int i=0;i<camSz;++i)
    {
        const auto& camId = colmapCameras[i].camera_id;
        ceresKs[camId].setIdentity();
        auto& K = ceresKs[camId];
        auto camPtr = camsPtr[camId];
        K(0, 0) = camPtr->params[0];
        K(1, 1) = camPtr->params[1];
        K(0, 2) = camPtr->params[2];
        K(1, 2) = camPtr->params[3];
    }
    std::map<int, Eigen::Matrix<double, 7, 1>> ceresPoses;
    std::map<int, Eigen::Matrix<double, 3, 3>> ceresRcws;
    std::map<int, Eigen::Matrix<double, 3, 1>> cerestcws;
    for(int i=0;i<imgSz;++i)
    {
        const auto& imgId = colmapImages[i].image_id;
        ceresPoses[imgId].setZero();
        auto& pose = ceresPoses[imgId];
        auto imgPtr = imagesPtr[imgId];
        pose(0) = imgPtr->qcw.w();
        pose(1) = imgPtr->qcw.x();
        pose(2) = imgPtr->qcw.y();
        pose(3) = imgPtr->qcw.z();
        pose(4) = imgPtr->tcw(0);
        pose(5) = imgPtr->tcw(1);
        pose(6) = imgPtr->tcw(2);
        ceresRcws[imgId] = imgPtr->qcw.toRotationMatrix();
        cerestcws[imgId] = imgPtr->tcw;
    }
    std::unordered_map<int, Eigen::Vector3d> ceresPoints;
    for(int i=0;i<pointSz;++i)
    {
        const auto& pId = colmapPoints[i].point3D_id;
        ceresPoints[pId].setZero();
        auto& point = ceresPoints[pId];
        auto pPtr = pointsPtr[pId];
        point = pPtr->point3D;
    }

    double maxD = 0.01;
    CeresProblem ceresPro;
    for(int i=0;i<imgSz;++i)
    {
        const auto& imgId = colmapImages[i].image_id;
        auto& ceresPose = ceresPoses[imgId];
        if(i == 0)
            ceresPro.addPose3DParameter(ceresPose.data(), true);
        else
            ceresPro.addPose3DParameter(ceresPose.data(), false);
    }
    for(int i=0;i<pointSz;++i)
    {
        const auto& pId = colmapPoints[i].point3D_id;
        auto& ceresPoint = ceresPoints[pId];
        if(ceresPoint(2) > maxD)
            continue;
        ceresPro.addPoint3DParameter(ceresPoint.data(), false);
    }
    for(int i=0;i<pointSz;++i)
    {
        const auto& pId = colmapPoints[i].point3D_id;
        auto& p = ceresPoints[pId];
        if(p(2) > maxD)
            continue;
        const auto& obs = pointsPtr[pId]->track;
        const auto& obSz = pointsPtr[pId]->track_length;
        for(int j=0;j<obSz;++j)
        {
            const auto& imgId = obs[j](0);
            const auto& uvId = obs[j](1);
            auto& ceresPose = ceresPoses[imgId];
            // Eigen::Vector3d Pc = ceresRcws[imgId] * p + cerestcws[imgId];
            // if(Pc(2) > )
            const auto& uv = imagesPtr[imgId]->points2D[uvId];
            const auto& K = ceresKs[imagesPtr[imgId]->camId];
            ceresPro.addUVFactor(uv, K, ceresPose.data(), p.data(), 1);
        }
    }
    ceresPro.solve();

    for(int i=0;i<imgSz;++i)
    {
        const auto& imgId = colmapImages[i].image_id;
        auto& pose = ceresPoses[imgId];
        auto imgPtr = imagesPtr[imgId];
        imgPtr->qcw.w() = pose(0);
        imgPtr->qcw.x() = pose(1);
        imgPtr->qcw.y() = pose(2);
        imgPtr->qcw.z() = pose(3);
        imgPtr->tcw(0) = pose(4);
        imgPtr->tcw(1) = pose(5);
        imgPtr->tcw(2) = pose(6);
    }
    for(int i=0;i<pointSz;++i)
    {
        const auto& pId = colmapPoints[i].point3D_id;
        auto& point = ceresPoints[pId];
        auto pPtr = pointsPtr[pId];
        pPtr->point3D = point;
    }
    std::string pthOut = "D:/gsWin/gaussian-splatting/gaussian-splatting/data/mask/sparse/1";
    colmapData.writeFromColmap(pthOut);
    return;
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
    //OPTIMIZE_LYJ::test_optimize_UV_Pose3d_P3d2();
    //OPTIMIZE_LYJ::ceres_Check_UV_Pose3d_P3d2();
    // OPTIMIZE_LYJ::test_optimize_UV2_Pose3d_Line3d();
    //OPTIMIZE_LYJ::ceres_Check_UV_Pose3d_Line3d();
     //main3();
    testColmapOptimize();
    return 0;
}