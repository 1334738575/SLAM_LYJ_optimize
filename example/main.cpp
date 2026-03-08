#include <iostream>
#include <vector>

#include <CeresCheck/CeresProblem/CeresProblem.h>
#include <Optimize_LYJ.h>
#include <Optimizer/optimizer.h>
#include <Factor/Factor.h>
#include <Variable/Variable.h>

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <IO/ColmapIO.h>
#include <IO/MeshIO.h>
#include <base/Pose.h>
#include <base/CameraModule.h>
#include <STLPlus/include/file_system.h>
#include <opencv2/opencv.hpp>


void testColmapOptimize()
{
    // std::string btmPath = "D:/SLAM_LYJ_Packages/SLAM_LYJ_qt/tmp/fuse_unbounded_post.ply";
    // COMMON_LYJ::BaseTriMesh btm;
    // COMMON_LYJ::readPLYMesh(btmPath, btm);
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
        //if(ceresPoint(2) > maxD)
        //    continue;
        ceresPro.addPoint3DParameter(ceresPoint.data(), false);
    }
    for(int i=0;i<pointSz;++i)
    {
        const auto& pId = colmapPoints[i].point3D_id;
        auto& p = ceresPoints[pId];
        //if(p(2) > maxD)
        //    continue;
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
    stlplus::folder_create(pthOut);
    colmapData.writeFromColmap(pthOut);
    return;
}

void showMatch(COMMON_LYJ::ColmapData& _colmapData, int _ind1, int _ind2, std::string _imgDir)
{
    std::vector<COMMON_LYJ::ColmapImage>& colmapImages = _colmapData.images_;
    std::vector<COMMON_LYJ::ColmapCamera>& colmapCameras = _colmapData.cameras_;
    std::vector<COMMON_LYJ::ColmapPoint>& colmapPoints = _colmapData.point3Ds_;
}
void testColmapOptimize2()
{
    std::string imageDir = "D:/tmp/colmapData/mask/dense/images/";
    using namespace OPTIMIZE_LYJ;
    std::string dataPath = "D:/tmp/colmapData/mask/dense/sparse/";
    std::string pth = dataPath + "0";
    COMMON_LYJ::ColmapData colmapData;
    colmapData.readFromColmap(pth);
    std::vector<COMMON_LYJ::ColmapImage>& colmapImages = colmapData.images_;
    std::vector<COMMON_LYJ::ColmapCamera>& colmapCameras = colmapData.cameras_;
    std::vector<COMMON_LYJ::ColmapPoint>& colmapPoints = colmapData.point3Ds_;
    int camSz = colmapData.num_cameras;
    int imgSz = colmapData.num_reg_images;
    int pointSz = colmapData.num_point3Ds;
    COMMON_LYJ::PinholeCamera camera(colmapCameras[0].width, colmapCameras[0].height, colmapCameras[0].params);
    //Eigen::Matrix3d K = camera.getK();
    std::vector<double> K = colmapCameras[0].params;
    std::map<int, int> imgId2Ind;
    std::map<int, COMMON_LYJ::Pose3D> allTcws;
    std::map<int, std::vector<Eigen::Vector2d>*> allKps;
    std::map<int, int> pointId2Ind;
    std::map<int, COMMON_LYJ::ColmapPoint*> pointsPtr;
    cv::Mat im1;
    cv::Mat im2;
    cv::Mat im3;
    int dstId1 = 1;
    int dstId2 = 30;
    int dstId3 = 48;
    for (int i = 0; i < imgSz; ++i)
    {
        int imgId = colmapImages[i].image_id;
        //if (imgId != dstId1 && imgId != dstId2)
        //    continue;
        const auto& qcw = colmapImages[i].qcw;
        allTcws[imgId].setR(qcw.toRotationMatrix());
        allTcws[imgId].sett(colmapImages[i].tcw);
        allKps[imgId] = &colmapImages[i].points2D;
        imgId2Ind[imgId] = i;
        std::string imagePath = imageDir + colmapImages[i].imgName;
        if(imgId == dstId1)
            im1 = cv::imread(imagePath);
        else if(imgId == dstId2)
            im2 = cv::imread(imagePath);
        else if (imgId == dstId3)
            im3 = cv::imread(imagePath);
        //std::cout << imagePath << std::endl;
    }
    std::vector<Eigen::Vector2i> mmms;
    for (int i = 0; i < pointSz; ++i)
    {
        const auto& obs = colmapPoints[i].track;
        //bool need = false;
        //Eigen::Vector2i mmm;
        //for (const auto& id2ind : imgId2Ind)
        //{
        //    bool found = false;
        //    for (int j = 0; j < colmapPoints[i].track_length; ++j)
        //    {
        //        int imgId = obs[j](0);
        //        if (imgId == id2ind.first)
        //        {
        //            found = true;
        //            if (imgId == dstId1)
        //                mmm(0) = obs[j](1);
        //            else if (imgId == dstId2)
        //                mmm(1) = obs[j](1);
        //            break;
        //        }
        //    }
        //    if (!found)
        //    {
        //        need = false;
        //        break;
        //    }
        //    need = true;
        //}
        //if (!need)
        //    continue;
        //mmms.push_back(mmm);
        pointsPtr[colmapPoints[i].point3D_id] = &colmapPoints[i];
        pointId2Ind[colmapPoints[i].point3D_id] = i;
    }
    //cv::Mat m12(im1.rows, im1.cols+im2.cols, CV_8UC3);
    //cv::Rect rect1(0, 0, im1.cols, im1.rows);
    //cv::Rect rect2(im1.cols, 0, im2.cols, im2.rows);
    //im1.copyTo(m12(rect1));
    //im2.copyTo(m12(rect2));
    //for (int i = 0; i < mmms.size(); ++i)
    //{
    //    Eigen::Vector2d uv1 = allKps[dstId1]->at(mmms[i](0));
    //    Eigen::Vector2d uv2 = allKps[dstId2]->at(mmms[i](1));
    //    cv::line(m12, cv::Point(uv1(0), uv1(1)), cv::Point(uv2(0) + im1.cols, uv2(1)), cv::Scalar(255, 0,0));
    //}
    //cv::imshow("222", m12);
    //cv::waitKey(0);
    //imgSz = 2;

    //OPTIMIZE_LYJ::OptimizerLargeSparse optimizer;
    OPTIMIZE_LYJ::OptimizeLargeSRBA optimizer;//观测多的时候不建议使用
    std::vector<std::shared_ptr<OPTIMIZE_LYJ::OptVarPose3d>> varTcws;
    std::map<int, int> varTId2Ori;
    std::map<int, int> ori2VarTInd;
    std::map<int, std::shared_ptr<OPTIMIZE_LYJ::OptVarPoint3d>> varPoints;//优化地图点原索引对应变量
    std::map<int, int> varPId2Ori;//变量id对应原地图点id
    std::map<int, int> ori2VarPInd;//优化地图点索引对应地图点索引
    int varId = 0;
    Eigen::Matrix<double, 3, 4> Tm;
    for (const auto& Tcw : allTcws)
    {
        std::shared_ptr<OptVarPose3d> p = std::make_shared<OptVarPose3d>(varId);
        Tcw.second.getMatrix34d(Tm);
        p->setData(Tm.data());
        if (varId == 0)
            p->setFixed(true);
        varTId2Ori[varId] = Tcw.first;
        ori2VarTInd[Tcw.first] = varId;
        optimizer.addVariable(p);
        varTcws.push_back(p);
        ++varId;
    }
    for (const auto& point : pointsPtr)
    {
        std::shared_ptr<OptVarPoint3d> p = std::make_shared<OptVarPoint3d>(varId);
        p->setData(point.second->point3D.data());
        varPId2Ori[varId] = point.first;
        ori2VarPInd[point.first] = varId - imgSz;
        optimizer.addVariable(p);
        varPoints[varId - imgSz] = p;
        ++varId;
    }
    
    auto funcGenerateScaleFactor = [&](double _ob, uint64_t _vId1, uint64_t _vId2, uint64_t& _fId)
        {
            std::shared_ptr<OptFactorAbr<double>> factorPtr = std::make_shared<OptFactorScale_Pose3d_Point3d>(_fId);
            OptFactorScale_Pose3d_Point3d* factor = dynamic_cast<OptFactorScale_Pose3d_Point3d*>(factorPtr.get());
            factor->setObs(_ob);
            std::vector<uint64_t> vIds;
            vIds.push_back(_vId1);
            vIds.push_back(_vId2);
            optimizer.addFactor(factorPtr, vIds);
            ++_fId;
        };
    auto funcGenerateFactor = [&](Eigen::Vector2d& _ob, uint64_t _vId1, uint64_t _vId2, uint64_t& _fId)
        {
            std::shared_ptr<OptFactorAbr<double>> factorPtr = std::make_shared<OptFactorUV_Pose3d_Point3d>(_fId);
            OptFactorUV_Pose3d_Point3d* factor = dynamic_cast<OptFactorUV_Pose3d_Point3d*>(factorPtr.get());
            factor->setObs(_ob.data(), K.data());
            std::vector<uint64_t> vIds;
            vIds.push_back(_vId1);
            vIds.push_back(_vId2);
            optimizer.addFactor(factorPtr, vIds);
            ++_fId;
        };
    uint64_t fId = 0;
    bool sAdded = false;
    for (int i = 0; i < pointSz; ++i)
    {
        const auto& pId = colmapPoints[i].point3D_id;
        if (ori2VarPInd.count(pId) == 0)
            continue;
        std::set<int> imgIdSet;
        uint64_t varPId = ori2VarPInd[pId];
        const auto& obs = pointsPtr[pId]->track;
        const auto& obSz = pointsPtr[pId]->track_length;
        for (int j = 0; j < obSz; ++j)
        {
            const auto& imgId = obs[j](0);
            const auto& uvId = obs[j](1);
            if (imgId2Ind.count(imgId) == 0 || imgIdSet.count(imgId) == 1)
                continue;
            imgIdSet.insert(imgId);
            uint64_t varTId = ori2VarTInd[imgId];
            if (!sAdded)
            {
                const auto& Tcw = allTcws[imgId];
                const auto& Pw = colmapPoints[i].point3D;
                Eigen::Vector3d Pc = Tcw * Pw;
                double ss = Pc.squaredNorm();
                funcGenerateScaleFactor(ss, varTId, varPId + imgSz, fId);
                sAdded = true;
            }
            Eigen::Vector2d& uv = allKps[imgId]->at(uvId);
            funcGenerateFactor(uv, varTId, varPId + imgSz, fId);
        }
    }
    optimizer.run();

    for (int i = 0; i < varTcws.size(); ++i)
    {
        int vId = varTcws[i]->getId();
        int ori = varTId2Ori[vId];
        int ind = imgId2Ind[ori];
        Eigen::Matrix<double, 3, 4> Tcw = varTcws[i]->getEigen();
        Eigen::Matrix3d Rcw = Tcw.block(0, 0, 3, 3);
        colmapImages[ind].qcw = Eigen::Quaterniond(Rcw);
        colmapImages[ind].tcw = Tcw.block(0, 3, 3, 1);
    }
    for (int i = 0; i < varPoints.size(); ++i)
    {
        int vId = varPoints[i]->getId();
        int ori = varPId2Ori[vId];
        int ind = pointId2Ind[ori];
        colmapPoints[ind].point3D = varPoints[i]->getEigen();
    }
    std::string pthOut = dataPath + "1";
    stlplus::folder_create(pthOut);
    colmapData.writeFromColmap(pthOut);
    return;
}

void testColmapOptimize3()
{
    std::string imageDir = "D:/tmp/colmapData/mask2/dense/images/";
    using namespace OPTIMIZE_LYJ;
    std::string dataPath = "D:/tmp/colmapData/mask2/dense/sparse/";
    std::string pth = dataPath + "0";
    COMMON_LYJ::ColmapData colmapData;
    colmapData.readFromColmap(pth);
    std::vector<COMMON_LYJ::ColmapImage>& colmapImages = colmapData.images_;
    std::vector<COMMON_LYJ::ColmapCamera>& colmapCameras = colmapData.cameras_;
    std::vector<COMMON_LYJ::ColmapPoint>& colmapPoints = colmapData.point3Ds_;
    int camSz = colmapData.num_cameras;
    int imgSz = colmapData.num_reg_images;
    int pointSz = colmapData.num_point3Ds;
    COMMON_LYJ::PinholeCamera camera(colmapCameras[0].width, colmapCameras[0].height, colmapCameras[0].params);
    std::vector<double> K = colmapCameras[0].params;


    struct OptImage
    {
        COMMON_LYJ::Pose3D Tcw;
        std::vector<Eigen::Vector2d>* kps;
        COMMON_LYJ::PinholeCamera cam;
    };
    struct ImageMatch
    {
        int img1;
        int img2;
        std::vector<Eigen::Vector2i> matches;
        std::vector<Eigen::Vector3d> points;
    };
    std::set<int> imgIds;
    //imgIds.insert(1);
    //imgIds.insert(30);
    for (int i = 1; i <= imgSz; i+=5)
        imgIds.insert(i);
    std::vector<OptImage> optImgs;
    std::map<int, int> imgId2Ind;
    std::vector<ImageMatch> imgMthes;
    for (int i = 0; i < imgSz; ++i)
    {
        auto& imgId = colmapImages[i].image_id;
        if (imgIds.count(imgId) == 0)
            continue;
        const auto& qcw = colmapImages[i].qcw;
        OptImage iii;
        iii.Tcw.setR(qcw.toRotationMatrix());
        iii.Tcw.sett(colmapImages[i].tcw);
        iii.kps = &colmapImages[i].points2D;
        imgId2Ind[imgId] = optImgs.size();
        optImgs.push_back(iii);
    }
    std::set<int64_t> mPairs;
    std::map<int64_t, int> pair2ind;
    for (int i = 0; i < pointSz; ++i)
    {
        const auto& obs = colmapPoints[i].track;
        for (int j = 0; j < colmapPoints[i].track_length; ++j)
        {
            const auto& img1 = obs[j](0);
            if (imgIds.count(img1) == 0)
                continue;
            for (int k = j+1; k < colmapPoints[i].track_length; ++k)
            {
                const auto& img2 = obs[k](0);
                if (img1 == img2)
                    continue;
                if (imgIds.count(img2) == 0)
                    continue;
                int64_t p = COMMON_LYJ::imagePair2Int64(img1, img2);
                if (mPairs.count(p))
                    continue;
                pair2ind[p] = mPairs.size();
                mPairs.insert(p);
            }
        }
    }
    imgMthes.resize(mPairs.size());
    for (int i = 0; i < mPairs.size(); ++i)
    {
        imgMthes[i].points.reserve(pointSz);
        imgMthes[i].matches.reserve(pointSz);
    }
    std::vector<Eigen::Vector3f> PwsTmp;
    for (int i = 0; i < pointSz; ++i)
    {
        const auto& obs = colmapPoints[i].track;
        auto& point = colmapPoints[i].point3D;
        for (int j = 0; j < colmapPoints[i].track_length; ++j)
        {
            const auto& img1 = obs[j](0);
            if (imgIds.count(img1) == 0)
                continue;
            const auto& uvId1 = obs[j](1);
            for (int k = j + 1; k < colmapPoints[i].track_length; ++k)
            {
                const auto& img2 = obs[k](0);
                const auto& uvId2 = obs[k](1);
                if (img1 == img2)
                    continue;
                if (imgIds.count(img2) == 0)
                    continue;
                int64_t p = COMMON_LYJ::imagePair2Int64(img1, img2);
                const auto& ind = pair2ind[p];
                if (img1 < img2)
                {
                    imgMthes[ind].img1 = imgId2Ind[img1];
                    imgMthes[ind].img2 = imgId2Ind[img2];
                    imgMthes[ind].matches.push_back(Eigen::Vector2i(uvId1, uvId2));
                }
                else
                {
                    imgMthes[ind].img1 = imgId2Ind[img2];
                    imgMthes[ind].img2 = imgId2Ind[img1];
                    imgMthes[ind].matches.push_back(Eigen::Vector2i(uvId2, uvId1));
                }
                imgMthes[ind].points.push_back(point);
                PwsTmp.push_back(point.cast<float>());;
            }
        }
    }
    COMMON_LYJ::BaseTriMesh btmTmp;
    btmTmp.setVertexs(PwsTmp);
    COMMON_LYJ::writePLYMesh("D:/tmp/OptTmp.ply", btmTmp);


    int optImgSz = optImgs.size();
    int imgMthSz = imgMthes.size();
    //OPTIMIZE_LYJ::OptimizerLargeSparse optimizer;
    OPTIMIZE_LYJ::OptimizeLargeSRBA optimizer;//观测多的时候不建议使用
    std::vector<std::shared_ptr<OPTIMIZE_LYJ::OptVarPose3d>> varTcws;
    std::map<int, std::shared_ptr<OPTIMIZE_LYJ::OptVarPoint3d>> varPoints;//优化地图点原索引对应变量
    int varId = 0;
    Eigen::Matrix<double, 3, 4> Tm;
    for (int i=0;i<optImgSz;++i)
    {
        auto& optImg = optImgs[i];
        std::shared_ptr<OptVarPose3d> p = std::make_shared<OptVarPose3d>(varId);
        optImg.Tcw.getMatrix34d(Tm);
        p->setData(Tm.data());
        if (varId == 0)
            p->setFixed(true);
        optimizer.addVariable(p);
        varTcws.push_back(p);
        ++varId;
    }
    for (int i = 0; i < imgMthSz; ++i)
    {
        auto& imgMth = imgMthes[i];
        auto& mths = imgMthes[i].matches;
        int mSz = mths.size();
        auto& points = imgMth.points;
        for (int j = 0; j < mSz; ++j)
        {
            std::shared_ptr<OptVarPoint3d> p = std::make_shared<OptVarPoint3d>(varId);
            p->setData(points[j].data());
            optimizer.addVariable(p);
            varPoints[varId - optImgSz] = p;
            ++varId;
        }
    }

    auto funcGenerateScaleFactor = [&](double _ob, uint64_t _vId1, uint64_t _vId2, uint64_t& _fId)
        {
            std::shared_ptr<OptFactorAbr<double>> factorPtr = std::make_shared<OptFactorScale_Pose3d_Point3d>(_fId);
            OptFactorScale_Pose3d_Point3d* factor = dynamic_cast<OptFactorScale_Pose3d_Point3d*>(factorPtr.get());
            factor->setObs(_ob);
            std::vector<uint64_t> vIds;
            vIds.push_back(_vId1);
            vIds.push_back(_vId2);
            optimizer.addFactor(factorPtr, vIds);
            ++_fId;
        };
    auto funcGenerateFactor = [&](Eigen::Vector2d& _ob, uint64_t _vId1, uint64_t _vId2, uint64_t& _fId)
        {
            std::shared_ptr<OptFactorAbr<double>> factorPtr = std::make_shared<OptFactorUV_Pose3d_Point3d>(_fId);
            OptFactorUV_Pose3d_Point3d* factor = dynamic_cast<OptFactorUV_Pose3d_Point3d*>(factorPtr.get());
            factor->setObs(_ob.data(), K.data());
            std::vector<uint64_t> vIds;
            vIds.push_back(_vId1);
            vIds.push_back(_vId2);
            optimizer.addFactor(factorPtr, vIds);
            ++_fId;
        };
    uint64_t fId = 0;
    bool sAdded = false;
    int PId = 0;
    std::vector<Eigen::Vector3f> PwsTmp2;
    for (int i = 0; i < imgMthSz; ++i)
    {
        auto& imgMth = imgMthes[i];
        auto& mths = imgMthes[i].matches;
        int mSz = mths.size();
        auto& points = imgMth.points;
        const int& imgId1 = imgMth.img1;
        const int& imgId2 = imgMth.img2;
        for (int j = 0; j < mSz; ++j)
        {
            auto& m = mths[j];
            auto& uv1 = optImgs[imgId1].kps->at(m(0));
            auto& uv2 = optImgs[imgId2].kps->at(m(1));
            if (!sAdded)
            {
                const auto& Tcw = optImgs[imgId1].Tcw;
                const auto& Pw = points[j];
                Eigen::Vector3d Pc = Tcw * Pw;
                double ss = Pc.squaredNorm();
                funcGenerateScaleFactor(ss, imgId1, PId + optImgSz, fId);
                sAdded = true;
            }
            funcGenerateFactor(uv1, imgId1, PId + optImgSz, fId);
            funcGenerateFactor(uv2, imgId2, PId + optImgSz, fId);
            ++PId;
            PwsTmp2.push_back(points[j].cast<float>());;
        }
    }
    COMMON_LYJ::BaseTriMesh btmTmp2;
    btmTmp2.setVertexs(PwsTmp2);
    COMMON_LYJ::writePLYMesh("D:/tmp/OptTmp2.ply", btmTmp2);


    std::vector<Eigen::Vector3f> PwsBf(varId - optImgSz);
    for (int i = 0; i < varPoints.size(); ++i)
    {
        int vId = varPoints[i]->getId();
        Eigen::Vector3d p = varPoints[i]->getEigen();
        Eigen::Vector3f pf = p.cast<float>();
        PwsBf[i] = pf;
    }
    COMMON_LYJ::BaseTriMesh btmBf;
    btmBf.setVertexs(PwsBf);
    COMMON_LYJ::writePLYMesh("D:/tmp/OptBf.ply", btmBf);

    optimizer.run();

    for (int i = 0; i < varTcws.size(); ++i)
    {
        Eigen::Matrix<double, 3, 4> Tcw = varTcws[i]->getEigen();
        optImgs[i].Tcw = COMMON_LYJ::Pose3D(Tcw);
    }
    std::vector<Eigen::Vector3f> PwsAf(varId - optImgSz, Eigen::Vector3f(0,0,0));
    for (int i = 0; i < varPoints.size(); ++i)
    {
        int vId = varPoints[i]->getId();
        Eigen::Vector3d p = varPoints[i]->getEigen();
        Eigen::Vector3f pf = p.cast<float>();
        if (pf.norm() > 30)
            continue;
        PwsAf[i] = pf;
        if (std::isnan(PwsAf.back().norm()))
            std::cout << "1111" << std::endl;
    }
    COMMON_LYJ::BaseTriMesh btmAf;
    btmAf.setVertexs(PwsAf);
    COMMON_LYJ::writePLYMesh("D:/tmp/OptAf.ply", btmAf);
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
    //testColmapOptimize();
    //testColmapOptimize2();
    testColmapOptimize3();
    return 0;
}