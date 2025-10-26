#include "OPTIMIZE_LYJ.h"
#include "Variable/Variable.h"
#include "Factor/Factor.h"
#include "Optimizer/Optimizer.h"
#include "CeresCheck/CeresCheck.h"
#include <functional>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <iostream>
#include <fstream>

namespace OPTIMIZE_LYJ
{
	OPTIMIZE_LYJ_API int optimize_version()
	{
		return 1;
	}
	OPTIMIZE_LYJ_API void test_optimize_P3d_P3d()
	{
		////双重释放，报错
		//{
		//	OptVarPoint3d var(0);
		//	Eigen::Vector3d data(1, 2, 3);
		//	var.setData(data.data());
		//	std::shared_ptr<OptVarAbr<double>> varSPtr = nullptr;
		//	varSPtr.reset(&var);
		//}

		if (false)
		{
			OptVarPoint3d var(0);
			Eigen::Vector3d data(1, 2, 3);
			var.setData(data.data());
			OptFactorP3d_P3d factor(0);
			Eigen::Vector3d obs(0, 0, 0);
			factor.setObs(obs.data());

			Eigen::Vector3d err;
			Eigen::Matrix3d jac;
			double *jacPtr = jac.data();
			OptVarAbr<double> *varPtr = &var;
			factor.calculateErrAndJac(err.data(), &jacPtr, 1, &varPtr);
			std::cout << "err: " << std::endl
					  << err << std::endl;
			std::cout << "jac: " << std::endl
					  << jac << std::endl;
		}

		std::shared_ptr<OptVarAbr<double>> varSPtr = std::make_shared<OptVarPoint3d>(0);
		Eigen::Vector3d data(1, 2, 3);
		varSPtr->setData(data.data());
		std::shared_ptr<OptFactorAbr<double>> factorSPtr = std::make_shared<OptFactorP3d_P3d>(0);
		Eigen::Vector3d obs(0, 0, 0);
		OptFactorP3d_P3d *factor = dynamic_cast<OptFactorP3d_P3d *>(factorSPtr.get());
		factor->setObs(obs.data());
		std::vector<uint64_t> vIds;
		vIds.push_back(0);
		{
			// OptimizerSmalld optimizer;
			OptimizerLargeSparse optimizer;
			optimizer.addVariable(varSPtr);
			optimizer.addFactor(factorSPtr, vIds);
			optimizer.run();
		}
		return;
	}
	OPTIMIZE_LYJ_API void test_optimize_Pose3d_Pose3d()
	{
		std::shared_ptr<OptVarAbr<double>> varSPtr = std::make_shared<OptVarPose3d>(0);
		Eigen::Matrix<double, 3, 4> data;
		data.setZero();
		data.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
		double r = 3.14 / 8;
		data(0, 0) = std::cos(r);
		data(0, 1) = -1 * std::sin(r);
		data(1, 0) = std::sin(r);
		data(1, 1) = std::cos(r);
		data(0, 3) = 1.0;
		data(1, 3) = 10.0;
		data(2, 3) = 222.0;
		varSPtr->setData(data.data());
		std::shared_ptr<OptFactorAbr<double>> factorSPtr = std::make_shared<OptFactorPose3d_Pose3d>(0);
		Eigen::Matrix<double, 3, 4> obs;
		obs.setZero();
		obs.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
		OptFactorPose3d_Pose3d *factor = dynamic_cast<OptFactorPose3d_Pose3d *>(factorSPtr.get());
		factor->setObs(obs.data());
		std::vector<uint64_t> vIds;
		vIds.push_back(varSPtr->getId());
		{
			OptimizerSmalld optimizer;
			// OptimizerLargeSparse optimizer;
			optimizer.addVariable(varSPtr);
			optimizer.addFactor(factorSPtr, vIds);
			optimizer.run();
		}
		return;
	}
	OPTIMIZE_LYJ_API void test_optimize_RelPose3d_Pose3d_Pose3d()
	{
		// Tw1
		std::shared_ptr<OptVarAbr<double>> varPtr1 = std::make_shared<OptVarPose3Eulard>(0);
		Eigen::Matrix<double, 3, 4> data;
		data.setZero();
		data.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
		double r = 3.14 / 8;
		data(0, 0) = std::cos(r);
		data(0, 1) = -1 * std::sin(r);
		data(1, 0) = std::sin(r);
		data(1, 1) = std::cos(r);
		data(0, 3) = 1.0;
		data(1, 3) = 10.0;
		data(2, 3) = 222.0;
		varPtr1->setData(data.data());
		varPtr1->setFixed(true);
		// Tw2
		std::shared_ptr<OptVarAbr<double>> varPtr2 = std::make_shared<OptVarPose3Eulard>(1);
		Eigen::Matrix<double, 3, 4> data2;
		data2.setZero();
		data2.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
		double r2 = 3.14 / 9;
		data2(0, 0) = std::cos(r2);
		data2(0, 1) = -1 * std::sin(r2);
		data2(1, 0) = std::sin(r2);
		data2(1, 1) = std::cos(r2);
		data2(0, 3) = 2.0;
		data2(1, 3) = 20.0;
		data2(2, 3) = 122.0;
		varPtr2->setData(data2.data());

		// T12
		std::shared_ptr<OptFactorAbr<double>> factorPtr = std::make_shared<OptFactorRelPose3d_Pose3d_Pose3d>(0);
		Eigen::Matrix<double, 3, 4> obs = OPTIMIZE_BASE::relPose(data, data2);
		double r3 = r2 - r2;
		obs(0, 0) = std::cos(r3);
		obs(0, 1) = -1 * std::sin(r3);
		obs(1, 0) = std::sin(r3);
		obs(1, 1) = std::cos(r3);
		obs(0, 3) += 2.0;
		obs(1, 3) += 20.0;
		obs(2, 3) += 22.0;
		OptFactorRelPose3d_Pose3d_Pose3d *factor = dynamic_cast<OptFactorRelPose3d_Pose3d_Pose3d *>(factorPtr.get());
		factor->setObs(obs.data());
		std::vector<uint64_t> vIds;
		vIds.push_back(varPtr1->getId());
		vIds.push_back(varPtr2->getId());

		OptVarPose3Eulard *v1 = dynamic_cast<OptVarPose3Eulard *>(varPtr1.get());
		OptVarPose3Eulard *v2 = dynamic_cast<OptVarPose3Eulard *>(varPtr2.get());
		std::cout << "init: " << std::endl
				  << "Tw1: " << *v1 << std::endl
				  << "Tw2: " << *v2 << std::endl
				  << "T12: " << *factor << std::endl;
		std::cout << "real T12: " << OPTIMIZE_BASE::relPose(varPtr1->getData(), varPtr2->getData()) << std::endl
				  << std::endl
				  << std::endl;
		// OptimizerSmalld optimizer;
		OptimizerLargeSparse optimizer;
		optimizer.addVariable(varPtr1);
		optimizer.addVariable(varPtr2);
		optimizer.addFactor(factorPtr, vIds);
		optimizer.run();

		std::cout << std::endl
				  << std::endl
				  << "final: " << std::endl
				  << "Tw1: " << *v1 << std::endl
				  << "Tw2: " << *v2 << std::endl
				  << "T12: " << *factor << std::endl;
		std::cout << "real T12: " << OPTIMIZE_BASE::relPose(varPtr1->getData(), varPtr2->getData()) << std::endl;
		return;
	}
	OPTIMIZE_LYJ_API void test_optimize_Plane_P()
	{
		Eigen::Vector3d Pw(1, 2, 3);
		Eigen::Vector3d P(4, 9, 3);
		Eigen::Vector3d nw(1, 2, 4);
		nw.normalize();
		Eigen::Vector4d planew;
		planew.head(3) = nw;
		planew(3) = -P.dot(nw);

		std::shared_ptr<OptVarAbr<double>> varPtr1 = std::make_shared<OptVarPoint3d>(0);
		varPtr1->setData(Pw.data());
		std::shared_ptr<OptFactorAbr<double>> factorPtr = std::make_shared<OptFactorPlane_Point3d>(0);
		OptFactorPlane_Point3d *factor = dynamic_cast<OptFactorPlane_Point3d *>(factorPtr.get());
		factor->setObs(planew.data());
		std::vector<uint64_t> vIds;
		vIds.push_back(varPtr1->getId());

		OptimizerSmalld optimizer;
		// OptimizerLargeSparse optimizer;
		optimizer.addVariable(varPtr1);
		optimizer.addFactor(factorPtr, vIds);
		optimizer.run();
		return;
	}
	OPTIMIZE_LYJ_API void test_optimize_UV_Pose3d_P3d()
	{
		Eigen::Matrix3d tK;
		tK.setIdentity();
		Eigen::Matrix<double, 3, 4> tTcw;
		tTcw.setZero();
		tTcw.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
		tTcw.block(0, 3, 3, 1) = Eigen::Vector3d(0, 2, 0);
		Eigen::Matrix<double, 3, 4> tTwc = OPTIMIZE_BASE::invPose(tTcw);
		Eigen::Matrix<double, 3, 4> tTcw2;
		tTcw2.setZero();
		tTcw2.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
		tTcw2.block(0, 3, 3, 1) = Eigen::Vector3d(0, -2, 0);
		Eigen::Matrix<double, 3, 4> tTwc2 = OPTIMIZE_BASE::invPose(tTcw2);
		Eigen::Vector3d tPw1(0, 0, 1);
		Eigen::Vector3d tPw2(0, 1, 1);
		Eigen::Vector3d tPw3(1, 1, 1);
		Eigen::Vector3d tPw4(1, 0, 1);
		Eigen::Vector3d tPw5(0, 0, 2);
		Eigen::Vector3d tPw6(0, 1, 2);
		Eigen::Vector3d tPw7(1, 1, 2);
		Eigen::Vector3d tPw8(1, 0, 2);

		uint64_t vId = 0;
		uint64_t fId = 0;
		std::vector<std::shared_ptr<OptVarAbr<double>>> Twcs;
		std::vector<std::shared_ptr<OptVarAbr<double>>> Pws;
		OptimizerSmalld optimizer;
		// OptimizerLargeSparse optimizer;

		auto funcGeneratePointVertex = [&](Eigen::Vector3d &_Pw, uint64_t &_vId, bool _fix = false)
		{
			std::shared_ptr<OptVarAbr<double>> varPtr = std::make_shared<OptVarPoint3d>(_vId);
			varPtr->setData(_Pw.data());
			varPtr->setFixed(_fix);
			optimizer.addVariable(varPtr);
			Pws.push_back(varPtr);
			++_vId;
		};
		auto funcGeneratePoseVertex = [&](Eigen::Matrix<double, 3, 4> &_Twc, uint64_t &_vId, bool _fix = false)
		{
			std::shared_ptr<OptVarAbr<double>> varPtr = std::make_shared<OptVarPose3d>(_vId);
			varPtr->setData(_Twc.data());
			varPtr->setFixed(_fix);
			optimizer.addVariable(varPtr);
			Twcs.push_back(varPtr);
			++_vId;
		};
		Eigen::Matrix<double, 3, 4> Twc = tTwc;
		Eigen::Matrix<double, 3, 4> Twc2 = tTwc2;
		// Twc2.setZero();
		Twc2.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
		double r = 3.14 / 8;
		Twc2(0, 0) = std::cos(r);
		Twc2(0, 1) = -1 * std::sin(r);
		Twc2(1, 0) = std::sin(r);
		Twc2(1, 1) = std::cos(r);
		Twc2(0, 3) = -1.0;
		Twc2(1, 3) = 0.0;
		Twc2(2, 3) = 0.0;
		funcGeneratePoseVertex(Twc, vId, true);
		funcGeneratePoseVertex(Twc2, vId);
		Eigen::Vector3d Pw1(tPw1[0], tPw1[1], tPw1[2]);
		Eigen::Vector3d Pw2(tPw2[0], tPw2[1], tPw2[2]);
		Eigen::Vector3d Pw3(tPw3[0], tPw3[1], tPw3[2]);
		Eigen::Vector3d Pw4(tPw4[0], tPw4[1], tPw4[2]);
		Eigen::Vector3d Pw5(tPw5[0], tPw5[1], tPw5[2]);
		Eigen::Vector3d Pw6(tPw6[0], tPw6[1], tPw6[2]);
		Eigen::Vector3d Pw7(tPw7[0], tPw7[1], tPw7[2]);
		Eigen::Vector3d Pw8(tPw8[0], tPw8[1], tPw8[2]);
		funcGeneratePointVertex(Pw1, vId, true);
		funcGeneratePointVertex(Pw2, vId, true);
		funcGeneratePointVertex(Pw3, vId, true);
		funcGeneratePointVertex(Pw4, vId, true);
		funcGeneratePointVertex(Pw5, vId, true);
		funcGeneratePointVertex(Pw6, vId, true);
		funcGeneratePointVertex(Pw7, vId, true);
		funcGeneratePointVertex(Pw8, vId, true);

		std::vector<double> K(4);
		K[0] = tK(0, 0);
		K[1] = tK(1, 1);
		K[2] = tK(0, 2);
		K[3] = tK(1, 2);
		auto funcGenerateFactor = [&](Eigen::Vector2d &_ob, uint64_t _vId1, uint64_t _vId2, uint64_t &_fId)
		{
			std::shared_ptr<OptFactorAbr<double>> factorPtr = std::make_shared<OptFactorUV_Pose3d_Point3d>(_fId);
			OptFactorUV_Pose3d_Point3d *factor = dynamic_cast<OptFactorUV_Pose3d_Point3d *>(factorPtr.get());
			factor->setObs(_ob.data(), K.data());
			std::vector<uint64_t> vIds;
			vIds.push_back(_vId1);
			vIds.push_back(_vId2);
			optimizer.addFactor(factorPtr, vIds);
			++_fId;
		};
		Eigen::Vector2d ob11 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw, tPw1.data());
		Eigen::Vector2d ob12 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw, tPw2.data());
		Eigen::Vector2d ob13 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw, tPw3.data());
		Eigen::Vector2d ob14 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw, tPw4.data());
		Eigen::Vector2d ob15 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw, tPw5.data());
		Eigen::Vector2d ob16 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw, tPw6.data());
		Eigen::Vector2d ob17 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw, tPw7.data());
		Eigen::Vector2d ob18 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw, tPw8.data());
		funcGenerateFactor(ob11, Twcs[0]->getId(), Pws[0]->getId(), fId);
		funcGenerateFactor(ob12, Twcs[0]->getId(), Pws[1]->getId(), fId);
		funcGenerateFactor(ob13, Twcs[0]->getId(), Pws[2]->getId(), fId);
		funcGenerateFactor(ob14, Twcs[0]->getId(), Pws[3]->getId(), fId);
		funcGenerateFactor(ob15, Twcs[0]->getId(), Pws[4]->getId(), fId);
		funcGenerateFactor(ob16, Twcs[0]->getId(), Pws[5]->getId(), fId);
		funcGenerateFactor(ob17, Twcs[0]->getId(), Pws[6]->getId(), fId);
		funcGenerateFactor(ob18, Twcs[0]->getId(), Pws[7]->getId(), fId);

		Eigen::Vector2d ob21 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw2, tPw1.data());
		Eigen::Vector2d ob22 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw2, tPw2.data());
		Eigen::Vector2d ob23 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw2, tPw3.data());
		Eigen::Vector2d ob24 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw2, tPw4.data());
		Eigen::Vector2d ob25 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw2, tPw5.data());
		Eigen::Vector2d ob26 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw2, tPw6.data());
		Eigen::Vector2d ob27 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw2, tPw7.data());
		Eigen::Vector2d ob28 = OPTIMIZE_BASE::Point2Image(K.data(), tTcw2, tPw8.data());
		funcGenerateFactor(ob21, Twcs[1]->getId(), Pws[0]->getId(), fId);
		funcGenerateFactor(ob22, Twcs[1]->getId(), Pws[1]->getId(), fId);
		funcGenerateFactor(ob23, Twcs[1]->getId(), Pws[2]->getId(), fId);
		funcGenerateFactor(ob24, Twcs[1]->getId(), Pws[3]->getId(), fId);
		funcGenerateFactor(ob25, Twcs[1]->getId(), Pws[4]->getId(), fId);
		funcGenerateFactor(ob26, Twcs[1]->getId(), Pws[5]->getId(), fId);
		funcGenerateFactor(ob27, Twcs[1]->getId(), Pws[6]->getId(), fId);
		funcGenerateFactor(ob28, Twcs[1]->getId(), Pws[7]->getId(), fId);

		optimizer.run();
		return;
	}

	using namespace std;
	using namespace Eigen;
	// 生成3D点云 ([-1,1]x[-1,1]x[2,5]范围)
	vector<Vector3d> generate3DPoints(int num_points)
	{
		vector<Vector3d> points;
		points.reserve(num_points);

		for (int i = 0; i < num_points; ++i)
		{
			Vector3d p = Vector3d::Random();
			p = p + Vector3d(0, 0, 2);
			points.push_back(p);
		}
		return points;
	}
	// 投影3D点到图像平面
	Vector2d project(const Vector3d &p3d, const Matrix3d &K, const Eigen::Matrix<double, 3, 4> &T_cam)
	{
		Matrix3d R = T_cam.block<3, 3>(0, 0);
		Vector3d t = T_cam.block<3, 1>(0, 3);
		Vector3d p_cam = R * p3d + t;
		Vector3d uv_hom = K * p_cam;
		return uv_hom.hnormalized();
	}
	// 验证重投影误差
	void verifyReprojection(const vector<Vector3d> &points3d,
							const vector<pair<Vector2d, Vector2d>> &matches,
							const Matrix3d &K,
							const Eigen::Matrix<double, 3, 4> &T1,
							const Eigen::Matrix<double, 3, 4> &T2)
	{
		double max_error = 0;
		for (size_t i = 0; i < points3d.size(); ++i)
		{
			Vector2d p1 = project(points3d[i], K, T1);
			Vector2d p2 = project(points3d[i], K, T2);

			double e1 = (p1 - matches[i].first).norm();
			double e2 = (p2 - matches[i].second).norm();
			max_error = max({max_error, e1, e2});
		}
		cout << "Max reprojection error: " << max_error << " pixels" << endl;
	}
	OPTIMIZE_LYJ_API void test_optimize_UV_Pose3d_P3d2()
	{
		Eigen::Matrix3d tK;
		tK << 500, 0, 320,
			0, 500, 240,
			0, 0, 1;
		Eigen::Matrix<double, 3, 4> tTcw;
		tTcw.setZero();
		tTcw.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
		tTcw.block(0, 3, 3, 1) = Eigen::Vector3d(0, 2, 0);
		Eigen::Matrix<double, 3, 4> tTwc = OPTIMIZE_BASE::invPose(tTcw);
		// tTwc = tTcw; //TODO
		Eigen::Matrix<double, 3, 4> tTcw2;
		tTcw2.block<3, 3>(0, 0) = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()).toRotationMatrix() *
								  Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY()).toRotationMatrix();
		tTcw2.block<3, 1>(0, 3) = Eigen::Vector3d(0.5, -0.2, 0.3);
		Eigen::Matrix<double, 3, 4> tTwc2 = OPTIMIZE_BASE::invPose(tTcw2);
		// tTwc2 = tTcw2; //TODO

		// 生成3D点
		const int num_points = 50;
		auto points3d = generate3DPoints(num_points);
		// 生成匹配点对
		vector<pair<Vector2d, Vector2d>> matches;
		for (const auto &p3d : points3d)
		{
			matches.emplace_back(
				project(p3d, tK, tTcw),
				project(p3d, tK, tTcw2));
		}
		// 验证数据正确性
		verifyReprojection(points3d, matches, tK, tTcw, tTcw2);

		Eigen::Vector3d lu(0, 0, 1);
		Eigen::Vector3d ru(640, 0, 1);
		Eigen::Vector3d rd(640, 480, 1);
		Eigen::Vector3d ld(0, 480, 1);
		auto invK = tK.inverse();
		lu = invK * lu;
		ru = invK * ru;
		rd = invK * rd;
		ld = invK * ld;
		std::ofstream Pwsf("Pws.txt");
		for (int i = 0; i < num_points; ++i)
		{
			Pwsf << points3d[i][0] << " " << points3d[i][1] << " " << points3d[i][2] << " "
				 << 255 << " " << 0 << " " << 0
				 << std::endl;
		}
		Pwsf << tTwc(0, 3) << " " << tTwc(1, 3) << " " << tTwc(2, 3) << " "
			 << 0 << " " << 255 << " " << 0
			 << std::endl;
		Eigen::Vector3d luw1 = tTwc.block(0, 0, 3, 3) * lu + tTwc.block(0, 3, 3, 1);
		Eigen::Vector3d ruw1 = tTwc.block(0, 0, 3, 3) * ru + tTwc.block(0, 3, 3, 1);
		Eigen::Vector3d rdw1 = tTwc.block(0, 0, 3, 3) * rd + tTwc.block(0, 3, 3, 1);
		Eigen::Vector3d ldw1 = tTwc.block(0, 0, 3, 3) * ld + tTwc.block(0, 3, 3, 1);
		Pwsf << luw1[0] << " " << luw1[1] << " " << luw1[2] << " "
			 << 0 << " " << 255 << " " << 0
			 << std::endl;
		Pwsf << ruw1[0] << " " << ruw1[1] << " " << ruw1[2] << " "
			 << 0 << " " << 255 << " " << 0
			 << std::endl;
		Pwsf << rdw1[0] << " " << rdw1[1] << " " << rdw1[2] << " "
			 << 0 << " " << 255 << " " << 0
			 << std::endl;
		Pwsf << ldw1[0] << " " << ldw1[1] << " " << ldw1[2] << " "
			 << 0 << " " << 255 << " " << 0
			 << std::endl;
		Pwsf << tTwc2(0, 3) << " " << tTwc2(1, 3) << " " << tTwc2(2, 3) << " "
			 << 0 << " " << 255 << " " << 0
			 << std::endl;
		Eigen::Vector3d luw2 = tTwc2.block(0, 0, 3, 3) * lu + tTwc2.block(0, 3, 3, 1);
		Eigen::Vector3d ruw2 = tTwc2.block(0, 0, 3, 3) * ru + tTwc2.block(0, 3, 3, 1);
		Eigen::Vector3d rdw2 = tTwc2.block(0, 0, 3, 3) * rd + tTwc2.block(0, 3, 3, 1);
		Eigen::Vector3d ldw2 = tTwc2.block(0, 0, 3, 3) * ld + tTwc2.block(0, 3, 3, 1);
		Pwsf << luw2[0] << " " << luw2[1] << " " << luw2[2] << " "
			 << 0 << " " << 255 << " " << 0
			 << std::endl;
		Pwsf << ruw2[0] << " " << ruw2[1] << " " << ruw2[2] << " "
			 << 0 << " " << 255 << " " << 0
			 << std::endl;
		Pwsf << rdw2[0] << " " << rdw2[1] << " " << rdw2[2] << " "
			 << 0 << " " << 255 << " " << 0
			 << std::endl;
		Pwsf << ldw2[0] << " " << ldw2[1] << " " << ldw2[2] << " "
			 << 0 << " " << 255 << " " << 0
			 << std::endl;
		for (int i = 0; i < num_points; ++i)
		{
			Pwsf << matches[i].first[0] << " " << matches[i].first[1] << " " << 1.0 << " "
				 << 0 << " " << 0 << " " << 255
				 << std::endl;
			Pwsf << matches[i].second[0] << " " << matches[i].second[1] << " " << 1.0 << " "
				 << 0 << " " << 0 << " " << 255
				 << std::endl;
		}
		Pwsf.close();

		std::ofstream truef("true.txt");
		truef << "true Twc1" << std::endl
			  << tTwc << std::endl;
		truef << "true Twc2" << std::endl
			  << tTwc2 << std::endl;
		for (int i = 0; i < num_points; ++i)
		{
			truef << "true Pw" << i << ":" << std::endl
				  << points3d[i] << std::endl;
		}
		truef.close();


		uint64_t vId = 0;
		uint64_t fId = 0;
		std::vector<std::shared_ptr<OptVarAbr<double>>> Twcs;
		std::vector<std::shared_ptr<OptVarAbr<double>>> Pws;
		// OptimizerSmalld optimizer;
		// OptimizerLargeSparse optimizer;
		OptimizeLargeSRBA optimizer;

		auto funcGeneratePointVertex = [&](Eigen::Vector3d &_Pw, uint64_t &_vId, bool _fix = false)
		{
			std::shared_ptr<OptVarAbr<double>> varPtr = std::make_shared<OptVarPoint3d>(_vId);
			varPtr->setData(_Pw.data());
			varPtr->setFixed(_fix);
			optimizer.addVariable(varPtr);
			Pws.push_back(varPtr);
			++_vId;
		};
		auto funcGeneratePoseVertex = [&](Eigen::Matrix<double, 3, 4> &_Twc, uint64_t &_vId, bool _fix = false)
		{
			std::shared_ptr<OptVarAbr<double>> varPtr = std::make_shared<OptVarPose3d>(_vId);
			varPtr->setData(_Twc.data());
			varPtr->setFixed(_fix);
			optimizer.addVariable(varPtr);
			Twcs.push_back(varPtr);
			++_vId;
		};
		Eigen::Matrix<double, 3, 4> Twc = tTwc;
		Eigen::Matrix<double, 3, 4> Twc2 = tTwc2;
		// Eigen::Matrix3d dRwc2 = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()).toRotationMatrix();
		// Twc2.block<3, 3>(0, 0) = dRwc2 * Twc2.block<3, 3>(0, 0);
		// Twc2(0, 3) += 1;
		// Twc2(1, 3) = 0.0;
		// Twc2(2, 3) = 0.0;
		funcGeneratePoseVertex(Twc, vId, true);
		funcGeneratePoseVertex(Twc2, vId, false);
		OptVarPose3d *v1 = dynamic_cast<OptVarPose3d *>(Twcs[0].get());
		OptVarPose3d *v2 = dynamic_cast<OptVarPose3d *>(Twcs[1].get());
		std::ofstream initf("init.txt");
		initf << "init: " << std::endl
			  << "Tw1: " << v1->getEigen() << std::endl
			  << "Tw2: " << v2->getEigen() << std::endl;
		for (int i = 0; i < num_points; ++i)
		{
			// Eigen::Vector3d Pw = (Eigen::Vector3d(points3d[i][0], points3d[i][1], points3d[i][2]));
			Eigen::Vector3d Pw = (Eigen::Vector3d(points3d[i][0] + 10, points3d[i][1] + 0.1, points3d[i][2]));
			funcGeneratePointVertex(Pw, vId, false);
			OptVarPoint3d *v = dynamic_cast<OptVarPoint3d *>(Pws[i].get());
			initf << "Pw" << i << ":" << std::endl
				  << v->getEigen() << std::endl;
		}
		initf.close();

		std::vector<double> K(4);
		K[0] = tK(0, 0);
		K[1] = tK(1, 1);
		K[2] = tK(0, 2);
		K[3] = tK(1, 2);
		auto funcGenerateFactor = [&](Eigen::Vector2d &_ob, uint64_t _vId1, uint64_t _vId2, uint64_t &_fId)
		{
			std::shared_ptr<OptFactorAbr<double>> factorPtr = std::make_shared<OptFactorUV_Pose3d_Point3d>(_fId);
			OptFactorUV_Pose3d_Point3d *factor = dynamic_cast<OptFactorUV_Pose3d_Point3d *>(factorPtr.get());
			factor->setObs(_ob.data(), K.data());
			std::vector<uint64_t> vIds;
			vIds.push_back(_vId1);
			vIds.push_back(_vId2);
			optimizer.addFactor(factorPtr, vIds);
			++_fId;
		};
		for (int i = 0; i < num_points; ++i)
		{
			auto &obs = matches[i];
			funcGenerateFactor(obs.first, Twcs[0]->getId(), Pws[i]->getId(), fId);
			funcGenerateFactor(obs.second, Twcs[1]->getId(), Pws[i]->getId(), fId);
		}

		optimizer.run();

		std::ofstream finalf("final.txt");
		finalf << "final: " << std::endl
			   << "Tw1: " << v1->getEigen() << std::endl
			   << "Tw2: " << v2->getEigen() << std::endl;
		for (int i = 0; i < num_points; ++i)
		{
			OptVarPoint3d *v = dynamic_cast<OptVarPoint3d *>(Pws[i].get());
			finalf << "Pw" << i << ":" << std::endl
				   << v->getEigen() << std::endl;
		}
		finalf.close();

		return;
	}
	OPTIMIZE_LYJ_API void ceres_Check_UV_Pose3d_P3d2()
	{
		OPTIMIZE_LYJ::OPTIMIZE_CERES::ceresCheckTcwUV();
		return;
	}
	

#define PI 3.14159265358979323846
	using namespace Eigen;
	// 定义三维点和二维点结构体
	struct Point3D { Vector3d coord; };
	struct Point2D { Vector2d coord; };
	struct Line3D { int from, to; };  // 边的起点和终点索引
	struct Line2D { Point2D p1, p2; };
	// 相机内参结构体（简化版）
	struct CameraIntrinsics {
		double fx = 500.0, fy = 500.0; // 焦距
		double cx = 320.0, cy = 240.0; // 主点
		int width = 640, height = 480; // 图像尺寸
	};
	// 生成立方体顶点和边
	void generate_cuboid(double _b, double _l, double _w, double _h, std::vector<Point3D>& vertices, std::vector<Line3D>& edges) {
		vertices.resize(8);
		double l = _l / 2.0;
		double w = _w / 2.0;
		double h = _h / 2.0;
		double bias = 20;
		// 立方体顶点（中心在原点）
		vertices[0].coord << -l + _b, -w + _b, -h + _b;
		vertices[1].coord << l + _b, -w + _b, -h + _b;
		vertices[2].coord << l + _b, w + _b, -h + _b;
		vertices[3].coord << -l + _b, w + _b, -h + _b;
		vertices[4].coord << -l + _b, -w + _b, h + _b;
		vertices[5].coord << l + _b, -w + _b, h + _b;
		vertices[6].coord << l + _b, w + _b, h + _b;
		vertices[7].coord << -l + _b, w + _b, h + _b;

		// 定义12条边
		edges = { {0,1}, {1,2}, {2,3}, {3,0},
				 {4,5}, {5,6}, {6,7}, {7,4},
				 {0,4}, {1,5}, {2,6}, {3,7} };
	}
	// 将世界坐标点变换到相机坐标系
	Vector3d transform_to_camera(const Vector3d& point,
		const Matrix3d& R, const Vector3d& t) {
		return R.transpose() * (point - t);
	}
	// 投影三维点到图像平面
	bool project_point(const Vector3d& p_cam, Point2D& p_pixel,
		const CameraIntrinsics& cam) {
		if (p_cam.z() <= 0) return false; // 深度为负，不可见

		// 透视投影
		double u = (cam.fx * p_cam.x() / p_cam.z()) + cam.cx;
		double v = (cam.fy * p_cam.y() / p_cam.z()) + cam.cy;

		// 检查是否在图像范围内
		if (u < 0 || u >= cam.width || v < 0 || v >= cam.height)
			return false;

		p_pixel.coord << u, v;
		//p_pixel.coord << p_cam.x() / p_cam.z(), p_cam.y() / p_cam.z();
		return true;
	}
	// 主函数：生成三个位姿下的二维线观测
	int genLineData(std::vector<Eigen::Matrix<double, 3, 4>>& _Twcs, std::vector<Eigen::Matrix<double, 6, 1>>& _lines, std::vector<std::vector<Eigen::Vector4d>>& _allObs, double _bT, double _bl,
		double _fx=500, double _fy = 500, double _cx = 320, double _cy = 240, int _w = 640, int _h = 480) {
		// 创建边长为2的立方体
		std::vector<Point3D> vertices;
		std::vector<Line3D> edges_3d;
		generate_cuboid(_bl, 1.0, 2.0, 3.0, vertices, edges_3d);

		_lines.resize(edges_3d.size());
		for (int i = 0; i < edges_3d.size(); ++i)
		{
			const Line3D& edge = edges_3d[i];
			const Vector3d& p1 = vertices[edge.from].coord;
			const Vector3d& p2 = vertices[edge.to].coord;
			_lines[i].head<3>() = p1;
			_lines[i].tail<3>() = p2;
		}

		// 定义三个相机位姿 [R|t] Twc
		std::vector<std::pair<Matrix3d, Vector3d>> poses = {
			{   // 正面视角
				AngleAxisd(0 + _bT, Vector3d::UnitY()).toRotationMatrix(),
				Vector3d(0, 0, -15)
			},
			{   // 右侧视角
				AngleAxisd(PI / 8 + _bT, Vector3d::UnitY()).toRotationMatrix(),
				Vector3d(-3, 0, -15)
			},
			{   // 俯视视角
				AngleAxisd(-PI / 8 - _bT, Vector3d::UnitX()).toRotationMatrix(),
				Vector3d(0, -2, -15)
			}
		};
		_Twcs.resize(poses.size());
		for (int i = 0; i < poses.size(); ++i)
		{
			_Twcs[i].block(0, 0, 3, 3) = poses[i].first;
			_Twcs[i].block(0, 3, 3, 1) = poses[i].second;
		}

		CameraIntrinsics cam{_fx, _fy, _cx, _cy, _w, _h}; // 使用默认内参

		_allObs.resize(poses.size());
		for (size_t pose_idx = 0; pose_idx < poses.size(); ++pose_idx) {
			auto& [R, t] = poses[pose_idx];
			std::vector<Line2D> visible_lines;

			// 处理每条3D边
			_allObs[pose_idx].resize(edges_3d.size());
			int cnt = 0;
			for (const Line3D& edge : edges_3d) {
				// 变换起点和终点到相机坐标系
				Vector3d p1_cam = transform_to_camera(
					vertices[edge.from].coord, R, t);
				Vector3d p2_cam = transform_to_camera(
					vertices[edge.to].coord, R, t);

				// 投影到图像平面
				Point2D p1_pixel, p2_pixel;
				bool visible1 = project_point(p1_cam, p1_pixel, cam);
				bool visible2 = project_point(p2_cam, p2_pixel, cam);

				// 仅当两端点均可见时才保留该边
				if (visible1 && visible2) {
					visible_lines.push_back({ p1_pixel, p2_pixel });
					_allObs[pose_idx][cnt] << p1_pixel.coord(0), p1_pixel.coord(1),
						p2_pixel.coord(0), p2_pixel.coord(1);
				}
				else {
					_allObs[pose_idx][cnt] << 0, 0, 0, 0;
				}
				++cnt;
			}

			// 输出当前位姿的观测结果
			std::cout << "Pose " << pose_idx + 1 << "观测到 "
				<< visible_lines.size() << " 条线段:\n";
			//for (const Line2D& line : visible_lines) {
			//	std::cout << " 线段: (" << line.p1.coord.x() << ", "
			//		<< line.p1.coord.y() << ") -> ("
			//		<< line.p2.coord.x() << ", "
			//		<< line.p2.coord.y() << ")\n";
			//}
		}

		return 0;
	}
	OPTIMIZE_LYJ_API void test_optimize_UV2_Pose3d_Line3d()
	{
		double fx = 500;
		double fy = 500;
		double cx = 320;
		double cy = 240;
		int w = 640;
		int h = 480;
		std::vector<Eigen::Matrix<double, 3, 4>> tTwcs;
		std::vector<Eigen::Matrix<double, 6, 1>> tline3Dws;
		std::vector<std::vector<Eigen::Vector4d>> allObs;
		genLineData(tTwcs, tline3Dws, allObs, 0, 0);
		Eigen::Matrix3d K;
		K << fx, 0, cx,
			0, fy, cy,
			0, 0, 1;
		Eigen::Matrix3d KK;
		OPTIMIZE_BASE::convertK2KK(K, KK);

		uint64_t vId = 0;
		uint64_t fId = 0;
		std::vector<std::shared_ptr<OptVarAbr<double>>> TwcVars;
		std::vector<std::shared_ptr<OptVarAbr<double>>> line3DwVars;
		OptimizerSmalld optimizer;
		//OptimizerLargeSparse optimizer;
		optimizer.setMaxIter(100);

		auto funcGenerateLineVertex = [&](Eigen::Vector4d& _line3Dw, uint64_t& _vId, bool _fix = false)
			{
				std::shared_ptr<OptVarAbr<double>> varPtr = std::make_shared<OptVarLine3d>(_vId);
				varPtr->setData(_line3Dw.data());
				varPtr->setFixed(_fix);
				optimizer.addVariable(varPtr);
				line3DwVars.push_back(varPtr);
				++_vId;
			};
		auto funcGeneratePoseVertex = [&](Eigen::Matrix<double, 3, 4>& _Twc, uint64_t& _vId, bool _fix = false)
			{
				std::shared_ptr<OptVarAbr<double>> varPtr = std::make_shared<OptVarPose3d>(_vId);
				varPtr->setData(_Twc.data());
				varPtr->setFixed(_fix);
				optimizer.addVariable(varPtr);
				TwcVars.push_back(varPtr);
				++_vId;
			};

		std::vector<Eigen::Matrix<double, 3, 4>> Twcs;
		std::vector<Eigen::Matrix<double, 6, 1>> line3Dws;
		std::vector<std::vector<Eigen::Vector4d>> allObsTmp;
		genLineData(Twcs, line3Dws, allObsTmp, 0.1, 0.1);
		std::vector<Eigen::Matrix<double, 3, 4>> Tcws(Twcs.size());
		std::vector<Eigen::Vector4d> orthws(line3Dws.size());
		for (size_t i = 0; i < Twcs.size(); i++)
		{
			if (i == 0) {
				Tcws[i] = OPTIMIZE_BASE::invPose(tTwcs[i]);
				funcGeneratePoseVertex(Tcws[i], vId, true);
			}
			else {
				Tcws[i] = OPTIMIZE_BASE::invPose(Twcs[i]);
				funcGeneratePoseVertex(Tcws[i], vId, false);
			}
		}
		for (size_t i = 0; i < line3Dws.size(); i++)
		{
			if (i == 0 || i > 4) {
				orthws[i] = OPTIMIZE_BASE::pp_to_orth(tline3Dws[i].head(3), tline3Dws[i].tail(3));
				funcGenerateLineVertex(orthws[i], vId, true);
			}
			else {
				orthws[i] = OPTIMIZE_BASE::pp_to_orth(line3Dws[i].head(3), line3Dws[i].tail(3));
				funcGenerateLineVertex(orthws[i], vId, false);
			}
		}

		auto funcGenerateFactor = [&](Eigen::Vector4d& _ob, uint64_t _vId1, uint64_t _vId2, uint64_t& _fId)
			{
				std::shared_ptr<OptFactorAbr<double>> factorPtr = std::make_shared<OptFactorLine2d_Pose3d_Line3d>(_fId);
				OptFactorLine2d_Pose3d_Line3d* factor = dynamic_cast<OptFactorLine2d_Pose3d_Line3d*>(factorPtr.get());
				factor->setObs(_ob, KK);
				std::vector<uint64_t> vIds;
				vIds.push_back(_vId1);
				vIds.push_back(_vId2);
				optimizer.addFactor(factorPtr, vIds);
				++_fId;
			};
		for (int i = 0; i < allObs.size(); ++i)
		{
			for (int j = 0; j < allObs[i].size(); ++j)
			{
				if (allObs[i][j].isZero())
					continue;
				funcGenerateFactor(allObs[i][j], TwcVars[i]->getId(), line3DwVars[j]->getId(), fId);

				//const Eigen::Matrix3d& Rcw = Tcws[i].block(0, 0, 3, 3);
				//const Eigen::Vector3d& tcw = Tcws[i].block(0, 3, 3, 1);

				//const Eigen::Vector2d& obsp = allObs[i][j].head(2);
				//const Eigen::Vector3d& linewsP = line3Dws[j].head(3);
				//Eigen::Vector3d sPc = Rcw * linewsP + tcw;
				//double z = sPc[2];
				//sPc /= z;
				//Eigen::Vector3d spc = K * sPc;
				////std::cout << obsp[0] - spc[0] << ", " << obsp[1] - spc[1] << std::endl;

				//const Eigen::Vector2d& obep = allObs[i][j].tail(2);
				//const Eigen::Vector3d& lineweP = line3Dws[j].tail(3);				
				//Eigen::Vector3d ePc = Rcw * lineweP + tcw;
				//double z2 = ePc[2];
				//ePc /= z2;
				//Eigen::Vector3d epc = K * ePc;
				////std::cout << obep[0] - epc[0] << ", " << obep[1] - epc[1] << std::endl;

				//Eigen::Vector3d v = lineweP - linewsP;
				//std::cout << linewsP << std::endl << std::endl;
				//std::cout << lineweP << std::endl << std::endl;
				//std::cout << v << std::endl << std::endl;
				//Eigen::Matrix<double, 6, 1> plkW = OPTIMIZE_BASE::line_to_plk(linewsP, v);
				//std::cout << plkW << std::endl << std::endl;
				//Eigen::Matrix<double, 6, 1> lineW = OPTIMIZE_BASE::orth_to_plk(orthws[j]);
				//std::cout << lineW << std::endl << std::endl;
				//Eigen::Matrix<double, 6, 1> lineC = OPTIMIZE_BASE::plk_to_pose(lineW, Rcw, tcw);
				//Eigen::Vector3d nc = lineC.head(3);
				//Eigen::Vector3d l2d = KK * nc;
				//double l_norm = l2d(0) * l2d(0) + l2d(1) * l2d(1);
				//double l_sqrtnorm = sqrt(l_norm);
				//double e1 = obsp(0) * l2d(0) + obsp(1) * l2d(1) + l2d(2);
				//double e2 = obep(0) * l2d(0) + obep(1) * l2d(1) + l2d(2);
				//Eigen::Vector2d err;
				//err(0) = e1 / l_sqrtnorm;
				//err(1) = e2 / l_sqrtnorm;
				//std::cout << err << std::endl;
				//if (err.norm() > 0.01)
				//	std::cout << "Large error!" << std::endl;

				continue;
			}
		}

		optimizer.run();
		return;
	}
	
	
	OPTIMIZE_LYJ_API void ceres_Check_UV_Pose3d_Line3d()
	{
		std::vector<Eigen::Matrix<double, 3, 4>> tTwcs;
		std::vector<Eigen::Matrix<double, 6, 1>> tline3Dws;
		std::vector<std::vector<Eigen::Vector4d>> allObs;
		genLineData(tTwcs, tline3Dws, allObs, 0, 0);
		std::vector<Eigen::Matrix<double, 3, 4>> Twcs;
		std::vector<Eigen::Matrix<double, 6, 1>> line3Dws;
		std::vector<std::vector<Eigen::Vector4d>> allObsTmp;
		genLineData(Twcs, line3Dws, allObsTmp, 0, 1);
		std::vector<Eigen::Matrix<double, 3, 4>> Tcws(Twcs.size());
		for (size_t i = 0; i < Twcs.size(); i++)
		{
			Tcws[i] = OPTIMIZE_BASE::invPose(Twcs[i]);
		}
		OPTIMIZE_LYJ::OPTIMIZE_CERES::ceresCheckTcwLineUV(Tcws, line3Dws, allObs);
		return;
	}
} // namespace OPTIMIZE_LYJ
