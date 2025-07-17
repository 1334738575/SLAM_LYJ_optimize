#include "OPTIMIZE_LYJ.h"
#include "Variable/Variable.h"
#include "Factor/Factor.h"
#include "Optimizer/Optimizer.h"
#include <functional>

#include <Eigen/Eigen>
#include <Eigen/Core>

namespace OPTIMIZE_LYJ
{
	OPTIMIZE_LYJ_API int optimize_version()
	{
		return 1;
	}
	OPTIMIZE_LYJ_API void test_optimize_P3d_P3d()
	{
		////À´÷ÿ Õ∑≈£¨±®¥Ì
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
		// Twc2.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
		// double r = 3.14 / 8;
		// Twc2(0, 0) = std::cos(r);
		// Twc2(0, 1) = -1 * std::sin(r);
		// Twc2(1, 0) = std::sin(r);
		// Twc2(1, 1) = std::cos(r);
		// Twc2(0, 3) = 1.0;
		// Twc2(1, 3) = 10.0;
		// Twc2(2, 3) = 222.0;
		funcGeneratePoseVertex(Twc, vId, true);
		funcGeneratePoseVertex(Twc2, vId);
		Eigen::Vector3d Pw1(tPw1[0], tPw1[1] + 0.2, tPw1[2]);
		Eigen::Vector3d Pw2(tPw2[0], tPw2[1] + 0.2, tPw2[2]);
		Eigen::Vector3d Pw3(tPw3[0], tPw3[1] + 0.2, tPw3[2]);
		Eigen::Vector3d Pw4(tPw4[0], tPw4[1] + 0.2, tPw4[2]);
		Eigen::Vector3d Pw5(tPw5[0], tPw5[1] + 0.2, tPw5[2]);
		Eigen::Vector3d Pw6(tPw6[0], tPw6[1] + 0.2, tPw6[2]);
		Eigen::Vector3d Pw7(tPw7[0], tPw7[1] + 0.2, tPw7[2]);
		Eigen::Vector3d Pw8(tPw8[0], tPw8[1] + 0.2, tPw8[2]);
		funcGeneratePointVertex(Pw1, vId);
		funcGeneratePointVertex(Pw2, vId);
		funcGeneratePointVertex(Pw3, vId);
		funcGeneratePointVertex(Pw4, vId);
		funcGeneratePointVertex(Pw5, vId);
		funcGeneratePointVertex(Pw6, vId);
		funcGeneratePointVertex(Pw7, vId);
		funcGeneratePointVertex(Pw8, vId);

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
} // namespace OPTIMIZE_LYJ
