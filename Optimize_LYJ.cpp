#include "OPTIMIZE_LYJ.h"
#include "Variable/Variable.h"
#include "Factor/Factor.h"
#include "Optimizer/Optimizer.h"

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

		if(false){
			OptVarPoint3d var(0);
			Eigen::Vector3d data(1, 2, 3);
			var.setData(data.data());
			OptFactorP3d_P3d factor(0);
			Eigen::Vector3d obs(0, 0, 0);
			factor.setObs(obs.data());

			Eigen::Vector3d err;
			Eigen::Matrix3d jac;
			double* jacPtr = jac.data();
			OptVarAbr<double>* varPtr = &var;
			factor.calculateErrAndJac(err.data(), &jacPtr, 1, &varPtr);
			std::cout << "err: " << std::endl << err << std::endl;
			std::cout << "jac: " << std::endl << jac << std::endl;
		}

		std::shared_ptr<OptVarAbr<double>> varSPtr = std::make_shared<OptVarPoint3d>(0);
		Eigen::Vector3d data(1, 2, 3);
		varSPtr->setData(data.data());
		std::shared_ptr<OptFactorAbr<double>> factorSPtr = std::make_shared<OptFactorP3d_P3d>(0);
		Eigen::Vector3d obs(0, 0, 0);
		OptFactorP3d_P3d* factor = dynamic_cast<OptFactorP3d_P3d*>(factorSPtr.get());
		factor->setObs(obs.data());
		std::vector<uint64_t> vIds;
		vIds.push_back(0);
		{
			//OptimizerSmalld optimizer;
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
		OptFactorPose3d_Pose3d* factor = dynamic_cast<OptFactorPose3d_Pose3d*>(factorSPtr.get());
		factor->setObs(obs.data());
		std::vector<uint64_t> vIds;
		vIds.push_back(varSPtr->getId());
		{
			//OptimizerSmalld optimizer;
			OptimizerLargeSparse optimizer;
			optimizer.addVariable(varSPtr);
			optimizer.addFactor(factorSPtr, vIds);
			optimizer.run();
		}
		return;
	}
	OPTIMIZE_LYJ_API void test_optimize_RelPose3d_Pose3d_Pose3d()
	{
		//Tw1
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
		//Tw2
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

		//T12
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
		OptFactorRelPose3d_Pose3d_Pose3d* factor = dynamic_cast<OptFactorRelPose3d_Pose3d_Pose3d*>(factorPtr.get());
		factor->setObs(obs.data());
		std::vector<uint64_t> vIds;
		vIds.push_back(varPtr1->getId());
		vIds.push_back(varPtr2->getId());

		OptVarPose3Eulard* v1 = dynamic_cast<OptVarPose3Eulard*>(varPtr1.get());
		OptVarPose3Eulard* v2 = dynamic_cast<OptVarPose3Eulard*>(varPtr2.get());
		std::cout << "init: " << std::endl << "Tw1: " << *v1 << std::endl << "Tw2: " << *v2 << std::endl << "T12: " << *factor << std::endl;
		std::cout << "real T12: " << OPTIMIZE_BASE::relPose(varPtr1->getData(), varPtr2->getData()) << std::endl << std::endl << std::endl;
		//OptimizerSmalld optimizer;
		OptimizerLargeSparse optimizer;
		optimizer.addVariable(varPtr1);
		optimizer.addVariable(varPtr2);
		optimizer.addFactor(factorPtr, vIds);
		optimizer.run();

		std::cout << std::endl << std::endl << "final: " << std::endl << "Tw1: " << *v1 << std::endl << "Tw2: " << *v2 << std::endl << "T12: " << *factor << std::endl;
		std::cout << "real T12: " << OPTIMIZE_BASE::relPose(varPtr1->getData(), varPtr2->getData()) << std::endl;
		return;
	}
} // namespace OPTIMIZE_LYJ
