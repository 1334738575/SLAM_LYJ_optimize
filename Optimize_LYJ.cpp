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
	OPTIMIZE_LYJ_API void test_optimize()
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
} // namespace OPTIMIZE_LYJ
