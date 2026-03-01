#ifndef OPTIMIZE_LYJ_OPTIMIZER_H
#define OPTIMIZE_LYJ_OPTIMIZER_H

#include "optimizerAbr.h"
#include <Eigen/Core>
#include <Eigen/Eigen>


namespace OPTIMIZE_LYJ
{
	class OPTIMIZE_LYJ_API OptimizerSmalld : public OptimizerAbr<double>
	{
	public:
		OptimizerSmalld();
		~OptimizerSmalld();

	private:

		// 通过 OptimizerAbr 继承，默认var和factor的id都是从0开始且连续 TODO
		bool init() override;

		bool generateAB(double& _err) override;

		bool solveDetX() override;

		bool updateX() override;


	private:
		Eigen::MatrixXd m_A;
		Eigen::VectorXd m_B;
		Eigen::VectorXd m_DetX;
	};


	class OPTIMIZE_LYJ_API OptimizerLargeSparse : public OptimizerAbr<double>
	{
	public:
		OptimizerLargeSparse();
		~OptimizerLargeSparse();

	private:

		// 通过 OptimizerAbr 继承
		bool init() override;

		bool generateAB(double& _err) override;

		bool solveDetX() override;

		bool updateX() override;

	private:
		Eigen::SparseMatrix<double> m_A;
		Eigen::VectorXd m_B;
		Eigen::VectorXd m_DetX;

	};


	class OPTIMIZE_LYJ_API OptimizeLargeSRBA : public OptimizerAbr<double>
	{
	public:
		OptimizeLargeSRBA();
		~OptimizeLargeSRBA();

	private:

		class FactorMat
		{
		public:
			FactorMat()
			{
			}
			~FactorMat() {}


		//private:
			std::shared_ptr<OptFactorAbr<double>> m_factor = nullptr;
			std::vector<Eigen::MatrixXd> m_jacs;
			Eigen::VectorXd m_err;
		};
		class EliminationMat
		{
		public:
			EliminationMat() {}
			~EliminationMat() {}

			// 非位姿jac，QR分解，变换位姿jac
			void QR()
			{
			}
			void solveElimination() 
			{
			}

		private:
			std::vector<Eigen::MatrixXd> m_jacs;
			Eigen::VectorXd m_err;
			std::vector<Eigen::MatrixXd> m_jacsRemand;
			Eigen::VectorXd m_errRemand;
		};


		// 通过 OptimizerAbr 继承
		bool init() override;

		bool generateAB(double& _err) override;

		bool solveDetX() override;

		bool updateX() override;

	private:
		Eigen::SparseMatrix<double> m_A;
		Eigen::VectorXd m_B;
		Eigen::VectorXd m_DetX;
		std::vector<FactorMat> m_factorMats;
		std::vector<EliminationMat> m_eliminationMats;
	};



}


#endif // !OPTIMIZE_LYJ_OPTIMIZER_H
