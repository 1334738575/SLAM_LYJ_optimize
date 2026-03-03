#ifndef OPTIMIZE_LYJ_OPTIMIZER_H
#define OPTIMIZE_LYJ_OPTIMIZER_H

#include "optimizerAbr.h"
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <set>


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

			int getEDim() const { return m_err.rows(); }


		//private:
			uint64_t m_id;
			//std::shared_ptr<OptFactorAbr<double>> m_factor = nullptr;
			std::vector<Eigen::MatrixXd> m_jacs;
			Eigen::VectorXd m_err;
			Factor2Var m_f2vs;
			std::vector<OptVarAbr<double>*> m_vars;
			bool m_valid = true;
		};
		class EliminationMat
		{
		public:
			EliminationMat(const std::vector<FactorMat*>& _fMatsIn, OptVarType _eliminationType, uint64_t _vId)
				:m_fMatsIn(_fMatsIn), m_eliminationType(_eliminationType), m_vId(_vId)
			{}
			~EliminationMat() {}

			// 非位姿jac，QR分解，变换位姿jac
			void QR()
			{
			}
			void solveElimination(std::vector<Eigen::Map<Eigen::VectorXd>> _dXs)
			{
			}

			std::vector<FactorMat*> m_fMatsIn;
			OptVarType m_eliminationType;
			uint64_t m_vId;
		//private:
			FactorMat* m_factorMatRemand;
			FactorMat m_factorMatEliminate;
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
		std::set<OptVarType> m_eliminationType;
		std::vector<int> m_cLocs;
		std::vector<int> m_rLocs;
	};



}


#endif // !OPTIMIZE_LYJ_OPTIMIZER_H
