#ifndef OPTIMIZE_LYJ_OPTIMIZER_H
#define OPTIMIZE_LYJ_OPTIMIZER_H

#include "optimizerAbr.h"
#include <Eigen/Core>
#include <Eigen/Eigen>


namespace OPTIMIZE_LYJ
{
	class OptimizerSmalld : public OptimizerAbr<double>
	{
	public:
		OptimizerSmalld();
		~OptimizerSmalld();

	private:

		// ͨ�� OptimizerAbr �̳У�Ĭ��var��factor��id���Ǵ�0��ʼ������ TODO
		bool generateAB(double& _err) override;

		bool solveDetX() override;

		bool updateX() override;


	private:
		Eigen::MatrixXd m_A;
		Eigen::VectorXd m_B;
		Eigen::VectorXd m_DetX;
	};


	class OptimizerLargeSparse : public OptimizerAbr<double>
	{
	public:
		OptimizerLargeSparse();
		~OptimizerLargeSparse();

	private:

		// ͨ�� OptimizerAbr �̳�
		bool generateAB(double& _err) override;

		bool solveDetX() override;

		bool updateX() override;


	private:
		Eigen::SparseMatrix<double> m_A;
		Eigen::VectorXd m_B;
		Eigen::VectorXd m_DetX;

	};


	class OptimizeLargeSRBA : public OptimizerAbr<double>
	{
	public:
		OptimizeLargeSRBA();
		~OptimizeLargeSRBA();

	private:


		// ͨ�� OptimizerAbr �̳�
		bool generateAB(double& _err) override;

		bool solveDetX() override;

		bool updateX() override;

	};



}


#endif // !OPTIMIZE_LYJ_OPTIMIZER_H
