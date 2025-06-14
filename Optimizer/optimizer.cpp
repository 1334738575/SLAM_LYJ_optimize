#include "optimizer.h"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace OPTIMIZE_LYJ
{

	OptimizerSmalld::OptimizerSmalld()
	{}
	OptimizerSmalld::~OptimizerSmalld()
	{}
	bool OptimizerSmalld::generateAB(double& _err)
	{
        int rows = 0;
        int cols = 0;
        std::vector<int> vLocs(m_vars.size() + 1, 0);
        int tmp;
        for (int i = 0; i < m_vars.size(); ++i) {
            cols += m_vars[i]->getTangentDim();
            vLocs[i + 1] = cols;
        }
        std::vector<int> fLocs(m_factors.size() + 1, 0);
        for (int i = 0; i < m_factors.size(); ++i) {
            rows += m_factors[i]->getEDim();
            fLocs[i + 1] = rows;
        }
        Eigen::MatrixXd Jac(rows, cols);
        Eigen::VectorXd Err(rows);

        int connectCnt = 0;
        std::vector<OptVarAbr<double>*> vars;
        std::vector<Eigen::MatrixXd> jacs;
        std::vector<double*> jacPtrs;
        int tanDim, eDim;
        for (int i = 0; i < m_factors.size(); ++i) {
			auto factor = m_factors[i];
			const auto& fId = factor->getId();
			const auto& fLoc = fLocs[fId];
            eDim = factor->getEDim();
            const auto& f2vs = m_factor2Vars[fId];
            connectCnt = f2vs.size();

            vars.resize(connectCnt);
            for (int j = 0; j < connectCnt; ++j)
                vars[j] = m_vars[f2vs.connectId(j)].get();

            jacs.resize(connectCnt);
            jacPtrs.resize(connectCnt);
            for (int j = 0; j < connectCnt; ++j) {
				const auto& vId = f2vs.connectId(j);
				const auto& vLoc = vLocs[vId];
                tanDim = m_vars[vId]->getTangentDim();
                jacs[j].resize(eDim, tanDim);
                jacPtrs[j] = jacs[j].data();
			}

			double* errPtr = Err.data() + fLoc;
			factor->calculateErrAndJac(errPtr, jacPtrs.data(), 1, vars.data());
        
            for (int j = 0; j < connectCnt; ++j) {
                const auto& vId = f2vs.connectId(j);
                const auto& vLoc = vLocs[vId];
                tanDim = m_vars[vId]->getTangentDim();
                for (int ii = 0; ii < eDim; ++ii) {
                    for (int jj = 0; jj < tanDim; ++jj) {
                        Jac(fLoc + ii, vLoc + jj) = jacs[j](ii, jj);
                    }
                }
            }
        }

        //std::cout << "Jac: " << std::endl << Jac << std::endl;
        //std::cout << "Err: " << std::endl << Err << std::endl;
        m_A = Jac.transpose() * Jac;
		m_B = -1 * Jac.transpose() * Err;
        _err = Err.norm();

		return true;
	}
	bool OptimizerSmalld::solveDetX()
	{
        //std::cout << "before: " << std::endl;
        //std::cout << "m_A: " << std::endl << m_A << std::endl;
        //std::cout << "m_B: " << std::endl << m_B << std::endl;
        int dim = m_A.rows();
        for (int i = 0; i < dim; ++i)
            m_A(i, i) += 1e-6;
        //std::cout << "after: " << std::endl;
        //std::cout << "m_A: " << std::endl << m_A << std::endl;
        //std::cout << "m_B: " << std::endl << m_B << std::endl;
        // 创建求解器
        Eigen::LDLT<Eigen::MatrixXd> solver;
        solver.compute(m_A);
        if (solver.info() != Eigen::Success) {
            //std::cerr << "Decomposition failed!" << std::endl;
            return false;
        }
        // 求解 Ax = b
        m_DetX = solver.solve(m_B);

        // 检查求解结果
        if (solver.info() != Eigen::Success) {
            //std::cerr << "Solving failed!" << std::endl;
            return false;
        }
        //std::cout << "The solution is:\n" << m_DetX << std::endl;
		return true;
	}
	bool OptimizerSmalld::updateX()
	{
        int cols = 0;
        std::vector<int> vLocs(m_vars.size() + 1, 0);
        int tmp;
        for (int i = 0; i < m_vars.size(); ++i) {
            cols += m_vars[i]->getTangentDim();
            vLocs[i + 1] = cols;
        }
		for (int i = 0; i < m_vars.size(); ++i) {
            auto var = m_vars[i];
			const auto& vId = var->getId();
			const auto& vLoc = vLocs[vId];
			const auto& tanDim = var->getTangentDim();
			double* detXPtr = m_DetX.data() + vLoc;
            var->update(detXPtr);
		}
		return true;
	}




    OptimizerLargeSparse::OptimizerLargeSparse()
    {
    }
    OptimizerLargeSparse::~OptimizerLargeSparse()
    {
    }
    bool OptimizerLargeSparse::generateAB(double& _err)
    {
        std::vector<Eigen::Triplet<double>> tripletLists;
        int rows = 0;
        int cols = 0;
        std::vector<int> vLocs(m_vars.size() + 1, 0);
        int tmp;
        for (int i = 0; i < m_vars.size(); ++i) {
            cols += m_vars[i]->getTangentDim();
            vLocs[i + 1] = cols;
        }
        std::vector<int> fLocs(m_factors.size() + 1, 0);
        for (int i = 0; i < m_factors.size(); ++i) {
            rows += m_factors[i]->getEDim();
            fLocs[i + 1] = rows;
        }
        Eigen::SparseMatrix<double> Jac(rows, cols);
        Eigen::VectorXd Err(rows);

        int connectCnt = 0;
        std::vector<OptVarAbr<double>*> vars;
        std::vector<Eigen::MatrixXd> jacs;
        std::vector<double*> jacPtrs;
        int tanDim, eDim;
        for (int i = 0; i < m_factors.size(); ++i) {
            auto factor = m_factors[i];
            const auto& fId = factor->getId();
            const auto& fLoc = fLocs[fId];
            eDim = factor->getEDim();
            const auto& f2vs = m_factor2Vars[fId];
            connectCnt = f2vs.size();

            vars.resize(connectCnt);
            for (int j = 0; j < connectCnt; ++j)
                vars[j] = m_vars[f2vs.connectId(j)].get();

            jacs.resize(connectCnt);
            jacPtrs.resize(connectCnt);
            for (int j = 0; j < connectCnt; ++j) {
                const auto& vId = f2vs.connectId(j);
                const auto& vLoc = vLocs[vId];
                tanDim = m_vars[vId]->getTangentDim();
                jacs[j].resize(eDim, tanDim);
                jacPtrs[j] = jacs[j].data();
            }

            double* errPtr = Err.data() + fLoc;
            factor->calculateErrAndJac(errPtr, jacPtrs.data(), 1, vars.data());

            for (int j = 0; j < connectCnt; ++j) {
                const auto& vId = f2vs.connectId(j);
                const auto& vLoc = vLocs[vId];
                tanDim = m_vars[vId]->getTangentDim();
                for (int ii = 0; ii < eDim; ++ii) {
                    for (int jj = 0; jj < tanDim; ++jj) {
                        tripletLists.emplace_back(fLoc + ii, vLoc + jj, jacs[j](ii, jj));
                    }
                }
            }
        }

        Jac.setFromTriplets(tripletLists.begin(), tripletLists.end());
        m_A = Jac.transpose() * Jac;
        m_B = -1 * Jac.transpose() * Err;
        _err = Err.norm();

        return true;
    }
    bool OptimizerLargeSparse::solveDetX()
    {
        int dim = m_A.rows();
        for (int i = 0; i < dim; ++i)
            m_A.coeffRef(i, i) = m_A.coeff(i,i) + 1e-6;
        // 创建求解器
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(m_A);
        if (solver.info() != Eigen::Success) {
            //std::cerr << "Decomposition failed!" << std::endl;
            return false;
        }
        // 求解 Ax = b
        m_DetX = solver.solve(m_B);

        // 检查求解结果
        if (solver.info() != Eigen::Success) {
            //std::cerr << "Solving failed!" << std::endl;
            return false;
        }
        //std::cout << "The solution is:\n" << m_DetX << std::endl;
        return true;
    }
    bool OptimizerLargeSparse::updateX()
    {
        int cols = 0;
        std::vector<int> vLocs(m_vars.size() + 1, 0);
        int tmp;
        for (int i = 0; i < m_vars.size(); ++i) {
            cols += m_vars[i]->getTangentDim();
            vLocs[i + 1] = cols;
        }
        for (int i = 0; i < m_vars.size(); ++i) {
            auto var = m_vars[i];
            const auto& vId = var->getId();
            const auto& vLoc = vLocs[vId];
            const auto& tanDim = var->getTangentDim();
            double* detXPtr = m_DetX.data() + vLoc;
            var->update(detXPtr);
        }
        return true;
    }



    OptimizeLargeSRBA::OptimizeLargeSRBA()
    {
    }
    OptimizeLargeSRBA::~OptimizeLargeSRBA()
    {
    }
    bool OptimizeLargeSRBA::generateAB(double& _err)
    {
        return false;
    }
    bool OptimizeLargeSRBA::solveDetX()
    {
        return false;
    }
    bool OptimizeLargeSRBA::updateX()
    {
        return false;
    }
}