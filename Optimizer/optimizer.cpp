#include "optimizer.h"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace OPTIMIZE_LYJ
{

    OptimizerSmalld::OptimizerSmalld()
    {
    }
    OptimizerSmalld::~OptimizerSmalld()
    {
    }
    bool OptimizerSmalld::init()
    {
        return true;
    }
    bool OptimizerSmalld::generateAB(double &_err)
    {
        int rows = 0;
        int cols = 0;
        std::vector<int> vLocs(this->m_vars.size() + 1, 0);
        int tmp;
        for (int i = 0; i < this->m_vars.size(); ++i)
        {
            if (!this->m_vars[i]->isFixed())
                cols += this->m_vars[i]->getTangentDim();
            vLocs[i + 1] = cols;
        }
        std::vector<int> fLocs(this->m_factors.size() + 1, 0);
        for (int i = 0; i < this->m_factors.size(); ++i)
        {
            auto factor = this->m_factors[i];
            const auto &fId = factor->getId();
            const auto &f2vs = this->m_factor2Vars[fId];
            if (checkEnable(this->m_vars, f2vs, factor))
            {
                factor->setEnable(true);
                rows += this->m_factors[i]->getEDim();
            }
            else
                factor->setEnable(false);
            fLocs[i + 1] = rows;
        }
        if (rows == 0 || cols == 0)
        {
            std::cout << "no factor to optimize" << std::endl;
            return false;
        }
        Eigen::MatrixXd Jac(rows, cols);
        Jac.setConstant(0);
        Eigen::VectorXd Err(rows);
        Err.setConstant(0);

        int connectCnt = 0;
        std::vector<OptVarAbr<double> *> vars;
        std::vector<Eigen::MatrixXd> jacs;
        std::vector<double *> jacPtrs;
        int tanDim, eDim;
        for (int i = 0; i < this->m_factors.size(); ++i)
        {
            auto factor = this->m_factors[i];
            if (!factor->isEnable())
                continue;
            const auto &fId = factor->getId();
            const auto &fLoc = fLocs[fId];
            eDim = factor->getEDim();
            const auto &f2vs = this->m_factor2Vars[fId];
            connectCnt = f2vs.size();

            vars.resize(connectCnt);
            for (int j = 0; j < connectCnt; ++j)
                vars[j] = this->m_vars[f2vs.connectId(j)].get();

            jacs.resize(connectCnt);
            jacPtrs.resize(connectCnt);
            for (int j = 0; j < connectCnt; ++j)
            {
                const auto &vId = f2vs.connectId(j);
                if (this->m_vars[vId]->isFixed())
                {
                    jacPtrs[j] = nullptr;
                }
                else
                {
                    const auto &vLoc = vLocs[vId];
                    tanDim = this->m_vars[vId]->getTangentDim();
                    jacs[j].resize(eDim, tanDim);
                    jacPtrs[j] = jacs[j].data();
                }
            }

            double *errPtr = Err.data() + fLoc;
            factor->calculateErrAndJac(errPtr, jacPtrs.data(), 1, vars.data());

            for (int j = 0; j < connectCnt; ++j)
            {
                const auto &vId = f2vs.connectId(j);
                if (this->m_vars[vId]->isFixed())
                {
                    continue;
                }
                const auto &vLoc = vLocs[vId];
                tanDim = this->m_vars[vId]->getTangentDim();
                for (int ii = 0; ii < eDim; ++ii)
                {
                    for (int jj = 0; jj < tanDim; ++jj)
                    {
                        Jac(fLoc + ii, vLoc + jj) = jacs[j](ii, jj);
                    }
                }
            }
        }

        // std::cout << Jac.rows() << " " << Jac.cols() << std::endl;
        //  std::cout << "Jac: " << std::endl << Jac << std::endl;
        //  std::cout << "Err: " << std::endl << Err << std::endl;
        m_A = Jac.transpose() * Jac;
        m_B = -1 * Jac.transpose() * Err;
        _err = 0;
        for (int i = 0; i < Err.rows(); ++i)
            _err += std::abs(Err(i));
        _err /= Err.rows();

        return true;
    }
    bool OptimizerSmalld::solveDetX()
    {
        // std::cout << "before: " << std::endl;
        // std::cout << "m_A: " << std::endl << m_A << std::endl;
        // std::cout << "m_B: " << std::endl << m_B << std::endl;
        int dim = m_A.rows();
        // for (int i = 0; i < dim; ++i)
        //     m_A(i, i) += 1e-6;
        //  std::cout << "after: " << std::endl;
        //  std::cout << "m_A: " << std::endl << m_A << std::endl;
        //  std::cout << "m_B: " << std::endl << m_B << std::endl;
        //   创建求解器
        Eigen::LDLT<Eigen::MatrixXd> solver;
        solver.compute(m_A);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "Decomposition failed!" << std::endl;
            return false;
        }
        // 求解 Ax = b
        this->m_DetX = solver.solve(m_B);

        // 检查求解结果
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "Solving failed!" << std::endl;
            return false;
        }
        // std::cout << "The solution is:\n" << m_DetX << std::endl;
        return true;
    }
    bool OptimizerSmalld::updateX()
    {
        int cols = 0;
        std::vector<int> vLocs(this->m_vars.size() + 1, 0);
        int tmp;
        for (int i = 0; i < this->m_vars.size(); ++i)
        {
            if (!this->m_vars[i]->isFixed())
                cols += this->m_vars[i]->getTangentDim();
            vLocs[i + 1] = cols;
        }
        for (int i = 0; i < this->m_vars.size(); ++i)
        {
            auto var = this->m_vars[i];
            if (var->isFixed())
                continue;
            const auto &vId = var->getId();
            const auto &vLoc = vLocs[vId];
            const auto &tanDim = var->getTangentDim();
            double *detXPtr = this->m_DetX.data() + vLoc;
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
    bool OptimizerLargeSparse::init()
    {
        return true;
    }
    bool OptimizerLargeSparse::generateAB(double &_err)
    {
        std::vector<Eigen::Triplet<double>> tripletLists;
        int rows = 0;
        int cols = 0;
        std::vector<int> vLocs(this->m_vars.size() + 1, 0);
        int tmp;
        for (int i = 0; i < this->m_vars.size(); ++i)
        {
            if (!this->m_vars[i]->isFixed())
                cols += this->m_vars[i]->getTangentDim();
            vLocs[i + 1] = cols;
        }
        std::vector<int> fLocs(this->m_factors.size() + 1, 0);
        for (int i = 0; i < this->m_factors.size(); ++i)
        {
            auto factor = this->m_factors[i];
            const auto &fId = factor->getId();
            const auto &f2vs = this->m_factor2Vars[fId];
            if (checkEnable(this->m_vars, f2vs, factor))
            {
                factor->setEnable(true);
                rows += this->m_factors[i]->getEDim();
            }
            else
                factor->setEnable(false);
            fLocs[i + 1] = rows;
        }
        if (rows == 0 || cols == 0)
        {
            std::cout << "no factor to optimize" << std::endl;
            return false;
        }
        tripletLists.reserve(cols * 100);
        Eigen::SparseMatrix<double> Jac(rows, cols);
        Eigen::VectorXd Err(rows);
        Err.setZero();

        int connectCnt = 0;
        std::vector<OptVarAbr<double> *> vars;
        std::vector<Eigen::MatrixXd> jacs;
        std::vector<double *> jacPtrs;
        int tanDim, eDim;
        for (int i = 0; i < this->m_factors.size(); ++i)
        {
            auto factor = this->m_factors[i];
            if (!factor->isEnable())
                continue;
            const auto &fId = factor->getId();
            const auto &fLoc = fLocs[fId];
            eDim = factor->getEDim();
            const auto &f2vs = this->m_factor2Vars[fId];
            connectCnt = f2vs.size();

            vars.resize(connectCnt);
            for (int j = 0; j < connectCnt; ++j)
                vars[j] = this->m_vars[f2vs.connectId(j)].get();

            jacs.resize(connectCnt);
            jacPtrs.resize(connectCnt);
            for (int j = 0; j < connectCnt; ++j)
            {
                const auto &vId = f2vs.connectId(j);
                if (this->m_vars[vId]->isFixed())
                {
                    jacPtrs[j] = nullptr;
                }
                else
                {
                    const auto &vLoc = vLocs[vId];
                    tanDim = this->m_vars[vId]->getTangentDim();
                    jacs[j].resize(eDim, tanDim);
                    jacPtrs[j] = jacs[j].data();
                }
            }

            double *errPtr = Err.data() + fLoc;
            factor->calculateErrAndJac(errPtr, jacPtrs.data(), 1, vars.data());

            for (int j = 0; j < connectCnt; ++j)
            {
                const auto &vId = f2vs.connectId(j);
                if (this->m_vars[vId]->isFixed())
                {
                    continue;
                }
                const auto &vLoc = vLocs[vId];
                tanDim = this->m_vars[vId]->getTangentDim();
                for (int ii = 0; ii < eDim; ++ii)
                {
                    for (int jj = 0; jj < tanDim; ++jj)
                    {
                        tripletLists.emplace_back(fLoc + ii, vLoc + jj, jacs[j](ii, jj));
                    }
                }
            }
        }

        Jac.setFromTriplets(tripletLists.begin(), tripletLists.end());
        m_A = Jac.transpose() * Jac;
        m_B = -1 * Jac.transpose() * Err;
        _err = 0;
        for (int i = 0; i < Err.rows(); ++i)
            _err += std::abs(Err(i));
        _err /= Err.rows();

        return true;
    }
    bool OptimizerLargeSparse::solveDetX()
    {
        // 创建求解器
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(m_A);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "Decomposition failed!" << std::endl;
            return false;
        }

        //Eigen::MatrixXd A(m_A);
        //// 2. 创建自伴随（实对称）特征值求解器，计算特征值和特征向量
        //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(A);
        //// 检查求解是否成功
        //if (eigensolver.info() != Eigen::Success) {
        //    std::cerr << "特征值求解失败！" << std::endl;
        //    return -1;
        //}
        //// 3. 获取结果
        //// 特征值（已按升序排列）
        //Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
        //// 4. 输出结果
        //std::cout << "特征值（升序）：\n" << eigenvalues.minCoeff() << "\n\n";

        // 求解 Ax = b
        m_DetX = solver.solve(m_B);
        // 检查求解结果
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "Solving failed!" << std::endl;
            return false;
        }
        // std::cout << "The solution is:\n" << m_DetX << std::endl;
        return true;
    }
    bool OptimizerLargeSparse::updateX()
    {
        int cols = 0;
        std::vector<int> vLocs(this->m_vars.size() + 1, 0);
        int tmp;
        for (int i = 0; i < this->m_vars.size(); ++i)
        {
            if (!this->m_vars[i]->isFixed())
                cols += this->m_vars[i]->getTangentDim();
            vLocs[i + 1] = cols;
        }
        for (int i = 0; i < this->m_vars.size(); ++i)
        {
            auto var = this->m_vars[i];
            if (var->isFixed())
                continue;
            const auto &vId = var->getId();
            const auto &vLoc = vLocs[vId];
            const auto &tanDim = var->getTangentDim();
            double *detXPtr = this->m_DetX.data() + vLoc;
            var->update(detXPtr);
        }
        return true;
    }

    OptimizeLargeSRBA::OptimizeLargeSRBA()
    {}
    OptimizeLargeSRBA::~OptimizeLargeSRBA()
    {}
    bool OptimizeLargeSRBA::init()
    {
        return true;
    }
    bool OptimizeLargeSRBA::generateAB(double &_err)
    {
        auto funcCalFactorMats = [&]()->bool
            {
                m_factorMats.resize(m_factors.size());
                int connectCnt = 0;
                std::vector<OptVarAbr<double>*> vars;
                std::vector<double*> jacPtrs;
                int tanDim, eDim;
                for (int i = 0; i < this->m_factors.size(); ++i)
                {
                    auto factor = this->m_factors[i];
                    if (!factor->isEnable())
                    {
                        std::cout << "error factor!" << std::endl;
                        return false;
                    }
                    std::vector<Eigen::MatrixXd>& jacs = m_factorMats[i].m_jacs;
                    m_factorMats[i].m_factor = factor;
                    const auto& fId = factor->getId();
                    eDim = factor->getEDim();
                    m_factorMats[i].m_err.resize(eDim);
                    const auto& f2vs = this->m_factor2Vars[fId];
                    connectCnt = f2vs.size();
                    vars.resize(connectCnt);
                    for (int j = 0; j < connectCnt; ++j)
                        vars[j] = this->m_vars[f2vs.connectId(j)].get();
                    jacs.resize(connectCnt);
                    jacPtrs.resize(connectCnt);
                    for (int j = 0; j < connectCnt; ++j)
                    {
                        const auto& vId = f2vs.connectId(j);
                        tanDim = this->m_vars[vId]->getTangentDim();
                        jacs[j].resize(eDim, tanDim);
                        jacPtrs[j] = jacs[j].data();
                    }
                    double* errPtr = m_factorMats[i].m_err.data();
                    factor->calculateErrAndJac(errPtr, jacPtrs.data(), 1, vars.data());


                }
            };
        return true;
    }
    bool OptimizeLargeSRBA::solveDetX()
    {

        return true;
    }
    bool OptimizeLargeSRBA::updateX()
    {

        return true;
    }
}