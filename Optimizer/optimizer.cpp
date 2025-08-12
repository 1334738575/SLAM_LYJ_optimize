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
        _err = Err.norm();

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
        _err = Err.norm();

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
    {
    }
    OptimizeLargeSRBA::~OptimizeLargeSRBA()
    {
    }
    bool OptimizeLargeSRBA::generateAB(double &_err)
    {
        // 默认id从0开始并连续
        int fId1 = 0;
        int fLoc1 = 0;
        int fId2 = 1;
        int fLoc2 = 6;
        std::vector<int> vLocs(this->m_vars.size() + 1, 0);
        for (int i = 0; i < this->m_vars.size(); ++i)
        {
            vLocs[i + 1] = this->m_vars[i]->getTangentDim() + vLocs[i];
        }
        std::vector<int> fLocs(this->m_factors.size() + 1, 0);
        for (int i = 0; i < this->m_factors.size(); ++i)
        {
            fLocs[i + 1] = this->m_factors[i]->getEDim() + fLocs[i];
        }
        int rows = fLocs.back();
        int cols = vLocs.back();
        if (rows == 0 || cols == 0)
        {
            std::cout << "no factor to optimize" << std::endl;
            return false;
        }
        Eigen::MatrixXd Jac(rows, cols);
        Jac.setZero();
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
                const auto &vLoc = vLocs[vId];
                tanDim = this->m_vars[vId]->getTangentDim();
                jacs[j].resize(eDim, tanDim);
                jacPtrs[j] = jacs[j].data();
            }

            double *errPtr = Err.data() + fLoc;
            factor->calculateErrAndJac(errPtr, jacPtrs.data(), 1, vars.data());

            for (int j = 0; j < connectCnt; ++j)
            {
                const auto &vId = f2vs.connectId(j);
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

        int PSize = this->m_vars.size() - 2;
        Eigen::Matrix<double, 4, 12> jacTmpT;
        jacTmpT.setZero();
        Eigen::Matrix<double, 4, 3> jacTmpP;
        jacTmpP.setZero();
        Eigen::Vector4d errTmp;
        errTmp.setZero();
        newJac.resize(PSize, 6);
        newJac.setZero();
        newErr.resize(PSize);
        newErr.setZero();
        newJac2.resize(3 * PSize, 12);
        newJac2.setZero();
        newJac3.resize(3 * PSize, 3 * PSize);
        newJac3.setZero();
        newErr2.resize(3 * PSize);
        newErr2.setZero();
        for (int i = 0; i < PSize; ++i)
        {
            int vId = i + 2;
            int r1 = 4 * i;
            int r2 = r1 + 2;
            int c1 = 0;
            int c2 = 6;
            int c3 = 12 + 3 * i;
            jacTmpT.block(0, 0, 2, 6) = Jac.block(r1, c1, 2, 6);
            jacTmpT.block(2, 6, 2, 6) = Jac.block(r2, c2, 2, 6);
            jacTmpP.block(0, 0, 2, 3) = Jac.block(r1, c3, 2, 3);
            jacTmpP.block(2, 0, 2, 3) = Jac.block(r2, c3, 2, 3);
            errTmp.block(0, 0, 2, 1) = Err.block(r1, 0, 2, 1);
            errTmp.block(2, 0, 2, 1) = Err.block(r2, 0, 2, 1);
            // ========== 列主元QR分解（数值稳定性更优） ==========
            Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 4, 3>> colPivQR(jacTmpP);
            Eigen::Matrix4d Q = colPivQR.householderQ();         // *Eigen::MatrixXd::Identity(jacTmpP.rows(), jacTmpP.cols());
            Eigen::Matrix<double, 4, 3> R = colPivQR.matrixQR(); // .triangularView<Eigen::Upper>();
            // Eigen::MatrixXd ttt = Q.transpose() * jacTmpP;
            // std::cout << ttt << std::endl;
            // std::cout << "列主元QR分解结果：\n"
            //     << "Q矩阵：\n" << Q << "\n"
            //     << "R矩阵：\n" << R << "\n\n";

            Eigen::Matrix4d Qt = Q.transpose();
            Eigen::Matrix<double, 4, 12> jacTmpT_QR = Qt * jacTmpT;
            Eigen::Matrix<double, 4, 3> jacTmpP_QR = Qt * jacTmpP;
            Eigen::Vector4d errTmp_QR = Qt * errTmp;
            newJac.block(i, 0, 1, 6) = jacTmpT_QR.block(3, 6, 1, 6);
            newErr(i) = errTmp_QR(3);
            newJac2.block(3 * i, 0, 3, 12) = jacTmpT_QR.block(0, 0, 3, 12);
            newJac3.block(3 * i, 3 * i, 3, 3) = jacTmpP_QR.block(0, 0, 3, 3);
            newErr2.block(3 * i, 0, 3, 1) = errTmp_QR.block(0, 0, 3, 1);
        }

        // std::cout << Jac.rows() << " " << Jac.cols() << std::endl;
        //  std::cout << "Jac: " << std::endl << Jac << std::endl;
        //  std::cout << "Err: " << std::endl << Err << std::endl;
        m_A = newJac.transpose() * newJac;
        m_B = -1 * newJac.transpose() * newErr;
        _err = newErr.norm() + newErr2.norm();
        return true;
    }
    bool OptimizeLargeSRBA::solveDetX()
    {
        // std::cout << "before: " << std::endl;
        // std::cout << "m_A: " << std::endl << m_A << std::endl;
        // std::cout << "m_B: " << std::endl << m_B << std::endl;
        int dim = m_A.rows();
        //  创建求解器
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

        Eigen::Matrix<double, 12, 1> detXTmp;
        detXTmp.setZero();
        detXTmp.block(6, 0, 6, 1) = this->m_DetX.block(0, 0, 6, 1);
        Eigen::VectorXd err = newJac2 * detXTmp - newErr2;
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> colPivQR(newJac3);
        // 使用QR分解求解Ax = b（最小二乘解）
        newDetX = colPivQR.solve(err);
        // std::cout << "线性方程组解：\n" << newDetX << "\n"
        //     << "残差范数：||Ax - b|| = "
        //     << (newJac3 * newDetX - err).norm() << "\n\n";

        return true;
    }
    bool OptimizeLargeSRBA::updateX()
    {
        double *detTPtr = this->m_DetX.data();
        this->m_vars[1]->update(detTPtr);
        for (int i = 2; i < this->m_vars.size(); ++i)
        {
            auto var = this->m_vars[i];
            double *detXPtr = newDetX.data() + 3 * (i - 2);
            var->update(detXPtr);
        }
        return true;
    }
}