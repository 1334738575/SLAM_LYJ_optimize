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
        //m_vars.clear();
        //m_var2Factors.clear();
        //m_factors.clear();
        //m_factor2Vars.clear();
        m_eliminationType.clear();
        for (int i = 0; i < m_vars.size(); ++i)
        {
            const auto& t = m_vars[i]->getType();
            if (t == VAR_T3D || t == VAR_T2D || t == VAR_IMU)
                continue;
            m_eliminationType.insert(t);
        }
        return true;
    }
    bool OptimizeLargeSRBA::generateAB(double &_err)
    {
        m_factorMats.clear();
        m_eliminationMats.clear();
        m_cLocs.clear();
        m_rLocs.clear();
        int vSz = m_vars.size();
        int fSz = m_factors.size();

        int incCnt = 0;
        for (int i = 0; i < vSz; ++i)
        {
            const auto& t = m_vars[i]->getType();
            const int& vId = m_vars[i]->getId();
            if (m_eliminationType.count(t))
                ++incCnt;
        }

        auto funcCalFactorMats = [&]()->bool
            {
                m_factorMats.resize(fSz + incCnt);
                int connectCnt = 0;
                std::vector<OptVarAbr<double>*> vars;
                std::vector<double*> jacPtrs;
                int tanDim, eDim;
                for (int i = 0; i < fSz; ++i)
                {
                    auto factor = this->m_factors[i];
                    if (!factor->isEnable())
                    {
                        std::cout << "error factor!" << std::endl;
                        return false;
                    }
                    const auto& fId = factor->getId();
                    eDim = factor->getEDim();
                    m_factorMats[i].m_err.resize(eDim);
                    const auto& f2vs = this->m_factor2Vars[fId];
                    std::vector<Eigen::MatrixXd>& jacs = m_factorMats[i].m_jacs;
                    m_factorMats[i].m_f2vs = f2vs;
                    m_factorMats[i].m_id = fId;
                    //m_factorMats[i].m_factor = factor;
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
                    m_factorMats[i].m_vars = vars;
                    double* errPtr = m_factorMats[i].m_err.data();
                    factor->calculateErrAndJac(errPtr, jacPtrs.data(), 1, vars.data());
                }
                return true;
            };
        funcCalFactorMats();

        int cols = 0;
        m_cLocs.resize(vSz + 1, 0);
        std::vector<FactorMat*> fMatsTmp;
        FactorMat fMatTmp;
        for (int i = 0; i < vSz; ++i)
        {
            const auto& t = m_vars[i]->getType();
            const int& vId = m_vars[i]->getId();
            fMatsTmp.clear();
            if (m_vars[i]->isFixed())
            {

            }
            else if (m_eliminationType.count(t))
            {
                //new factor
                Var2Factor v2fs = m_var2Factors[vId];
                int cSz = v2fs.size();
                m_eliminationMats.emplace_back(t, vId);
                auto& fMatsIn = m_eliminationMats.back().m_fMatsIn;
                fMatsIn.resize(cSz);
                for (int j = 0; j < cSz; ++j)
                {
                    const auto& fId = v2fs.connectId(j);
                    //FactorMat* fMat = &m_factorMats[fId];
                    //fMatsTmp.push_back(fMat);
                    //fMat->m_valid = false;
                    //fMatsIn.push_back(&m_factorMats[fId]);
                    //fMatsIn.back()->m_valid = false;
                    fMatsIn[j] = &m_factorMats[fId];
                    fMatsIn[j]->m_valid = false;
                }
                //m_eliminationMats.emplace_back(fMatsTmp, t, vId);
                //fMatTmp.m_id = m_factorMats.size();
                //m_factorMats.push_back(fMatTmp);
                //m_eliminationMats.back().m_factorMatRemand = &m_factorMats.back();
            }
            else
            {
                cols += m_vars[i]->getTangentDim();
            }
            m_cLocs[i + 1] = cols;
        }

        int incSz = m_eliminationMats.size();
        for (int i = 0; i < incSz; ++i)
        {
            m_factorMats[fSz + i].m_id = fSz + i;
            m_eliminationMats[i].m_factorMatRemand = &m_factorMats[fSz + i];
            m_eliminationMats[i].QR();
        }

        int fSzNew = m_factorMats.size();
        m_rLocs.resize(fSzNew + 1, 0);
        int rows = 0;
        for (int i = 0; i < fSzNew; ++i)
        {
            if (m_factorMats[i].m_valid)
                rows += m_factorMats[i].getEDim();
            m_rLocs[i + 1] = rows;
        }

        std::vector<Eigen::Triplet<double>> tripletLists;
        tripletLists.reserve(cols * 100);
        Eigen::SparseMatrix<double> Jac(rows, cols);
        Eigen::VectorXd Err(rows);
        Err.setZero();
        for (int i = 0; i < fSzNew; ++i)
        {
            int sr = m_rLocs[i];
            int er = m_rLocs[i + 1];
            if (sr == er)
                continue;
            //const auto& vars = m_factorMats[i].m_vars;
            const auto& f2vs = m_factorMats[i].m_f2vs;
            int cSz = f2vs.size();
            for (int j = 0; j < cSz; ++j)
            {
                const auto& vId = f2vs.connectId(j);
                int sc = m_cLocs[vId];
                int ec = m_cLocs[vId + 1];
                if (sc == ec)
                    continue;
                const auto& jac = m_factorMats[i].m_jacs[j];
                const auto& err = m_factorMats[i].m_err;
                for (int r = 0; r < (er - sr); ++r)
                {
                    for (int c = 0; c < (ec - sc); ++c)
                    {
                        tripletLists.emplace_back(sr + r, sc + c, jac(r, c));
                    }
                    Err(sr + r) = err(r);
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
    bool OptimizeLargeSRBA::solveDetX()
    {
        // 创建求解器
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(m_A);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "Decomposition failed!" << std::endl;
            return false;
        }
        //std::cout << m_A.rows() << " " << m_A.cols() << std::endl;
        //std::cout << m_B.rows() << std::endl;
        //std::cout << m_B.cols() << std::endl;

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
    bool OptimizeLargeSRBA::updateX()
    {
        for (int i = 0; i < this->m_vars.size(); ++i)
        {
            int sc = m_cLocs[i];
            int ec = m_cLocs[i + 1];
            if (sc == ec)
                continue;
            auto var = this->m_vars[i];
            double* detXPtr = this->m_DetX.data() + sc;
            var->update(detXPtr);
        }

        int eSz = m_eliminationMats.size();
        std::vector<Eigen::Map<Eigen::VectorXd>> dXs;
        Eigen::VectorXd dXEli;
        for (int i = 0; i < eSz; ++i)
        {
            dXs.clear();
            const auto& vars = m_eliminationMats[i].m_factorMatEliminate.m_vars;
            const auto& f2vs = m_eliminationMats[i].m_factorMatEliminate.m_f2vs;
            const auto& vIdEli = m_eliminationMats[i].m_vId;
            int cSz = f2vs.size();
            int eliDim = m_vars[vIdEli]->getTangentDim();
            dXEli.resize(eliDim);
            //dXs.resize(cSz);
            for (int j = 0; j < cSz; ++j)
            {
                const auto& vId = f2vs.connectId(j);
                if (vId == vIdEli)
                {
                    Eigen::Map<Eigen::VectorXd> dX(dXEli.data(), eliDim);
                    dXs.push_back(dX);
                    continue;
                }
                int sc = m_cLocs[vId];
                int ec = m_cLocs[vId + 1];
                Eigen::Map<Eigen::VectorXd> dX(m_DetX.data() + sc, ec - sc);
                dXs.push_back(dX);
            }
            m_eliminationMats[i].solveElimination(dXs);
        }
        return true;
    }
}