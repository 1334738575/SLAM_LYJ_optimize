#ifndef OPTIMIZE_LYJ_OPTIMIZER_H
#define OPTIMIZE_LYJ_OPTIMIZER_H

#include "optimizerAbr.h"
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Dense>
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
				auto& jacsRemand = m_factorMatRemand->m_jacs;
				auto& errRemand = m_factorMatRemand->m_err;
				auto& f2vsRemand = m_factorMatRemand->m_f2vs;
				auto& varsRemand = m_factorMatRemand->m_vars;
				auto& jacsEliminate = m_factorMatEliminate.m_jacs;
				auto& errEliminate = m_factorMatEliminate.m_err;
				auto& f2vsEliminate = m_factorMatEliminate.m_f2vs;
				auto& varsEliminate = m_factorMatEliminate.m_vars;
				int iSz = m_fMatsIn.size();
				std::map<uint64_t, OptVarAbr<double>*> vIdMaps;
				const auto& vIdEliminate = m_vId;
				OptVarAbr<double>* varEliminate;
				int rows = 0;
				std::vector<int> rLocs(iSz + 1, 0);
				for (int i = 0; i < iSz; ++i)
				{
					const auto& f2vs = m_fMatsIn[i]->m_f2vs;
					const auto& vars = m_fMatsIn[i]->m_vars;
					int cSz = f2vs.size();
					rows += m_fMatsIn[i]->m_err.rows();
					rLocs[i + 1] = rows;
					for (int j = 0; j < cSz; ++j)
					{
						int vId = f2vs.connectId(j);
						if (m_vId == vId)
						{
							varEliminate = vars[i];
							continue;
						}
						vIdMaps[vId] = vars[i];
					}
				}
				int remandSz = vIdMaps.size();
				jacsRemand.resize(remandSz);
				std::vector<uint64_t> vIdsRemand(remandSz);
				varsRemand.resize(remandSz);
				int ind = 0;
				std::map<uint64_t, int> id2Loc;
				int cols = 0;
				std::vector<int> cLocs(remandSz + 1, 0);
				for (auto& vIdMap : vIdMaps)
				{
					const auto& vId = vIdMap.first;
					auto& var = vIdMap.second;
					vIdsRemand[ind] = vId;
					varsRemand[ind] = var;
					int vDim = var->getTangentDim();
					id2Loc[vId] = ind;
					++ind;
					cols += vDim;
					cLocs[ind] = cols;
				}
				f2vsRemand = Factor2Var(vIdsRemand);
				vIdsRemand.push_back(m_vId);
				f2vsEliminate = Factor2Var(vIdsRemand);
				varsEliminate = varsRemand;
				varsEliminate.push_back(varEliminate);
				jacsEliminate.resize(remandSz + 1);

				Eigen::MatrixXd jac1;
				jac1.resize(rows, cols);
				jac1.setZero();
				int cols2 = varEliminate->getTangentDim();
				Eigen::MatrixXd jac2;
				jac2.resize(rows, cols2);
				jac2.setZero();
				Eigen::VectorXd err1;
				err1.resize(rows);
				err1.setZero();
				for (int i = 0; i < iSz; ++i)
				{
					auto& jacs = m_fMatsIn[i]->m_jacs;
					const auto& f2vs = m_fMatsIn[i]->m_f2vs;
					auto& err = m_fMatsIn[i]->m_err;
					int cSz = f2vs.size();
					int sr = rLocs[i];
					int er = rLocs[i + 1];
					for (int j = 0; j < cSz; ++j)
					{
						int vId = f2vs.connectId(j);
						auto& jac = jacs[j];
						int sc, ec;
						if (m_vId == vId)
						{
							sc = 0;
							ec = cols2;
							jac2.block(sr, sc, er - sr, ec - sc) = jac;
						}
						else
						{
							int loc = id2Loc[vId];
							sc = cLocs[loc];
							ec = cLocs[loc + 1];
							jac1.block(sr, sc, er - sr, ec - sc) = jac;
						}
					}
					err1.block(sr, 0, er - sr, 1) = err;
				}

				Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(jac2);
				// 获取 Q、R 和列置换索引（pivots）
				Eigen::MatrixXd Qt = qr.householderQ().transpose();
				//Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
				//Eigen::VectorXi pivots = qr.colsPermutation().indices(); // 列置换索引
				Eigen::MatrixXd Qtjac1 = Qt * jac1;
				Eigen::MatrixXd Qtjac2 = Qt * jac2;
				Eigen::VectorXd Qterr1 = Qt * err1;

				for (int i = 0; i < remandSz; ++i)
				{
					int sc = cLocs[i];
					int ec = cLocs[i + 1];
					jacsEliminate[i] = Qtjac1.block(0, sc, cols2, ec - sc);
				}
				jacsEliminate[remandSz] = Qtjac2.block(0, 0, cols2, cols2);
				errEliminate = Qterr1.block(0, 0, cols2, 1);

				int rows2 = rows - cols2;
				for (int i = 0; i < remandSz; ++i)
				{
					int sc = cLocs[i];
					int ec = cLocs[i + 1];
					jacsRemand[i] = Qtjac1.block(cols2, sc, rows2, ec - sc);
				}
				errRemand = Qterr1.block(cols2, 0, rows2, 1);
			}
			void solveElimination(std::vector<Eigen::Map<Eigen::VectorXd>> _dXs)
			{
				const auto& f2vs = m_factorMatEliminate.m_f2vs;
				int cSz = f2vs.size();
				const auto& jacs = m_factorMatEliminate.m_jacs;
				Eigen::VectorXd err = m_factorMatEliminate.m_err;
				const auto& vars = m_factorMatEliminate.m_vars;
				int ind = -1;
				for (int i = 0; i < cSz; ++i)
				{
					const auto& vId = f2vs.connectId(i);
					if (m_vId == vId)
					{
						ind = i;
						continue;
					}
					err += (jacs[i] * _dXs[i]);
				}
				Eigen::VectorXd dX = jacs[ind].inverse() * err;
				vars[ind]->update(dX.data());
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
