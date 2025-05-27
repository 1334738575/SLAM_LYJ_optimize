#ifndef OPTIMIZE_OPTIMIZERBALABR_H
#define OPTIMIZE_OPTIMIZERBALABR_H

#include "Factor/Factor.h"
#include <map>
#include <Eigen/Eigen>


namespace OPTIMIZE_LYJ
{
	template<typename T>
	class OptimizerBALAbr
	{
	public:
		class OptConnect
		{
		public:
			OptConnect() {};
			OptConnect(int _num) {
				m_connectIds.assign(_num, UINT64_MAX);
				m_valids.assign(_num, false);
				m_locs.resize(_num);
			}
			OptConnect(const std::vector<uint64_t>& _ids) {
				int num = _ids.size();
				m_connectIds = _ids;
				m_valids.assign(num, true);
				m_locs.resize(num);
			}
			~OptConnect() {
				//if (m_connectIds) {
				//	delete m_connectIds;
				//	delete m_valids;
				//	delete m_locs;
				//}
			}
			inline int size() const { return m_connectIds.size(); }
			void reset(int _num) {
				m_connectIds.assign(_num, UINT64_MAX);
				m_valids.assign(_num, false);
				m_locs.resize(_num);
			}
			void addConnectId(const uint64_t& _id) {
				m_connectIds.push_back(_id);
				m_valids.push_back(true);
				m_locs.push_back(std::pair<int, int>(-1, -1));
			}
			void addConnectId(const uint64_t& _id, const int& _l1, const int& _l2) {
				m_connectIds.push_back(_id);
				m_valids.push_back(true);
				m_locs.push_back(std::pair<int, int>(_l1, _l2));
			}
			inline void setConnectId(const int _i, const uint64_t& _id) { m_connectIds[_i] = _id; }
			inline uint64_t& connectId(const int _i) { return m_connectIds[_i]; }
			inline const uint64_t& connectId(const int _i) const { return m_connectIds[_i]; }
			inline void setValid(const int _i, const bool& _valid) { m_valids[_i] = _valid; }
			inline bool& valid(const int _i) { return m_valids[_i]; }
			inline const bool& valid(const int _i) const { return m_valids[_i]; }
			inline void setLoc(const int _i, const int& _l1, const int& _l2) {
				m_locs[_i].first = _l1;
				m_locs[_i].second = _l2;
			}
			inline std::pair<int, int>& loc(const int _i) { return m_locs[_i]; }
			inline const std::pair<int, int>& loc(const int _i) const { return m_locs[_i]; }

		private:
			std::vector<uint64_t> m_connectIds;
			std::vector<bool> m_valids;
			std::vector<std::pair<int, int>> m_locs;
		};
		typedef OptConnect Var2Factor;
		typedef OptConnect Factor2Var;
		typedef OptConnect Var2Var;
		//typedef OptConnect Pose2Othervar;
		//typedef OptConnect Othervar2Pose;

		//colmajor
		template<typename T>
		class MatInner
		{
		public:
			MatInner() {}
			MatInner(const int _r, const int _c) : m_r(_r), m_c(_c)
			{
				m_data = new T[_r * _c];
				reset();
			}
			~MatInner() {
				if (m_data)
					delete m_data;
			}
			inline void reset() { memset(m_data, 0, sizeof(T) * m_r * m_c); }
			inline T* getData() { return m_data; }
			inline const T* getData() const { return m_data; }
			inline void setData(T* _data) { memcpy(m_data, _data, sizeof(T) * m_r * m_c); }
			inline const int r() const { return m_r; }
			inline const int c() const { return m_c; }
			Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> getEigenMap() {
				return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(m_data, m_r, m_c);
			}
		private:
			T* m_data = nullptr;
			int m_r = -1;
			int m_c = -1;
		};
		class JacManager
		{
		public:
			JacManager() {
			}
			~JacManager() {}

			inline void addJac(std::shared_ptr<MatInner<T>> _jac)
			{
				m_jacs.push_back(_jac);
			}

			//非位姿jac，QR分解，变换位姿jac
			void QR() {

			}

		private:
			std::vector<std::shared_ptr<MatInner<T>>> m_jacs;
			std::shared_ptr<MatInner<T>> m_Q = nullptr;
		};
		class Jacs
		{
		public:
			Jacs(const int& _eDim, const std::vector<int>& _vDims) {
				m_jacs.resize(_vDims.size());
				for (int i = 0; i < _vDims.size(); i++)
					m_jacs[i].reset(new MatInner<T>(_eDim, _vDims[i]));
			}
			~JacManager() {}

		private:
			std::vector<std::shared_ptr<MatInner<T>>> m_jacs;
		};

		OptimizerAbr() {}
		~OptimizerAbr() {
			//std::cout << "release optimizerAbr" << std::endl;
		}

		virtual bool run() {
			for (int i = 0; i < m_maxIterNum; ++i) {
				T err = -1;
				if (!generateAB(err))
					return false;
				if (!solveDetX())
					return false;
				if (!updateX())
					return false;
				if (isFinish(i, err))
					break;
			}
			return true;
		}

		virtual bool generateAB(T& _err) {
			//compute all jac

			//qr choosed variable

			//shape A B
		}

		virtual bool solveDetX() {
			//solve X not qr

			//solve X by qr
		}

		virtual bool updateX() {

		}

		virtual bool isFinish(const int _i, const T _err) {
			m_lastErr = _err;
			std::cout << "iter: " << _i << ", err: " << _err << std::endl;
			if (_err <= m_minErrTh || _i >= m_maxIterNum)
				return true;
			return false;
		}

		virtual bool addVariable(std::shared_ptr<OptVarAbr<T>> _var)
		{
			m_vars.push_back(_var);
			return true;
		}

		virtual bool addFactor(std::shared_ptr<OptFactorAbr<T>> _factor, const std::vector<uint64_t>& _vIds)
		{
			m_factors.push_back(_factor);
			const uint64_t fId = m_factors.size();
			m_factor2Vars[fId] = Factor2Var(_vIds);
			for (int i = 0; i < _vIds.size(); ++i)
				m_var2Factors[_vIds[i]].addConnectId(fId);
			return true;
		}

	protected:
		bool generateMap() {
			int varNum = m_vars.size();
			m_var2Vars.clear();
			m_poseLoc.clear();
			m_landmarkLoc.clear();
			m_otherLoc.clear();
			m_poses.clear();
			m_pose2Factors.clear();
			m_landmarks.clear();
			m_landmark2Factors.clear();
			m_others.clear();
			m_other2Factors.clear();
			m_jacManagers.clear();
			m_var2Vars.resize(varNum);
			m_poseLoc.assign(varNum, UINT64_MAX);
			m_landmarkLoc.assign(varNum, UINT64_MAX);
			m_otherLoc.assign(varNum, UINT64_MAX);
			m_poses.reserve(varNum);
			m_pose2Factors.reserve(varNum);
			m_landmarks.reserve(varNum);
			m_landmark2Factors.reserve(varNum);
			m_others.reserve(varNum);
			m_other2Factors.reserve(varNum);
			m_jacManagers.resize(varNum);
			for (int i = 0; i < varNum; ++i) {
				const auto& var = m_vars[i];
				if (var->getType() == VAR_DEAULT) {
					std::cout << "error!!!" << std::endl;
					return false;
				}
				else if (var->getType() == VAR_T3D || var->getType() == VAR_T2D)
				{
					m_poseLoc[i] = i;
					m_poses.push_back(var);
					m_pose2Factors.push_back(m_var2Factors[i]);
				}
				else if (var->getType() == VAR_POINT3D || var->getType() == VAR_POINT2D ||
					var->getType() == VAR_LINE3D || var->getType() == VAR_LINE2D ||
					var->getType() == VAR_PLANE3D)
				{
					m_landmarkLoc[i] = i;
					m_landmarks.push_back(var);
					m_landmark2Factors.push_back(m_var2Factors[i]);
				}
				else
				{
					m_otherLoc[i] = i;
					m_others.push_back(var);
					m_other2Factors.push_back(m_var2Factors[i]);
				}
			}
			int factorNum = m_factors.size();
			m_jacs.clear();
			m_jacs.reserve(factorNum);
			for (int i = 0; i < factorNum; ++i) {
				const auto factor = m_factors[i];
				const int eDim = m_factors[i]->getEDim();
				const std::vector<int>& vDims = m_factors[i]->getVDims();
				m_jacs.emplace_back(eDim, vDims);
				const auto& f2vs = m_factor2Vars[i];
				int conNum = f2vs.size();
				for (int ci = 0; ci < conNum; ++ci) {
					const auto& conId1 = f2vs.connectId(ci);
					for (int cj = ci + 1; cj < conNum; ++cj) {
						const auto& conId2 = f2vs.connectId(cj);
						m_var2Vars[conId1].addConnectId(conId2, i, cj);
						m_var2Vars[conId2].addConnectId(conId1, i, ci);
					}
				}
			}

			for (int i = 0; i < varNum; ++i) {
				auto& jacManager = m_jacManagers[i];
				const auto& v2vs = m_var2Vars[i];
				int conNum = v2vs.size();
				for (int ci = 0; ci < conNum; ++ci) {
					const auto& conId = v2vs.connectId(ci);
					const auto& loc = v2vs.loc(ci);
					auto jac = m_jacs[loc.first].getJac(loc.second);
					jacManager.addJac(jac);
				}
			}

			return true;
		}

	protected:
		int m_maxIterNum = 30;
		T m_minErrTh = 1e-6;
		T m_lastErr = -1;

		std::vector<std::shared_ptr<OptVarAbr<T>>> m_vars;
		std::map<uint64_t, Var2Factor> m_var2Factors;

		std::vector<Var2Var> m_var2Vars;
		std::vector<uint64_t> m_poseLoc;
		std::vector<uint64_t> m_landmarkLoc;
		std::vector<uint64_t> m_otherLoc;
		std::vector<std::shared_ptr<OptVarAbr<T>>> m_poses;
		std::vector<Var2Factor> m_pose2Factors;
		std::vector<std::shared_ptr<OptVarAbr<T>>> m_landmarks;
		std::vector<Var2Factor> m_landmark2Factors;
		std::vector<std::shared_ptr<OptVarAbr<T>>> m_others;
		std::vector<Var2Factor> m_other2Factors;
		std::vector<JacManager> m_jacManagers;


		std::vector<std::shared_ptr<OptFactorAbr<T>>> m_factors;
		std::map<uint64_t, Factor2Var> m_factor2Vars;

		std::vector<Jacs> m_jacs;
	};




}


#endif //OPTIMIZE_OPTIMIZERBALABR_H