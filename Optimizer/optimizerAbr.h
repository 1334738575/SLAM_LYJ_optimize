#ifndef OPTIMIZE_OPTIMIZERABR_H
#define OPTIMIZE_OPTIMIZERABR_H

#include "Factor/Factor.h"
#include <map>
#include <memory>
#include <Optimize_LYJ_Defines.h>
#include <chrono>

namespace OPTIMIZE_LYJ
{

	template <typename T>
	class OPTIMIZE_LYJ_API OptimizerAbr
	{
	public:
		class OptConnect
		{
		public:
			OptConnect() {};
			OptConnect(int _num)
			{
				m_connectIds.assign(_num, UINT64_MAX);
				m_valids.assign(_num, false);
				m_locs.resize(_num);
			}
			OptConnect(const std::vector<uint64_t> &_ids)
			{
				int num = _ids.size();
				m_connectIds = _ids;
				m_valids.assign(num, true);
				m_locs.resize(num);
			}
			~OptConnect()
			{
				// if (m_connectIds) {
				//	delete m_connectIds;
				//	delete m_valids;
				//	delete m_locs;
				// }
			}
			inline int size() const { return m_connectIds.size(); }
			void reset(int _num)
			{
				m_connectIds.assign(_num, UINT64_MAX);
				m_valids.assign(_num, false);
				m_locs.resize(_num);
			}
			inline void addConnectId(const uint64_t &_id)
			{
				m_connectIds.push_back(_id);
				m_valids.push_back(true);
				m_locs.push_back(std::pair<int, int>(-1, -1));
			}
			inline void setConnectId(const int _i, const uint64_t &_id) { m_connectIds[_i] = _id; }
			inline uint64_t &connectId(const int _i) { return m_connectIds[_i]; }
			inline const uint64_t &connectId(const int _i) const { return m_connectIds[_i]; }
			inline void setValid(const int _i, const bool &_valid) { m_valids[_i] = _valid; }
			inline bool valid(const int _i) { return m_valids[_i]; }
			inline const bool &valid(const int _i) const { return m_valids[_i]; }
			inline void setLoc(const int _i, const int &_l1, const int &_l2)
			{
				m_locs[_i].first = _l1;
				m_locs[_i].second = _l2;
			}
			inline std::pair<int, int> &loc(const int _i) { return m_locs[_i]; }
			inline const std::pair<int, int> &loc(const int _i) const { return m_locs[_i]; }

		private:
			std::vector<uint64_t> m_connectIds;
			std::vector<bool> m_valids;
			std::vector<std::pair<int, int>> m_locs;
		};
		typedef OptConnect Var2Factor;
		typedef OptConnect Factor2Var;

		OptimizerAbr() {}
		~OptimizerAbr()
		{
			// std::cout << "release optimizerAbr" << std::endl;
		}

		virtual bool run()
		{
			m_curIter = 0;
			for (m_curIter = 0; m_curIter < m_maxIterNum; ++m_curIter)
			{
				auto n = std::chrono::high_resolution_clock::now();
				T err = -1;
				if (!init())
					return false;
				auto nn = std::chrono::high_resolution_clock::now();
				if (!generateAB(err))
					return false;
				auto n2 = std::chrono::high_resolution_clock::now();
				if (!solveDetX())
					return false;
				auto n3 = std::chrono::high_resolution_clock::now();
				if (isFinish(m_curIter, err))
					break;
				auto n4 = std::chrono::high_resolution_clock::now();
				if (!updateX())
					return false;
				auto n5 = std::chrono::high_resolution_clock::now();
				double tIni = std::chrono::duration_cast<std::chrono::milliseconds>(nn - n).count();
				double tGen = std::chrono::duration_cast<std::chrono::milliseconds>(n2 - nn).count();
				double tSlv = std::chrono::duration_cast<std::chrono::milliseconds>(n3 - n2).count();
				double tfin = std::chrono::duration_cast<std::chrono::milliseconds>(n4 - n3).count();
				double tupd = std::chrono::duration_cast<std::chrono::milliseconds>(n5 - n4).count();
				std::cout << "init: " << tIni << ",\tgenerate: " << tGen << ",\tsolve: " << tSlv << ",\tfinish: " << tfin << ",\tupdate: " << tupd << std::endl;
			}
			return true;
		}

		virtual bool init() = 0;

		virtual bool generateAB(T &_err) = 0;

		virtual bool solveDetX() = 0;

		virtual bool updateX() = 0;

		virtual bool isFinish(const int _i, const T _err)
		{
			double detErr = std::abs(m_lastErr - _err);
			m_lastErr = _err;
			std::cout << "iter: " << _i << ", err: " << _err << std::endl;
			if (_err <= m_minErrTh || _i >= m_maxIterNum || detErr < 1e-6)
				return true;
			return false;
		}

		virtual bool addVariable(std::shared_ptr<OptVarAbr<T>> _var)
		{
			m_vars.push_back(_var);
			return true;
		}

		virtual bool addFactor(std::shared_ptr<OptFactorAbr<T>> _factor, const std::vector<uint64_t> &_vIds)
		{
			m_factors.push_back(_factor);
			const uint64_t &fId = _factor->getId();
			m_factor2Vars[fId] = Factor2Var(_vIds);
			for (int i = 0; i < _vIds.size(); ++i)
				m_var2Factors[_vIds[i]].addConnectId(fId);
			return true;
		}

		static bool checkEnable(const std::vector<std::shared_ptr<OptVarAbr<T>>> _vars, const Factor2Var &_factor2Vars, std::shared_ptr<OptFactorAbr<T>> _factor)
		{
			int connectCnt = _factor2Vars.size();
			bool isEnable = false;
			for (int j = 0; j < connectCnt; ++j)
			{
				if (!_vars[_factor2Vars.connectId(j)]->isFixed())
					isEnable = true;
			}
			_factor->setEnable(isEnable);
			return isEnable;
		}
		inline void setMaxIter(int _maxIter) { m_maxIterNum = _maxIter; }

	protected:
		int m_maxIterNum = 30;
		int m_curIter = 0;
		T m_minErrTh = 1e-6;
		T m_lastErr = -1;

		std::vector<std::shared_ptr<OptVarAbr<T>>> m_vars;
		std::map<uint64_t, Var2Factor> m_var2Factors;
		std::vector<std::shared_ptr<OptFactorAbr<T>>> m_factors;
		std::map<uint64_t, Factor2Var> m_factor2Vars;
	};

}

#endif // OPTIMIZE_OPTIMIZERABR_H