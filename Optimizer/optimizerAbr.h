#ifndef OPTIMIZE_OPTIMIZERABR_H
#define OPTIMIZE_OPTIMIZERABR_H

#include "Factor/Factor.h"
#include <map>
#include <memory>

namespace OPTIMIZE_LYJ
{

	template <typename T>
	class OptimizerAbr
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

		// template <typename T>
		class MatInner
		{
		public:
			MatInner() {}
			MatInner(const int _r, const int _c) : m_r(_r), m_c(_c)
			{
				m_data = new T[_r * _c];
				reset();
			}
			~MatInner()
			{
				if (m_data)
					delete m_data;
			}
			inline void reset() { memset(m_data, 0, sizeof(T) * m_r * m_c); }
			inline T *getData() { return m_data; }
			inline const T *getData() const { return m_data; }
			inline void setData(T *_data) { memcpy(m_data, _data, sizeof(T) * m_r * m_c); }
			inline const int r() const { return m_r; }
			inline const int c() const { return m_c; }

		private:
			T *m_data = nullptr;
			int m_r = -1;
			int m_c = -1;
		};
		class JacManager
		{
		public:
			JacManager(const uint64_t _id, const Var2Factor &_v2f,
					   const std::vector<OptFactorAbr<T>> &_factors, const std::vector<Factor2Var> &_factor2Vars)
			{
				m_id = _id;
				m_jacs.resize(_v2f.size());
				for (int i = 0; i < _v2f.size(); i++)
				{
					const uint64_t &fId = _v2f.connectId(i);
					const Factor2Var &f2v = _factor2Vars[fId];
					const OptFactorAbr<T> &f = _factors[fId];
					const int r = f.getEDim();
					const std::vector<int> vDims = f.getVDims();
					for (int j = 0; j < f2v.size(); j++)
					{
						if (_id != f2v.connectId(j))
							continue;
						const int &c = vDims[j];
						m_jacs[i] = MatInner(r, c);
					}
				}
			}
			~JacManager() {}

			// 非位姿jac，QR分解，变换位姿jac
			void QR()
			{
			}

		private:
			uint64_t m_id = UINT64_MAX;
			std::vector<MatInner> m_jacs;
			// std::vector<MatInner<T>> m_jacs;
		};

		OptimizerAbr() {}
		~OptimizerAbr()
		{
			// std::cout << "release optimizerAbr" << std::endl;
		}

		virtual bool run()
		{
			for (int i = 0; i < m_maxIterNum; ++i)
			{
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

		virtual bool generateAB(T &_err) = 0;

		virtual bool solveDetX() = 0;

		virtual bool updateX() = 0;

		virtual bool isFinish(const int _i, const T _err)
		{
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

	protected:
		int m_maxIterNum = 30;
		T m_minErrTh = 1e-6;
		T m_lastErr = -1;

		std::vector<std::shared_ptr<OptVarAbr<T>>> m_vars;
		std::map<uint64_t, Var2Factor> m_var2Factors;
		std::vector<std::shared_ptr<OptFactorAbr<T>>> m_factors;
		std::map<uint64_t, Factor2Var> m_factor2Vars;
		std::vector<JacManager> m_jacManagers;
	};

}

#endif // OPTIMIZE_OPTIMIZERABR_H