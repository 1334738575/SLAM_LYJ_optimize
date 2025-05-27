#ifndef OPTIMIZE_FACTOR_H
#define OPTIMIZE_FACTOR_H

#include <vector>
#include "Factor/FactorAbr.h"
// #include "Variable/VariableAbr.h"

namespace OPTIMIZE_LYJ
{
    class OptFactorP3d_P3d : public OptFactor<double, 3, 3>
    {
    public:
        OptFactorP3d_P3d(const uint64_t _id) : OptFactor(_id, FACTOR_UNDEFINE_0) {}
        ~OptFactorP3d_P3d()
        {
            if (m_obs)
                delete m_obs;
        }
        void setObs(double *_obs)
        {
            if (m_obs == nullptr)
                m_obs = new double[3];
            memcpy(m_obs, _obs, sizeof(double) * 3);
        }

        bool calculateErrAndJac(double *_err, double **_jac, double _w, OptVarAbr<double> **_values) const override
        {
            if (!checkVDims(_values))
                return false;
            auto varData = _values[0]->getData();
            if (_err == nullptr)
                return false;
            for (size_t i = 0; i < 3; i++)
            {
                _err[i] = (m_obs[i] - varData[i]) * _w;
            }
            if (_jac[0])
            {
                _jac[0][0] = -1;
                _jac[0][1] = 0;
                _jac[0][2] = 0;
                _jac[0][3] = 0;
                _jac[0][4] = -1;
                _jac[0][5] = 0;
                _jac[0][6] = 0;
                _jac[0][7] = 0;
                _jac[0][8] = -1;
            }
            return true;
        }

    private:
        double *m_obs = nullptr;
    };

}

#endif // OPTIMIZE_FACTOR_H