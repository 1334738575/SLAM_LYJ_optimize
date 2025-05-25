#ifndef OPTIMIZE_FACTOR_H
#define OPTIMIZE_FACTOR_H

#include <vector>
#include "Variable/Variable.h"

namespace OPTIMIZE_LYJ
{

    enum OptFactorType
    {
        FACTOR_DEAULT = 0,
        FACTOR_UV_T3DPOINT3D,
        FACTOR_UV_T2D,
        FACTOR_POINT3D_T3D,
        FACTOR_UV2_T3DLINE3D,
        FACTOR_UV2_T2DLINE2D,
        FACTOR_UV_T3DPOINT3D_WITH_PLANE3D,
        FACTOR_PLANE3D_T3D_WITH_UV,
        FACTOR_T3D_T3DIMU,
        FACTOR_UNDEFINE_0 = 100,
        FACTOR_UNDEFINE_1,
        FACTOR_UNDEFINE_2
    };

    template<typename T>
    class OptFactorAbr
    {
    public:
        OptFactorAbr(const uint64_t _id, const OptFactorType _type) : m_errId(_id), m_type(_type) {}
        ~OptFactorAbr() {
            //if (m_err)
                //delete m_err;
        }

        inline const uint64_t& getId() const { return m_errId; }
        //inline T* getError() { return m_err; }
        virtual const int getEDim() const = 0;
        inline const int getVNum() const { return m_vDims.size(); }
        const std::vector<int>& getVDims() const { return m_vDims; };

        bool checkVDims(OptVarAbr<T>** _values) const {
            for (size_t i = 0; i < m_vDims.size(); ++i) {
                if (m_vDims[i] != _values[i]->getDim())
                    return false;
            }
            return true;
        }
        virtual bool calculateErrAndJac(T* _err, T** _jacs, T _w, OptVarAbr<T>** _values) const = 0;
    protected:
        const uint64_t m_errId = UINT64_MAX;
        const OptFactorType m_type = FACTOR_DEAULT;
        //T* m_err = nullptr;
        std::vector<int> m_vDims;
    };

    template<typename T, int EDIM, int... VDIMS>
    class OptFactor : public OptFactorAbr<T>
    {
    protected:
    public:
        OptFactor(const uint64_t _id, const OptFactorType _type) : OptFactorAbr(_id, _type)
        {
            //m_err = new T[EDIM];
            //memset(m_err, 0, sizeof(T) * EDIM);
            m_vDims = std::vector<int>{ VDIMS... };
        }
        ~OptFactor() {}

        const int getEDim() const override { return EDIM; }
    };

    class OptFactorP3d_P3d : public OptFactor<double, 3, 3>
    {
    public:
        OptFactorP3d_P3d(const uint64_t _id) : OptFactor(_id, FACTOR_UNDEFINE_0) {}
        ~OptFactorP3d_P3d() {
            if (m_obs)
                delete m_obs;
        }
        void setObs(double* _obs)
        {
            if (m_obs == nullptr)
                m_obs = new double[3];
            memcpy(m_obs, _obs, sizeof(double) * 3);
        }

        bool calculateErrAndJac(double* _err, double** _jac, double _w, OptVarAbr<double>** _values) const override
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
            if (_jac[0]) {
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
        double* m_obs = nullptr;
    };


}

#endif //OPTIMIZE_FACTOR_H