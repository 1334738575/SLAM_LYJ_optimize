#ifndef OPTIMIZE_FACTORABR_H
#define OPTIMIZE_FACTORABR_H

#include <vector>
#include "Variable/VariableAbr.h"

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
        FACTOR_T3D_T3D,
        FACTOR_RELT3D_T3D_T3D,
        FACTOR_UNDEFINE_0 = 100,
        FACTOR_UNDEFINE_1,
        FACTOR_UNDEFINE_2
    };

    template <typename T>
    class OptFactorAbr
    {
    public:
        OptFactorAbr(const uint64_t _id, const OptFactorType _type) : m_errId(_id), m_type(_type) {}
        ~OptFactorAbr()
        {
            // if (m_err)
            // delete m_err;
        }

        inline const uint64_t &getId() const { return m_errId; }
        // inline T* getError() { return m_err; }
        virtual const int getEDim() const = 0;
        inline const int getVNum() const { return m_vDims.size(); }
        const std::vector<int> &getVDims() const { return m_vDims; };

        void setEnable(bool _enable) { m_enable = _enable ? 1 : 0; }
        bool isEnable() const { return m_enable == 1; }
        bool checkVDims(OptVarAbr<T> **_values) const
        {
            for (size_t i = 0; i < m_vDims.size(); ++i)
            {
                if (m_vDims[i] != _values[i]->getTangentDim())
                    return false;
            }
            return true;
        }
        virtual bool calculateErrAndJac(T *_err, T **_jacs, T _w, OptVarAbr<T> **_values) const = 0;

    protected:
        const uint64_t m_errId = UINT64_MAX;
        const OptFactorType m_type = FACTOR_DEAULT;
        // T* m_err = nullptr;
        std::vector<int> m_vDims;
        char m_enable = 1;
    };

    template <typename T, int EDIM, int... VDIMS>
    class OptFactor : public OptFactorAbr<T>
    {
    protected:
    public:
        OptFactor(const uint64_t _id, const OptFactorType _type) : OptFactorAbr<T>(_id, _type)
        {
            // m_err = new T[EDIM];
            // memset(m_err, 0, sizeof(T) * EDIM);
            this->m_vDims = std::vector<int>{VDIMS...};
        }
        ~OptFactor() {}

        const int getEDim() const override { return EDIM; }
    };

}

#endif // OPTIMIZE_FACTORABR_H