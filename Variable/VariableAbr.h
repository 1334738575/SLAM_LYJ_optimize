#ifndef OPTIMIZE_VARIABLEABR_H
#define OPTIMIZE_VARIABLEABR_H

#include <iostream>

namespace OPTIMIZE_LYJ
{
    enum OptVarType
    {
        VAR_DEAULT = 0,
        VAR_T3D,
        VAR_T2D,
        VAR_POINT3D,
        VAR_POINT2D,
        VAR_LINE3D,
        VAR_LINE2D,
        VAR_PLANE3D,
        VAR_IMU,
        VAR_UNDEFINE_0 = 100,
        VAR_UNDEFINE_1,
        VAR_UNDEFINE_2
    };

    template <typename T>
    class OptVarAbr
    {
    public:
        OptVarAbr(const uint64_t _id, const OptVarType _type) : m_vId(_id), m_type(_type) {}
        ~OptVarAbr()
        {
            if (m_data)
                delete m_data;
        }

        inline const uint64_t &getId() const { return m_vId; }
        inline const OptVarType &getType() const { return m_type; }
        inline void setId(uint64_t _id) { m_vId = _id; }
        virtual void setData(T *_data) = 0;
        inline T *getData() { return m_data; }
        inline const T *getData() const { return m_data; }
        virtual int getDim() const = 0;
        virtual int getTangentDim() const = 0;
        virtual bool update(T *_detX) = 0;

    protected:
        T *m_data = nullptr;
        const uint64_t m_vId = UINT64_MAX;
        const OptVarType m_type = VAR_DEAULT;
    };

    template <typename T, int DIM, int TANDIM>
    class OptVar : public OptVarAbr<T>
    {
    public:
        OptVar(const uint64_t _id, const OptVarType _type) : OptVarAbr(_id, _type)
        {
            m_data = new T[DIM];
            memset(m_data, 0, sizeof(T) * DIM);
        }
        ~OptVar() {}

        void setData(T *_data) override
        {
            memcpy(m_data, _data, sizeof(T) * DIM);
        }
        int getDim() const override
        {
            return DIM;
        }
        int getTangentDim() const override
        {
            return TANDIM;
        }

    protected:
    };

}

#endif // OPTIMIZE_VARIABLEABR_H