#ifndef OPTIMIZE_VARIABLE_H
#define OPTIMIZE_VARIABLE_H

#include <iostream>
#include "VariableAbr.h"

namespace OPTIMIZE_LYJ
{
    class OptVarPoint3d : public OptVar<double, 3, 3>
    {
    public:
        OptVarPoint3d(const uint16_t _id) : OptVar(_id, VAR_POINT3D) {}
        ~OptVarPoint3d() {}

        bool update(double *_detX) override
        {
            for (int i = 0; i < 3; ++i)
            {
                m_data[i] += _detX[i];
            }
            return true;
        }
        friend std::ostream &operator<<(std::ostream &os, const OptVarPoint3d &cls)
        {
            std::cout << "(" << cls.m_data[0] << ", " << cls.m_data[1] << ", " << cls.m_data[2] << ")";
            return os;
        }

    private:
    };

}

#endif // OPTIMIZE_VARIABLE_H