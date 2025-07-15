#ifndef OPTIMIZE_FACTOR_H
#define OPTIMIZE_FACTOR_H

#include <vector>
#include "Factor/FactorAbr.h"
// #include "Variable/VariableAbr.h"
#include "OptFunc.h"

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

    // 计算SO(3)右雅可比逆矩阵
    static Eigen::Matrix3d computeJacobianInvSO3(const Eigen::Vector3d &phi)
    {
        const double theta = phi.norm();
        if (theta < 1e-6)
            return Eigen::Matrix3d::Identity();

        const Eigen::Matrix3d phi_hat = OPTIMIZE_BASE::skew_symmetric(phi);
        const Eigen::Matrix3d J_inv = Eigen::Matrix3d::Identity() - 0.5 * phi_hat + (1.0 / (theta * theta) - (1 + cos(theta)) / (2 * theta * sin(theta))) * phi_hat * phi_hat;

        return J_inv;
    }
    // 将变换矩阵转换为李代数向量
    static Eigen::Matrix<double, 6, 1> logSE3(const Eigen::Matrix4d &T)
    {
        Eigen::Matrix<double, 6, 1> xi;
        Eigen::Matrix3d R = T.block(0, 0, 3, 3);
        const Eigen::AngleAxisd aa(R);

        // 旋转部分
        const Eigen::Vector3d phi = aa.angle() * aa.axis();
        const Eigen::Matrix3d J_inv = computeJacobianInvSO3(phi);

        // 平移部分
        const Eigen::Vector3d rho = J_inv * T.block(0, 3, 3, 1);

        xi << rho, phi;
        return xi;
    }
    // 计算SE(3)右雅可比逆矩阵
    static Eigen::Matrix<double, 6, 6> computeJacobianInvSE3(const Eigen::Matrix<double, 6, 1> &xi)
    {
        const Eigen::Vector3d rho = xi.head<3>();
        const Eigen::Vector3d phi = xi.tail<3>();
        const Eigen::Matrix3d J_rot_inv = computeJacobianInvSO3(phi);

        Eigen::Matrix<double, 6, 6> J_inv = Eigen::Matrix<double, 6, 6>::Zero();
        J_inv.block<3, 3>(0, 0) = J_rot_inv;
        J_inv.block<3, 3>(3, 3) = J_rot_inv;
        J_inv.block<3, 3>(0, 3) = OPTIMIZE_BASE::skew_symmetric(rho) * J_rot_inv;

        return J_inv;
    }
    class OptFactorPose3d_Pose3d : public OptFactor<double, 6, 6>
    {
    public:
        OptFactorPose3d_Pose3d(const uint64_t _id) : OptFactor<double, 6, 6>(_id, OptFactorType::FACTOR_T3D_T3D) {}
        ~OptFactorPose3d_Pose3d()
        {
            if (m_obs)
                delete m_obs;
        }
        void setObs(double *_obs)
        {
            if (m_obs == nullptr)
                m_obs = new double[12];
            memcpy(m_obs, _obs, sizeof(double) * 12);
        }
        bool calculateErrAndJac(double *_err, double **_jac, double _w, OptVarAbr<double> **_values) const override
        {
            if (!checkVDims(_values))
                return false;
            auto varData = _values[0]->getData();
            if (_err == nullptr)
                return false;

            //// 计算残差：log(prior^{-1} * current)
            // Eigen::Matrix<double, 4, 4> T;
            // T.setIdentity();
            // Eigen::Matrix<double, 4, 4> Tpri;
            // Tpri.setIdentity();
            // T.block(0, 0, 3, 3) = Eigen::Map<const Eigen::Matrix3d>(varData, 3, 3);
            // T.block(0, 3, 3, 1) = Eigen::Map<const Eigen::Vector3d>(varData + 9, 3);
            // Tpri.block(0, 0, 3, 3) = Eigen::Map<const Eigen::Matrix3d>(m_obs, 3, 3);
            // Tpri.block(0, 3, 3, 1) = Eigen::Map<const Eigen::Vector3d>(m_obs + 9, 3);
            // const Eigen::Matrix4d error = Tpri.inverse() * T;
            // Eigen::Matrix<double, 6, 1> residual = logSE3(error);
            //// 计算雅可比矩阵
            // memcpy(_err, residual.data(), sizeof(double) * 6);
            // Eigen::Matrix<double, 6, 6> jacobian = computeJacobianInvSE3(residual);
            // if (_jac[0])
            //{
            //     memcpy(_jac[0], jacobian.data(), sizeof(double) * 36);
            // }
            // return true;

            Eigen::Matrix<double, 6, 1> err;
            err.setZero();
            Eigen::Matrix<double, 4, 4> T;
            T.setIdentity();
            Eigen::Matrix<double, 4, 4> Tpri;
            Tpri.setIdentity();
            T.block(0, 0, 3, 3) = Eigen::Map<const Eigen::Matrix3d>(varData, 3, 3);
            T.block(0, 3, 3, 1) = Eigen::Map<const Eigen::Vector3d>(varData + 9, 3);
            Tpri.block(0, 0, 3, 3) = Eigen::Map<const Eigen::Matrix3d>(m_obs, 3, 3);
            Tpri.block(0, 3, 3, 1) = Eigen::Map<const Eigen::Vector3d>(m_obs + 9, 3);
            // Eigen::Matrix<double, 6, 6> jac = OPTIMIZE_BASE::compute_prior_jacobian(T, Tpri, err);
            Eigen::Matrix<double, 6, 6> jac;
            OPTIMIZE_BASE::cal_jac_errT_T(Tpri, T, err, jac);
            memcpy(_err, err.data(), sizeof(double) * 6);
            if (_jac[0])
            {
                memcpy(_jac[0], jac.data(), sizeof(double) * 36);
            }
            return true;
        }

    private:
        double *m_obs = nullptr;
    };

    class OptFactorRelPose3d_Pose3d_Pose3d : public OptFactor<double, 6, 6, 6>
    {
    public:
        OptFactorRelPose3d_Pose3d_Pose3d(const uint64_t _id) : OptFactor<double, 6, 6, 6>(_id, OptFactorType::FACTOR_RELT3D_T3D_T3D) {}
        ~OptFactorRelPose3d_Pose3d_Pose3d()
        {
            if (m_obs)
                delete m_obs;
        }
        void setObs(double *_obs)
        {
            if (m_obs == nullptr)
                m_obs = new double[12];
            memcpy(m_obs, _obs, sizeof(double) * 12);
        }

        bool calculateErrAndJac(double *_err, double **_jac, double _w, OptVarAbr<double> **_values) const override
        {
            if (!checkVDims(_values))
                return false;
            if (_err == nullptr)
                return false;
            auto Tw1 = _values[0]->getData();
            auto Tw2 = _values[1]->getData();
            auto T12 = m_obs;

            Eigen::Matrix3d Rw1 = Eigen::Map<const Eigen::Matrix3d>(Tw1, 3, 3);
            Eigen::Vector3d tw1 = Eigen::Map<const Eigen::Vector3d>(Tw1 + 9, 3);
            Eigen::Matrix3d Rw2 = Eigen::Map<const Eigen::Matrix3d>(Tw2, 3, 3);
            Eigen::Vector3d tw2 = Eigen::Map<const Eigen::Vector3d>(Tw2 + 9, 3);
            Eigen::Matrix3d R12 = Eigen::Map<const Eigen::Matrix3d>(T12, 3, 3);
            Eigen::Vector3d t12 = Eigen::Map<const Eigen::Vector3d>(T12 + 9, 3);

            Eigen::Matrix<double, 6, 1> err;
            err.setZero();
            Eigen::Matrix<double, 6, 6> jacT12_Tw1;
            jacT12_Tw1.setIdentity();
            Eigen::Matrix<double, 6, 6> jacT12_Tw2;
            jacT12_Tw2.setIdentity();

            OPTIMIZE_BASE::computeJacRelPose3d_Pose3d_Pose3d(
                Rw1, tw1,
                Rw2, tw2,
                R12, t12,
                Eigen::Vector3d::Zero(), // 假设相机中心在原点
                Eigen::Vector3d::Zero(), // 假设相机中心在原点
                jacT12_Tw1, jacT12_Tw2, err);
            // std::cout << "jacT12_Tw1: " << std::endl << jacT12_Tw1 << std::endl;
            // std::cout << "jacT12_Tw2: " << std::endl << jacT12_Tw2 << std::endl;
            // std::cout << "err: " << std::endl << err << std::endl;

            memcpy(_err, err.data(), sizeof(double) * 6);
            if (_jac[0])
                memcpy(_jac[0], jacT12_Tw1.data(), sizeof(double) * 36);
            if (_jac[1])
                memcpy(_jac[1], jacT12_Tw2.data(), sizeof(double) * 36);
            return true;
        }

        friend std::ostream &operator<<(std::ostream &os, const OptFactorRelPose3d_Pose3d_Pose3d &cls)
        {
            std::cout << cls.m_obs[0] << " \t" << cls.m_obs[3] << " \t" << cls.m_obs[6] << " \t" << cls.m_obs[9] << std::endl
                      << cls.m_obs[1] << " \t" << cls.m_obs[4] << " \t" << cls.m_obs[7] << " \t" << cls.m_obs[10] << std::endl
                      << cls.m_obs[2] << " \t" << cls.m_obs[5] << " \t" << cls.m_obs[8] << " \t" << cls.m_obs[11] << std::endl;
            return os;
        }

    private:
        double *m_obs = nullptr;
    };

}

#endif // OPTIMIZE_FACTOR_H