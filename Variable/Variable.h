#ifndef OPTIMIZE_VARIABLE_H
#define OPTIMIZE_VARIABLE_H

#include <iostream>
#include "VariableAbr.h"
#include "OptFunc.h"
#include <Eigen/Eigen>

namespace OPTIMIZE_LYJ
{

    class OptVarPoint3d : public OptVar<double, 3, 3>
    {
    public:
        OptVarPoint3d(const uint64_t _id) : OptVar(_id, VAR_POINT3D) {}
        ~OptVarPoint3d() {}

        // 通过 OptVar 继承
        bool update(double *_detX) override
        {
            if (std::isnan(_detX[0]) || std::isnan(_detX[1]) || std::isnan(_detX[2]))
            {
                std::cout << _detX[0] << " " << _detX[1] << " " << _detX[2] << std::endl;
                //std::cout << "OptVarPoint3d update error: detX contains NaN!" << std::endl;
                //return false;
            }
            //std::cout << "before: " << *this << std::endl;
            for (int i = 0; i < 3; ++i)
            {
                m_data[i] += _detX[i];
            }
            //std::cout << "after: " << *this << std::endl;
            return true;
        }

		Eigen::Vector3d getEigen() const
		{
			return Eigen::Map<const Eigen::Vector3d>(m_data, 3);
		}
        friend std::ostream &operator<<(std::ostream &os, const OptVarPoint3d &cls)
        {
            std::cout << "(" << cls.m_data[0] << ", " << cls.m_data[1] << ", " << cls.m_data[2] << ")";
            return os;
        }

    private:
    };

    static Eigen::Matrix3d computeJacobianSO3(const Eigen::Vector3d &phi)
    {
        const double theta = phi.norm();
        if (theta < 1e-6)
            return Eigen::Matrix3d::Identity();

        const Eigen::Matrix3d phi_hat = OPTIMIZE_BASE::skew_symmetric(phi);
        const Eigen::Matrix3d J = Eigen::Matrix3d::Identity() + (1 - cos(theta)) / (theta * theta) * phi_hat + (theta - sin(theta)) / (theta * theta * theta) * phi_hat * phi_hat;

        return J;
    }
    static Eigen::Matrix4d expSE3(const Eigen::Matrix<double, 6, 1> &xi)
    {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        const Eigen::Vector3d rho = xi.head<3>();
        const Eigen::Vector3d phi = xi.tail<3>();

        // 旋转部分
        const Eigen::Matrix3d R = Eigen::AngleAxisd(phi.norm(), phi.normalized()).toRotationMatrix();

        // 平移部分
        const Eigen::Matrix3d J = computeJacobianSO3(phi);
        T.block(0, 0, 3, 3) = R;
        T.block(0, 3, 3, 1) = J * rho;

        return T;
    }
    // 12-> R, t; 6->dt, dR
    class OptVarPose3d : public OptVar<double, 12, 6>
    {
    public:
        OptVarPose3d(const uint64_t _id) : OptVar(_id, VAR_T3D) {}
        ~OptVarPose3d() {}

        // 通过 OptVar 继承
        bool update(double *_detX) override
        {
            //std::cout << "before T: " << *this << std::endl;

            // Eigen::Matrix<double, 6, 1> delta = Eigen::Map<Eigen::Matrix<double, 6, 1>>(_detX, 6);
            // Eigen::Matrix4d delta_pose = expSE3(delta);
            // Eigen::Matrix3d dR = delta_pose.block(0, 0, 3, 3);
            // Eigen::Vector3d dett = delta_pose.block(0, 3, 3, 1);
            // Eigen::Map<Eigen::Vector3d> t(m_data + 9, 3);
            // Eigen::Map<Eigen::Matrix3d> R(m_data, 3, 3);
            //// 更新旋转部分（李代数）
            // R = dR * R;
            //// 更新平移部分
            // t = dett + dR * t;
            // std::cout << R << std::endl;
            // std::cout << t << std::endl;
            // return true;

            // Eigen::Matrix<double, 6, 1> detX = Eigen::Map<Eigen::Matrix<double, 6, 1>>(_detX, 6);
            // Eigen::Matrix4d dT = OPTIMIZE_BASE::se3_exp(detX);
            // Eigen::Map<Eigen::Vector3d> t(m_data + 9, 3);
            // Eigen::Map<Eigen::Matrix3d> R(m_data, 3, 3);
            // Eigen::Vector3d dett = dT.block(0, 3, 3, 1);
            // Eigen::Matrix3d dR = dT.block(0, 0, 3, 3);
            // // // 更新旋转部分（李代数）
            // // R = dR * R;
            // // // 更新平移部分
            // // t = dett + dR * t;
            // // 更新旋转部分（李代数）
            // R = R * dR;
            // // 更新平移部分
            // t = t + R * dett;
            // std::cout << "T: " << *this << std::endl;
            // return true;

            //// 左乘，不稳定，无法最优
            //Eigen::Map<Eigen::Vector3d> dett(_detX, 3);
            //// Eigen::Map<Eigen::Vector3d> detr(_detX + 3, 3);
            //Eigen::Map<Eigen::Vector3d> t(m_data + 9, 3);
            //Eigen::Map<Eigen::Matrix3d> R(m_data, 3, 3);
            //Eigen::Map<Eigen::Vector3d> detr(_detX + 3, 3);
            //Eigen::Vector3d axis = detr.normalized();
            //double theta = detr.norm();
            //Eigen::Matrix3d dR = Eigen::AngleAxisd(theta, axis).toRotationMatrix();
            //// Eigen::Matrix3d dR = OPTIMIZE_BASE::ExpSO3(_detX[3], _detX[4], _detX[5]);
            //R = dR * R;
            //t = dett + dR * t;
            ////std::cout << R << std::endl;
            ////std::cout << t << std::endl;
            //return true;

            // 右乘
            Eigen::Map<Eigen::Vector3d> dett(_detX, 3);
            Eigen::Map<Eigen::Vector3d> t(m_data + 9, 3);
            Eigen::Map<Eigen::Matrix3d> R(m_data, 3, 3);
            Eigen::Map<Eigen::Vector3d> detr(_detX + 3, 3);
            Eigen::Vector3d axis = detr.normalized();
            double theta = detr.norm();
            Eigen::Matrix3d dR = Eigen::AngleAxisd(theta, axis).toRotationMatrix();
            R = R * dR;
            t = t + R * dett;

            //std::cout << "after T: " << *this << std::endl;
            return true;
        }

        Eigen::Matrix<double, 3, 4> getEigen() const
        {
            return Eigen::Map<const Eigen::Matrix<double, 3, 4>>(m_data, 3, 4);
        }
        friend std::ostream &operator<<(std::ostream &os, const OptVarPose3d &cls)
        {
            std::cout << cls.m_data[0] << " \t" << cls.m_data[3] << " \t" << cls.m_data[6] << " \t" << cls.m_data[9] << std::endl
                      << cls.m_data[1] << " \t" << cls.m_data[4] << " \t" << cls.m_data[7] << " \t" << cls.m_data[10] << std::endl
                      << cls.m_data[2] << " \t" << cls.m_data[5] << " \t" << cls.m_data[8] << " \t" << cls.m_data[11] << std::endl;
            return os;
        }
    };


    class OptVarPose3Eulard : public OptVarPose3d
    {
    public:
        OptVarPose3Eulard(const uint64_t _id) : OptVarPose3d(_id) {}
        ~OptVarPose3Eulard() {}

        // 通过 OptVar 继承
        bool update(double *_detX) override
        {
            Eigen::Map<Eigen::Vector3d> t(m_data + 9, 3);
            Eigen::Map<Eigen::Matrix3d> R(m_data, 3, 3);
            Eigen::Vector3d dett = Eigen::Map<Eigen::Vector3d>(_detX + 3, 3);
            Eigen::Matrix3d dR = OPTIMIZE_BASE::EulerToRot(_detX[0], _detX[1], _detX[2]);
            // 更新旋转部分（李代数）
            R = dR * R;
            // 更新平移部分
            t = dett + dR * t;
            // std::cout <<"T" << m_vId<< std::endl << *this << std::endl;
            return true;
        }

    private:
    };

}

#endif // OPTIMIZE_VARIABLE_H