#include "OptFunc.h"

namespace OPTIMIZE_LYJ
{
    namespace OPTIMIZE_BASE
    {

        Eigen::Matrix<double, 3, 4> relPose(const Eigen::Matrix<double, 3, 4> &_Tw1, const Eigen::Matrix<double, 3, 4> &_Tw2)
        {
            Eigen::Matrix<double, 3, 4> T12;
            T12.setZero();
            T12.block(0, 0, 3, 3) = _Tw1.block(0, 0, 3, 3).transpose() * _Tw2.block(0, 0, 3, 3);
            T12.block(0, 3, 3, 1) = _Tw1.block(0, 0, 3, 3).transpose() * (_Tw2.block(0, 3, 3, 1) - _Tw1.block(0, 3, 3, 1));
            return T12;
        }

        Eigen::Matrix<double, 3, 4> relPose(const double *const _Tw1, const double *const _Tw2)
        {
            Eigen::Map<const Eigen::Matrix<double, 3, 4>> Tw1(_Tw1);
            Eigen::Map<const Eigen::Matrix<double, 3, 4>> Tw2(_Tw2);
            Eigen::Matrix<double, 3, 4> T12;
            T12.setZero();
            T12.block(0, 0, 3, 3) = Tw1.block(0, 0, 3, 3).transpose() * Tw2.block(0, 0, 3, 3);
            T12.block(0, 3, 3, 1) = Tw1.block(0, 0, 3, 3).transpose() * (Tw2.block(0, 3, 3, 1) - Tw1.block(0, 3, 3, 1));
            return T12;
        }

        Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d &v)
        {
            Eigen::Matrix3d S;
            S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
            return S;
        }
        Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R)
        {
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
            return svd.matrixU() * svd.matrixV();
        }
        Eigen::Matrix3d ExpSO3(const double x, const double y, const double z)
        {
            const double d2 = x * x + y * y + z * z;
            const double d = sqrt(d2);
            Eigen::Matrix3d W;
            W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
            if (d < 1e-6)
            {
                Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W + 0.5 * W * W;
                return NormalizeRotation(res);
            }
            else
            {
                Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W * sin(d) / d + W * W * (1.0 - cos(d)) / d2;
                return NormalizeRotation(res);
            }
        }
        Eigen::Vector3d Lnso3(const Eigen::Matrix3d &R)
        {
            Eigen::Vector3d theta;

            return Eigen::Vector3d();
        }
        v6d orth_to_plk(const v4d &orth)
        {
            v6d plk;
            v3d theta = orth.head(3);
            double phi = orth[3];
            double s1 = sin(theta[0]);
            double c1 = cos(theta[0]);
            double s2 = sin(theta[1]);
            double c2 = cos(theta[1]);
            double s3 = sin(theta[2]);
            double c3 = cos(theta[2]);
            m33 R;
            R << c2 * c3, s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3,
                c2 * s3, s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3,
                -s2, s1 * c2, c1 * c2;
            // double w1 = cos(phi);
            // double w2 = sin(phi);
            // plk.head(3) = R.col(0) * w1;
            // plk.tail(3) = R.col(1) * w2;
            double d = phi;
            plk.head(3) = -R.col(0) * d;
            plk.tail(3) = R.col(1);
            return plk;
        }
        v6d plk_to_pose(const v6d &plk_w, const m33 &Rcw, const v3d &tcw)
        {
            v3d nw = plk_w.head(3);
            v3d vw = plk_w.tail(3);
            v3d nc = Rcw * nw + skew_symmetric(tcw) * Rcw * vw;
            v3d vc = Rcw * vw;
            v6d plk_c;
            plk_c.head(3) = nc;
            plk_c.tail(3) = vc;
            return plk_c;
        }
        v6d orth_to_line(const v4d &orth)
        {
            v6d line;

            v3d theta = orth.head(3);
            double phi = orth[3];
            // todo:: SO3
            double s1 = std::sin(theta[0]);
            double c1 = std::cos(theta[0]);
            double s2 = std::sin(theta[1]);
            double c2 = std::cos(theta[1]);
            double s3 = std::sin(theta[2]);
            double c3 = std::cos(theta[2]);
            m33 R;
            R << c2 * c3, s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3,
                c2 * s3, s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3,
                -s2, s1 * c2, c1 * c2;

            // double w1 = std::cos(phi);
            // double w2 = std::sin(phi);
            // double d = w1 / w2; // 原点到直线的距离
            double d = phi;

            line.head(3) = -R.col(2) * d;
            line.tail(3) = R.col(1);

            return line;
        }
        v4d line_to_orth(const v3d &p, const v3d &v)
        {
            v4d orth;
            v3d n = p.cross(v);
            v3d u1 = n / n.norm();
            v3d u2 = v / v.norm();
            v3d u3 = u1.cross(u2);
            orth[0] = std::atan2(u2(2), u3(2));
            orth[1] = std::asin(-u1(2));
            orth[2] = std::atan2(u1(1), u1(0));

            // v2d w(n.norm(), v.norm());
            // w = w / w.norm();
            // orth[3] = std::asin(w(1));
            orth[3] = p.cross(u2).norm();

            return orth;
        }
        // 普吕克与正交转换
        v4d plk_to_orth(const v3d &n, const v3d &v)
        {
            v4d orth;
            v3d u1 = n / n.norm();
            v3d u2 = v / v.norm();
            v3d u3 = u1.cross(u2);
            // todo:: use SO3
            orth[0] = std::atan2(u2(2), u3(2));
            orth[1] = std::asin(-u1(2));
            orth[2] = std::atan2(u1(1), u1(0));

            // TemVec2 w(n.norm(), v.norm());
            // w = w / w.norm();
            // orth[3] = std::asin(w(1));
            orth[3] = n.cross(v).norm();

            return orth;
        }

        void cal_jac_errUV_Tcw_Pw(const Eigen::Matrix<double, 3, 4> &Tcw, const Eigen::Matrix3d &K,
                                  const Eigen::Vector3d &Pw, const Eigen::Vector2d &uv,
                                  Eigen::Vector2d &err, Eigen::Matrix<double, 2, 6> &jac, const double w, const double invalidErr)
        {
            Eigen::Vector3d Pc = Tcw.block(0, 0, 3, 3) * Pw + Tcw.block(0, 3, 3, 1);
            err(0) = uv(0) - K(0, 0) * Pc(0) / Pc(2) - K(0, 2);
            err(1) = uv(1) - K(1, 1) * Pc(1) / Pc(2) - K(1, 2);
            if (err.norm() > invalidErr)
            {
                err(0) *= exp(-1 * 2.3 * abs(err(0)) / invalidErr);
                err(1) *= exp(-1 * 2.3 * abs(err(1)) / invalidErr);
            }
            err(0) *= w;
            err(1) *= w;

            /*
            -fx/Z   0       fxX/Z2
            0       -fy/Z   fyY/Z2
            */
            /*
            1   0   0   0   Z   -Y
            0   1   0   -Z  0   x
            0   0   1   Y   -X  0
            */
            jac.setZero();
            Eigen::Matrix<double, 2, 3> dedPc;
            dedPc << -1 * K(0, 0) / Pc(2), 0, K(0, 0) * Pc(0) / (Pc(2) * Pc(2)),
                0, -1 * K(1, 1) / Pc(2), K(1, 1) * Pc(1) / (Pc(2) * Pc(2));
            Eigen::Matrix<double, 3, 6> dPcdT;
            dPcdT.block(0, 0, 3, 3).setIdentity();
            dPcdT.block(0, 3, 3, 3) << 0, Pc(2), -1 * Pc(1),
                -1 * Pc(2), 0, Pc(0),
                Pc(1), -1 * Pc(0), 0;
            jac = dedPc * dPcdT;
        }
        //---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - ;
        void cal_jac_errT_T(const m44 &priTcw, const m44 &Tcw, v6d &err, m66 &jac)
        {
            // 弃用
            m33 Rcc = Tcw.block(0, 0, 3, 3) * priTcw.block(0, 0, 3, 3).transpose();
            v3d t = Rcc * priTcw.block(0, 3, 3, 1);
            v3d tcc = Tcw.block(0, 3, 3, 1) - t;
            Eigen::AngleAxisd ang(Rcc);
            v3d a = ang.axis();
            double theta = ang.angle();
            err.block(3, 0, 3, 1) = a * theta;
            err.block(0, 0, 3, 1) = tcc;
            jac.setIdentity();
            m33 dtdt = m33::Identity();
            jac.block(0, 0, 3, 3) = dtdt;
            m33 dtdR = skew_symmetric(t);
            jac.block(0, 3, 3, 3) = -1 * dtdR;
            m33 dRdR = m33::Identity() + (1 - cos(theta)) * (a * a.transpose() - m33::Identity()) + sin(theta) * skew_symmetric(a);
            jac.block(3, 3, 3, 3) = dRdR;
        }
        //---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - ;
        void cal_jac_errL2D_Tcw_L3D(const m34 &Tcw, const v4d &lineOrth, v2d &err, m26 &jacT, m24 &jacL, const m33 &KK, const v4d &obs)
        {
            v6d lineW = orth_to_plk(lineOrth);
            m33 Rcw = Tcw.block(0, 0, 3, 3);
            v3d tcw = Tcw.block(0, 3, 3, 1);
            v6d lineC = plk_to_pose(lineW, Rcw, tcw);
            v3d nc = lineC.head(3);
            v3d l2d = KK * nc;
            double l_norm = l2d(0) * l2d(0) + l2d(1) * l2d(1);
            double l_sqrtnorm = sqrt(l_norm);
            double l_trinorm = l_norm * l_sqrtnorm;
            double e1 = obs(0) * l2d(0) + obs(1) * l2d(1) + l2d(2);
            double e2 = obs(2) * l2d(0) + obs(3) * l2d(1) + l2d(2);
            err(0) = e1 / l_sqrtnorm;
            err(1) = e2 / l_sqrtnorm;

            m23 jaco_e_l_Tmp;
            jaco_e_l_Tmp << (obs(0) / l_sqrtnorm - l2d(0) * e1 / l_trinorm), (obs(1) / l_sqrtnorm - l2d(1) * e1 / l_trinorm), 1.0 / l_sqrtnorm,
                (obs(2) / l_sqrtnorm - l2d(0) * e2 / l_trinorm), (obs(3) / l_sqrtnorm - l2d(1) * e2 / l_trinorm), 1.0 / l_sqrtnorm;
            m23 jaco_e_l = jaco_e_l_Tmp;
            m36 jaco_l_Lc;
            jaco_l_Lc.setZero();
            jaco_l_Lc.block(0, 0, 3, 3) = KK;
            m26 jaco_e_Lc;
            jaco_e_Lc = jaco_e_l * jaco_l_Lc;

            v3d ncc = Rcw * lineW.head(3);
            v3d dcc = Rcw * lineW.tail(3);
            m33 nccc = skew_symmetric(ncc);
            m33 dccc = skew_symmetric(dcc);
            m66 jaco_Lc_pose;
            jaco_Lc_pose.setZero();
            jaco_Lc_pose.block(0, 0, 3, 3) = -1 * dccc;
            jaco_Lc_pose.block(0, 3, 3, 3) = -1 * nccc - skew_symmetric(tcw) * dccc;
            jaco_Lc_pose.block(3, 3, 3, 3) = -1 * dccc;
            jacT = jaco_e_Lc * jaco_Lc_pose;

            m66 invTwc;
            invTwc << Rcw, skew_symmetric(tcw) * Rcw,
                m33::Zero(), Rcw;
            v3d nw = lineW.head(3);
            v3d vw = lineW.tail(3);
            v3d u1 = nw / nw.norm();
            v3d u2 = vw / vw.norm();
            v3d u3 = u1.cross(u2);
            v2d w(nw.cross(vw).norm(), 1);
            // v2d wT(nw.norm(), vw.norm());
            // v2d w = wT / wT.norm();
            m64 jaco_Lw_orth;
            jaco_Lw_orth.setZero();
            jaco_Lw_orth.block(3, 0, 3, 1) = w(1) * u3;
            jaco_Lw_orth.block(0, 1, 3, 1) = -w(0) * u3;
            jaco_Lw_orth.block(0, 2, 3, 1) = w(0) * u2;
            jaco_Lw_orth.block(3, 2, 3, 1) = -w(1) * u1;
            jaco_Lw_orth.block(0, 3, 3, 1) = -w(1) * u1;
            jaco_Lw_orth.block(3, 3, 3, 1) = w(0) * u2;
            jacL = jaco_e_Lc * invTwc * jaco_Lw_orth;
        }
        //---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ----;
        void update_Tcw(m34 &Tcw, const v6d &detX, const double rate)
        {
            v3d dett = detX.block(0, 0, 3, 1) * rate;
            v3d detr = detX.block(3, 0, 3, 1) * rate;
            m33 Rcw = Tcw.block(0, 0, 3, 3);
            v3d tcw = Tcw.block(0, 3, 3, 1);
            m33 detR = ExpSO3(detr(0), detr(1), detr(2));
            Tcw.block(0, 0, 3, 3) = detR * Rcw;
            Tcw.block(0, 3, 3, 1) = dett + detR * tcw;
        }
        //---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - ;
        void update_lineOrth(v4d &orthW, const v4d &detX, const double rate)
        {
            v4d otrhW;
            v3d theta = orthW.block(0, 0, 3, 1);
            double phi = orthW[3];
            double s1 = sin(theta[0]);
            double c1 = cos(theta[0]);
            double s2 = sin(theta[1]);
            double c2 = cos(theta[1]);
            double s3 = sin(theta[2]);
            double c3 = cos(theta[2]);
            m33 R;
            R << c2 * c3, s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3,
                c2 * s3, s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3,
                -s2, s1 * c2, c1 * c2;
            double w1 = phi;
            double w2 = 1;
            // double w1 = cos(phi);
            // double w2 = sin(phi);
            v3d detTheta = detX.block(0, 0, 3, 1) * rate;
            double detPhi = detX(3) * rate;
            m33 Rz;
            Rz << cos(detTheta(2)), -sin(detTheta(2)), 0,
                sin(detTheta(2)), cos(detTheta(2)), 0,
                0, 0, 1;
            m33 Ry;
            Ry << cos(detTheta(1)), 0, sin(detTheta(1)),
                0, 1, 0,
                -sin(detTheta(1)), 0, cos(detTheta(1));
            m33 Rx;
            Rx << 1, 0, 0,
                0, cos(detTheta(0)), -sin(detTheta(0)),
                0, sin(detTheta(0)), cos(detTheta(0));
            m33 Rf = R * Rx * Ry * Rz;
            m22 W;
            W << w1, -w2, w2, w1;
            m22 Wf;
            Wf << w1 + detPhi, -1, 1, w1 + detPhi;
            // m22 detW;
            // detW << cos(detPhi), -sin(detPhi), sin(detPhi), cos(detPhi);
            // m22 Wf = W * detW;
            v4d thetaPlus;
            v3d u1 = Rf.col(0);
            v3d u2 = Rf.col(1);
            v3d u3 = Rf.col(2);
            thetaPlus[0] = atan2(u2(2), u3(2));
            thetaPlus[1] = asin(-u1(2));
            thetaPlus[2] = atan2(u1(1), u1(0));
            thetaPlus[3] = Wf(0, 0);
            // thetaPlus[3] = asin(Wf(1,0))
            v6d line = orth_to_line(thetaPlus);
            v3d P1 = line.head(3);
            v3d V = line.tail(3);
            v3d P2 = P1 + V * 3;
            P1 = P2 - V * 6;
        }

        // 3D点投影到像素坐标（带深度校验）
        Eigen::Vector2d project(const Eigen::Vector3d &P_c,
                                double fx, double fy,
                                double cx, double cy,
                                double min_z)
        {
            const double &z = P_c.z();
            // if (z < min_z) { // 处理深度过小的无效情况
            //     std::cerr << "Warning: Invalid depth " << z << " during projection!" << std::endl;
            //     return Eigen::Vector2d(-1, -1); // 返回无效坐标
            // }
            return Eigen::Vector2d(
                fx * P_c.x() / z + cx,
                fy * P_c.y() / z + cy);
        }
        // 计算雅可比矩阵（2x6，对应像素误差关于SE3的导数）
        Eigen::Matrix<double, 2, 6> compute_se3_jacobian(
            const Eigen::Matrix3d &R_cw, // 相机位姿（世界到相机的变换）
            const Eigen::Vector3d &t_cw, // 相机位姿（世界到相机的变换）
            const Eigen::Vector3d &P_w,  // 世界坐标系下的3D点
            double fx, double fy,        // 相机内参
            double cx, double cy,
            double min_z) // 最小深度阈值
        {
            // Step 1: 将点变换到相机坐标系
            const Eigen::Vector3d P_c = R_cw * P_w + t_cw;
            const double &x = P_c.x();
            const double &y = P_c.y();
            const double &z = P_c.z();

            //// 检查深度有效性
            // if (z < min_z) {
            //     std::cerr << "Error: Negative depth! Jacobian is invalid." << std::endl;
            //     return Eigen::Matrix<double, 2, 6>::Zero();
            // }

            // Step 2: 计算投影导数 de/dPc
            Eigen::Matrix<double, 2, 3> de_dpc;
            const double z_inv = 1.0 / z;
            const double z_inv2 = z_inv * z_inv;
            de_dpc << -fx * z_inv, 0, fx * x * z_inv2,
                0, -fy * z_inv, fy * y * z_inv2;

            // Step 3: 计算点坐标关于李代数的导数 dPc/dξ
            Eigen::Matrix<double, 3, 6> dpc_dxi;
            dpc_dxi.leftCols<3>() = Eigen::Matrix3d::Identity(); // 平移部分
            dpc_dxi.rightCols<3>() = -skew_symmetric(P_c);       // 旋转部分（反对称矩阵）

            // Step 4: 链式法则组合雅可比矩阵
            return de_dpc * dpc_dxi;
        }

        // 生成3x3反对称矩阵
        Eigen::Matrix3d hat(const Eigen::Vector3d &v)
        {
            Eigen::Matrix3d M;
            M << 0, -v.z(), v.y(),
                v.z(), 0, -v.x(),
                -v.y(), v.x(), 0;
            return M;
        }
        // 从反对称矩阵恢复向量
        Eigen::Vector3d vee(const Eigen::Matrix3d &M)
        {
            return Eigen::Vector3d(M(2, 1), M(0, 2), M(1, 0));
        }
        // SE3 对数映射（将变换矩阵转换为李代数）
        se3 se3_log(const SE3 &T)
        {
            Eigen::Matrix3d R = T.block<3, 3>(0, 0);
            Eigen::Vector3d t = T.block<3, 1>(0, 3);

            // 计算旋转向量 θω
            Eigen::AngleAxisd angle_axis(R);
            double theta = angle_axis.angle();
            Eigen::Vector3d omega = angle_axis.axis();

            // 处理θ接近0的情况
            constexpr double kEpsilon = 1e-6;
            if (theta < kEpsilon)
            {
                se3 ret;
                ret.setZero();
                ret.head<3>() = t;
                return ret;
            }

            // 计算平移部分 ρ
            const Eigen::Matrix3d phi_hat = OPTIMIZE_BASE::skew_symmetric(omega * theta);
            Eigen::Matrix3d J_inv = Eigen::Matrix3d::Identity() - 0.5 * phi_hat + (1.0 / (theta * theta) - (1 + cos(theta)) / (2 * theta * sin(theta))) * phi_hat * phi_hat;
            Eigen::Vector3d rho = J_inv * t;

            // 组装 se3 向量 [ρ; ωθ]
            se3 xi;
            xi << rho, omega * theta;
            return xi;
        }
        // SE3 指数映射（将李代数转换为变换矩阵）
        SE3 se3_exp(const se3 &xi)
        {
            Eigen::Vector3d rho = xi.head<3>();
            Eigen::Vector3d omega_theta = xi.tail<3>();
            // 计算旋转部分
            double theta = omega_theta.norm();
            Eigen::Vector3d omega = omega_theta.normalized();
            Eigen::Matrix3d R = Eigen::AngleAxisd(theta, omega).toRotationMatrix();
            // Eigen::Matrix3d R = ExpSO3(omega_theta(0), omega_theta(1), omega_theta(2));
            Eigen::Vector3d t;
            if (theta < 1e-6)
            {
                t = rho;
            }
            else
            {
                // 计算平移部分
                const Eigen::Matrix3d phi_hat = OPTIMIZE_BASE::skew_symmetric(omega_theta);
                const Eigen::Matrix3d J = Eigen::Matrix3d::Identity() + (1 - cos(theta)) / (theta * theta) * phi_hat + (theta - sin(theta)) / (theta * theta * theta) * phi_hat * phi_hat;
                // const Eigen::Matrix3d phi_hat = OPTIMIZE_BASE::skew_symmetric(omega);
                // Eigen::Matrix3d J = Eigen::Matrix3d::Identity()
                //	- 0.5 * phi_hat + (1.0 / (theta * theta) - (1 + cos(theta)) / (2 * theta * sin(theta))) * phi_hat * phi_hat;
                t = J * rho;
            }
            // 构造变换矩阵
            SE3 T;
            T.block<3, 3>(0, 0) = R;
            T.block<3, 1>(0, 3) = t;
            T.row(3) << 0, 0, 0, 1; // 齐次坐标
            return T;
        }
        // 构造伴随矩阵 ad(ξ)
        Eigen::Matrix<double, 6, 6> ad_se3(const se3 &xi)
        {
            Eigen::Vector3d rho = xi.head<3>();
            Eigen::Vector3d omega = xi.tail<3>();

            Eigen::Matrix<double, 6, 6> ad;
            ad.block<3, 3>(0, 0) = hat(omega);
            ad.block<3, 3>(0, 3) = hat(rho);
            ad.block<3, 3>(3, 3) = hat(omega);
            ad.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
            return ad;
        }
        // 计算先验位姿误差的雅可比矩阵
        Eigen::Matrix<double, 6, 6> compute_prior_jacobian(
            const SE3 &T_current, // 当前位姿 (4x4)
            const SE3 &T_prior,   // 先验位姿 (4x4)
            se3 &_err)
        {
            // 计算相对变换 T_rel = T_prior^{-1} * T_current
            SE3 T_rel = T_prior.inverse() * T_current;

            // 计算李代数误差 ξ = log(T_rel)
            _err = se3_log(T_rel);

            // 计算右雅可比矩阵的逆近似: J_r^{-1} ≈ I + 0.5 * ad(ξ)
            Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Identity();
            if (_err.norm() > 1e-6)
            { // 避免小量计算带来的数值不稳定
                J += 0.5 * ad_se3(_err);
            }

            return J;
        }

        // 反对称矩阵生成
        Eigen::Matrix3d skew(const Eigen::Vector3d &v)
        {
            Eigen::Matrix3d S;
            S << 0, -v.z(), v.y(),
                v.z(), 0, -v.x(),
                -v.y(), v.x(), 0;
            return S;
        }
        // 计算SE3的伴随矩阵
        Eigen::Matrix<double, 6, 6> adjoint_SE3(const SE3 &T)
        {
            Eigen::Matrix3d R = T.block<3, 3>(0, 0);
            Eigen::Vector3d t = T.block<3, 1>(0, 3);
            Eigen::Matrix<double, 6, 6> adj;
            adj.block<3, 3>(0, 0) = R;
            adj.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
            adj.block<3, 3>(3, 0) = skew(t) * R;
            adj.block<3, 3>(3, 3) = R;
            return adj;
        }
        // 三维线投影到图像直线
        Eigen::Vector3d projectLineToImage(
            const PluckerLine &L_camera,
            const Eigen::Matrix3d &K_invT)
        {
            // 提取方向向量并投影
            Eigen::Vector3d l_homog = K_invT * L_camera.n;
            return l_homog.normalized(); // 齐次坐标归一化
        }
        void computeJacobians(
            const SE3 &T,                 // 当前位姿
            const PluckerLine &L_world,   // 世界坐标系下的线
            const Eigen::Vector3d &l_obs, // 观测到的图像线
            Eigen::Matrix<double, 3, 6> &J_pose,
            Eigen::Matrix<double, 3, 6> &J_line)
        {
            // Step 1: 变换到相机坐标系
            Eigen::Matrix<double, 6, 6> adj_T = adjoint_SE3(T);
            PluckerLine L_camera;
            L_camera.n = adj_T.block<3, 3>(0, 0) * L_world.n + adj_T.block<3, 3>(0, 3) * L_world.v;
            L_camera.v = adj_T.block<3, 3>(3, 0) * L_world.n + adj_T.block<3, 3>(3, 3) * L_world.v;

            // Step 2: 投影到图像平面
            Eigen::Matrix3d K_invT; // = /* 相机内参的逆转置 */;
            Eigen::Vector3d l_pred = projectLineToImage(L_camera, K_invT);

            // Step 3: 误差关于投影线的导数
            Eigen::Matrix3d de_dl = -Eigen::Matrix3d::Identity();

            // Step 4: 投影线关于Plücker坐标的导数
            Eigen::Matrix<double, 3, 3> dl_dLc = K_invT.block<3, 3>(0, 0);

            // Step 5: Plücker坐标关于位姿的导数
            Eigen::Matrix<double, 3, 6> dLc_dxi;
            dLc_dxi.block<3, 3>(0, 0) = -skew(L_camera.n);
            dLc_dxi.block<3, 3>(0, 3) = -skew(L_camera.v);

            // Step 6: Plücker坐标关于线参数的导数
            Eigen::Matrix<double, 3, 6> dLc_dLw = adj_T.block<3, 6>(0, 0);

            // 组合雅可比矩阵
            J_pose = de_dl * dl_dLc * dLc_dxi;
            J_line = de_dl * dl_dLc * dLc_dLw;
        }

        Eigen::Matrix3d EulerToRot(const double &r, const double &p, const double &w)
        {
            double cr = std::cos(r);
            double sr = std::sin(r);
            double cp = std::cos(p);
            double sp = std::sin(p);
            double cw = std::cos(w);
            double sw = std::sin(w);
            Eigen::Matrix3d rotate;
            rotate << cr * cp, cr * sp * sw - sr * cw, cr * sp * cw + sr * sw,
                sr * cp, sr * sp * sw + cr * cw, sr * sp * cw - cr * sw,
                -sp, cp * sw, cp * cw;
            return rotate;
        }
        void computeJacRelPose3d_Pose3d_Pose3d(
            const Eigen::Matrix3d &Rw1, const Eigen::Vector3d &tw1,
            const Eigen::Matrix3d &Rw2, const Eigen::Vector3d &tw2,
            const Eigen::Matrix3d &R12, const Eigen::Vector3d &t12,
            const Eigen::Vector3d &cw1, const Eigen::Vector3d &cw2,
            Eigen::Matrix<double, 6, 6> &jacT12_Tw1, Eigen::Matrix<double, 6, 6> &jacT12_Tw2,
            Eigen::Matrix<double, 6, 1> &err)
        {
            typedef double ValueType;

            const auto &RwA00 = Rw1(0, 0);
            const auto &RwA01 = Rw1(0, 1);
            const auto &RwA02 = Rw1(0, 2);
            const auto &RwA10 = Rw1(1, 0);
            const auto &RwA11 = Rw1(1, 1);
            const auto &RwA12 = Rw1(1, 2);
            const auto &RwA20 = Rw1(2, 0);
            const auto &RwA21 = Rw1(2, 1);
            const auto &RwA22 = Rw1(2, 2);
            const auto &twA0 = tw1(0);
            const auto &twA1 = tw1(1);
            const auto &twA2 = tw1(2);

            const auto &RwB00 = Rw2(0, 0);
            const auto &RwB01 = Rw2(0, 1);
            const auto &RwB02 = Rw2(0, 2);
            const auto &RwB10 = Rw2(1, 0);
            const auto &RwB11 = Rw2(1, 1);
            const auto &RwB12 = Rw2(1, 2);
            const auto &RwB20 = Rw2(2, 0);
            const auto &RwB21 = Rw2(2, 1);
            const auto &RwB22 = Rw2(2, 2);
            const auto &twB0 = tw2(0);
            const auto &twB1 = tw2(1);
            const auto &twB2 = tw2(2);

            const auto &RAB00 = R12(0, 0);
            const auto &RAB01 = R12(0, 1);
            const auto &RAB02 = R12(0, 2);
            const auto &RAB10 = R12(1, 0);
            const auto &RAB11 = R12(1, 1);
            const auto &RAB12 = R12(1, 2);
            const auto &RAB20 = R12(2, 0);
            const auto &RAB21 = R12(2, 1);
            const auto &RAB22 = R12(2, 2);
            const auto &tAB0 = t12(0);
            const auto &tAB1 = t12(1);
            const auto &tAB2 = t12(2);

            const auto &centerA0 = cw1(0);
            const auto &centerA1 = cw1(1);
            const auto &centerA2 = cw1(2);
            const auto &centerB0 = cw2(0);
            const auto &centerB1 = cw2(1);
            const auto &centerB2 = cw2(2);

            ValueType err0 = atan2(RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22), RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22));
            ValueType err1 = -asin(RAB00 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB10 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB20 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22));
            ValueType err2 = atan2(RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22));
            ValueType err3 = -RwA00 * twA0 + RwA00 * twB0 - RwA10 * twA1 + RwA10 * twB1 - RwA20 * twA2 + RwA20 * twB2 + tAB0 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + tAB1 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + tAB2 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22);
            ValueType err4 = -RwA01 * twA0 + RwA01 * twB0 - RwA11 * twA1 + RwA11 * twB1 - RwA21 * twA2 + RwA21 * twB2 + tAB0 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + tAB1 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + tAB2 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22);
            ValueType err5 = -RwA02 * twA0 + RwA02 * twB0 - RwA12 * twA1 + RwA12 * twB1 - RwA22 * twA2 + RwA22 * twB2 + tAB0 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + tAB1 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + tAB2 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22);

            ValueType jac_e_0_T1_0 = (RAB00 * (RwA00 * RwB10 - RwA10 * RwB00) + RAB10 * (RwA00 * RwB11 - RwA10 * RwB01) + RAB20 * (RwA00 * RwB12 - RwA10 * RwB02)) * (-RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) - RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) - RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22)) / (pow(RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22), 2) + pow(RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22), 2)) + (RAB00 * (RwA01 * RwB10 - RwA11 * RwB00) + RAB10 * (RwA01 * RwB11 - RwA11 * RwB01) + RAB20 * (RwA01 * RwB12 - RwA11 * RwB02)) * (RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22)) / (pow(RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22), 2) + pow(RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22), 2));
            ValueType jac_e_0_T1_1 = (RAB00 * (-RwA00 * RwB20 + RwA20 * RwB00) + RAB10 * (-RwA00 * RwB21 + RwA20 * RwB01) + RAB20 * (-RwA00 * RwB22 + RwA20 * RwB02)) * (-RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) - RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) - RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22)) / (pow(RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22), 2) + pow(RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22), 2)) + (RAB00 * (-RwA01 * RwB20 + RwA21 * RwB00) + RAB10 * (-RwA01 * RwB21 + RwA21 * RwB01) + RAB20 * (-RwA01 * RwB22 + RwA21 * RwB02)) * (RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22)) / (pow(RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22), 2) + pow(RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22), 2));
            ValueType jac_e_0_T1_2 = (RAB00 * (RwA10 * RwB20 - RwA20 * RwB10) + RAB10 * (RwA10 * RwB21 - RwA20 * RwB11) + RAB20 * (RwA10 * RwB22 - RwA20 * RwB12)) * (-RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) - RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) - RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22)) / (pow(RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22), 2) + pow(RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22), 2)) + (RAB00 * (RwA11 * RwB20 - RwA21 * RwB10) + RAB10 * (RwA11 * RwB21 - RwA21 * RwB11) + RAB20 * (RwA11 * RwB22 - RwA21 * RwB12)) * (RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22)) / (pow(RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22), 2) + pow(RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22), 2));
            ValueType jac_e_0_T1_3 = 0;
            ValueType jac_e_0_T1_4 = 0;
            ValueType jac_e_0_T1_5 = 0;
            ValueType jac_e_0_T2_0 = (RAB00 * (-RwA00 * RwB10 + RwA10 * RwB00) + RAB10 * (-RwA00 * RwB11 + RwA10 * RwB01) + RAB20 * (-RwA00 * RwB12 + RwA10 * RwB02)) * (-RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) - RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) - RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22)) / (pow(RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22), 2) + pow(RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22), 2)) + (RAB00 * (-RwA01 * RwB10 + RwA11 * RwB00) + RAB10 * (-RwA01 * RwB11 + RwA11 * RwB01) + RAB20 * (-RwA01 * RwB12 + RwA11 * RwB02)) * (RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22)) / (pow(RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22), 2) + pow(RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22), 2));
            ValueType jac_e_0_T2_1 = (RAB00 * (RwA00 * RwB20 - RwA20 * RwB00) + RAB10 * (RwA00 * RwB21 - RwA20 * RwB01) + RAB20 * (RwA00 * RwB22 - RwA20 * RwB02)) * (-RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) - RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) - RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22)) / (pow(RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22), 2) + pow(RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22), 2)) + (RAB00 * (RwA01 * RwB20 - RwA21 * RwB00) + RAB10 * (RwA01 * RwB21 - RwA21 * RwB01) + RAB20 * (RwA01 * RwB22 - RwA21 * RwB02)) * (RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22)) / (pow(RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22), 2) + pow(RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22), 2));
            ValueType jac_e_0_T2_2 = (RAB00 * (-RwA10 * RwB20 + RwA20 * RwB10) + RAB10 * (-RwA10 * RwB21 + RwA20 * RwB11) + RAB20 * (-RwA10 * RwB22 + RwA20 * RwB12)) * (-RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) - RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) - RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22)) / (pow(RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22), 2) + pow(RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22), 2)) + (RAB00 * (-RwA11 * RwB20 + RwA21 * RwB10) + RAB10 * (-RwA11 * RwB21 + RwA21 * RwB11) + RAB20 * (-RwA11 * RwB22 + RwA21 * RwB12)) * (RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22)) / (pow(RAB00 * (RwA00 * RwB00 + RwA10 * RwB10 + RwA20 * RwB20) + RAB10 * (RwA00 * RwB01 + RwA10 * RwB11 + RwA20 * RwB21) + RAB20 * (RwA00 * RwB02 + RwA10 * RwB12 + RwA20 * RwB22), 2) + pow(RAB00 * (RwA01 * RwB00 + RwA11 * RwB10 + RwA21 * RwB20) + RAB10 * (RwA01 * RwB01 + RwA11 * RwB11 + RwA21 * RwB21) + RAB20 * (RwA01 * RwB02 + RwA11 * RwB12 + RwA21 * RwB22), 2));
            ValueType jac_e_0_T2_3 = 0;
            ValueType jac_e_0_T2_4 = 0;
            ValueType jac_e_0_T2_5 = 0;
            ValueType jac_e_1_T1_0 = -(RAB00 * (RwA02 * RwB10 - RwA12 * RwB00) + RAB10 * (RwA02 * RwB11 - RwA12 * RwB01) + RAB20 * (RwA02 * RwB12 - RwA12 * RwB02)) / sqrt(1 - pow(RAB00 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB10 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB20 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2));
            ValueType jac_e_1_T1_1 = -(RAB00 * (-RwA02 * RwB20 + RwA22 * RwB00) + RAB10 * (-RwA02 * RwB21 + RwA22 * RwB01) + RAB20 * (-RwA02 * RwB22 + RwA22 * RwB02)) / sqrt(1 - pow(RAB00 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB10 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB20 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2));
            ValueType jac_e_1_T1_2 = -(RAB00 * (RwA12 * RwB20 - RwA22 * RwB10) + RAB10 * (RwA12 * RwB21 - RwA22 * RwB11) + RAB20 * (RwA12 * RwB22 - RwA22 * RwB12)) / sqrt(1 - pow(RAB00 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB10 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB20 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2));
            ValueType jac_e_1_T1_3 = 0;
            ValueType jac_e_1_T1_4 = 0;
            ValueType jac_e_1_T1_5 = 0;
            ValueType jac_e_1_T2_0 = -(RAB00 * (-RwA02 * RwB10 + RwA12 * RwB00) + RAB10 * (-RwA02 * RwB11 + RwA12 * RwB01) + RAB20 * (-RwA02 * RwB12 + RwA12 * RwB02)) / sqrt(1 - pow(RAB00 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB10 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB20 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2));
            ValueType jac_e_1_T2_1 = -(RAB00 * (RwA02 * RwB20 - RwA22 * RwB00) + RAB10 * (RwA02 * RwB21 - RwA22 * RwB01) + RAB20 * (RwA02 * RwB22 - RwA22 * RwB02)) / sqrt(1 - pow(RAB00 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB10 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB20 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2));
            ValueType jac_e_1_T2_2 = -(RAB00 * (-RwA12 * RwB20 + RwA22 * RwB10) + RAB10 * (-RwA12 * RwB21 + RwA22 * RwB11) + RAB20 * (-RwA12 * RwB22 + RwA22 * RwB12)) / sqrt(1 - pow(RAB00 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB10 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB20 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2));
            ValueType jac_e_1_T2_3 = 0;
            ValueType jac_e_1_T2_4 = 0;
            ValueType jac_e_1_T2_5 = 0;
            ValueType jac_e_2_T1_0 = (RAB01 * (RwA02 * RwB10 - RwA12 * RwB00) + RAB11 * (RwA02 * RwB11 - RwA12 * RwB01) + RAB21 * (RwA02 * RwB12 - RwA12 * RwB02)) * (RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22)) / (pow(RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2) + pow(RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2)) + (-RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) - RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) - RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22)) * (RAB02 * (RwA02 * RwB10 - RwA12 * RwB00) + RAB12 * (RwA02 * RwB11 - RwA12 * RwB01) + RAB22 * (RwA02 * RwB12 - RwA12 * RwB02)) / (pow(RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2) + pow(RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2));
            ValueType jac_e_2_T1_1 = (RAB01 * (-RwA02 * RwB20 + RwA22 * RwB00) + RAB11 * (-RwA02 * RwB21 + RwA22 * RwB01) + RAB21 * (-RwA02 * RwB22 + RwA22 * RwB02)) * (RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22)) / (pow(RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2) + pow(RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2)) + (-RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) - RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) - RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22)) * (RAB02 * (-RwA02 * RwB20 + RwA22 * RwB00) + RAB12 * (-RwA02 * RwB21 + RwA22 * RwB01) + RAB22 * (-RwA02 * RwB22 + RwA22 * RwB02)) / (pow(RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2) + pow(RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2));
            ValueType jac_e_2_T1_2 = (RAB01 * (RwA12 * RwB20 - RwA22 * RwB10) + RAB11 * (RwA12 * RwB21 - RwA22 * RwB11) + RAB21 * (RwA12 * RwB22 - RwA22 * RwB12)) * (RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22)) / (pow(RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2) + pow(RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2)) + (-RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) - RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) - RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22)) * (RAB02 * (RwA12 * RwB20 - RwA22 * RwB10) + RAB12 * (RwA12 * RwB21 - RwA22 * RwB11) + RAB22 * (RwA12 * RwB22 - RwA22 * RwB12)) / (pow(RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2) + pow(RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2));
            ValueType jac_e_2_T1_3 = 0;
            ValueType jac_e_2_T1_4 = 0;
            ValueType jac_e_2_T1_5 = 0;
            ValueType jac_e_2_T2_0 = (RAB01 * (-RwA02 * RwB10 + RwA12 * RwB00) + RAB11 * (-RwA02 * RwB11 + RwA12 * RwB01) + RAB21 * (-RwA02 * RwB12 + RwA12 * RwB02)) * (RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22)) / (pow(RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2) + pow(RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2)) + (-RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) - RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) - RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22)) * (RAB02 * (-RwA02 * RwB10 + RwA12 * RwB00) + RAB12 * (-RwA02 * RwB11 + RwA12 * RwB01) + RAB22 * (-RwA02 * RwB12 + RwA12 * RwB02)) / (pow(RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2) + pow(RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2));
            ValueType jac_e_2_T2_1 = (RAB01 * (RwA02 * RwB20 - RwA22 * RwB00) + RAB11 * (RwA02 * RwB21 - RwA22 * RwB01) + RAB21 * (RwA02 * RwB22 - RwA22 * RwB02)) * (RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22)) / (pow(RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2) + pow(RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2)) + (-RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) - RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) - RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22)) * (RAB02 * (RwA02 * RwB20 - RwA22 * RwB00) + RAB12 * (RwA02 * RwB21 - RwA22 * RwB01) + RAB22 * (RwA02 * RwB22 - RwA22 * RwB02)) / (pow(RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2) + pow(RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2));
            ValueType jac_e_2_T2_2 = (RAB01 * (-RwA12 * RwB20 + RwA22 * RwB10) + RAB11 * (-RwA12 * RwB21 + RwA22 * RwB11) + RAB21 * (-RwA12 * RwB22 + RwA22 * RwB12)) * (RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22)) / (pow(RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2) + pow(RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2)) + (-RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) - RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) - RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22)) * (RAB02 * (-RwA12 * RwB20 + RwA22 * RwB10) + RAB12 * (-RwA12 * RwB21 + RwA22 * RwB11) + RAB22 * (-RwA12 * RwB22 + RwA22 * RwB12)) / (pow(RAB01 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB11 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB21 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2) + pow(RAB02 * (RwA02 * RwB00 + RwA12 * RwB10 + RwA22 * RwB20) + RAB12 * (RwA02 * RwB01 + RwA12 * RwB11 + RwA22 * RwB21) + RAB22 * (RwA02 * RwB02 + RwA12 * RwB12 + RwA22 * RwB22), 2));
            ValueType jac_e_2_T2_3 = 0;
            ValueType jac_e_2_T2_4 = 0;
            ValueType jac_e_2_T2_5 = 0;
            ValueType jac_e_3_T1_0 = -RwA00 * twA1 + RwA00 * twB1 - RwA00 * (centerA1 - twA1) + RwA10 * twA0 - RwA10 * twB0 - RwA10 * (-centerA0 + twA0) + tAB0 * (RwA00 * RwB10 - RwA10 * RwB00) + tAB1 * (RwA00 * RwB11 - RwA10 * RwB01) + tAB2 * (RwA00 * RwB12 - RwA10 * RwB02);
            ValueType jac_e_3_T1_1 = RwA00 * twA2 - RwA00 * twB2 - RwA00 * (-centerA2 + twA2) - RwA20 * twA0 + RwA20 * twB0 - RwA20 * (centerA0 - twA0) + tAB0 * (-RwA00 * RwB20 + RwA20 * RwB00) + tAB1 * (-RwA00 * RwB21 + RwA20 * RwB01) + tAB2 * (-RwA00 * RwB22 + RwA20 * RwB02);
            ValueType jac_e_3_T1_2 = -RwA10 * twA2 + RwA10 * twB2 - RwA10 * (centerA2 - twA2) + RwA20 * twA1 - RwA20 * twB1 - RwA20 * (-centerA1 + twA1) + tAB0 * (RwA10 * RwB20 - RwA20 * RwB10) + tAB1 * (RwA10 * RwB21 - RwA20 * RwB11) + tAB2 * (RwA10 * RwB22 - RwA20 * RwB12);
            ValueType jac_e_3_T1_3 = -RwA00;
            ValueType jac_e_3_T1_4 = -RwA10;
            ValueType jac_e_3_T1_5 = -RwA20;
            ValueType jac_e_3_T2_0 = RwA00 * (centerB1 - twB1) + RwA10 * (-centerB0 + twB0) + tAB0 * (-RwA00 * RwB10 + RwA10 * RwB00) + tAB1 * (-RwA00 * RwB11 + RwA10 * RwB01) + tAB2 * (-RwA00 * RwB12 + RwA10 * RwB02);
            ValueType jac_e_3_T2_1 = RwA00 * (-centerB2 + twB2) + RwA20 * (centerB0 - twB0) + tAB0 * (RwA00 * RwB20 - RwA20 * RwB00) + tAB1 * (RwA00 * RwB21 - RwA20 * RwB01) + tAB2 * (RwA00 * RwB22 - RwA20 * RwB02);
            ValueType jac_e_3_T2_2 = RwA10 * (centerB2 - twB2) + RwA20 * (-centerB1 + twB1) + tAB0 * (-RwA10 * RwB20 + RwA20 * RwB10) + tAB1 * (-RwA10 * RwB21 + RwA20 * RwB11) + tAB2 * (-RwA10 * RwB22 + RwA20 * RwB12);
            ValueType jac_e_3_T2_3 = RwA00;
            ValueType jac_e_3_T2_4 = RwA10;
            ValueType jac_e_3_T2_5 = RwA20;
            ValueType jac_e_4_T1_0 = -RwA01 * twA1 + RwA01 * twB1 - RwA01 * (centerA1 - twA1) + RwA11 * twA0 - RwA11 * twB0 - RwA11 * (-centerA0 + twA0) + tAB0 * (RwA01 * RwB10 - RwA11 * RwB00) + tAB1 * (RwA01 * RwB11 - RwA11 * RwB01) + tAB2 * (RwA01 * RwB12 - RwA11 * RwB02);
            ValueType jac_e_4_T1_1 = RwA01 * twA2 - RwA01 * twB2 - RwA01 * (-centerA2 + twA2) - RwA21 * twA0 + RwA21 * twB0 - RwA21 * (centerA0 - twA0) + tAB0 * (-RwA01 * RwB20 + RwA21 * RwB00) + tAB1 * (-RwA01 * RwB21 + RwA21 * RwB01) + tAB2 * (-RwA01 * RwB22 + RwA21 * RwB02);
            ValueType jac_e_4_T1_2 = -RwA11 * twA2 + RwA11 * twB2 - RwA11 * (centerA2 - twA2) + RwA21 * twA1 - RwA21 * twB1 - RwA21 * (-centerA1 + twA1) + tAB0 * (RwA11 * RwB20 - RwA21 * RwB10) + tAB1 * (RwA11 * RwB21 - RwA21 * RwB11) + tAB2 * (RwA11 * RwB22 - RwA21 * RwB12);
            ValueType jac_e_4_T1_3 = -RwA01;
            ValueType jac_e_4_T1_4 = -RwA11;
            ValueType jac_e_4_T1_5 = -RwA21;
            ValueType jac_e_4_T2_0 = RwA01 * (centerB1 - twB1) + RwA11 * (-centerB0 + twB0) + tAB0 * (-RwA01 * RwB10 + RwA11 * RwB00) + tAB1 * (-RwA01 * RwB11 + RwA11 * RwB01) + tAB2 * (-RwA01 * RwB12 + RwA11 * RwB02);
            ValueType jac_e_4_T2_1 = RwA01 * (-centerB2 + twB2) + RwA21 * (centerB0 - twB0) + tAB0 * (RwA01 * RwB20 - RwA21 * RwB00) + tAB1 * (RwA01 * RwB21 - RwA21 * RwB01) + tAB2 * (RwA01 * RwB22 - RwA21 * RwB02);
            ValueType jac_e_4_T2_2 = RwA11 * (centerB2 - twB2) + RwA21 * (-centerB1 + twB1) + tAB0 * (-RwA11 * RwB20 + RwA21 * RwB10) + tAB1 * (-RwA11 * RwB21 + RwA21 * RwB11) + tAB2 * (-RwA11 * RwB22 + RwA21 * RwB12);
            ValueType jac_e_4_T2_3 = RwA01;
            ValueType jac_e_4_T2_4 = RwA11;
            ValueType jac_e_4_T2_5 = RwA21;
            ValueType jac_e_5_T1_0 = -RwA02 * twA1 + RwA02 * twB1 - RwA02 * (centerA1 - twA1) + RwA12 * twA0 - RwA12 * twB0 - RwA12 * (-centerA0 + twA0) + tAB0 * (RwA02 * RwB10 - RwA12 * RwB00) + tAB1 * (RwA02 * RwB11 - RwA12 * RwB01) + tAB2 * (RwA02 * RwB12 - RwA12 * RwB02);
            ValueType jac_e_5_T1_1 = RwA02 * twA2 - RwA02 * twB2 - RwA02 * (-centerA2 + twA2) - RwA22 * twA0 + RwA22 * twB0 - RwA22 * (centerA0 - twA0) + tAB0 * (-RwA02 * RwB20 + RwA22 * RwB00) + tAB1 * (-RwA02 * RwB21 + RwA22 * RwB01) + tAB2 * (-RwA02 * RwB22 + RwA22 * RwB02);
            ValueType jac_e_5_T1_2 = -RwA12 * twA2 + RwA12 * twB2 - RwA12 * (centerA2 - twA2) + RwA22 * twA1 - RwA22 * twB1 - RwA22 * (-centerA1 + twA1) + tAB0 * (RwA12 * RwB20 - RwA22 * RwB10) + tAB1 * (RwA12 * RwB21 - RwA22 * RwB11) + tAB2 * (RwA12 * RwB22 - RwA22 * RwB12);
            ValueType jac_e_5_T1_3 = -RwA02;
            ValueType jac_e_5_T1_4 = -RwA12;
            ValueType jac_e_5_T1_5 = -RwA22;
            ValueType jac_e_5_T2_0 = RwA02 * (centerB1 - twB1) + RwA12 * (-centerB0 + twB0) + tAB0 * (-RwA02 * RwB10 + RwA12 * RwB00) + tAB1 * (-RwA02 * RwB11 + RwA12 * RwB01) + tAB2 * (-RwA02 * RwB12 + RwA12 * RwB02);
            ValueType jac_e_5_T2_1 = RwA02 * (-centerB2 + twB2) + RwA22 * (centerB0 - twB0) + tAB0 * (RwA02 * RwB20 - RwA22 * RwB00) + tAB1 * (RwA02 * RwB21 - RwA22 * RwB01) + tAB2 * (RwA02 * RwB22 - RwA22 * RwB02);
            ValueType jac_e_5_T2_2 = RwA12 * (centerB2 - twB2) + RwA22 * (-centerB1 + twB1) + tAB0 * (-RwA12 * RwB20 + RwA22 * RwB10) + tAB1 * (-RwA12 * RwB21 + RwA22 * RwB11) + tAB2 * (-RwA12 * RwB22 + RwA22 * RwB12);
            ValueType jac_e_5_T2_3 = RwA02;
            ValueType jac_e_5_T2_4 = RwA12;
            ValueType jac_e_5_T2_5 = RwA22;

            jacT12_Tw1 << jac_e_0_T1_0, jac_e_0_T1_1, jac_e_0_T1_2, jac_e_0_T1_3, jac_e_0_T1_4, jac_e_0_T1_5,
                jac_e_1_T1_0, jac_e_1_T1_1, jac_e_1_T1_2, jac_e_1_T1_3, jac_e_1_T1_4, jac_e_1_T1_5,
                jac_e_2_T1_0, jac_e_2_T1_1, jac_e_2_T1_2, jac_e_2_T1_3, jac_e_2_T1_4, jac_e_2_T1_5,
                jac_e_3_T1_0, jac_e_3_T1_1, jac_e_3_T1_2, jac_e_3_T1_3, jac_e_3_T1_4, jac_e_3_T1_5,
                jac_e_4_T1_0, jac_e_4_T1_1, jac_e_4_T1_2, jac_e_4_T1_3, jac_e_4_T1_4, jac_e_4_T1_5,
                jac_e_5_T1_0, jac_e_5_T1_1, jac_e_5_T1_2, jac_e_5_T1_3, jac_e_5_T1_4, jac_e_5_T1_5;
            jacT12_Tw2 << jac_e_0_T2_0, jac_e_0_T2_1, jac_e_0_T2_2, jac_e_0_T2_3, jac_e_0_T2_4, jac_e_0_T2_5,
                jac_e_1_T2_0, jac_e_1_T2_1, jac_e_1_T2_2, jac_e_1_T2_3, jac_e_1_T2_4, jac_e_1_T2_5,
                jac_e_2_T2_0, jac_e_2_T2_1, jac_e_2_T2_2, jac_e_2_T2_3, jac_e_2_T2_4, jac_e_2_T2_5,
                jac_e_3_T2_0, jac_e_3_T2_1, jac_e_3_T2_2, jac_e_3_T2_3, jac_e_3_T2_4, jac_e_3_T2_5,
                jac_e_4_T2_0, jac_e_4_T2_1, jac_e_4_T2_2, jac_e_4_T2_3, jac_e_4_T2_4, jac_e_4_T2_5,
                jac_e_5_T2_0, jac_e_5_T2_1, jac_e_5_T2_2, jac_e_5_T2_3, jac_e_5_T2_4, jac_e_5_T2_5;
            err << err0, err1, err2, err3, err4, err5;
        }

    }

    // 内部均以Twc为基础，右乘更新
    namespace OPTIMIZE_BASE_TWC
    {
        void cal_jac_errT_T(const m34 &priTwc, const m34 &Twc, v6d &err, m66 &jac)
        {
            m33 Rcc = priTwc.block(0, 0, 3, 3).transpose() * Twc.block(0, 0, 3, 3);
            v3d tcc = priTwc.block(0, 0, 3, 3).transpose() * (Twc.block(0, 3, 3, 1) - priTwc.block(0, 3, 3, 1));
            Eigen::AngleAxisd ang(Rcc);
            v3d a = ang.axis();
            double theta = ang.angle();
            err.block(3, 0, 3, 1) = a * theta;
            err.block(0, 0, 3, 1) = tcc;
            jac.setIdentity();
            jac.block(0, 0, 3, 3) = priTwc.block(0, 0, 3, 3).transpose();
            jac.block(3, 3, 3, 3) = Rcc;
        }

        void cal_jac_errUV_Tcw_Pw(const m34 &Twc, const m33 &K,
                                  const v3d &Pw, const v2d &uv,
                                  v2d &err, m26 &jac)
        {
            m33 Rcw = Twc.block(0, 0, 3, 3).transpose();
            v3d t = Pw - Twc.block(0, 3, 3, 1);
            v3d Pc = Rcw * t;
            err(0) = uv(0) - K(0, 0) * Pc(0) / Pc(2) - K(0, 2);
            err(1) = uv(1) - K(1, 1) * Pc(1) / Pc(2) - K(1, 2);

            /*
            -fx/Z   0       fxX/Z2
            0       -fy/Z   fyY/Z2
            */
            /*
            RwcT * (Pw - twc)^
            */
            jac.setZero();
            Eigen::Matrix<double, 2, 3> dedPc;
            dedPc << -1 * K(0, 0) / Pc(2), 0, K(0, 0) * Pc(0) / (Pc(2) * Pc(2)),
                0, -1 * K(1, 1) / Pc(2), K(1, 1) * Pc(1) / (Pc(2) * Pc(2));
            Eigen::Matrix<double, 3, 6> dPcdT;
            dPcdT.block(0, 0, 3, 3) = -1 * Rcw;
            dPcdT.block(0, 3, 3, 3) = OPTIMIZE_BASE::skew_symmetric(t);
            jac = dedPc * dPcdT;
        }

        void cal_jac_errPlane_Pw(const v4d &planew, const v3d &Pw, double &err, v3d &jac)
        {
            err = planew(0) * Pw(0) + planew(1) * Pw(1) + planew(2) * Pw(2) + planew(3);
            jac = planew.head(3);
        }

    }
}