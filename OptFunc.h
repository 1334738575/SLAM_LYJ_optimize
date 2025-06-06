#ifndef OPTIMIZE_FUNC_H
#define OPTIMIZE_FUNC_H

#include <Eigen/Eigen>
#include <Eigen/Core>

namespace OPTIMIZE_LYJ
{
    using m3d = Eigen::Matrix3d;
    using m33 = Eigen::Matrix3d;
    using m34 = Eigen::Matrix<double, 3, 4>;
    using v3d = Eigen::Vector3d;
    using v4d = Eigen::Vector4d;
    using v6d = Eigen::Matrix<double, 6, 1>;
    using m2d = Eigen::Matrix2d;
    using m22 = Eigen::Matrix2d;
    using m23 = Eigen::Matrix<double, 2, 3>;
    using m26 = Eigen::Matrix<double, 2, 6>;
    using m24 = Eigen::Matrix<double, 2, 4>;
    using m36 = Eigen::Matrix<double, 3, 6>;
    using m64 = Eigen::Matrix<double, 6, 4>;
    using m66 = Eigen::Matrix<double, 6, 6>;
    using v2d = Eigen::Vector2d;

    static Eigen::Matrix3d skew_symmetric(const Eigen::Matrix3d &v)
    {
        Eigen::Matrix3d S;
        S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
        return S;
    }
    static Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R)
    {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        return svd.matrixU() * svd.matrixV();
    }
    static Eigen::Matrix3d ExpSO3(const double x, const double y, const double z)
    {
        const double d2 = x * x + y * y + z * z;
        const double d = sqrt(d2);
        Eigen::Matrix3d W;
        W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
        if (d < 1e-5)
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
    static v6d orth_to_plk(const v4d &orth)
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
        line.head(3) = -R.col(0) * d;
        line.tail(3) = R.col(1);
        return plk;
    }
    static v6d plk_to_pose(const v6d &plk_w, const m33 &Rcw, const v3d &tcw)
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
    static v6d orth_to_line(const v4d &orth)
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
        // double d = w1 / w2; // ԭ�㵽ֱ�ߵľ���
        double d = phi;

        line.head(3) = -R.col(2) * d;
        line.tail(3) = R.col(1);

        return line;
    }
    static v4d line_to_orth(const v3d &p, const v3d &v)
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
        orth[3] = p.corss(u2).norm();

        return orth;
    }
    // ������������ת��
    static v4d plk_to_orth(const v3d &n, const v3d &v)
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
        orth[3] = n.corss(v).norm();

        return orth;
    }

    void cal_jac_errUV_Tcw_Pw(const Eigen::Matrix<double, 3, 4> &Tcw, const Eigen::Matrix3d &K,
                              const Eigen::Vector3d &Pw, const Eigen::Vector2d &uv,
                              Eigen::Vector2d &err, Eigen::Matrix<double, 2, 6> &jac, const double w, const double invalidErr)
    {
        Eigen::Vector3d Pc = Tcw * Pw;
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
    void cal_jac_errT_T(const m34 &priTcw, const m34 &Tcw, v6d &err, m66 &jac)
    {
        m3d Rcc = Tcw.block(0, 0, 3, 3) * priTcw.block(0, 0, 3, 3);
        v3d t = Tcw.block(0, 0, 3, 3) * priTcw.block(0, 3, 3, 1);
        v3d tcc = Tcw.block(0, 3, 3, 1) + t;
        Eigen::AngleAxisd ang(Rcc);
        v3d a = ang.axis();
        double theta = ang.angle();
        err.block(3, 0, 3, 1) = a * theta / 2;
        err.block(0, 0, 3, 1) = tcc / 2;
        jac.setIdentity();
        m3d dtdt = m3d::Identity();
        jac.block(0, 0, 3, 3) = dtdt;
        m3d dtdR = skew_symmetric(t);
        jac.block(0, 3, 3, 1) = -1 * dtdR;
        double halfThetaCot = theta / 2 * cos(theta / 2) * cos(theta / 2);
        m3d dRdR = halfThetaCot * m3d::Identity() + (1 - halfThetaCot) * (a * a.transpose()) - theta / 2 * skew_symmetric(a);
        jac.block(3, 3, 3, 3) = dRdR;
    }
    //---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - ;
    void cal_jac_errL2D_Tcw_L3D(const m34 &Tcw, const v4d &lineOrth, v2d &err, m26 &jacT, m24 &jacL, const m3d &KK, const v4d &obs)
    {
        v6d lineW = orth_to_plk(lineOrth);
        m3d Rcw = Tcw.block(0, 0, 3, 3);
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
        m2d Wf;
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
}

#endif