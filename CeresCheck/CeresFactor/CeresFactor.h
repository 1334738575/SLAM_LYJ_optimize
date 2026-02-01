#pragma once
#ifdef HAS_CERES


#include "CeresCheck/CeresParameter/CeresParameter.h"
#include <ceres/rotation.h>
#include <Eigen/Eigen>


struct CeresAutoFactorUV
{
	const Eigen::Vector2d& ob_;
	const Eigen::Matrix3d& K_;
	double w_ = 1;
	CeresAutoFactorUV(const Eigen::Vector2d& _ob, const Eigen::Matrix3d& _K, double _w=1)
		:ob_(_ob), K_(_K), w_(_w)
	{};

	template<typename T>
	bool operator()(const T* const _Tcw, const T* const _Point, T* _residual) const
	{
		const T* qcw = _Tcw;
		const T* tcw = _Tcw + 4;

		T Pc[3];
		ceres::QuaternionRotatePoint(qcw, _Point, Pc);
		for (int i = 0; i < 3; ++i)
			Pc[i] += tcw[i];
		Pc[0] /= Pc[2];
		Pc[1] /= Pc[2];
		T uv[2];
		uv[0] = T(K_(0, 0)) * Pc[0] + T(K_(0, 2));
		uv[1] = T(K_(1, 1)) * Pc[1] + T(K_(1, 2));
		_residual[0] = T(w_) * (T(ob_(0)) - uv[0]);
		_residual[1] = T(w_) * (T(ob_(1)) - uv[1]);

		return true;
	}
};

struct CeresAutoFactorUVGScale
{
	const Eigen::Vector2d& ob_;
	const Eigen::Matrix3d& K_;
	double w_ = 1;
	CeresAutoFactorUVGScale(const Eigen::Vector2d& _ob, const Eigen::Matrix3d& _K, double _w = 1)
		:ob_(_ob), K_(_K), w_(_w)
	{
	};

	template<typename T>
	bool operator()(const T* const _Tcw, const T* const _Point, const T* const _gScale, T* _residual) const
	{
		const T* qcw = _Tcw;
		const T* tcw = _Tcw + 4;

		T Pc[3];
		ceres::QuaternionRotatePoint(qcw, _Point, Pc);
		for (int i = 0; i < 3; ++i)
			Pc[i] += (T(_gScale[0]) * tcw[i]);
		Pc[0] /= Pc[2];
		Pc[1] /= Pc[2];
		T uv[2];
		uv[0] = T(K_(0, 0)) * Pc[0] + T(K_(0, 2));
		uv[1] = T(K_(1, 1)) * Pc[1] + T(K_(1, 2));
		_residual[0] = T(w_) * (T(ob_(0)) - uv[0]);
		_residual[1] = T(w_) * (T(ob_(1)) - uv[1]);

		return true;
	}
};

struct CeresAutoFactorStereoUVGScale
{
	const Eigen::Vector2d& obl_;
	const Eigen::Matrix3d& Kl_;
	const Eigen::Vector2d& obr_;
	const Eigen::Matrix3d& Kr_;
	const Eigen::Quaterniond& qrl_;
	const Eigen::Vector3d& trl_;
	double w_ = 1;
	CeresAutoFactorStereoUVGScale(
		const Eigen::Vector2d& _obl, const Eigen::Matrix3d& _Kl,
		const Eigen::Vector2d& _obr, const Eigen::Matrix3d& _Kr,
		const Eigen::Quaterniond& _qrl, const Eigen::Vector3d& _trl,
		double _w = 1)
		:obl_(_obl), Kl_(_Kl), obr_(_obr), Kr_(_Kr), qrl_(_qrl), trl_(_trl), w_(_w)
	{
	};

	template<typename T>
	bool operator()(const T* const _Tlw, const T* const _Point, const T* const _gScale, T* _residual) const
	{
		const T* qlw = _Tlw;
		const T* tlw = _Tlw + 4;

		T Pl[3];
		ceres::QuaternionRotatePoint(qlw, _Point, Pl);
		for (int i = 0; i < 3; ++i)
			Pl[i] += (T(_gScale[0]) * tlw[i]);
		T uvl[2];
		uvl[0] = T(Kl_(0, 0)) * Pl[0] / Pl[2] + T(Kl_(0, 2));
		uvl[1] = T(Kl_(1, 1)) * Pl[1] / Pl[2] + T(Kl_(1, 2));
		_residual[0] = T(w_) * (T(obl_(0)) - uvl[0]);
		_residual[1] = T(w_) * (T(obl_(1)) - uvl[1]);

		T qrl[4];
		qrl[0] = T(qrl_.w());
		qrl[1] = T(qrl_.x());
		qrl[2] = T(qrl_.y());
		qrl[3] = T(qrl_.z());
		T Pr[3];
		ceres::QuaternionRotatePoint(qrl, Pl, Pr);
		for (int i = 0; i < 3; ++i)
			Pr[i] += (T(_gScale[0]) * T(trl_[i]));
		T uvr[2];
		uvr[0] = T(Kr_(0, 0)) * Pr[0] / Pr[2] + T(Kr_(0, 2));
		uvr[1] = T(Kr_(1, 1)) * Pr[1] / Pr[2] + T(Kr_(1, 2));
		_residual[2] = T(w_) * (T(obr_(0)) - uvr[0]);
		_residual[3] = T(w_) * (T(obr_(1)) - uvr[1]);

		return true;
	}
};

struct CeresAutoFactorLine3D
{
	const Eigen::Vector4d& ob_;
	const Eigen::Matrix3d& KK_;
	double w_ = 1;
	CeresAutoFactorLine3D(const Eigen::Vector4d& _ob, const Eigen::Matrix3d& _KK, double _w = 1)
		:ob_(_ob), KK_(_KK), w_(_w)
	{
	}

	static void convertK2KK(const Eigen::Matrix3d& _K, Eigen::Matrix3d& _KK)
	{
		const double& fx = _K(0, 0);
		const double& fy = _K(1, 1);
		_KK.setZero();
		Eigen::Matrix3d KK = _K.inverse();
		_KK = KK.transpose();
		_KK *= (fx * fy);
	}
	template<typename T>
	Eigen::Matrix<T, 6, 1> orth_to_plk(const T* const orth)const 
	{
		Eigen::Matrix<T, 6, 1> plk;
		const T* const theta = orth;
		T phi = orth[3];
		T s1 = sin(theta[0]);
		T c1 = cos(theta[0]);
		T s2 = sin(theta[1]);
		T c2 = cos(theta[1]);
		T s3 = sin(theta[2]);
		T c3 = cos(theta[2]);
		Eigen::Matrix<T, 3, 3> R;
		R << c2 * c3, s1* s2* c3 - c1 * s3, c1* s2* c3 + s1 * s3,
			c2* s3, s1* s2* s3 + c1 * c3, c1* s2* s3 - s1 * c3,
			-s2, s1* c2, c1* c2;
		T w1 = cos(phi);
		T w2 = sin(phi);
		plk.head(3) = R.col(0) * w1;
		plk.tail(3) = R.col(1) * w2;
		return plk;
	}
	template<typename T>
	Eigen::Matrix<T, 3, 3> skew_symmetric(const Eigen::Matrix<T, 3, 1>& v)const
	{
		Eigen::Matrix<T, 3, 3> S;
		S(0, 0) = T(0);
		S(0, 1) = -v(2);
		S(0, 2) = v(1);
		S(1, 0) = v(2);
		S(1, 1) = T(0);
		S(1, 2) = -v(0);
		S(2, 0) = -v(1);
		S(2, 1) = v(0);
		S(2, 2) = T(0);
		return S;
	}
	template<typename T>
	Eigen::Matrix<T, 6, 1> plk_to_pose(const Eigen::Matrix<T, 6, 1>& plk_w, const Eigen::Matrix<T, 3, 3>& Rcw, const Eigen::Matrix<T, 3, 1>& tcw)const
	{
		Eigen::Matrix<T, 3, 1> nw = plk_w.head(3);
		Eigen::Matrix<T, 3, 1> vw = plk_w.tail(3);
		Eigen::Matrix<T, 3, 1> nc = Rcw * nw + skew_symmetric<T>(tcw) * Rcw * vw;
		Eigen::Matrix<T, 3, 1> vc = Rcw * vw;
		Eigen::Matrix<T, 6, 1> plk_c;
		plk_c.head(3) = nc;
		plk_c.tail(3) = vc;
		return plk_c;
	}
	template<typename T>
	bool operator()(const T* const _Tcw, const T* const _line, T* _residual)const
	{
		Eigen::Matrix<T, 6, 1> lineW = orth_to_plk<T>(_line);
		Eigen::Quaternion<T> q;
		q.x() = _Tcw[1];
		q.y() = _Tcw[2];
		q.z() = _Tcw[3];
		q.w() = _Tcw[0];
		Eigen::Matrix<T, 3, 3> Rcw = q.toRotationMatrix();
		const T* t = _Tcw + 4;
		Eigen::Matrix<T, 3, 1> tcw;
		tcw(0) = t[0];
		tcw(1) = t[1];
		tcw(2) = t[2];
		Eigen::Matrix<T, 6, 1> lineC = plk_to_pose<T>(lineW, Rcw, tcw);
		Eigen::Matrix<T, 3, 1> nc = lineC.head(3);
		Eigen::Matrix<T, 3, 1> l2d = KK_.cast<T>() * nc;
		T l_norm = l2d(0) * l2d(0) + l2d(1) * l2d(1);
		T l_sqrtnorm = sqrt(l_norm);
		T l_trinorm = l_norm * l_sqrtnorm;
		T e1 = T(ob_(0)) * l2d(0) + T(ob_(1)) * l2d(1) + l2d(2);
		T e2 = T(ob_(2)) * l2d(0) + T(ob_(3)) * l2d(1) + l2d(2);
		_residual[0] = T(w_) * e1 / l_sqrtnorm;
		_residual[1] = T(w_) * e2 / l_sqrtnorm;
		return true;
	}
};




















#endif // HAS_CERES
