#pragma once
#ifdef HAS_CERES


#include "CeresCheck/CeresParameter/CeresParameter.h"
#include <ceres/rotation.h>
#include <Eigen/Eigen>


struct CeresAutoFactorUV
{
	const Eigen::Vector2d& ob_;
	const Eigen::Matrix3d& K_;
	CeresAutoFactorUV(const Eigen::Vector2d& _ob, const Eigen::Matrix3d& _K)
		:ob_(_ob), K_(_K) 
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
		_residual[0] = T(ob_(0)) - uv[0];
		_residual[1] = T(ob_(1)) - uv[1];

		return true;
	}
};




















#endif // HAS_CERES
