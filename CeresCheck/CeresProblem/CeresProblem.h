#pragma once
#ifdef HAS_CERES

#include "CeresCheck/CeresFactor/CeresFactor.h"





class CeresProblem : protected ceres::Problem
{
public:
	CeresProblem();
	~CeresProblem();

	void addPoint3DParameter(double* _point3D, bool _bFix=false);
	void addPose3DParameter(double* _pose3D, bool _bFix = false);

	void addUVFactor(const Eigen::Vector2d& _uv, const Eigen::Matrix3d& _K, double* _Tcw, double* _Pw);


	bool solve();

private:
	ceres::Solver::Options slvOpt_;
	ceres::Solver::Summary slvSmry;
};











#endif // HAS_CERES
