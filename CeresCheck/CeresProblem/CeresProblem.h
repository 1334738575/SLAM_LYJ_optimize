#pragma once
#ifdef HAS_CERES

#include "CeresCheck/CeresFactor/CeresFactor.h"


// export
#ifdef WIN32
#ifdef _MSC_VER
#define CERES_LYJ_API __declspec(dllexport)
#else
#define CERES_LYJ_API
#endif
#else
#define CERES_LYJ_API
#endif


class CERES_LYJ_API CeresProblem : protected ceres::Problem
{
public:
	CeresProblem();
	~CeresProblem();

	void addPoint3DParameter(double* _point3D, bool _bFix=false);
	void addPose3DParameter(double* _pose3D, bool _bFix = false);
	void addGlobalScaleParameter(double* _gScale, double _up=1000, double _down=0.001, bool _bFix = false);
	void addLine3DParameter(double* _line3D, bool _bFix = false);
	void addRay3DParameter(double* _ray3D, bool _bFix = false);

	void addUVFactor(const Eigen::Vector2d& _uv, const Eigen::Matrix3d& _K, double* _Tcw, double* _Pw, double _w=1);
	void addUVGScaleFactor(const Eigen::Vector2d& _uv, const Eigen::Matrix3d& _K,
		double* _Tcw, double* _Pw, double* _gScale, double _w = 1);
	void addStereoUVGScaleFactor(
		const Eigen::Vector2d& _uvl, const Eigen::Matrix3d& _Kl,
		const Eigen::Vector2d& _uvr, const Eigen::Matrix3d& _Kr,
		const Eigen::Quaterniond& _qrl, const Eigen::Vector3d& _trl,
		double* _Tlw, double* _Pw, double* _gScale, double _w = 1);
	void addLine3DFactor(const Eigen::Vector4d& _ob, const Eigen::Matrix3d& _KK,
		double* _Tcw, double* _line3D, double _w = 1);

	bool solve();

private:
	ceres::Solver::Options slvOpt_;
	ceres::Solver::Summary slvSmry;
};











#endif // HAS_CERES
