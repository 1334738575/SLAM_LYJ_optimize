#ifdef HAS_CERES


#include "CeresCheck/CeresProblem/CeresProblem.h"

CeresProblem::CeresProblem()
{}
CeresProblem::~CeresProblem()
{}

void CeresProblem::addPoint3DParameter(double* _point3D, bool _bFix)
{
	this->AddParameterBlock(_point3D, 3);
	this->SetManifold(_point3D, new CeresPoint3DParameter());
	if (_bFix)
		this->SetParameterBlockConstant(_point3D);
}
void CeresProblem::addPose3DParameter(double* _pose3D, bool _bFix)
{
	this->AddParameterBlock(_pose3D, 7);
	this->SetManifold(_pose3D, new CeresPose3DParameter());
	if (_bFix)
		this->SetParameterBlockConstant(_pose3D);
}
void CeresProblem::addGlobalScaleParameter(double* _gScale, double _up, double _down, bool _bFix)
{
	this->AddParameterBlock(_gScale, 1);
	this->SetManifold(_gScale, new CeresGlobalScaleParameter());
	if (_bFix)
		this->SetParameterBlockConstant(_gScale);
	this->SetParameterUpperBound(_gScale, 0, _up);
	this->SetParameterLowerBound(_gScale, 0, _down);
}
void CeresProblem::addLine3DParameter(double* _line3D, bool _bFix)
{
	this->AddParameterBlock(_line3D, 4);
	this->SetManifold(_line3D, new CeresLine3DParameter());
	if (_bFix)
		this->SetParameterBlockConstant(_line3D);
}
void CeresProblem::addRay3DParameter(double* _ray3D, bool _bFix)
{
	this->AddParameterBlock(_ray3D, 5);
	this->SetManifold(_ray3D, new CeresRay3DParameter2());
	if (_bFix)
		this->SetParameterBlockConstant(_ray3D);
}

void CeresProblem::addUVFactor(const Eigen::Vector2d& _uv, const Eigen::Matrix3d& _K, double* _Tcw, double* _Pw, double _w)
{
	ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CeresAutoFactorUV, 2, 7, 3>(
		new CeresAutoFactorUV(_uv, _K, _w)
	);
	this->AddResidualBlock(cost_function, nullptr, _Tcw, _Pw);
}
void CeresProblem::addUVGScaleFactor(const Eigen::Vector2d& _uv, const Eigen::Matrix3d& _K, 
	double* _Tcw, double* _Pw, double* _gScale, double _w)
{
	ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CeresAutoFactorUVGScale, 2, 7, 3, 1>(
		new CeresAutoFactorUVGScale(_uv, _K, _w)
	);
	this->AddResidualBlock(cost_function, nullptr, _Tcw, _Pw, _gScale);
}
void CeresProblem::addStereoUVGScaleFactor(
	const Eigen::Vector2d& _uvl, const Eigen::Matrix3d& _Kl, 
	const Eigen::Vector2d& _uvr, const Eigen::Matrix3d& _Kr, 
	const Eigen::Quaterniond& _qrl, const Eigen::Vector3d& _trl, 
	double* _Tlw, double* _Pw, double* _gScale, double _w)
{
	ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CeresAutoFactorStereoUVGScale, 4, 7, 3, 1>(
		new CeresAutoFactorStereoUVGScale(_uvl, _Kl, _uvr, _Kr, _qrl, _trl, _w)
	);
	this->AddResidualBlock(cost_function, nullptr, _Tlw, _Pw, _gScale);
}
void CeresProblem::addLine3DFactor(const Eigen::Vector4d& _ob, const Eigen::Matrix3d& _KK, double* _Tcw, double* _line3D, double _w)
{
	ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CeresAutoFactorLine3D, 4, 7, 4>(
		new CeresAutoFactorLine3D(_ob, _KK, _w)
	);
	this->AddResidualBlock(cost_function, nullptr, _Tcw, _line3D);
}

bool CeresProblem::solve()
{
	slvOpt_.minimizer_progress_to_stdout = true;
	slvOpt_.max_num_iterations = 100;
	slvOpt_.linear_solver_type = ceres::SPARSE_SCHUR;
	ceres::Solve(slvOpt_, this, &slvSmry);
	std::cout << slvSmry.FullReport() << std::endl;
	return true;
}



#endif // HAS_CERES
