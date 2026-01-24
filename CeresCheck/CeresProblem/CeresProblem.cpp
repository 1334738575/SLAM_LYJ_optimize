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

void CeresProblem::addUVFactor(const Eigen::Vector2d& _uv, const Eigen::Matrix3d& _K, double* _Tcw, double* _Pw)
{
	ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CeresAutoFactorUV, 2, 7, 3>(
		new CeresAutoFactorUV(_uv, _K)
	);
	this->AddResidualBlock(cost_function, nullptr, _Tcw, _Pw);
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
