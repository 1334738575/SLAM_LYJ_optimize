#include <iostream>
#include <Optimize_LYJ.h>
#include <Eigen/Core>
#include <Eigen/Eigen>

Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> testEigenMap(double* _data)
{
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> mat(_data, 2, 3);
    //std::cout << "eigen map: " << std::endl << mat << std::endl;
    return mat;
}

int main(int argc, char *argv[])
{
    //double data[6] = { 1, 2, 3, 4, 5, 6 };
    //auto mat = testEigenMap(data);
    //std::cout << "eigen map2: " << std::endl << mat << std::endl;
    //return 0;
    std::cout << "Optimize Version: " << OPTIMIZE_LYJ::optimize_version() << std::endl;
    OPTIMIZE_LYJ::test_optimize();
    return 0;
}