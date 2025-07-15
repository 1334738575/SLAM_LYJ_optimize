#ifndef OPTIMIZE_LYJ_H
#define OPTIMIZE_LYJ_H

#include "Optimize_LYJ_Defines.h"

namespace OPTIMIZE_LYJ
{
    OPTIMIZE_LYJ_API int optimize_version();

    OPTIMIZE_LYJ_API void test_optimize_P3d_P3d();
    OPTIMIZE_LYJ_API void test_optimize_Pose3d_Pose3d();
    OPTIMIZE_LYJ_API void test_optimize_RelPose3d_Pose3d_Pose3d();
    OPTIMIZE_LYJ_API void test_optimize_Plane_P();
    OPTIMIZE_LYJ_API void test_optimize_UV_Pose3d_P3d();
}

#endif