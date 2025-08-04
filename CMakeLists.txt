cmake_minimum_required(VERSION 3.20)
project(OPTIMIZE_LYJ)

# ################################setting##################################
set(CMAKE_CXX_STANDARD 17)

option(USE_CERES "use ceres" ON)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

if(WIN32)
    if(MSVC)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")

        # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Od /Ob0")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG")
        add_link_options(/DEBUG) # 生成调试信息（测试在visual studio中有效）
    endif()
elseif(LINUX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g")
endif()

set(${PROJECT_NAME}_LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/Output)
set(${PROJECT_NAME}_CMAKE_FILE)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

# ################################cmake##################################
if(DEFINED ${PROJECT_NAME}_CMAKE_FILE)
    include(${${PROJECT_NAME}_CMAKE_FILE})
elseif(EXISTS ${CMAKE_CURRENT_LIST_DIR}/../cmake/FuncLyj.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/FuncLyj.cmake)
else()
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/FuncLyj.cmake)
endif()

# ################################package##################################
set(ALL_INCLUDE_DIRS)
set(ALL_LIBRARIES)
set(ALL_BINARY_DIRS)

set(Eigen3_DIR "D:/SLAM_LYJ/dependencies/Eigen/share/eigen3/cmake")
set(Ceres_DIR "D:/SLAM_LYJ/dependencies/Ceres/lib/cmake/Ceres")

# set(Eigen3_DIR "D:/SLAM_LYJ/dependencies/Eigen334/install2/share/eigen3/cmake")
# set(Eigen3_DIR "D:/SLAM_LYJ/dependencies/Eigen334/install2/share/eigen3/cmake")
# set(Eigen3_DIR "C:\\Users\\DELL\\Desktop\\code\\eigen\\install3\\share\\eigen3\\cmake")
find_package(Eigen3)
list(APPEND ALL_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
if(USE_CERES)
    # message(FATAL_ERROR 1111)
    find_package(Ceres REQUIRED)
    list(APPEND ALL_LIBRARIES Ceres::ceres)
    list(APPEND ALL_BINARY_DIRS "D:/SLAM_LYJ/dependencies/Ceres/bin;")
    list(APPEND ALL_BINARY_DIRS "D:/SLAM_LYJ/dependencies/Glog/bin;")
    list(APPEND ALL_BINARY_DIRS "D:/SLAM_LYJ/dependencies/Gflags/Lib")
endif()

# list(APPEND ALL_LIBRARIES Eigen3::Eigen)

include_directories(${ALL_INCLUDE_DIRS})

# ################################code##################################
set(SLAM_LYJ_OPTIMIZE_SRC_FILES)
GroupFiles(SLAM_LYJ_OPTIMIZE_SRC_FILES ${CMAKE_CURRENT_LIST_DIR} ".")

# ################################library##################################
add_library(${PROJECT_NAME} SHARED ${SLAM_LYJ_OPTIMIZE_SRC_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${ALL_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC

    ${ALL_LIBRARIES}
    # Eigen3::Eigen
)

# ################################output##################################
if(DEFINED LIBRARY_LYJ_OUTPUT_PATH)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_LYJ_OUTPUT_PATH}
        ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_LYJ_OUTPUT_PATH}
        RUNTIME_OUTPUT_DIRECTORY ${LIBRARY_LYJ_OUTPUT_PATH}
    )
endif()

# ################################example##################################
add_executable(test example/main.cpp)
target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test PRIVATE ${PROJECT_NAME})
set_target_properties(test PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=${ALL_BINARY_DIRS}")

# ################################install##################################
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)

# 安装库和头文件
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib # DLL 在 Windows 下会被安装到 bin/
    RUNTIME DESTINATION bin # 适用于 DLL 的运行时组件
    INCLUDES DESTINATION include
)

# #file
# install(FILES test.h
# DESTINATION include
# PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
# #directory
# install(DIRECTORY ./          # 注意末尾的斜杠
# DESTINATION include
# FILES_MATCHING PATTERN "*.h" # 仅匹配 .h 文件
# PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
# )
install(DIRECTORY ./
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.h" # 包含所有 .h 文件

    # 排除指定目录（递归匹配）
    PATTERN "build" EXCLUDE # 排除所有名为 build 的目录及内容
    PATTERN "cmake" EXCLUDE # 排除所有名为 cmake 的目录及内容
    PATTERN "install" EXCLUDE # 排除所有名为 install 的目录及内容
    PATTERN "Lib" EXCLUDE # 排除所有名为 Lib 的目录及内容
    PATTERN ".git" EXCLUDE # 排除所有名为 Lib 的目录及内容
    PATTERN ".vscode" EXCLUDE # 排除所有名为 Lib 的目录及内容
    PATTERN "example" EXCLUDE # 排除所有名为 Lib 的目录及内容
    PATTERN "Output" EXCLUDE # 排除所有名为 Lib 的目录及内容
    PERMISSIONS
    OWNER_READ GROUP_READ WORLD_READ
)

# 安装后删除空目录（可选）
install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/cleanup.cmake")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cleanup.cmake [[
file(REMOVE_RECURSE "${CMAKE_INSTALL_PREFIX}/include/build")
file(REMOVE_RECURSE "${CMAKE_INSTALL_PREFIX}/include/cmake")
]])

# set(TARGETNAME ${PROJECT_NAME}Targets)
# 导出目标配置
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake
)

# 生成包配置文件
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake
)
write_basic_package_version_file(
    ${PROJECT_NAME}ConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION lib/cmake
)