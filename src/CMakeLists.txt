cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++17 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4 REQUIRED)
# find_package(G2O REQUIRED)

# g2o 使用thirdparty中的
# include_directories(../3rdparty/g2o/)
# include_directories(../3rdparty/g2o/build)
# set(g2o_libs
#         ${CMAKE_SOURCE_DIR}/../3rdparty/g2o/lib/libg2o_stuff.so
#         ${CMAKE_SOURCE_DIR}/../3rdparty/g2o/lib/libg2o_core.so
#         ${CMAKE_SOURCE_DIR}/../3rdparty/g2o/lib/libg2o_solver_dense.so
#         ${CMAKE_SOURCE_DIR}/../3rdparty/g2o/lib/libg2o_solver_csparse.so
#         ${CMAKE_SOURCE_DIR}/../3rdparty/g2o/lib/libg2o_csparse_extension.so
#         ${CMAKE_SOURCE_DIR}/../3rdparty/g2o/lib/libg2o_types_sba.so
#         )
# find_package(G2O REQUIRED)
# include_directories(${G2O_INCLUDE_DIRS})
# find_package(Sophus REQUIRED)
# include_directories("../3rdparty/Sophus")

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        # ${G2O_INCLUDE_DIRS}
        # ${Sophus_INCLUDE_DIRS}
        # "/usr/include/eigen3/"
)

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(orb_move orb_move.cpp)
target_link_libraries(orb_move ${OpenCV_LIBS})

# # add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2 
# add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
# target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

# # # add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2
# add_executable(triangulation triangulation.cpp)
# target_link_libraries(triangulation ${OpenCV_LIBS})

# add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
# target_link_libraries(pose_estimation_3d2d
# # ${g2o_libs}
#         ${OpenCV_LIBS})

# add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
# target_link_libraries(pose_estimation_3d3d
# # ${g2o_libs}
#         ${OpenCV_LIBS})
