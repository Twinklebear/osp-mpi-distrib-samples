cmake_minimum_required(VERSION 3.5)
project(osp_mpi_distrib_samples)

set(CMAKE_CXX_STANDARD 14)

find_package(ospray REQUIRED)
find_package(MPI REQUIRED)
include_directories(${OSPRAY_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})

add_subdirectory(simple)
add_subdirectory(regions)

