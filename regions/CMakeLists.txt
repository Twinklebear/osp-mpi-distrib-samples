add_executable(regions regions.cpp)
target_link_libraries(regions ${OSPRAY_LIBRARIES} ${MPI_CXX_LIBRARIES})

