add_executable(partition partition.cpp)
target_link_libraries(partition ${OSPRAY_LIBRARIES} ${MPI_CXX_LIBRARIES})

