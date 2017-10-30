add_executable(simple simple.cpp)
target_link_libraries(simple ${OSPRAY_LIBRARIES} ${MPI_CXX_LIBRARIES})

