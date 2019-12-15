project(minpt)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fopenmp")

add_executable(minpt minpt.cpp)