cmake_minimum_required(VERSION 3.16)
project(pcl_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Only the components the harness actually uses; pulling all of PCL would
# drag in visualization and its VTK dependency for no reason.
find_package(PCL 1.12 REQUIRED COMPONENTS common filters)

add_executable(pcl_bench main.cpp)
target_include_directories(pcl_bench PRIVATE ${PCL_INCLUDE_DIRS})
target_link_directories(pcl_bench PRIVATE ${PCL_LIBRARY_DIRS})
target_compile_definitions(pcl_bench PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(pcl_bench PRIVATE ${PCL_LIBRARIES})
