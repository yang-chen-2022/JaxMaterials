###################
# Main CMake file #
###################

cmake_minimum_required(VERSION 3.23)

project(LinearElasticity)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_language(CUDA)
find_package(CUDAToolkit)

# All code is in the src subdirectory
add_subdirectory(src)

# Obtain google test framework
include(FetchContent)
FetchContent_Declare(
  googletest
  DOWNLOAD_EXTRACT_TIMESTAMP True
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip  
)
FetchContent_MakeAvailable(googletest)
