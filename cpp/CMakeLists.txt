cmake_minimum_required(VERSION 3.28)
project(MyAlphaGo)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(GTP-Core/cpp)
include_directories(GTP-Core/cpp/src)

set(Torch_DIR "home/nana/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

target_link_libraries(GoTraining "${TORCH_LIBRARIES}" "${HDF5_LIBRARIES}")

if (TORCH_CUDA_FOUND)
    message(STATUS "Compiling with CUDA support.")
    target_compile_definitions(GoTraining PRIVATE -DUSE_CUDA)
endif()

add_executable(MyAlphaGo
        src/main.cpp
        # other MyAlphaGo source files...
)

set_property(TARGET MyAlphaGo PROPERTY CXX_STANDARD 20)
set_property(TARGET MyAlphaGo PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
