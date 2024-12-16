cmake_minimum_required(VERSION 3.28)
project(MyAlphaGo)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(GTP-Core/cpp)
include_directories(GTP-Core/cpp/src)

add_executable(MyAlphaGo
        src/main.cpp
        # other MyAlphaGo source files...
)