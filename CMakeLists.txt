cmake_minimum_required(VERSION 3.0)
project(example)

find_package(Torch REQUIRED)
find_package(OpenCV 4.1.0 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(example mnist-code.cpp main.h)

target_link_libraries(example ${OpenCV_LIBS})
target_link_libraries(example "${TORCH_LIBRARIES}")

set_property(TARGET example PROPERTY CXX_STANDARD 11)