cmake_minimum_required(VERSION 3.15...3.30)

project(
    MocapDrone
    VERSION 1.0
    LANGUAGES CXX
    )

find_package(lcm REQUIRED)
include(${LCM_USE_FILE})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

