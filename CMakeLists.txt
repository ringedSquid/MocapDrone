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



# Handle types
FILE(
    GLOB 
    type_definitions 
    "${CMAKE_CURRENT_LIST_DIR}/./types/*.lcm"
    )

lcm_wrap_types(
    CPP_HEADERS cpp_headers
    ${type_definitions}
    )

lcm_add_library(type_definitions-cpp CPP ${cpp_headers})

target_include_directories(
    type_definitions-cpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )


add_executable(camera_feed_process "src/camera_feed_process.cpp")
lcm_target_link_libraries(camera_feed_process type_definitions-cpp ${LCM_NAMESPACE}lcm ${OpenCV_LIBS})

