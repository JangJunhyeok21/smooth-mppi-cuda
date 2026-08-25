cmake_minimum_required(VERSION 3.8)
project(smppi_cuda_controller LANGUAGES C CXX CUDA)

# ROS 2 Humble's rosidl generators are installed for the system Python.  A
# shell with Conda active may otherwise select Conda's incompatible empy
# package and fail while generating this package's messages.
set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH
  "Python interpreter used by ROS 2 code generation" FORCE)
set(PYTHON_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH
  "Legacy Python interpreter used by ROS 2 code generation" FORCE)

# colcon / VS Code tasks 에서 Debug 또는 Release 를 선택할 수 있도록
# 사용자가 지정하지 않은 경우에만 Release 를 기본값으로 둔다.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# 1. 아키텍처 자동 감지 및 분기 로직
# Jetson 환경(ARM64)인지 데스크톱(x86_64)인지 프로세서 종류로 판단합니다.
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Detected ARM64 architecture (Assuming Jetson Orin Nano).")
    set(CMAKE_CUDA_ARCHITECTURES "87") # Jetson Orin Nano
else()
    message(STATUS "Detected x86_64 architecture (Assuming ThinkPad).")
    # Ampere/Ada cubins plus compute_89 PTX, which the installed driver can
    # JIT for newer GPUs. CUDA 12.6 cannot compile sm_120 directly.
    set(CMAKE_CUDA_ARCHITECTURES "86;89")
endif()

# CUDA 언어 활성화
enable_language(CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(rosidl_default_generators REQUIRED)
# [수정 3] fastrtps_cpp 종속성 제거 (기본 typesupport가 알아서 처리함)
find_package(std_msgs REQUIRED)
# f1_msgs is used only by the separate predictor; the MPPI node consumes the
# package-owned DynamicObstacleTrajectory contract.
find_package(f1_msgs QUIET)

# CUDA 라이브러리 찾기
find_package(CUDAToolkit REQUIRED)
# The predictor uses the same lightweight ONNX Runtime C++ path as the
# real-time MDN_SLCMPC implementation.  Override this cache entry on Orin with
# the root of an ONNX Runtime build that includes CUDA/TensorRT providers.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(SMPPI_ONNXRUNTIME_DEFAULT "/opt/onnxruntime/onnxruntime-linux-aarch64-1.20.1")
else()
  set(SMPPI_ONNXRUNTIME_DEFAULT "/opt/onnxruntime/onnxruntime-linux-x64-1.20.1")
endif()
set(SMPPI_ONNXRUNTIME_ROOT "${SMPPI_ONNXRUNTIME_DEFAULT}"
    CACHE PATH "ONNX Runtime installation root (set this on Orin)")
if(NOT EXISTS "${SMPPI_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h" OR
   NOT EXISTS "${SMPPI_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
  message(FATAL_ERROR
    "ONNX Runtime not found under ${SMPPI_ONNXRUNTIME_ROOT}; set "
    "-DSMPPI_ONNXRUNTIME_ROOT=<onnxruntime-install-root>")
endif()

# 라이브러리 생성 (.cpp 대신 .cu 포함)
add_library(smppi_lib STATIC
  src/mppi_core.cu
)

# GCC line directive 경고 억제 및 고속 수학 함수 적용
target_compile_options(smppi_lib PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:Clang>>:-Xcompiler=-Wno-gnu-line-directive>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# 헤더 경로 설정
target_include_directories(smppi_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDA_INCLUDE_DIRS}
)

# 실행 파일 생성
add_executable(smppi_node
  src/smppi_node.cpp
)
add_executable(dynamic_obstacle_predictor_node
  src/dynamic_obstacle_predictor_node.cpp)
set_property(TARGET dynamic_obstacle_predictor_node PROPERTY CXX_STANDARD 17)
target_include_directories(dynamic_obstacle_predictor_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${SMPPI_ONNXRUNTIME_ROOT}/include)

add_executable(mppi_step_parity model_tuning/real_car_v2/mppi_step_parity.cu)
target_include_directories(mppi_step_parity PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(mppi_step_parity smppi_lib ${CUDA_LIBRARIES})

add_executable(mppi_horizon80_benchmark
  model_tuning/real_car_v2/benchmark_mppi_horizon80.cu)
target_include_directories(mppi_horizon80_benchmark PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(mppi_horizon80_benchmark smppi_lib ${CUDA_LIBRARIES})

target_compile_options(smppi_node PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:Clang>>:-Xcompiler=-Wno-gnu-line-directive>
)

target_include_directories(smppi_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 커스텀 메시지 생성 인터페이스
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MppiTrajectory.msg"
  "msg/KfState.msg"
  "msg/DynamicObstacleTrajectory.msg"
  DEPENDENCIES std_msgs
)

# [수정 3 적용] 기본 C++ typesupport만 연결
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(smppi_node "${cpp_typesupport_target}")
target_link_libraries(dynamic_obstacle_predictor_node
  "${cpp_typesupport_target}"
  ${SMPPI_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
set_target_properties(dynamic_obstacle_predictor_node PROPERTIES
  BUILD_RPATH "${SMPPI_ONNXRUNTIME_ROOT}/lib"
  INSTALL_RPATH "${SMPPI_ONNXRUNTIME_ROOT}/lib")

set(SMPPI_NODE_DEPENDENCIES
  ament_index_cpp
  rclcpp
  ackermann_msgs
  nav_msgs
  visualization_msgs
  geometry_msgs
  sensor_msgs
)
ament_target_dependencies(smppi_node ${SMPPI_NODE_DEPENDENCIES})
ament_target_dependencies(dynamic_obstacle_predictor_node
  rclcpp nav_msgs ament_index_cpp visualization_msgs geometry_msgs)
if(f1_msgs_FOUND)
  ament_target_dependencies(dynamic_obstacle_predictor_node f1_msgs)
  target_compile_definitions(dynamic_obstacle_predictor_node PRIVATE SMPPI_HAS_F1_MSGS=1)
endif()

# 라이브러리 링크 (CUDA 링크)
target_link_libraries(smppi_node
  smppi_lib
  Eigen3::Eigen
  ${CUDA_LIBRARIES}
)

# path_publisher 실행 파일
add_executable(path_publisher
  src/path_publisher.cpp
)

target_include_directories(path_publisher PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(path_publisher
  ament_index_cpp
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
)

install(TARGETS smppi_node dynamic_obstacle_predictor_node path_publisher mppi_step_parity
  mppi_horizon80_benchmark
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(TARGETS smppi_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# Runtime predictor is the C++ target above.  Keep only the Python data
# collection/opponent-driving utilities in the installed runtime scripts.
install(PROGRAMS
  scripts/map2_opponent_driver.py
  scripts/collect_mppi_mdn_episode.py
  scripts/simulated_perception_obstacles.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config data msg
  DESTINATION share/${PROJECT_NAME})
  
ament_export_dependencies(rosidl_default_runtime)

ament_package()
