cmake_minimum_required(VERSION 3.8)
project(cuda_mppi_controller)

# CUDA 언어 활성화
enable_language(CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# CUDA 라이브러리 찾기 (보통 시스템 경로에 있음)
find_package(CUDA REQUIRED)

# 라이브러리 생성 (.cpp 대신 .cu 포함)
add_library(cuda_mppi_lib
  src/mppi_core.cu
)

# 헤더 경로 설정
target_include_directories(cuda_mppi_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDA_INCLUDE_DIRS}
)

# 실행 파일 생성
add_executable(cuda_mppi_node
  src/cuda_mppi_node.cpp
)

target_include_directories(cuda_mppi_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(cuda_mppi_node
  rclcpp
  ackermann_msgs
  nav_msgs
  visualization_msgs
  geometry_msgs
)

# 라이브러리 링크 (OpenMP 대신 CUDA 링크)
target_link_libraries(cuda_mppi_node
  cuda_mppi_lib
  ${CUDA_LIBRARIES}
)

install(TARGETS cuda_mppi_node cuda_mppi_lib
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

ament_package()