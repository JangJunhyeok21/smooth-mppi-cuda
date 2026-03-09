cmake_minimum_required(VERSION 3.8)
project(cuda_mppi_controller)

# [수정 1] GPU 연산 속도를 위해 반드시 Release 모드 사용
set(CMAKE_BUILD_TYPE Release) 

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

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
find_package(rosidl_default_generators REQUIRED)
# [수정 3] fastrtps_cpp 종속성 제거 (기본 typesupport가 알아서 처리함)
find_package(std_msgs REQUIRED)

# CUDA 라이브러리 찾기
find_package(CUDA REQUIRED)

# 라이브러리 생성 (.cpp 대신 .cu 포함)
add_library(cuda_mppi_lib
  src/mppi_core.cu
)

# GCC line directive 경고 억제 및 고속 수학 함수 적용
target_compile_options(cuda_mppi_lib PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:Clang>>:-Xcompiler=-Wno-gnu-line-directive>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
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

target_compile_options(cuda_mppi_node PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:Clang>>:-Xcompiler=-Wno-gnu-line-directive>
)

target_include_directories(cuda_mppi_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 커스텀 메시지 생성 인터페이스
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MppiTrajectory.msg"
  DEPENDENCIES std_msgs
)

# [수정 3 적용] 기본 C++ typesupport만 연결
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(cuda_mppi_node "${cpp_typesupport_target}")

ament_target_dependencies(cuda_mppi_node
  rclcpp
  ackermann_msgs
  nav_msgs
  visualization_msgs
  geometry_msgs
)

# 라이브러리 링크 (CUDA 링크)
target_link_libraries(cuda_mppi_node
  cuda_mppi_lib
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
  rclcpp
  nav_msgs
  geometry_msgs
)

install(TARGETS cuda_mppi_node cuda_mppi_lib path_publisher
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config data msg
  DESTINATION share/${PROJECT_NAME})
  
ament_export_dependencies(rosidl_default_runtime)

ament_package()