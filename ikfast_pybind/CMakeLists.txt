cmake_minimum_required(VERSION 3.15.0)
project(ikfast_pybind)
add_compile_options(-std=c++17)
# cmake_policy(SET CMP0148 NEW)
set(Python3_FIND_VIRTUALENV "FIRST")

# First find path of python executable
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
# Manually specify the path to NumPy headers
message(STATUS "python3 path: ${Python3_EXECUTABLE} ")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NumPy_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (NOT NumPy_INCLUDE_DIR)
    message(FATAL_ERROR "NumPy headers not found")
endif()

# Get pybind11 path
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_VENV_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "pybind11 path: ${PYBIND11_VENV_CMAKE_PATH}")
set(pybind11_DIR ${PYBIND11_VENV_CMAKE_PATH})
find_package(pybind11 REQUIRED)
find_package(LAPACK)
find_package(Eigen3)


include_directories(
  include
)

add_library(${PROJECT_NAME}_ik
  src/${PROJECT_NAME}/ikfast71_panda_arm.cpp
)
set_target_properties(${PROJECT_NAME}_ik PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${PROJECT_NAME}_ik
  ${LAPACK_LIBRARIES}
)


# Declare pybind library
add_library(${PROJECT_NAME} SHARED
  src/${PROJECT_NAME}/ik.cpp
)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
message(STATUS "${pybind11_INCLUDE_DIRS}")
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
  ${PROJECT_NAME}_ik
  Python3::Python
  ${pybind11_LIBRARIES}
)

add_executable(${PROJECT_NAME}_test 
  src/${PROJECT_NAME}/test.cpp
)
set_target_properties(${PROJECT_NAME}_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${PROJECT_NAME}_test SYSTEM PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_test
  ${PROJECT_NAME}_ik
)

install(TARGETS ${PROJECT_NAME} 
  RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/pybind/panda_ikfast
  LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/pybind/panda_ikfast
  ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/pybind/panda_ikfast
)

install(
    CODE "execute_process(COMMAND pip install ${CMAKE_CURRENT_SOURCE_DIR})"
)