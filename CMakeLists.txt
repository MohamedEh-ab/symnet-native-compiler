cmake_minimum_required(VERSION 3.18)
project(SymNet
        VERSION 2.1.0
        DESCRIPTION "Constant-folding, SIMD-friendly C++17 engine for the SymNet DSL"
        LANGUAGES CXX)

# ----------------------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------------------
option(SYMNET_NATIVE_ARCH    "Compile in-tree targets with -march=native (non-portable binaries)" ON)
option(SYMNET_STRICT_FP      "Add -ffp-contract=off for bit-reproducible floating point"           OFF)
option(SYMNET_BUILD_CLI      "Build the symnet_cli executable"                                     ON)
option(SYMNET_BUILD_PYTHON   "Build the Python extension module (needs pybind11)"                  ON)
option(SYMNET_FETCH_PYBIND11 "Download pybind11 with FetchContent if it is not found"              ON)
option(SYMNET_BUILD_TESTS    "Build and register the tests"                                        ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

# ----------------------------------------------------------------------------------------
# Internal build flags. They reach in-tree targets only ($<BUILD_INTERFACE>), so an installed
# SymNet::native_cpp never forces -march=native onto downstream projects.
# ----------------------------------------------------------------------------------------
add_library(symnet_native_flags INTERFACE)
if(MSVC)
  target_compile_options(symnet_native_flags INTERFACE /W4 /O2
      $<$<BOOL:${SYMNET_NATIVE_ARCH}>:/arch:AVX2>
      $<$<BOOL:${SYMNET_STRICT_FP}>:/fp:strict>)
else()
  target_compile_options(symnet_native_flags INTERFACE
      -Wall -Wextra -Wpedantic
      $<$<NOT:$<CONFIG:Debug>>:-O3>
      $<$<BOOL:${SYMNET_NATIVE_ARCH}>:-march=native>
      $<$<BOOL:${SYMNET_STRICT_FP}>:-ffp-contract=off>)
endif()

# ----------------------------------------------------------------------------------------
# Header-only library:  target_link_libraries(app PRIVATE SymNet::native_cpp)
# ----------------------------------------------------------------------------------------
add_library(symnet_native_cpp INTERFACE)
add_library(SymNet::native_cpp ALIAS symnet_native_cpp)
set_target_properties(symnet_native_cpp PROPERTIES EXPORT_NAME native_cpp)
target_include_directories(symnet_native_cpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(symnet_native_cpp INTERFACE cxx_std_17)
target_link_libraries(symnet_native_cpp INTERFACE $<BUILD_INTERFACE:symnet_native_flags>)

# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------
if(SYMNET_BUILD_CLI)
  add_executable(symnet_cli src/main.cpp)
  target_link_libraries(symnet_cli PRIVATE SymNet::native_cpp)
  install(TARGETS symnet_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# ----------------------------------------------------------------------------------------
# Python extension module `symnet_native`
# ----------------------------------------------------------------------------------------
if(SYMNET_BUILD_PYTHON)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module QUIET)
  if(Python_FOUND)
    set(PYBIND11_FINDPYTHON ON)  # make pybind11 reuse the interpreter found above

    # `pip install pybind11` is invisible to CMake; ask the interpreter where its config lives.
    if(NOT pybind11_DIR)
      execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                      OUTPUT_VARIABLE _symnet_pb11_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                      RESULT_VARIABLE _symnet_pb11_rc ERROR_QUIET)
      if(_symnet_pb11_rc EQUAL 0 AND EXISTS "${_symnet_pb11_dir}")
        set(pybind11_DIR "${_symnet_pb11_dir}" CACHE PATH "pybind11 CMake config directory")
      endif()
    endif()

    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND AND SYMNET_FETCH_PYBIND11)
      message(STATUS "pybind11 not found locally - fetching v2.13.6 with FetchContent")
      include(FetchContent)
      FetchContent_Declare(pybind11
          GIT_REPOSITORY https://github.com/pybind/pybind11.git
          GIT_TAG        v2.13.6
          GIT_SHALLOW    TRUE)
      FetchContent_MakeAvailable(pybind11)
      set(pybind11_FOUND TRUE)
    endif()

    if(pybind11_FOUND)
      pybind11_add_module(symnet_native MODULE
          src/bindings.cpp src/bind_dsl.cpp src/bind_finite_field.cpp)
      target_link_libraries(symnet_native PRIVATE SymNet::native_cpp)
      set_target_properties(symnet_native PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
      set(SYMNET_PYTHON_INSTALL_DIR
          "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages"
          CACHE STRING "Install destination of the Python module (relative to the install prefix)")
      install(TARGETS symnet_native LIBRARY DESTINATION ${SYMNET_PYTHON_INSTALL_DIR})
    else()
      message(WARNING "pybind11 not found and SYMNET_FETCH_PYBIND11=OFF: skipping the Python module. "
                      "Run `pip install pybind11` or pass -Dpybind11_DIR=<dir>.")
    endif()
  else()
    message(WARNING "Python 3 interpreter/development files not found: skipping the Python module.")
  endif()
endif()

# ----------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------
if(SYMNET_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------------------
# Install headers + exported CMake target for downstream find_package / link users
# ----------------------------------------------------------------------------------------
install(DIRECTORY include/symnet DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS symnet_native_cpp EXPORT SymNetTargets)
install(EXPORT SymNetTargets
        NAMESPACE SymNet::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SymNet)

message(STATUS "SymNet ${PROJECT_VERSION}: ${CMAKE_BUILD_TYPE}, native arch=${SYMNET_NATIVE_ARCH}, "
               "CLI=${SYMNET_BUILD_CLI}, Python=${SYMNET_BUILD_PYTHON}, tests=${SYMNET_BUILD_TESTS}")
