cmake_minimum_required(VERSION 3.18)

project(ffs_gwc_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "RTX 5080 SM120")
endif()

find_package(CUDAToolkit REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT SDK root containing include/ and lib/")
set(TENSORRT_PYTHON_DIR "" CACHE PATH "dp3 site-packages directory containing tensorrt_libs/")
set(TENSORRT_INCLUDE_HINTS
  ${TENSORRT_ROOT}/include
  ${TENSORRT_ROOT}/include/x86_64-linux-gnu
  ${TENSORRT_PYTHON_DIR}/tensorrt/include
  $ENV{CONDA_PREFIX}/include
  $ENV{CONDA_PREFIX}/lib/python3.10/site-packages/tensorrt/include
)
set(TENSORRT_LIBRARY_HINTS
  ${TENSORRT_ROOT}/lib
  ${TENSORRT_ROOT}/lib64
  ${TENSORRT_ROOT}/lib/x86_64-linux-gnu
  ${TENSORRT_PYTHON_DIR}/tensorrt_libs
  $ENV{CONDA_PREFIX}/lib/python3.10/site-packages/tensorrt_libs
)

find_path(NVINFER_INCLUDE_DIR NvInfer.h PATHS ${TENSORRT_INCLUDE_HINTS} NO_DEFAULT_PATH)
if(NOT NVINFER_INCLUDE_DIR)
  message(FATAL_ERROR "TensorRT headers not found. Set -DTENSORRT_ROOT or provide TENSORRT_PYTHON_DIR plus SDK headers.")
endif()
find_file(NVINFER_LIB_VERSIONED
  NAMES libnvinfer.so.10 libnvinfer.so.10.*
  PATHS ${TENSORRT_LIBRARY_HINTS}
  NO_DEFAULT_PATH
)
if(NVINFER_LIB_VERSIONED)
  set(NVINFER_LIB "${NVINFER_LIB_VERSIONED}")
else()
  find_library(NVINFER_LIB NAMES nvinfer PATHS ${TENSORRT_LIBRARY_HINTS} NO_DEFAULT_PATH)
endif()
if(NOT NVINFER_LIB)
  message(FATAL_ERROR "libnvinfer.so.10 not found. Set -DTENSORRT_ROOT or TENSORRT_PYTHON_DIR.")
endif()

add_library(ffs_gwc_plugin SHARED
  src/gwc_volume_plugin.cpp
  src/depth_kernels.cu
)
set_target_properties(ffs_gwc_plugin PROPERTIES
  OUTPUT_NAME "ffs_gwc_plugin"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_WITH_INSTALL_RPATH ON
  BUILD_RPATH_USE_ORIGIN ON
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)
target_compile_options(ffs_gwc_plugin PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20015>
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20013>
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20011>
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20091>
)
target_include_directories(ffs_gwc_plugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${NVINFER_INCLUDE_DIR}
)
target_link_libraries(ffs_gwc_plugin PRIVATE ${NVINFER_LIB} CUDA::cudart)

install(TARGETS ffs_gwc_plugin LIBRARY DESTINATION lib)
