#ifndef __MATH_FUNCTINS_H__
#define __MATH_FUNCTINS_H__

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <cblas.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#include <algorithm>

#include <glog/logging.h>
#define TENSORRT_BLOCK 512

inline int TENSORRT_GET_BLOCKS(const int N) {
  return (N + TENSORRT_BLOCK - 1) / TENSORRT_BLOCK;
}
cudaError_t PReLUForward(const int count, const int channels, const int dim, const float* bottom_data,
  float* top_data, void* mDeviceKernel, const int div_factor);

#endif