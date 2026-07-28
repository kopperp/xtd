/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Catch accidental prior inclusion of <stdfloat> before defines.h
#if defined(_GLIBCXX_STDFLOAT) || defined(_LIBCPP_STDFLOAT) || defined(_STDFLOAT_)
    #error "<stdfloat> was included before 'xtd/internal/defines.h'. Please include 'defines.h' first."
#endif

// XTD_DEVICE_FUNCTION
#if defined(__CUDACC__) || defined(__HIPCC__)
// CUDA or HIP/ROCm compiler
#define XTD_DEVICE_FUNCTION __host__ __device__
#else
// SYCL or standard C++ code
#define XTD_DEVICE_FUNCTION
#endif

// XTD_TARGET_...
#if defined(__CUDA_ARCH__)
// CUDA device code
#define XTD_TARGET_CUDA
#elif defined(__HIP_DEVICE_COMPILE__)
// HIP/ROCm device code
#define XTD_TARGET_HIP
#elif defined(__SYCL_DEVICE_ONLY__)
// SYCL device code
#define XTD_TARGET_SYCL
#else
// standard C++ code
#define XTD_TARGET_CPU
#endif

// backend runtime include files
#if defined(XTD_TARGET_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(XTD_TARGET_HIP)
#include <hip/hip_runtime.h>
#endif

#if defined(XTD_TARGET_SYCL)
#include <sycl/sycl.hpp>
#endif

#ifndef XTD_HAS_STDFLOAT16 // Remove this? Mainly for debugging
#if __has_include(<stdfloat>) && defined(__STDCPP_FLOAT16_T__)
    #include <stdfloat>
    #define XTD_HAS_STDFLOAT16 1
#else
    #define XTD_HAS_STDFLOAT16 0
#endif
#endif

// Prevent accidental downstream includes of <stdfloat>
#if defined(__GNUC__) || defined(__clang__)
    #pragma GCC poison stdfloat
#endif

#include "xtd/internal/float16.h"
