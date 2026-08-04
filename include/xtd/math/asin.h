/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <cmath>

#include "xtd/internal/defines.h"

namespace xtd {
  /* Computes the inverse sine (measured in radians) of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16 asin(float16 arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::asinf(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    // return ::hasin(arg);
    // AMD uses a suboptimal range reduction, use full float evaluation
    return float16(::asinf(static_cast<float>(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::half_precision::asin(arg);
#elif XTD_HAS_STDFLOAT16
    if (std::is_constant_evaluated()) {
      return static_cast<float16>(std::asin(static_cast<float>(arg)));
    } else {
#if defined(__clang__) && __has_builtin(__builtin_asinf16) && defined(__AVX512FP16__)
      return __builtin_asinf16(std::bit_cast<std::float16_t>(arg));
#elif defined(__clang__)
      return __builtin_asinf(static_cast<float>(arg));
#else
      return std::asin(static_cast<float>(arg));
#endif
    }
#else
    // standard C/C++ code
    return float16(std::asin(static_cast<float_t>(arg)));
#endif
  }

  /* Computes the inverse sine (measured in radians) of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float asin(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asinf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asinf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asin(arg);
#else
    // standard C/C++ code
    return ::asinf(arg);
#endif
  }

  /* Computes the inverse sine (measured in radians) of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double asin(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asin(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asin(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asin(arg);
#else
    // standard C/C++ code
    return ::asin(arg);
#endif
  }

  /* Computes the inverse sine (measured in radians) of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double asin(std::integral auto arg) {
    return xtd::asin(static_cast<double>(arg));
  }

  /* Computes the inverse sine (measured in radians) of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float asinf(std::floating_point auto arg) {
    return xtd::asin(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float asinf(std::integral auto arg) {
    return xtd::asin(static_cast<float>(arg));
  }

  /* Computes the inverse sine (measured in radians) of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16 asinf(xtd::floating_point auto arg) {
    return xtd::asin(static_cast<float16>(arg));
  }

}  // namespace xtd
