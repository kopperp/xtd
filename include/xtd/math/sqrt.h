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
  /* Computes the square root of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16 sqrt(float16 arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hsqrt(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hsqrt(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::half_precision::sqrt(arg);
#elif XTD_HAS_STDFLOAT16
    if (std::is_constant_evaluated()) {
      return static_cast<float16>(std::sqrt(static_cast<float>(arg)));
    } else {
#if defined(__clang__) && __has_builtin(__builtin_sqrtf16) && defined(__AVX512FP16__)
      return __builtin_sqrtf16(std::bit_cast<std::float16_t>(arg));
#elif defined(__clang__)
      return __builtin_sqrtf(static_cast<float>(arg));
#else
      return std::sqrt(static_cast<float>(arg));
#endif
    }
#else
    // standard C/C++ code
    return float16(std::sqrt(static_cast<float_t>(arg)));
#endif
  }

  /* Computes the square root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sqrt(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sqrtf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sqrtf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sqrt(arg);
#else
    // standard C/C++ code
    return ::sqrtf(arg);
#endif
  }

  /* Computes the square root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sqrt(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sqrt(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sqrt(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sqrt(arg);
#else
    // standard C/C++ code
    return ::sqrt(arg);
#endif
  }

  /* Computes the square root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sqrt(std::integral auto arg) {
    return xtd::sqrt(static_cast<double>(arg));
  }

  /* Computes the square root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sqrtf(std::floating_point auto arg) {
    return xtd::sqrt(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float sqrtf(std::integral auto arg) {
    return xtd::sqrt(static_cast<float>(arg));
  }

  /* Computes the square root of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16 sqrtf(xtd::floating_point auto arg) {
    return xtd::sqrt(static_cast<float16>(arg));
  }

}  // namespace xtd
