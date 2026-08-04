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
  /* Computes the hyperbolic tangent of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16 tanh(float16 arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::htanh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    // HIP is missing this specific function
    return float16(::tanh(static_cast<float_t>(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::half_precision::tanh(arg);
#elif XTD_HAS_STDFLOAT16
    if (std::is_constant_evaluated()) {
      return static_cast<float16>(std::tanh(static_cast<float>(arg)));
    } else {
#if defined(__clang__) && __has_builtin(__builtin_tanhf16) && defined(__AVX512FP16__)
      return __builtin_tanhf16(std::bit_cast<std::float16_t>(arg));
#elif defined(__clang__)
      return __builtin_tanhf(static_cast<float>(arg));
#else
      return std::tanh(static_cast<float>(arg));
#endif
    }
#else
    // standard C/C++ code
    return float16(std::tanh(static_cast<float_t>(arg)));
#endif
  }

  /* Computes the hyperbolic tangent of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float tanh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tanhf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tanhf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tanh(arg);
#else
    // standard C/C++ code
    return ::tanhf(arg);
#endif
  }

  /* Computes the hyperbolic tangent of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double tanh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tanh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tanh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tanh(arg);
#else
    // standard C/C++ code
    return ::tanh(arg);
#endif
  }

  /* Computes the hyperbolic tangent of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double tanh(std::integral auto arg) {
    return xtd::tanh(static_cast<double>(arg));
  }

  /* Computes the hyperbolic tangent of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float tanhf(std::floating_point auto arg) {
    return xtd::tanh(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float tanhf(std::integral auto arg) {
    return xtd::tanh(static_cast<float>(arg));
  }

  /* Computes the hyperbolic tangent of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16 tanhf(xtd::floating_point auto arg) {
    return xtd::tanh(static_cast<float16>(arg));
  }

}  // namespace xtd
