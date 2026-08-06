/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <cmath>

#include "xtd/internal/defines.h"
#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include "xtd/math/cosh.inl"
#endif

namespace xtd {

  /* Computes the hyperbolic cosine of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16 cosh(float16 arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::coshf(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __float2half(::coshf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cosh(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
    // Mask off the sign bit (bit 15) to evaluate |arg|
    std::uint16_t abs_bits = std::bit_cast<std::uint16_t>(arg) & 0x7FFF;
    return std::bit_cast<float16>(coshf16_lut[abs_bits]);
/*
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
#if defined(__clang__) && __has_builtin(__builtin_coshf16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::cosh(static_cast<float>(arg));
    } return __builtin_coshf16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_coshf)
    if (std::is_constant_evaluated()) {
      return std::cosh(static_cast<float>(arg));
    } return __builtin_coshf(static_cast<float>(arg));
#else
    return std::cosh(static_cast<float>(arg));
#endif
#else
    return std::cosh(static_cast<float_t>(arg));
*/
#endif
  }

  /* Computes the hyperbolic cosine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float cosh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::coshf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::coshf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cosh(arg);
#else
    // standard C/C++ code
    return ::coshf(arg);
#endif
  }

  /* Computes the hyperbolic cosine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cosh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cosh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cosh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cosh(arg);
#else
    // standard C/C++ code
    return ::cosh(arg);
#endif
  }

  /* Computes the hyperbolic cosine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cosh(std::integral auto arg) {
    return xtd::cosh(static_cast<double>(arg));
  }

  /* Computes the hyperbolic cosine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float coshf(std::floating_point auto arg) {
    return xtd::cosh(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float coshf(std::integral auto arg) {
    return xtd::cosh(static_cast<float>(arg));
  }

}  // namespace xtd
