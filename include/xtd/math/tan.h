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
#include "xtd/math/tan.inl"
#endif

namespace xtd {
  /* Computes the tangent of arg (measured in radians), in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16 tan(float16 arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::tanf(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __float2half(::tanf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::half_precision::tan(arg);
#else
    // standard C/C++ code
    // XOR the sign bit (bit 15) to evaluate -arg
    std::uint16_t arg_bits = std::bit_cast<std::uint16_t>(arg);
    std::uint16_t abs_bits = arg_bits & 0x7FFF;
    std::uint16_t sign_bit = arg_bits & 0x8000;

    return std::bit_cast<float16>(static_cast<std::uint16_t>(tanf16_lut[abs_bits] ^ sign_bit));
/*
#elif XTD_HAS_STDFLOAT16
#if defined(__clang__) && __has_builtin(__builtin_tanf16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::tan(static_cast<float>(arg));
    } return __builtin_tanf16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_tanf)
    if (std::is_constant_evaluated()) {
      return std::tan(static_cast<float>(arg));
    } return __builtin_tanf(static_cast<float>(arg));
#else
    return std::tan(static_cast<float>(arg));
#endif
#else
    return std::tan(static_cast<float_t>(arg));
*/
#endif
  }

  /* Computes the tangent of arg (measured in radians), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float tan(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tanf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tanf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tan(arg);
#else
    // standard C/C++ code
    return ::tanf(arg);
#endif
  }

  /* Computes the tangent of arg (measured in radians), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double tan(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tan(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tan(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tan(arg);
#else
    // standard C/C++ code
    return ::tan(arg);
#endif
  }

  /* Computes the tangent of arg (measured in radians), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double tan(std::integral auto arg) {
    return xtd::tan(static_cast<double>(arg));
  }

  /* Computes the tangent of arg (measured in radians), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float tanf(std::floating_point auto arg) {
    return xtd::tan(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float tanf(std::integral auto arg) {
    return xtd::tan(static_cast<float>(arg));
  }

  /* Computes the tangent of arg (measured in radians), in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16 tanf(xtd::floating_point auto arg) {
    return xtd::tan(static_cast<float16>(arg));
  }

}  // namespace xtd
