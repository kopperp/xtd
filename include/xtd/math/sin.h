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
#include "xtd/math/sin.inl"
#endif

namespace xtd {

  /* Computes the sine of arg (measured in radians), in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16 sin(float16 arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hsin(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    // AMD uses a suboptimal range reduction, use full float evaluation
    return __float2half(::sinf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sin(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
    // XOR the sign bit (bit 15) to evaluate -arg
    std::uint16_t arg_bits = std::bit_cast<std::uint16_t>(arg);
    std::uint16_t abs_bits = arg_bits & 0x7FFF;
    std::uint16_t sign_bit = arg_bits & 0x8000;

    return std::bit_cast<float16>(static_cast<std::uint16_t>(sinf16_lut[abs_bits] ^ sign_bit));
/*
#elif XTD_HAS_STDFLOAT16
#if defined(__clang__) && __has_builtin(__builtin_sinf16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::sin(static_cast<float>(arg));
    } return __builtin_sinf16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_sinf)
    if (std::is_constant_evaluated()) {
      return std::sin(static_cast<float>(arg));
    } return __builtin_sinf(static_cast<float>(arg));
#else
    return std::sin(static_cast<float>(arg));
#endif
#else
    return std::sin(static_cast<float_t>(arg));
*/
#endif
  }


  /* Computes the sine of arg (measured in radians), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sin(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sinf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sinf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sin(arg);
#else
    // standard C/C++ code
    return ::sinf(arg);
#endif
  }

  /* Computes the sine of arg (measured in radians), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sin(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sin(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sin(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sin(arg);
#else
    // standard C/C++ code
    return ::sin(arg);
#endif
  }

  /* Computes the sine of arg (measured in radians), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sin(std::integral auto arg) {
    return xtd::sin(static_cast<double>(arg));
  }

  /* Computes the sine of arg (measured in radians), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sinf(std::floating_point auto arg) {
    return xtd::sin(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float sinf(std::integral auto arg) {
    return xtd::sin(static_cast<float>(arg));
  }

  /* Computes the sine of arg (measured in radians), in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16 sinf(xtd::floating_point auto arg) {
    return xtd::sin(static_cast<float16>(arg));
  }

}  // namespace xtd
