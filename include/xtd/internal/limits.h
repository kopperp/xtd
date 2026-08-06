/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#include <limits>

// XTD concepts
#include "xtd/internal/concepts.h"
#include "xtd/internal/float16.h"

// Mirror std::limits
#if !defined(__FLT16_DIG__)
#define __FLT16_HAS_QUIET_NAN__ 1
#define __FLT16_MIN_EXP__ (-13)
#define __FLT16_MAX_10_EXP__ 4
#define __FLT16_DECIMAL_DIG__ 5
#define __FLT16_DENORM_MIN__ 5.96046447753906250000000000000000000e-8f
#define __FLT16_MIN_10_EXP__ (-4)
#define __FLT16_IS_IEC_60559__ 1
#define __FLT16_MIN__ 6.10351562500000000000000000000000000e-5f
#define __FLT16_MAX_EXP__ 16
#define __FLT16_EPSILON__ 9.76562500000000000000000000000000000e-4f
#define __FLT16_NORM_MAX__ 6.55040000000000000000000000000000000e+4f
#define __FLT16_MAX__ 6.55040000000000000000000000000000000e+4f
#define __FLT16_HAS_INFINITY__ 1
#define __FLT16_HAS_DENORM__ 1
#define __FLT16_MANT_DIG__ 11
#define __FLT16_DIG__ 3

// Default values.  Should be overridden in configuration files if necessary.
#ifndef __glibcxx_float16_has_denorm_loss
#  define __glibcxx_float16_has_denorm_loss false
#endif
#ifndef __glibcxx_float16_traps
#  define __glibcxx_float16_traps false
#endif
#ifndef __glibcxx_float16_tinyness_before
#  define __glibcxx_float16_tinyness_before false
#endif

namespace std {
  /// numeric_limits<float16> specialization.
  template<>
    struct numeric_limits<float16>
    {
      static constexpr bool is_specialized = true;
      static constexpr float16 min() noexcept { return static_cast<float>(__FLT16_MIN__); }
      static constexpr float16 max() noexcept { return static_cast<float>(__FLT16_MAX__); }

#if __cplusplus >= 201103L
      static constexpr float16 lowest() noexcept { return static_cast<float>(-__FLT16_MAX__); }
#endif

      static constexpr int digits = __FLT16_MANT_DIG__;
      static constexpr int digits10 = __FLT16_DIG__;
#if __cplusplus >= 201103L
#define __glibcxx_max_digits10(T) \
  (2 + (T) * 643L / 2136)
      static constexpr int max_digits10
	 = __glibcxx_max_digits10 (__FLT16_MANT_DIG__);
#undef __glibcxx_max_digits10
#endif
      static constexpr bool is_signed = true;
      static constexpr bool is_integer = false;
      static constexpr bool is_exact = false;
      static constexpr int radix = __FLT_RADIX__;

      static constexpr float16
      epsilon() noexcept { return static_cast<float>(__FLT16_EPSILON__); }

      static constexpr float16 round_error() noexcept { return std::bit_cast<float16>(static_cast<uint16_t>(0.5)); }

      static constexpr int min_exponent = __FLT16_MIN_EXP__;
      static constexpr int min_exponent10 = __FLT16_MIN_10_EXP__;
      static constexpr int max_exponent = __FLT16_MAX_EXP__;
      static constexpr int max_exponent10 = __FLT16_MAX_10_EXP__;

      static constexpr bool has_infinity = __FLT16_HAS_INFINITY__;
      static constexpr bool has_quiet_NaN = __FLT16_HAS_QUIET_NAN__;
      static constexpr bool has_signaling_NaN = has_quiet_NaN;
      static constexpr float_denorm_style has_denorm = bool(__FLT16_HAS_DENORM__) ? denorm_present : denorm_absent;
      static constexpr bool has_denorm_loss = __glibcxx_float16_has_denorm_loss;

      // static constexpr float16 infinity() noexcept { return std::bit_cast<float16>(static_cast<uint16_t>(__builtin_huge_val())); }
      // static constexpr float16 quiet_NaN() noexcept { return std::bit_cast<float16>(static_cast<uint16_t>(__builtin_nan(""))); }
      // static constexpr float16 signaling_NaN() noexcept { return std::bit_cast<float16>(static_cast<uint16_t>(__builtin_nans(""))); }
      static constexpr float16 infinity() noexcept { return std::bit_cast<float16>(static_cast<uint16_t>(0x7C00)); }
      static constexpr float16 quiet_NaN() noexcept { return std::bit_cast<float16>(static_cast<uint16_t>(0x7E00)); }
      static constexpr float16 signaling_NaN() noexcept { return std::bit_cast<float16>(static_cast<uint16_t>(0x7D00)); }
      static constexpr float16 denorm_min() noexcept { return static_cast<float>(__FLT16_DENORM_MIN__); }

      static constexpr bool is_iec559 = has_infinity && has_quiet_NaN && has_denorm == denorm_present;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = false;

      static constexpr bool traps = __glibcxx_float16_traps;
      static constexpr bool tinyness_before = __glibcxx_float16_tinyness_before;
      static constexpr float_round_style round_style = round_to_nearest;
    };
} // namespace std

#undef __glibcxx_float16_has_denorm_loss
#undef __glibcxx_float16_traps
#undef __glibcxx_float16_tinyness_before

#endif
