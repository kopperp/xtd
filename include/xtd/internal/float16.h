/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <cmath>

#include <cstdint>   // For uint16_t
#include <bit>       // For std::bit_cast
// #include <stdfloat>

#include "xtd/internal/defines.h"

#if defined(XTD_TARGET_CUDA)
#include <cuda_fp16.h>
#elif defined(XTD_TARGET_HIP)
#include <hip/hip_fp16.h>
#elif defined(XTD_TARGET_SYCL)
#include <sycl/sycl.hpp>
#endif

class float16 {
public:
    constexpr float16() noexcept = default;

    // Accept raw bit data
    // XTD_DEVICE_FUNCTION explicit constexpr float16(uint16_t raw_bits) noexcept
    //     : bits_(raw_bits) {}

    /* ========================================================================
     * CONSTRUCTOR FROM FLOAT
     * ===================================================================== */
    XTD_DEVICE_FUNCTION constexpr float16(float f) noexcept
#if defined(XTD_TARGET_CUDA)
        // CUDA device code
        : bits_(__half_as_ushort(__float2half(f))) {}
#elif defined(XTD_TARGET_HIP)
        // HIP/ROCm device code
        // : bits_(__half_as_ushort(__float2half(f))) {}
        : bits_(std::bit_cast<uint16_t>(__float2half_rn(f))) {}
#elif defined(XTD_TARGET_SYCL)
        // SYCL device code
        // : bits_(std::bit_cast<uint16_t>(static_cast<sycl::half>(f))) {}
        : bits_(sycl::half2ushort_rn(sycl::float2half_rn(f))) {}
#else
#if XTD_HAS_STDFLOAT16
        // standard C/C++ code
        : bits_(std::bit_cast<uint16_t>(static_cast<std::float16_t>(f))) {}
#else
    {   // Helper union to for bitwise re-interpret
        union float32 {
            std::uint32_t bits_;
            float float_;
        };

        float32 f32{};
        f32.float_ = f;
        bits_ = float_to_half( f32.bits_ );
    }
#endif
#endif

    /* ========================================================================
     * CONVERSION BACK TO FLOAT
     * ===================================================================== */
    XTD_DEVICE_FUNCTION constexpr operator float() const noexcept {
#if defined(XTD_TARGET_CUDA)
        // CUDA device code
        return __half2float(__ushort_as_half(bits_));
#elif defined(XTD_TARGET_HIP)
        // HIP/ROCm device code
        // return __half2float(__ushort_as_half(bits_));
        return __half2float(std::bit_cast<__half>(bits_));
#elif defined(XTD_TARGET_SYCL)
        // SYCL device code
        return static_cast<float>(sycl::ushort2half_rn(bits_));
#else
#if XTD_HAS_STDFLOAT16
        // standard C/C++ code
        return static_cast<float>(std::bit_cast<std::float16_t>(bits_));
#else
        // Helper union to for bitwise re-interpret
        union float32 {
            std::uint32_t bits_;
            float float_;
        };

        float32 f32{};
        f32.bits_ = half_to_float(bits_);
        return f32.float_;
#endif
#endif
    }

    /* ========================================================================
     * COMPATIBILITY CONSTRUCTOR TO FLOAT16
     * ===================================================================== */
    // NON-EXPLICIT constructor allows implicit conversion from std::float16_t (_Float16)
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    XTD_DEVICE_FUNCTION constexpr float16(__half f) noexcept
        : bits_(__half_as_ushort(f)) {}
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    XTD_DEVICE_FUNCTION constexpr float16(__half f) noexcept
        : bits_(__half_as_ushort(f)) {}
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    XTD_DEVICE_FUNCTION constexpr float16(sycl::half f) noexcept
        : bits_(sycl::half2ushort_rn(f)) {}
#endif
#if XTD_HAS_STDFLOAT16
    // standard C/C++ code
    XTD_DEVICE_FUNCTION constexpr float16(std::float16_t f) noexcept
        : bits_(std::bit_cast<uint16_t>(f)) {}
#endif

    /* ========================================================================
     * COMPATIBILITY CONVERTER FROM FLOAT16
     * ===================================================================== */
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    XTD_DEVICE_FUNCTION constexpr operator __half() const noexcept {
        return __ushort_as_half(bits_);
    }
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    XTD_DEVICE_FUNCTION constexpr operator __half() const noexcept {
        return __ushort_as_half(bits_);
    }
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    XTD_DEVICE_FUNCTION constexpr operator sycl::half() const noexcept {
        return sycl::ushort2half_rn(bits_);
    }
#endif
#if XTD_HAS_STDFLOAT16
    // standard C/C++ code
    XTD_DEVICE_FUNCTION constexpr operator std::float16_t() const noexcept {
        return std::bit_cast<std::float16_t>(bits_);
    }
#endif

private:
    // Needs initialization to be compatible with <C++20
    // uint16_t bits_{0};
    uint16_t bits_;

    /* Copyright <2020> <Feng Wang>
     * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
     * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
     * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
     * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
     */
    XTD_DEVICE_FUNCTION constexpr inline static std::uint16_t float_to_half( const std::uint32_t f ) noexcept
    {
        const std::uint32_t one = ( 0x00000001 );
        const std::uint32_t f_s_mask = ( 0x80000000 );
        const std::uint32_t f_e_mask = ( 0x7f800000 );
        const std::uint32_t f_m_mask = ( 0x007fffff );
        const std::uint32_t f_m_hidden_bit = ( 0x00800000 );
        const std::uint32_t f_m_round_bit = ( 0x00001000 );
        const std::uint32_t f_snan_mask = ( 0x7fc00000 );
        const std::uint32_t f_e_pos = ( 0x00000017 );
        const std::uint32_t h_e_pos = ( 0x0000000a );
        const std::uint32_t h_e_mask = ( 0x00007c00 );
        const std::uint32_t h_snan_mask = ( 0x00007e00 );
        const std::uint32_t h_e_mask_value = ( 0x0000001f );
        const std::uint32_t f_h_s_pos_offset = ( 0x00000010 );
        const std::uint32_t f_h_bias_offset = ( 0x00000070 );
        const std::uint32_t f_h_m_pos_offset = ( 0x0000000d );
        const std::uint32_t h_nan_min = ( 0x00007c01 );
        const std::uint32_t f_h_e_biased_flag = ( 0x0000008f );
        const std::uint32_t f_s = ( f & f_s_mask );
        const std::uint32_t f_e = ( f & f_e_mask );
        const std::uint16_t h_s = ( f_s >> f_h_s_pos_offset );
        const std::uint32_t f_m = ( f & f_m_mask );
        const std::uint16_t f_e_amount = ( f_e >> f_e_pos );
        const std::uint32_t f_e_half_bias = ( f_e_amount - f_h_bias_offset );
        const std::uint32_t f_snan = ( f & f_snan_mask );
        const std::uint32_t f_m_round_mask = ( f_m & f_m_round_bit );
        const std::uint32_t f_m_round_offset = ( f_m_round_mask << one );
        const std::uint32_t f_m_rounded = ( f_m + f_m_round_offset );
        const std::uint32_t f_m_denorm_sa = ( one - f_e_half_bias );
        const std::uint32_t f_m_with_hidden = ( f_m_rounded | f_m_hidden_bit );
        const std::uint32_t f_m_denorm = ( f_m_with_hidden >> f_m_denorm_sa );
        const std::uint32_t h_m_denorm = ( f_m_denorm >> f_h_m_pos_offset );
        const std::uint32_t f_m_rounded_overflow = ( f_m_rounded & f_m_hidden_bit );
        const std::uint32_t m_nan = ( f_m >> f_h_m_pos_offset );
        const std::uint32_t h_em_nan = ( h_e_mask | m_nan );
        const std::uint32_t h_e_norm_overflow_offset = ( f_e_half_bias + 1 );
        const std::uint32_t h_e_norm_overflow = ( h_e_norm_overflow_offset << h_e_pos );
        const std::uint32_t h_e_norm = ( f_e_half_bias << h_e_pos );
        const std::uint32_t h_m_norm = ( f_m_rounded >> f_h_m_pos_offset );
        const std::uint32_t h_em_norm = ( h_e_norm | h_m_norm );
        const std::uint32_t is_h_ndenorm_msb = ( f_h_bias_offset - f_e_amount );
        const std::uint32_t is_f_e_flagged_msb = ( f_h_e_biased_flag - f_e_half_bias );
        const std::uint32_t is_h_denorm_msb = ( ~is_h_ndenorm_msb );
        const std::uint32_t is_f_m_eqz_msb = ( f_m - 1 );
        const std::uint32_t is_h_nan_eqz_msb = ( m_nan - 1 );
        const std::uint32_t is_f_inf_msb = ( is_f_e_flagged_msb & is_f_m_eqz_msb );
        const std::uint32_t is_f_nan_underflow_msb = ( is_f_e_flagged_msb & is_h_nan_eqz_msb );
        const std::uint32_t is_e_overflow_msb = ( h_e_mask_value - f_e_half_bias );
        const std::uint32_t is_h_inf_msb = ( is_e_overflow_msb | is_f_inf_msb );
        const std::uint32_t is_f_nsnan_msb = ( f_snan - f_snan_mask );
        const std::uint32_t is_m_norm_overflow_msb = ( -f_m_rounded_overflow );
        const std::uint32_t is_f_snan_msb = ( ~is_f_nsnan_msb );
        const std::uint32_t h_em_overflow_result = _uint32_sels( is_m_norm_overflow_msb, h_e_norm_overflow, h_em_norm );
        const std::uint32_t h_em_nan_result = _uint32_sels( is_f_e_flagged_msb, h_em_nan, h_em_overflow_result );
        const std::uint32_t h_em_nan_underflow_result = _uint32_sels( is_f_nan_underflow_msb, h_nan_min, h_em_nan_result );
        const std::uint32_t h_em_inf_result = _uint32_sels( is_h_inf_msb, h_e_mask, h_em_nan_underflow_result );
        const std::uint32_t h_em_denorm_result = _uint32_sels( is_h_denorm_msb, h_m_denorm, h_em_inf_result );
        const std::uint32_t h_em_snan_result = _uint32_sels( is_f_snan_msb, h_snan_mask, h_em_denorm_result );
        const std::uint32_t h_result = ( h_s | h_em_snan_result );
        return ( std::uint16_t )( h_result );
    }

    XTD_DEVICE_FUNCTION constexpr inline static std::uint32_t half_to_float( const std::uint16_t h ) noexcept
    {
        const std::uint32_t h_e_mask = ( 0x00007c00 );
        const std::uint32_t h_m_mask = ( 0x000003ff );
        const std::uint32_t h_s_mask = ( 0x00008000 );
        const std::uint32_t h_f_s_pos_offset = ( 0x00000010 );
        const std::uint32_t h_f_e_pos_offset = ( 0x0000000d );
        const std::uint32_t h_f_bias_offset = ( 0x0001c000 );
        const std::uint32_t f_e_mask = ( 0x7f800000 );
        const std::uint32_t f_m_mask = ( 0x007fffff );
        const std::uint32_t h_f_e_denorm_bias = ( 0x0000007e );
        const std::uint32_t h_f_m_denorm_sa_bias = ( 0x00000008 );
        const std::uint32_t f_e_pos = ( 0x00000017 );
        const std::uint32_t h_e_mask_minus_one = ( 0x00007bff );
        const std::uint32_t h_e = ( h & h_e_mask );
        const std::uint32_t h_m = ( h & h_m_mask );
        const std::uint32_t h_s = ( h & h_s_mask );
        const std::uint32_t h_e_f_bias = ( h_e + h_f_bias_offset );
        const std::uint32_t h_m_nlz = h_m ? _uint32_cntlz( h_m ) : 32;
        const std::uint32_t f_s = ( h_s << h_f_s_pos_offset );
        const std::uint32_t f_e = ( h_e_f_bias << h_f_e_pos_offset );
        const std::uint32_t f_m = ( h_m << h_f_e_pos_offset );
        const std::uint32_t f_em = ( f_e | f_m );
        const std::uint32_t h_f_m_sa = ( h_m_nlz - h_f_m_denorm_sa_bias );
        const std::uint32_t f_e_denorm_unpacked = ( h_f_e_denorm_bias - h_f_m_sa );
        const std::uint32_t h_f_m = ( h_m << h_f_m_sa );
        const std::uint32_t f_m_denorm = ( h_f_m & f_m_mask );
        const std::uint32_t f_e_denorm = ( f_e_denorm_unpacked << f_e_pos );
        const std::uint32_t f_em_denorm = ( f_e_denorm | f_m_denorm );
        const std::uint32_t f_em_nan = ( f_e_mask | f_m );
        const std::uint32_t is_e_eqz_msb = ( h_e - 1 );
        const std::uint32_t is_m_nez_msb = ( -h_m );
        const std::uint32_t is_e_flagged_msb = ( h_e_mask_minus_one - h_e );
        const std::uint32_t is_zero_msb = ( is_e_eqz_msb & ~is_m_nez_msb );
        const std::uint32_t is_inf_msb = ( is_e_flagged_msb & ~is_m_nez_msb );
        const std::uint32_t is_denorm_msb = ( is_m_nez_msb & is_e_eqz_msb );
        const std::uint32_t is_nan_msb = ( is_e_flagged_msb & is_m_nez_msb );
        const std::uint32_t is_zero = ( ( ( std::int32_t )is_zero_msb ) >> 31 );
        const std::uint32_t f_zero_result = ( f_em & ~is_zero );
        const std::uint32_t f_denorm_result = _uint32_sels( is_denorm_msb, f_em_denorm, f_zero_result );
        const std::uint32_t f_inf_result = _uint32_sels( is_inf_msb, f_e_mask, f_denorm_result );
        const std::uint32_t f_nan_result = _uint32_sels( is_nan_msb, f_em_nan, f_inf_result );
        const std::uint32_t f_result = ( f_s | f_nan_result );
        return ( f_result );
    }

    XTD_DEVICE_FUNCTION constexpr inline static std::uint32_t _uint32_sels( std::uint32_t test, std::uint32_t a, std::uint32_t b ) noexcept
        {
            const std::uint32_t mask = ( ( ( std::int32_t )test ) >> 31 );
            const std::uint32_t sel_a = ( a & mask );
            const std::uint32_t sel_b = ( b & ~mask );
            const std::uint32_t result = ( sel_a | sel_b );
            return ( result );
        }

    XTD_DEVICE_FUNCTION constexpr inline static std::uint32_t _uint32_cntlz( std::uint32_t x ) noexcept
        {
#ifdef __GNUC__
            if ( x == 0 ) return 0x00000020;
            std::uint32_t is_x_nez_msb = ( -x );
            std::uint32_t nlz = __builtin_clz( x );
            std::uint32_t result = _uint32_sels( is_x_nez_msb, nlz, 0x00000020 );
            return ( result );
#else
            if ( x == 0 ) return 0x00000020;
            const std::uint32_t x0 = ( x >> 1 );
            const std::uint32_t x1 = ( x | x0 );
            const std::uint32_t x2 = ( x1 >> 2 );
            const std::uint32_t x3 = ( x1 | x2 );
            const std::uint32_t x4 = ( x3 >> 4 );
            const std::uint32_t x5 = ( x3 | x4 );
            const std::uint32_t x6 = ( x5 >> 8 );
            const std::uint32_t x7 = ( x5 | x6 );
            const std::uint32_t x8 = ( x7 >> 16 );
            const std::uint32_t x9 = ( x7 | x8 );
            const std::uint32_t xA = ( ~x9 );
            const std::uint32_t xB = ( xA >> 1 );
            const std::uint32_t xC = ( xB & 0x55555555 );
            const std::uint32_t xD = ( xA - xC );
            const std::uint32_t xE = ( xD & 0x33333333 );
            const std::uint32_t xF = ( xD >> 2 );
            const std::uint32_t x10 = ( xF & 0x33333333 );
            const std::uint32_t x11 = ( xE + x10 );
            const std::uint32_t x12 = ( x11 >> 4 );
            const std::uint32_t x13 = ( x11 + x12 );
            const std::uint32_t x14 = ( x13 & 0x0f0f0f0f );
            const std::uint32_t x15 = ( x14 >> 8 );
            const std::uint32_t x16 = ( x14 + x15 );
            const std::uint32_t x17 = ( x16 >> 16 );
            const std::uint32_t x18 = ( x16 + x17 );
            const std::uint32_t x19 = ( x18 & 0x0000003f );
            return ( x19 );
#endif // NOT __GNUC__
        }
};
