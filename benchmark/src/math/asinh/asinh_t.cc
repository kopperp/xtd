/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <vector>

// Catch2 headers
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/asinh.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 2;
constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::asinh", "[asinh][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::asinh(float16)") {
      benchmark<float16, float16, xtd::asinh>(device);
    }

    SECTION("float xtd::asinh(float)") {
      benchmark<float, float, xtd::asinh>(device);
    }

    SECTION("double xtd::asinh(double)") {
      benchmark<double, double, xtd::asinh>(device);
    }

    SECTION("double xtd::asinh(int)") {
      benchmark<double, int, xtd::asinh>(device);
    }

    SECTION("float16 xtd::asinhf(float16)") {
      benchmark<float16, float16, xtd::asinhf>(device);
    }

    SECTION("float xtd::asinhf(float)") {
      benchmark<float, float, xtd::asinhf>(device);
    }

    SECTION("float xtd::asinhf(double)") {
      benchmark<float, double, xtd::asinhf>(device);
    }

    SECTION("float xtd::asinhf(int)") {
      benchmark<float, int, xtd::asinhf>(device);
    }
  }
}
