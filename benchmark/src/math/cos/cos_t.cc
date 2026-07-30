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
#include "xtd/math/cos.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 1;
constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::cos", "[cos][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::cos(float16)") {
      benchmark<float16, float16, xtd::cos>(device);
    }

    SECTION("float xtd::cos(float)") {
      benchmark<float, float, xtd::cos>(device);
    }

    SECTION("double xtd::cos(double)") {
      benchmark<double, double, xtd::cos>(device);
    }

    SECTION("double xtd::cos(int)") {
      benchmark<double, int, xtd::cos>(device);
    }

    SECTION("float16 xtd::cosf(float16)") {
      benchmark<float16, float16, xtd::cosf>(device);
    }

    SECTION("float xtd::cosf(float)") {
      benchmark<float, float, xtd::cosf>(device);
    }

    SECTION("float xtd::cosf(double)") {
      benchmark<float, double, xtd::cosf>(device);
    }

    SECTION("float xtd::cosf(int)") {
      benchmark<float, int, xtd::cosf>(device);
    }
  }
}
