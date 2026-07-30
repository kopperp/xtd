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
#include "xtd/math/ceil.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 0;
constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::ceil", "[ceil][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::ceil(float16)") {
      benchmark<float16, float16, xtd::ceil>(device);
    }

    SECTION("float xtd::ceil(float)") {
      benchmark<float, float, xtd::ceil>(device);
    }

    SECTION("double xtd::ceil(double)") {
      benchmark<double, double, xtd::ceil>(device);
    }

    SECTION("double xtd::ceil(int)") {
      benchmark<double, int, xtd::ceil>(device);
    }

    SECTION("float16 xtd::ceilf(float16)") {
      benchmark<float16, float16, xtd::ceilf>(device);
    }

    SECTION("float xtd::ceilf(float)") {
      benchmark<float, float, xtd::ceilf>(device);
    }

    SECTION("float xtd::ceilf(double)") {
      benchmark<float, double, xtd::ceilf>(device);
    }

    SECTION("float xtd::ceilf(int)") {
      benchmark<float, int, xtd::ceilf>(device);
    }
  }
}
