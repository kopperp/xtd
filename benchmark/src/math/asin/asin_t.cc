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
#include "xtd/math/asin.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 1;
constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::asin", "[asin][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::asin(float16)") {
      benchmark<float16, float16, xtd::asin>(device);
    }

    SECTION("float xtd::asin(float)") {
      benchmark<float, float, xtd::asin>(device);
    }

    SECTION("double xtd::asin(double)") {
      benchmark<double, double, xtd::asin>(device);
    }

    SECTION("double xtd::asin(int)") {
      benchmark<double, int, xtd::asin>(device);
    }

    SECTION("float16 xtd::asinf(float16)") {
      benchmark<float16, float16, xtd::asinf>(device);
    }

    SECTION("float xtd::asinf(float)") {
      benchmark<float, float, xtd::asinf>(device);
    }

    SECTION("float xtd::asinf(double)") {
      benchmark<float, double, xtd::asinf>(device);
    }

    SECTION("float xtd::asinf(int)") {
      benchmark<float, int, xtd::asinf>(device);
    }
  }
}
