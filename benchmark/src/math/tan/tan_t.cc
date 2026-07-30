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
#include "xtd/math/tan.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 1;
constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::tan", "[tan][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::tan(float16)") {
      benchmark<float16, float16, xtd::tan>(device);
    }

    SECTION("float xtd::tan(float)") {
      benchmark<float, float, xtd::tan>(device);
    }

    SECTION("double xtd::tan(double)") {
      benchmark<double, double, xtd::tan>(device);
    }

    SECTION("double xtd::tan(int)") {
      benchmark<double, int, xtd::tan>(device);
    }

    SECTION("float16 xtd::tanf(float16)") {
      benchmark<float16, float16, xtd::tanf>(device);
    }

    SECTION("float xtd::tanf(float)") {
      benchmark<float, float, xtd::tanf>(device);
    }

    SECTION("float xtd::tanf(double)") {
      benchmark<float, double, xtd::tanf>(device);
    }

    SECTION("float xtd::tanf(int)") {
      benchmark<float, int, xtd::tanf>(device);
    }
  }
}
