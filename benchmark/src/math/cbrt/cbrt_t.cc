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
#include "xtd/math/cbrt.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 1;
constexpr int ulps_single = 1;
constexpr int ulps_double = 4;

TEST_CASE("xtd::cbrt", "[cbrt][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::cbrt(float16)") {
      benchmark<float16, float16, xtd::cbrt>(device);
    }

    SECTION("float xtd::cbrt(float)") {
      benchmark<float, float, xtd::cbrt>(device);
    }

    SECTION("double xtd::cbrt(double)") {
      benchmark<double, double, xtd::cbrt>(device);
    }

    SECTION("double xtd::cbrt(int)") {
      benchmark<double, int, xtd::cbrt>(device);
    }

    SECTION("float16 xtd::cbrtf(float16)") {
      benchmark<float16, float16, xtd::cbrtf>(device);
    }

    SECTION("float xtd::cbrtf(float)") {
      benchmark<float, float, xtd::cbrtf>(device);
    }

    SECTION("float xtd::cbrtf(double)") {
      benchmark<float, double, xtd::cbrtf>(device);
    }

    SECTION("float xtd::cbrtf(int)") {
      benchmark<float, int, xtd::cbrtf>(device);
    }
  }
}
