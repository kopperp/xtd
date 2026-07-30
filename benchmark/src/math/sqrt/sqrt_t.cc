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
#include "xtd/math/sqrt.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 0;
constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::sqrt", "[sqrt][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::sqrt(float16)") {
      benchmark<float16, float16, xtd::sqrt>(device);
    }

    SECTION("float xtd::sqrt(float)") {
      benchmark<float, float, xtd::sqrt>(device);
    }

    SECTION("double xtd::sqrt(double)") {
      benchmark<double, double, xtd::sqrt>(device);
    }

    SECTION("double xtd::sqrt(int)") {
      benchmark<double, int, xtd::sqrt>(device);
    }

    SECTION("float16 xtd::sqrtf(float16)") {
      benchmark<float16, float16, xtd::sqrtf>(device);
    }

    SECTION("float xtd::sqrtf(float)") {
      benchmark<float, float, xtd::sqrtf>(device);
    }

    SECTION("float xtd::sqrtf(double)") {
      benchmark<float, double, xtd::sqrtf>(device);
    }

    SECTION("float xtd::sqrtf(int)") {
      benchmark<float, int, xtd::sqrtf>(device);
    }
  }
}
