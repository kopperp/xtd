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
#include "xtd/math/fabs.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 0;
constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::fabs", "[fabs][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::fabs(float16)") {
      benchmark<float16, float16, xtd::fabs>(device);
    }

    SECTION("float xtd::fabs(float)") {
      benchmark<float, float, xtd::fabs>(device);
    }

    SECTION("double xtd::fabs(double)") {
      benchmark<double, double, xtd::fabs>(device);
    }

    SECTION("double xtd::fabs(int)") {
      benchmark<double, int, xtd::fabs>(device);
    }

    SECTION("float16 xtd::fabsf(float16)") {
      benchmark<float16, float16, xtd::fabsf>(device);
    }

    SECTION("float xtd::fabsf(float)") {
      benchmark<float, float, xtd::fabsf>(device);
    }

    SECTION("float xtd::fabsf(double)") {
      benchmark<float, double, xtd::fabsf>(device);
    }

    SECTION("float xtd::fabsf(int)") {
      benchmark<float, int, xtd::fabsf>(device);
    }
  }
}
