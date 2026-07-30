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
#include "xtd/math/floor.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 0;
constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::floor", "[floor][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::floor(float16)") {
      benchmark<float16, float16, xtd::floor>(device);
    }

    SECTION("float xtd::floor(float)") {
      benchmark<float, float, xtd::floor>(device);
    }

    SECTION("double xtd::floor(double)") {
      benchmark<double, double, xtd::floor>(device);
    }

    SECTION("double xtd::floor(int)") {
      benchmark<double, int, xtd::floor>(device);
    }

    SECTION("float16 xtd::floorf(float16)") {
      benchmark<float16, float16, xtd::floorf>(device);
    }

    SECTION("float xtd::floorf(float)") {
      benchmark<float, float, xtd::floorf>(device);
    }

    SECTION("float xtd::floorf(double)") {
      benchmark<float, double, xtd::floorf>(device);
    }

    SECTION("float xtd::floorf(int)") {
      benchmark<float, int, xtd::floorf>(device);
    }
  }
}
