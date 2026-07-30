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
#include "xtd/math/trunc.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 0;
constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::trunc", "[trunc][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::trunc(float16)") {
      benchmark<float16, float16, xtd::trunc>(device);
    }

    SECTION("float xtd::trunc(float)") {
      benchmark<float, float, xtd::trunc>(device);
    }

    SECTION("double xtd::trunc(double)") {
      benchmark<double, double, xtd::trunc>(device);
    }

    SECTION("double xtd::trunc(int)") {
      benchmark<double, int, xtd::trunc>(device);
    }

    SECTION("float16 xtd::truncf(float16)") {
      benchmark<float16, float16, xtd::truncf>(device);
    }

    SECTION("float xtd::truncf(float)") {
      benchmark<float, float, xtd::truncf>(device);
    }

    SECTION("float xtd::truncf(double)") {
      benchmark<float, double, xtd::truncf>(device);
    }

    SECTION("float xtd::truncf(int)") {
      benchmark<float, int, xtd::truncf>(device);
    }
  }
}
