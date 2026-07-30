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
#include "xtd/math/atan.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 1;
constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::atan", "[atan][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::atan(float16)") {
      benchmark<float16, float16, xtd::atan>(device);
    }

    SECTION("float xtd::atan(float)") {
      benchmark<float, float, xtd::atan>(device);
    }

    SECTION("double xtd::atan(double)") {
      benchmark<double, double, xtd::atan>(device);
    }

    SECTION("double xtd::atan(int)") {
      benchmark<double, int, xtd::atan>(device);
    }

    SECTION("float16 xtd::atanf(float16)") {
      benchmark<float16, float16, xtd::atanf>(device);
    }

    SECTION("float xtd::atanf(float)") {
      benchmark<float, float, xtd::atanf>(device);
    }

    SECTION("float xtd::atanf(double)") {
      benchmark<float, double, xtd::atanf>(device);
    }

    SECTION("float xtd::atanf(int)") {
      benchmark<float, int, xtd::atanf>(device);
    }
  }
}
