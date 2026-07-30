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
#include "xtd/math/atan2.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

// constexpr int ulps_half = 1;
constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::atan2", "[atan2][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    /*
    SECTION("float16 xtd::atan2(float16, float16)") {
      benchmark<float16, float16, xtd::atan2>(device);
    }
    */

    SECTION("float xtd::atan2(float, float)") {
      benchmark<float, float, xtd::atan2>(device);
    }

    SECTION("double xtd::atan2(double, double)") {
      benchmark<double, double, xtd::atan2>(device);
    }

    SECTION("double xtd::atan2(int, int)") {
      benchmark<double, int, xtd::atan2>(device);
    }

    /*
    SECTION("float16 xtd::atan2f(float16, float16)") {
      benchmark<float16, float16, xtd::atan2f>(device);
    }
    */

    SECTION("float xtd::atan2f(float, float)") {
      benchmark<float, float, xtd::atan2f>(device);
    }

    SECTION("float xtd::atan2f(double, double)") {
      benchmark<float, double, xtd::atan2f>(device);
    }

    SECTION("float xtd::atan2f(int, int)") {
      benchmark<float, int, xtd::atan2f>(device);
    }
  }
}
