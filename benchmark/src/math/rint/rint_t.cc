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
#include "xtd/math/rint.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 0;
constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::rint", "[rint][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::rint(float16)") {
      benchmark<float16, float16, xtd::rint>(device);
    }

    SECTION("float xtd::rint(float)") {
      benchmark<float, float, xtd::rint>(device);
    }

    SECTION("double xtd::rint(double)") {
      benchmark<double, double, xtd::rint>(device);
    }

    SECTION("double xtd::rint(int)") {
      benchmark<double, int, xtd::rint>(device);
    }

    SECTION("float16 xtd::rintf(float16)") {
      benchmark<float16, float16, xtd::rintf>(device);
    }

    SECTION("float xtd::rintf(float)") {
      benchmark<float, float, xtd::rintf>(device);
    }

    SECTION("float xtd::rintf(double)") {
      benchmark<float, double, xtd::rintf>(device);
    }

    SECTION("float xtd::rintf(int)") {
      benchmark<float, int, xtd::rintf>(device);
    }
  }
}
