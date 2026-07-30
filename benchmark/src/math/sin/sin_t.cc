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
#include "xtd/math/sin.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 1;
constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::sin", "[sin][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::sin(float16)") {
      benchmark<float16, float16, xtd::sin>(device);
    }

    SECTION("float xtd::sin(float)") {
      benchmark<float, float, xtd::sin>(device);
    }

    SECTION("double xtd::sin(double)") {
      benchmark<double, double, xtd::sin>(device);
    }

    SECTION("double xtd::sin(int)") {
      benchmark<double, int, xtd::sin>(device);
    }

    SECTION("float16 xtd::sinf(float16)") {
      benchmark<float16, float16, xtd::sinf>(device);
    }

    SECTION("float xtd::sinf(float)") {
      benchmark<float, float, xtd::sinf>(device);
    }

    SECTION("float xtd::sinf(double)") {
      benchmark<float, double, xtd::sinf>(device);
    }

    SECTION("float xtd::sinf(int)") {
      benchmark<float, int, xtd::sinf>(device);
    }
  }
}
