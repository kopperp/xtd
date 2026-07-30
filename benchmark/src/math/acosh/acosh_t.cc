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
#include "xtd/math/acosh.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 2;
constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::acosh", "[acosh][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::acosh(float16)") {
      benchmark<float16, float16, xtd::acosh>(device);
    }

    SECTION("float xtd::acosh(float)") {
      benchmark<float, float, xtd::acosh>(device);
    }

    SECTION("double xtd::acosh(double)") {
      benchmark<double, double, xtd::acosh>(device);
    }

    SECTION("double xtd::acosh(int)") {
      benchmark<double, int, xtd::acosh>(device);
    }

    SECTION("float xtd::acoshf(float)") {
      benchmark<float, float, xtd::acoshf>(device);
    }

    SECTION("float xtd::acoshf(double)") {
      benchmark<float, double, xtd::acoshf>(device);
    }

    SECTION("float xtd::acoshf(int)") {
      benchmark<float, int, xtd::acoshf>(device);
    }
  }
}
