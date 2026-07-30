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
#include "xtd/math/cosh.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

// constexpr int ulps_half = 2;
constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::cosh", "[cosh][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    /* FIXME: half precision is insufficient
    SECTION("float16 xtd::cosh(float16)") {
      benchmark<float16, float16, xtd::cosh>(device);
    }
    */

    SECTION("float xtd::cosh(float)") {
      benchmark<float, float, xtd::cosh>(device);
    }

    SECTION("double xtd::cosh(double)") {
      benchmark<double, double, xtd::cosh>(device);
    }

    SECTION("double xtd::cosh(int)") {
      benchmark<double, int, xtd::cosh>(device);
    }

    /*
    SECTION("float16 xtd::coshf(float16)") {
      benchmark<float16, float16, xtd::coshf>(device);
    }
    */

    SECTION("float xtd::coshf(float)") {
      benchmark<float, float, xtd::coshf>(device);
    }

    SECTION("float xtd::coshf(double)") {
      benchmark<float, double, xtd::coshf>(device);
    }

    SECTION("float xtd::coshf(int)") {
      benchmark<float, int, xtd::coshf>(device);
    }
  }
}
