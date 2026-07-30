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
#include "xtd/math/acos.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 1;
constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::acos", "[acos][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::acos(float16)") {
      benchmark<float16, float16, xtd::acos>(device);
    }

    SECTION("float xtd::acos(float)") {
      benchmark<float, float, xtd::acos>(device);
    }

    SECTION("double xtd::acos(double)") {
      benchmark<double, double, xtd::acos>(device);
    }

    SECTION("double xtd::acos(int)") {
      benchmark<double, int, xtd::acos>(device);
    }

    SECTION("float xtd::acosf(float16)") {
      benchmark<float16, float16, xtd::acosf>(device);
    }

    SECTION("float xtd::acosf(float)") {
      benchmark<float, float, xtd::acosf>(device);
    }

    SECTION("float xtd::acosf(double)") {
      benchmark<float, double, xtd::acosf>(device);
    }

    SECTION("float xtd::acosf(int)") {
      benchmark<float, int, xtd::acosf>(device);
    }
  }
}
