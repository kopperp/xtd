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
#include "xtd/math/log.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 1;
constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::log", "[log][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::log(float16)") {
      benchmark<float16, float16, xtd::log>(device);
    }

    SECTION("float xtd::log(float)") {
      benchmark<float, float, xtd::log>(device);
    }

    SECTION("double xtd::log(double)") {
      benchmark<double, double, xtd::log>(device);
    }

    SECTION("double xtd::log(int)") {
      benchmark<double, int, xtd::log>(device);
    }

    SECTION("float16 xtd::logf(float16)") {
      benchmark<float16, float16, xtd::logf>(device);
    }

    SECTION("float xtd::logf(float)") {
      benchmark<float, float, xtd::logf>(device);
    }

    SECTION("float xtd::logf(double)") {
      benchmark<float, double, xtd::logf>(device);
    }

    SECTION("float xtd::logf(int)") {
      benchmark<float, int, xtd::logf>(device);
    }
  }
}
