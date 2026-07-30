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
#include "xtd/math/log10.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 2;
constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::log10", "[log10][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::log10(float16)") {
      benchmark<float16, float16, xtd::log10>(device);
    }

    SECTION("float xtd::log10(float)") {
      benchmark<float, float, xtd::log10>(device);
    }

    SECTION("double xtd::log10(double)") {
      benchmark<double, double, xtd::log10>(device);
    }

    SECTION("double xtd::log10(int)") {
      benchmark<double, int, xtd::log10>(device);
    }

    SECTION("float16 xtd::log10f(float16)") {
      benchmark<float16, float16, xtd::log10f>(device);
    }

    SECTION("float xtd::log10f(float)") {
      benchmark<float, float, xtd::log10f>(device);
    }

    SECTION("float xtd::log10f(double)") {
      benchmark<float, double, xtd::log10f>(device);
    }

    SECTION("float xtd::log10f(int)") {
      benchmark<float, int, xtd::log10f>(device);
    }
  }
}
