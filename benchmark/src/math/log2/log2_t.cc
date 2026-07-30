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
#include "xtd/math/log2.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 1;
constexpr int ulps_single = 1;
constexpr int ulps_double = 2;

TEST_CASE("xtd::log2", "[log2][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::log2(float16)") {
      benchmark<float16, float16, xtd::log2>(device);
    }

    SECTION("float xtd::log2(float)") {
      benchmark<float, float, xtd::log2>(device);
    }

    SECTION("double xtd::log2(double)") {
      benchmark<double, double, xtd::log2>(device);
    }

    SECTION("double xtd::log2(int)") {
      benchmark<double, int, xtd::log2>(device);
    }

    SECTION("float16 xtd::log2f(float16)") {
      benchmark<float16, float16, xtd::log2f>(device);
    }

    SECTION("float xtd::log2f(float)") {
      benchmark<float, float, xtd::log2f>(device);
    }

    SECTION("float xtd::log2f(double)") {
      benchmark<float, double, xtd::log2f>(device);
    }

    SECTION("float xtd::log2f(int)") {
      benchmark<float, int, xtd::log2f>(device);
    }
  }
}
