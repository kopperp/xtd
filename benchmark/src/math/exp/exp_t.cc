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
#include "xtd/math/exp.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

// constexpr int ulps_half = 1;
constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::exp", "[exp][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    /* FIXME: half precision is insufficient
    SECTION("float16 xtd::exp(float16)") {
      benchmark<float16, float16, xtd::exp>(device);
    }
    */

    SECTION("float xtd::exp(float)") {
      benchmark<float, float, xtd::exp>(device);
    }

    SECTION("double xtd::exp(double)") {
      benchmark<double, double, xtd::exp>(device);
    }

    SECTION("double xtd::exp(int)") {
      benchmark<double, int, xtd::exp>(device);
    }

    /* FIXME: half precision is insufficient
    SECTION("float16 xtd::expf(float16)") {
      benchmark<float16, float16, xtd::expf>(device);
    }
    */

    SECTION("float xtd::expf(float)") {
      benchmark<float, float, xtd::expf>(device);
    }

    SECTION("float xtd::expf(double)") {
      benchmark<float, double, xtd::expf>(device);
    }

    SECTION("float xtd::expf(int)") {
      benchmark<float, int, xtd::expf>(device);
    }
  }
}
