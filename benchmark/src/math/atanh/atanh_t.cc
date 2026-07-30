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
#include "xtd/math/atanh.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 2;
constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::atanh", "[atanh][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::atanh(float16)") {
      benchmark<float16, float16, xtd::atanh>(device);
    }

    SECTION("float xtd::atanh(float)") {
      benchmark<float, float, xtd::atanh>(device);
    }

    SECTION("double xtd::atanh(double)") {
      benchmark<double, double, xtd::atanh>(device);
    }

    SECTION("double xtd::atanh(int)") {
      benchmark<double, int, xtd::atanh>(device);
    }

    SECTION("float16 xtd::atanhf(float16)") {
      benchmark<float16, float16, xtd::atanhf>(device);
    }

    SECTION("float xtd::atanhf(float)") {
      benchmark<float, float, xtd::atanhf>(device);
    }

    SECTION("float xtd::atanhf(double)") {
      benchmark<float, double, xtd::atanhf>(device);
    }

    SECTION("float xtd::atanhf(int)") {
      benchmark<float, int, xtd::atanhf>(device);
    }
  }
}
