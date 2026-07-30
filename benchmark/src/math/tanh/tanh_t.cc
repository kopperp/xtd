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
#include "xtd/math/tanh.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/benchmark.h"

constexpr int ulps_half = 2;
constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::tanh", "[tanh][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float16 xtd::tanh(float16)") {
      benchmark<float16, float16, xtd::tanh>(device);
    }

    SECTION("float xtd::tanh(float)") {
      benchmark<float, float, xtd::tanh>(device);
    }

    SECTION("double xtd::tanh(double)") {
      benchmark<double, double, xtd::tanh>(device);
    }

    SECTION("double xtd::tanh(int)") {
      benchmark<double, int, xtd::tanh>(device);
    }

    SECTION("float16 xtd::tanhf(float16)") {
      benchmark<float16, float16, xtd::tanhf>(device);
    }

    SECTION("float xtd::tanhf(float)") {
      benchmark<float, float, xtd::tanhf>(device);
    }

    SECTION("float xtd::tanhf(double)") {
      benchmark<float, double, xtd::tanhf>(device);
    }

    SECTION("float xtd::tanhf(int)") {
      benchmark<float, int, xtd::tanhf>(device);
    }
  }
}
