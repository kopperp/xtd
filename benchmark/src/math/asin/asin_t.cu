/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/asin.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"


TEST_CASE("xtd::asin", "[asin][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::asin(float16)") {
          benchmark<float16, float16, xtd::asin>(device);
        }

        SECTION("float xtd::asin(float)") {
          benchmark<float, float, xtd::asin>(device);
        }

        SECTION("double xtd::asin(double)") {
          benchmark<double, double, xtd::asin>(device);
        }

        SECTION("double xtd::asin(int)") {
          benchmark<double, int, xtd::asin>(device);
        }

        SECTION("float16 xtd::asinf(float16)") {
          benchmark<float16, float16, xtd::asinf>(device);
        }

        SECTION("float xtd::asinf(float)") {
          benchmark<float, float, xtd::asinf>(device);
        }

        SECTION("float xtd::asinf(double)") {
          benchmark<float, double, xtd::asinf>(device);
        }

        SECTION("float xtd::asinf(int)") {
          benchmark<float, int, xtd::asinf>(device);
        }
      }
    }
  }
}
