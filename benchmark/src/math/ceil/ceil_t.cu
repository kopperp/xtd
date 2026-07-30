/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/ceil.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"


TEST_CASE("xtd::ceil", "[ceil][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::ceil(float16)") {
          benchmark<float16, float16, xtd::ceil>(device);
        }

        SECTION("float xtd::ceil(float)") {
          benchmark<float, float, xtd::ceil>(device);
        }

        SECTION("double xtd::ceil(double)") {
          benchmark<double, double, xtd::ceil>(device);
        }

        SECTION("double xtd::ceil(int)") {
          benchmark<double, int, xtd::ceil>(device);
        }

        SECTION("float16 xtd::ceilf(float16)") {
          benchmark<float16, float16, xtd::ceilf>(device);
        }

        SECTION("float xtd::ceilf(float)") {
          benchmark<float, float, xtd::ceilf>(device);
        }

        SECTION("float xtd::ceilf(double)") {
          benchmark<float, double, xtd::ceilf>(device);
        }

        SECTION("float xtd::ceilf(int)") {
          benchmark<float, int, xtd::ceilf>(device);
        }
      }
    }
  }
}
