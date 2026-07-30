/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/trunc.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"


TEST_CASE("xtd::trunc", "[trunc][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::trunc(float16)") {
          benchmark<float16, float16, xtd::trunc>(device);
        }

        SECTION("float xtd::trunc(float)") {
          benchmark<float, float, xtd::trunc>(device);
        }

        SECTION("double xtd::trunc(double)") {
          benchmark<double, double, xtd::trunc>(device);
        }

        SECTION("double xtd::trunc(int)") {
          benchmark<double, int, xtd::trunc>(device);
        }

        SECTION("float16 xtd::truncf(float16)") {
          benchmark<float16, float16, xtd::truncf>(device);
        }

        SECTION("float xtd::truncf(float)") {
          benchmark<float, float, xtd::truncf>(device);
        }

        SECTION("float xtd::truncf(double)") {
          benchmark<float, double, xtd::truncf>(device);
        }

        SECTION("float xtd::truncf(int)") {
          benchmark<float, int, xtd::truncf>(device);
        }
      }
    }
  }
}
