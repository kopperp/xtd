/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/sqrt.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"


TEST_CASE("xtd::sqrt", "[sqrt][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::sqrt(float16)") {
          benchmark<float16, float16, xtd::sqrt>(device);
        }

        SECTION("float xtd::sqrt(float)") {
          benchmark<float, float, xtd::sqrt>(device);
        }

        SECTION("double xtd::sqrt(double)") {
          benchmark<double, double, xtd::sqrt>(device);
        }

        SECTION("double xtd::sqrt(int)") {
          benchmark<double, int, xtd::sqrt>(device);
        }

        SECTION("float16 xtd::sqrtf(float16)") {
          benchmark<float16, float16, xtd::sqrtf>(device);
        }

        SECTION("float xtd::sqrtf(float)") {
          benchmark<float, float, xtd::sqrtf>(device);
        }

        SECTION("float xtd::sqrtf(double)") {
          benchmark<float, double, xtd::sqrtf>(device);
        }

        SECTION("float xtd::sqrtf(int)") {
          benchmark<float, int, xtd::sqrtf>(device);
        }
      }
    }
  }
}
