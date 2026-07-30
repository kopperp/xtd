/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/cbrt.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"


TEST_CASE("xtd::cbrt", "[cbrt][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::cbrt(float16)") {
          benchmark<float16, float16, xtd::cbrt>(device);
        }

        SECTION("float xtd::cbrt(float)") {
          benchmark<float, float, xtd::cbrt>(device);
        }

        SECTION("double xtd::cbrt(double)") {
          benchmark<double, double, xtd::cbrt>(device);
        }

        SECTION("double xtd::cbrt(int)") {
          benchmark<double, int, xtd::cbrt>(device);
        }

        SECTION("float16 xtd::cbrtf(float16)") {
          benchmark<float16, float16, xtd::cbrtf>(device);
        }

        SECTION("float xtd::cbrtf(float)") {
          benchmark<float, float, xtd::cbrtf>(device);
        }

        SECTION("float xtd::cbrtf(double)") {
          benchmark<float, double, xtd::cbrtf>(device);
        }

        SECTION("float xtd::cbrtf(int)") {
          benchmark<float, int, xtd::cbrtf>(device);
        }
      }
    }
  }
}
