/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/cosh.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"

// 
TEST_CASE("xtd::cosh", "[cosh][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        /* FIXME: half precision is insufficient
        SECTION("float16 xtd::cosh(float16)") {
          benchmark<float16, float16, xtd::cosh>(device);
        }
        */

        SECTION("float xtd::cosh(float)") {
          benchmark<float, float, xtd::cosh>(device);
        }

        SECTION("double xtd::cosh(double)") {
          benchmark<double, double, xtd::cosh>(device);
        }

        SECTION("double xtd::cosh(int)") {
          benchmark<double, int, xtd::cosh>(device);
        }

        /* FIXME: half precision is insufficient
        SECTION("float16 xtd::coshf(float16)") {
          benchmark<float16, float16, xtd::coshf>(device);
        }
        */

        SECTION("float xtd::coshf(float)") {
          benchmark<float, float, xtd::coshf>(device);
        }

        SECTION("float xtd::coshf(double)") {
          benchmark<float, double, xtd::coshf>(device);
        }

        SECTION("float xtd::coshf(int)") {
          benchmark<float, int, xtd::coshf>(device);
        }
      }
    }
  }
}
