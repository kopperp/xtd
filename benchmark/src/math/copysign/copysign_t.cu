/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/copysign.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"

// 
TEST_CASE("xtd::copysign", "[copysign][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        /* FIXME: half precision is insufficient
        SECTION("float16 xtd::copysign(float16, float16)") {
          benchmark<float16, float16, xtd::copysign>(device);
        }
        */

        SECTION("float xtd::copysign(float, float)") {
          benchmark<float, float, xtd::copysign>(device);
        }

        SECTION("double xtd::copysign(double, double)") {
          benchmark<double, double, xtd::copysign>(device);
        }

        SECTION("double xtd::copysign(int, int)") {
          benchmark<double, int, xtd::copysign>(device);
        }

        /* FIXME: half precision is insufficient
        SECTION("float16 xtd::copysignf(float16, float16)") {
          benchmark<float16, float16, xtd::copysignf>(device);
        }
        */

        SECTION("float xtd::copysignf(float, float)") {
          benchmark<float, float, xtd::copysignf>(device);
        }

        SECTION("float xtd::copysignf(double, double)") {
          benchmark<float, double, xtd::copysignf>(device);
        }

        SECTION("float xtd::copysignf(int, int)") {
          benchmark<float, int, xtd::copysignf>(device);
        }
      }
    }
  }
}
