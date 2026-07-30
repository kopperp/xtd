/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/asinh.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"


TEST_CASE("xtd::asinh", "[asinh][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::asinh(float16)") {
          benchmark<float16, float16, xtd::asinh>(device);
        }

        SECTION("float xtd::asinh(float)") {
          benchmark<float, float, xtd::asinh>(device);
        }

        SECTION("double xtd::asinh(double)") {
          benchmark<double, double, xtd::asinh>(device);
        }

        SECTION("double xtd::asinh(int)") {
          benchmark<double, int, xtd::asinh>(device);
        }

        SECTION("float16 xtd::asinhf(float16)") {
          benchmark<float16, float16, xtd::asinhf>(device);
        }

        SECTION("float xtd::asinhf(float)") {
          benchmark<float, float, xtd::asinhf>(device);
        }

        SECTION("float xtd::asinhf(double)") {
          benchmark<float, double, xtd::asinhf>(device);
        }

        SECTION("float xtd::asinhf(int)") {
          benchmark<float, int, xtd::asinhf>(device);
        }
      }
    }
  }
}
