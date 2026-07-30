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
#include "common/hip/platform.h"
#include "common/hip/benchmark.h"


TEST_CASE("xtd::asinh", "[asinh][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
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
