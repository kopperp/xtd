/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/cos.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/benchmark.h"


TEST_CASE("xtd::cos", "[cos][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::cos(float16)") {
          benchmark<float16, float16, xtd::cos>(device);
        }

        SECTION("float xtd::cos(float)") {
          benchmark<float, float, xtd::cos>(device);
        }

        SECTION("double xtd::cos(double)") {
          benchmark<double, double, xtd::cos>(device);
        }

        SECTION("double xtd::cos(int)") {
          benchmark<double, int, xtd::cos>(device);
        }

        SECTION("float16 xtd::cosf(float16)") {
          benchmark<float16, float16, xtd::cosf>(device);
        }

        SECTION("float xtd::cosf(float)") {
          benchmark<float, float, xtd::cosf>(device);
        }

        SECTION("float xtd::cosf(double)") {
          benchmark<float, double, xtd::cosf>(device);
        }

        SECTION("float xtd::cosf(int)") {
          benchmark<float, int, xtd::cosf>(device);
        }
      }
    }
  }
}
