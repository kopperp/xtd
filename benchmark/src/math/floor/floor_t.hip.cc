/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/floor.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/benchmark.h"


TEST_CASE("xtd::floor", "[floor][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::floor(float16)") {
          benchmark<float16, float16, xtd::floor>(device);
        }

        SECTION("float xtd::floor(float)") {
          benchmark<float, float, xtd::floor>(device);
        }

        SECTION("double xtd::floor(double)") {
          benchmark<double, double, xtd::floor>(device);
        }

        SECTION("double xtd::floor(int)") {
          benchmark<double, int, xtd::floor>(device);
        }

        SECTION("float16 xtd::floorf(float16)") {
          benchmark<float16, float16, xtd::floorf>(device);
        }

        SECTION("float xtd::floorf(float)") {
          benchmark<float, float, xtd::floorf>(device);
        }

        SECTION("float xtd::floorf(double)") {
          benchmark<float, double, xtd::floorf>(device);
        }

        SECTION("float xtd::floorf(int)") {
          benchmark<float, int, xtd::floorf>(device);
        }
      }
    }
  }
}
