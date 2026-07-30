/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/fabs.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/benchmark.h"


TEST_CASE("xtd::fabs", "[fabs][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::fabs(float16)") {
          benchmark<float16, float16, xtd::fabs>(device);
        }

        SECTION("float xtd::fabs(float)") {
          benchmark<float, float, xtd::fabs>(device);
        }

        SECTION("double xtd::fabs(double)") {
          benchmark<double, double, xtd::fabs>(device);
        }

        SECTION("double xtd::fabs(int)") {
          benchmark<double, int, xtd::fabs>(device);
        }

        SECTION("float16 xtd::fabsf(float16)") {
          benchmark<float16, float16, xtd::fabsf>(device);
        }

        SECTION("float xtd::fabsf(float)") {
          benchmark<float, float, xtd::fabsf>(device);
        }

        SECTION("float xtd::fabsf(double)") {
          benchmark<float, double, xtd::fabsf>(device);
        }

        SECTION("float xtd::fabsf(int)") {
          benchmark<float, int, xtd::fabsf>(device);
        }
      }
    }
  }
}
