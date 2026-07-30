/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/acosh.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/benchmark.h"


TEST_CASE("xtd::acosh", "[acosh][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::acosh(float16)") {
          benchmark<float16, float16, xtd::acosh>(device);
        }

        SECTION("float xtd::acosh(float)") {
          benchmark<float, float, xtd::acosh>(device);
        }

        SECTION("double xtd::acosh(double)") {
          benchmark<double, double, xtd::acosh>(device);
        }

        SECTION("double xtd::acosh(int)") {
          benchmark<double, int, xtd::acosh>(device);
        }

        SECTION("float xtd::acoshf(float)") {
          benchmark<float, float, xtd::acoshf>(device);
        }

        SECTION("float xtd::acoshf(double)") {
          benchmark<float, double, xtd::acoshf>(device);
        }

        SECTION("float xtd::acoshf(int)") {
          benchmark<float, int, xtd::acoshf>(device);
        }
      }
    }
  }
}
