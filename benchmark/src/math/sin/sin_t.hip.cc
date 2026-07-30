/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/sin.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/benchmark.h"


TEST_CASE("xtd::sin", "[sin][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::sin(float16)") {
          benchmark<float16, float16, xtd::sin>(device);
        }

        SECTION("float xtd::sin(float)") {
          benchmark<float, float, xtd::sin>(device);
        }

        SECTION("double xtd::sin(double)") {
          benchmark<double, double, xtd::sin>(device);
        }

        SECTION("double xtd::sin(int)") {
          benchmark<double, int, xtd::sin>(device);
        }

        SECTION("float16 xtd::sinf(float16)") {
          benchmark<float16, float16, xtd::sinf>(device);
        }

        SECTION("float xtd::sinf(float)") {
          benchmark<float, float, xtd::sinf>(device);
        }

        SECTION("float xtd::sinf(double)") {
          benchmark<float, double, xtd::sinf>(device);
        }

        SECTION("float xtd::sinf(int)") {
          benchmark<float, int, xtd::sinf>(device);
        }
      }
    }
  }
}
