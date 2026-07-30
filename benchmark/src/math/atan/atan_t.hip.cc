/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/atan.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/benchmark.h"


TEST_CASE("xtd::atan", "[atan][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::atan(float16)") {
          benchmark<float16, float16, xtd::atan>(device);
        }

        SECTION("float xtd::atan(float)") {
          benchmark<float, float, xtd::atan>(device);
        }

        SECTION("double xtd::atan(double)") {
          benchmark<double, double, xtd::atan>(device);
        }

        SECTION("double xtd::atan(int)") {
          benchmark<double, int, xtd::atan>(device);
        }

        SECTION("float16 xtd::atanf(float16)") {
          benchmark<float16, float16, xtd::atanf>(device);
        }

        SECTION("float xtd::atanf(float)") {
          benchmark<float, float, xtd::atanf>(device);
        }

        SECTION("float xtd::atanf(double)") {
          benchmark<float, double, xtd::atanf>(device);
        }

        SECTION("float xtd::atanf(int)") {
          benchmark<float, int, xtd::atanf>(device);
        }
      }
    }
  }
}
