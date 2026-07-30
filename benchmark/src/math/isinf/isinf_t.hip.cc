/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/isinf.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/benchmark.h"

TEST_CASE("xtd::isinf", "[isinf][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("int xtd::isinf(float16)") {
          benchmark<int, float16, xtd::isinf>(device);
        }

        SECTION("int xtd::isinf(float)") {
          benchmark<int, float, xtd::isinf>(device);
        }

        SECTION("int xtd::isinf(double)") {
          benchmark<int, double, xtd::isinf>(device);
        }

        SECTION("int xtd::isinf(int)") {
          benchmark<int, int, xtd::isinf>(device);
        }
      }
    }
  }
}
