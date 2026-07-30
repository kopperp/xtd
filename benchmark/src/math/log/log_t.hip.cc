/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/log.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/benchmark.h"


TEST_CASE("xtd::log", "[log][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::log(float16)") {
          benchmark<float16, float16, xtd::log>(device);
        }

        SECTION("float xtd::log(float)") {
          benchmark<float, float, xtd::log>(device);
        }

        SECTION("double xtd::log(double)") {
          benchmark<double, double, xtd::log>(device);
        }

        SECTION("double xtd::log(int)") {
          benchmark<double, int, xtd::log>(device);
        }

        SECTION("float16 xtd::logf(float16)") {
          benchmark<float16, float16, xtd::logf>(device);
        }

        SECTION("float xtd::logf(float)") {
          benchmark<float, float, xtd::logf>(device);
        }

        SECTION("float xtd::logf(double)") {
          benchmark<float, double, xtd::logf>(device);
        }

        SECTION("float xtd::logf(int)") {
          benchmark<float, int, xtd::logf>(device);
        }
      }
    }
  }
}
