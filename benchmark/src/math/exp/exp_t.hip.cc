/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/exp.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/benchmark.h"

// 
TEST_CASE("xtd::exp", "[exp][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        /* FIXME: half precision is insufficient
        SECTION("float16 xtd::exp(float16)") {
          benchmark<float16, float16, xtd::exp>(device);
        }
        */

        SECTION("float xtd::exp(float)") {
          benchmark<float, float, xtd::exp>(device);
        }

        SECTION("double xtd::exp(double)") {
          benchmark<double, double, xtd::exp>(device);
        }

        SECTION("double xtd::exp(int)") {
          benchmark<double, int, xtd::exp>(device);
        }

        /* FIXME: half precision is insufficient
        SECTION("float16 xtd::expf(float16)") {
          benchmark<float16, float16, xtd::expf>(device);
        }
        */

        SECTION("float xtd::expf(float)") {
          benchmark<float, float, xtd::expf>(device);
        }

        SECTION("float xtd::expf(double)") {
          benchmark<float, double, xtd::expf>(device);
        }

        SECTION("float xtd::expf(int)") {
          benchmark<float, int, xtd::expf>(device);
        }
      }
    }
  }
}
