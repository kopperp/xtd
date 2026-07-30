/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/rint.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"


TEST_CASE("xtd::rint", "[rint][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::rint(float16)") {
          benchmark<float16, float16, xtd::rint>(device);
        }

        SECTION("float xtd::rint(float)") {
          benchmark<float, float, xtd::rint>(device);
        }

        SECTION("double xtd::rint(double)") {
          benchmark<double, double, xtd::rint>(device);
        }

        SECTION("double xtd::rint(int)") {
          benchmark<double, int, xtd::rint>(device);
        }

        SECTION("float16 xtd::rintf(float16)") {
          benchmark<float16, float16, xtd::rintf>(device);
        }

        SECTION("float xtd::rintf(float)") {
          benchmark<float, float, xtd::rintf>(device);
        }

        SECTION("float xtd::rintf(double)") {
          benchmark<float, double, xtd::rintf>(device);
        }

        SECTION("float xtd::rintf(int)") {
          benchmark<float, int, xtd::rintf>(device);
        }
      }
    }
  }
}
