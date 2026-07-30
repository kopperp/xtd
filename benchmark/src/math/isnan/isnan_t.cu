/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/isnan.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"

TEST_CASE("xtd::isnan", "[isnan][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("int xtd::isnan(float16)") {
          benchmark<int, float16, xtd::isnan>(device);
        }

        SECTION("int xtd::isnan(float)") {
          benchmark<int, float, xtd::isnan>(device);
        }

        SECTION("int xtd::isnan(double)") {
          benchmark<int, double, xtd::isnan>(device);
        }

        SECTION("int xtd::isnan(int)") {
          benchmark<int, int, xtd::isnan>(device);
        }
      }
    }
  }
}
