/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/tan.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"


TEST_CASE("xtd::tan", "[tan][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::tan(float16)") {
          benchmark<float16, float16, xtd::tan>(device);
        }

        SECTION("float xtd::tan(float)") {
          benchmark<float, float, xtd::tan>(device);
        }

        SECTION("double xtd::tan(double)") {
          benchmark<double, double, xtd::tan>(device);
        }

        SECTION("double xtd::tan(int)") {
          benchmark<double, int, xtd::tan>(device);
        }

        SECTION("float16 xtd::tanf(float16)") {
          benchmark<float16, float16, xtd::tanf>(device);
        }

        SECTION("float xtd::tanf(float)") {
          benchmark<float, float, xtd::tanf>(device);
        }

        SECTION("float xtd::tanf(double)") {
          benchmark<float, double, xtd::tanf>(device);
        }

        SECTION("float xtd::tanf(int)") {
          benchmark<float, int, xtd::tanf>(device);
        }
      }
    }
  }
}
