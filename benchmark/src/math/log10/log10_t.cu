/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/log10.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"


TEST_CASE("xtd::log10", "[log10][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::log10(float16)") {
          benchmark<float16, float16, xtd::log10>(device);
        }

        SECTION("float xtd::log10(float)") {
          benchmark<float, float, xtd::log10>(device);
        }

        SECTION("double xtd::log10(double)") {
          benchmark<double, double, xtd::log10>(device);
        }

        SECTION("double xtd::log10(int)") {
          benchmark<double, int, xtd::log10>(device);
        }

        SECTION("float16 xtd::log10f(float16)") {
          benchmark<float16, float16, xtd::log10f>(device);
        }

        SECTION("float xtd::log10f(float)") {
          benchmark<float, float, xtd::log10f>(device);
        }

        SECTION("float xtd::log10f(double)") {
          benchmark<float, double, xtd::log10f>(device);
        }

        SECTION("float xtd::log10f(int)") {
          benchmark<float, int, xtd::log10f>(device);
        }
      }
    }
  }
}
