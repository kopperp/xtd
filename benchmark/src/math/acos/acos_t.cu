/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/acos.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"


TEST_CASE("xtd::acos", "[acos][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::acos(float16)") {
          benchmark<float16, float16, xtd::acos>(device);
        }

        SECTION("float xtd::acos(float)") {
          benchmark<float, float, xtd::acos>(device);
        }

        SECTION("double xtd::acos(double)") {
          benchmark<double, double, xtd::acos>(device);
        }

        SECTION("double xtd::acos(int)") {
          benchmark<double, int, xtd::acos>(device);
        }

        SECTION("float xtd::acosf(float16)") {
          benchmark<float16, float16, xtd::acosf>(device);
        }

        SECTION("float xtd::acosf(float)") {
          benchmark<float, float, xtd::acosf>(device);
        }

        SECTION("float xtd::acosf(double)") {
          benchmark<float, double, xtd::acosf>(device);
        }

        SECTION("float xtd::acosf(int)") {
          benchmark<float, int, xtd::acosf>(device);
        }
      }
    }
  }
}
