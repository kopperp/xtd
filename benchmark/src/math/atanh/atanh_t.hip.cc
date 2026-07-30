/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/atanh.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/benchmark.h"


TEST_CASE("xtd::atanh", "[atanh][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::atanh(float16)") {
          benchmark<float16, float16, xtd::atanh>(device);
        }

        SECTION("float xtd::atanh(float)") {
          benchmark<float, float, xtd::atanh>(device);
        }

        SECTION("double xtd::atanh(double)") {
          benchmark<double, double, xtd::atanh>(device);
        }

        SECTION("double xtd::atanh(int)") {
          benchmark<double, int, xtd::atanh>(device);
        }

        SECTION("float16 xtd::atanhf(float16)") {
          benchmark<float16, float16, xtd::atanhf>(device);
        }

        SECTION("float xtd::atanhf(float)") {
          benchmark<float, float, xtd::atanhf>(device);
        }

        SECTION("float xtd::atanhf(double)") {
          benchmark<float, double, xtd::atanhf>(device);
        }

        SECTION("float xtd::atanhf(int)") {
          benchmark<float, int, xtd::atanhf>(device);
        }
      }
    }
  }
}
