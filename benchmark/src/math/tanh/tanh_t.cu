/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch_amalgamated.hpp>

// xtd headers
#include "xtd/math/tanh.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/benchmark.h"


TEST_CASE("xtd::tanh", "[tanh][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::tanh(float16)") {
          benchmark<float16, float16, xtd::tanh>(device);
        }

        SECTION("float xtd::tanh(float)") {
          benchmark<float, float, xtd::tanh>(device);
        }

        SECTION("double xtd::tanh(double)") {
          benchmark<double, double, xtd::tanh>(device);
        }

        SECTION("double xtd::tanh(int)") {
          benchmark<double, int, xtd::tanh>(device);
        }

        SECTION("float16 xtd::tanhf(float16)") {
          benchmark<float16, float16, xtd::tanhf>(device);
        }

        SECTION("float xtd::tanhf(float)") {
          benchmark<float, float, xtd::tanhf>(device);
        }

        SECTION("float xtd::tanhf(double)") {
          benchmark<float, double, xtd::tanhf>(device);
        }

        SECTION("float xtd::tanhf(int)") {
          benchmark<float, int, xtd::tanhf>(device);
        }
      }
    }
  }
}
