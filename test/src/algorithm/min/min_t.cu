/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <algorithm>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/algorithm/min.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "byval_min.h"

TEST_CASE("xtd::min", "[min][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::min(float16, float16)") {
          // NVCC suffers from internal template mangling errors on stateless lambdas
          // constexpr auto float16_min = +[](float16 a, float16 b) -> float16 { return float16(std::min(static_cast<float>(a), static_cast<float>(b))); };
          // validate<float16, float16, xtd::min, float16_min>(device);
          validate<float16, float16, xtd::min, byval::min>(device);
        }

        SECTION("float xtd::min(float, float)") {
          validate<float, float, xtd::min, byval::min>(device);
        }

        SECTION("double xtd::min(double, double)") {
          validate<double, double, xtd::min, byval::min>(device);
        }

        SECTION("int xtd::min(int, int)") {
          validate<int, int, xtd::min, byval::min>(device);
        }

        SECTION("long xtd::min(long, long)") {
          validate<long, long, xtd::min, byval::min>(device);
        }

        SECTION("long long xtd::min(long long, long long)") {
          validate<long long, long long, xtd::min, byval::min>(device);
        }
      }
    }
  }
}
