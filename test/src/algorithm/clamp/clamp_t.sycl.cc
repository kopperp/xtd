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
#include "xtd/algorithm/clamp.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate_clamp.h"
#include "byval_clamp.h"

TEST_CASE("xtd::clamp", "[clamp][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
        SECTION("float16 xtd::clamp(float16, float16)") {
          validate_clamp<float16, float16, xtd::clamp, [](float16 x, float16 lo, float16 hi) -> float16 { return float16(std::clamp(static_cast<float>(x), static_cast<float>(lo), static_cast<float>(hi))); }>(platform, device);
        }

          SECTION("float xtd::clamp(float, float)") {
            validate_clamp<float, float, xtd::clamp, byval::clamp>(platform, device);
          }

          SECTION("double xtd::clamp(double, double)") {
            validate_clamp<double, double, xtd::clamp, byval::clamp>(platform, device);
          }

          SECTION("int xtd::clamp(int, int)") {
            validate_clamp<int, int, xtd::clamp, byval::clamp>(platform, device);
          }

          SECTION("long xtd::clamp(long, long)") {
            validate_clamp<long, long, xtd::clamp, byval::clamp>(platform, device);
          }

          SECTION("long long xtd::clamp(long long, long long)") {
            validate_clamp<long long, long long, xtd::clamp, byval::clamp>(platform, device);
          }
        }
      }
    }
  }
}
