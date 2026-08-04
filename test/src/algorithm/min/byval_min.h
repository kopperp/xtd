/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <algorithm>

// xtd headers
#include "xtd/concepts/arithmetic.h"

namespace byval {
  template <xtd::arithmetic T>
  T min(T a, T b) {
    return std::min(a, b);
  }

  inline float16 min(float16 a, float16 b) {
    return float16(std::min(static_cast<float>(a), static_cast<float>(b)));
  }
}  // namespace byval
