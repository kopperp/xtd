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
  T abs(T x, T lo, T hi) {
    return std::abs(x);
  }

  inline float16 abs(float16 x) {
    return float16(std::abs(static_cast<float>(x)));
  }
}  // namespace byval

