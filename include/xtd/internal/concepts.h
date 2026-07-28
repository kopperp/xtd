/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Simone Balducci <simone.balducci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace xtd {

  template <typename T>
  concept Numeric = requires {
    std::is_arithmetic_v<T>;
    requires sizeof(T) <= 8;
  };

  template <typename T>
  concept floating_point = !std::integral<T> && (
    std::floating_point<T> ||
    (std::numeric_limits<T>::is_specialized && !std::numeric_limits<T>::is_integer) ||
    requires(T a) {
      { static_cast<float>(a) } -> std::same_as<float>;
      { T(a) } -> std::same_as<T>;
    }
  );

}  // namespace xtd
