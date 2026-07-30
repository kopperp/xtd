/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <concepts>
#include <iomanip>
#include <iostream>
#include <span>

// Catch2 headers
#include <catch_amalgamated.hpp>

// XTD concepts
#include "xtd/concepts.h"

// benchmark header
#include "common/halton.h"
#include "common/cpu/inputs.h"

namespace test::cpu {

  template <typename T>
  struct detailed {
    constexpr detailed(T value) : value_{value} {
    }

    T value_;
  };

  template <xtd::floating_point T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    std::ostringstream buffer;
    buffer << std::fixed << std::setprecision(std::numeric_limits<T>::max_digits10) << val.value_ << " ["
           << std::hexfloat << val.value_ << "]";
    out << buffer.str();
    return out;
  }

  template <std::integral T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    out << val.value_;
    return out;
  }

  template <xtd::floating_point ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType)>
  inline void benchmark(const Device& device) {
    std::span<const InputType> values = inputs().values<InputType>();
    const size_t n = values.size();

    BENCHMARK(std::move(std::string(Catch::getResultCapture().getCurrentTestName())), i) {
      // Prevent optimization of the function call across the entire loop
      return XtdFunc(values[i % n]);
    };
    (void)device;
  }

  template <std::integral ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType)>
  inline void benchmark(const Device& device) {
    std::span<const InputType> values = inputs().values<InputType>();
    const size_t n = values.size();

    BENCHMARK(std::move(std::string(Catch::getResultCapture().getCurrentTestName())), i) {
      // Prevent optimization of the function call across the entire loop
      return XtdFunc(values[i % n]);
    };
    (void)device;
  }

  template <xtd::floating_point ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType, InputType)>
  inline void benchmark(const Device& device) {
    std::span<const InputType> values = inputs().values<InputType>();
    const size_t size = values.size();

    std::vector<std::pair<InputType, InputType>> input_pairs;
    input_pairs.reserve(size);
    for (size_t t = 0; t < size; ++t) {
      auto [i, j] = halton<2>(t, size);
      input_pairs.emplace_back(values[i], values[j]);
    }

    BENCHMARK(std::move(std::string(Catch::getResultCapture().getCurrentTestName())), i) {
      // Prevent optimization of the function call across the entire loop
      auto const& [x, y] = input_pairs[i % size];
      return XtdFunc(x, y);
    };
    (void)device;
  }

}  // namespace test::cpu
