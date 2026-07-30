/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <iomanip>
#include <iostream>
#include <vector>

// HIP headers
#include <hip/hip_runtime.h>

// Catch2 headers
#include <catch_amalgamated.hpp>

// XTD concepts
#include "xtd/concepts.h"

// test headers
#include "common/halton.h"
#include "common/hip/device.h"
#include "common/hip/hip_check.h"
#include "common/hip/inputs.h"

namespace test::hip {

  template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType)>
  __global__ static void kernel(InputType const* input, ResultType* result, int size) {
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int t = thread; t < size; t += stride) {
      result[t] = XtdFunc(input[t]);
    }
  }

  template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType, InputType)>
  __global__ static void kernel(std::pair<InputType, InputType> const* input_pairs, ResultType* result, unsigned int size) {
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (unsigned int t = thread; t < size; t += stride) {
      // generate a low-discrepancy deterministic sequence over [0, size)*[0, size)
      auto const& [x, y] = input_pairs[t];
      result[t] = static_cast<ResultType>(XtdFunc(x, y));
    }
  }

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
  inline void benchmark(const Device& device, int ulps = 0) {
    hipStream_t queue = device.queue();
    const Inputs& input = inputs(device);
    std::span<const InputType> values = input.values<InputType>();
    const size_t size = input.size();

    // allocate memory for the results and fill it with zeroes
    ResultType* result;
    HIP_CHECK(hipMallocAsync(&result, size * sizeof(ResultType), queue));
    // HIP_CHECK(hipMemsetAsync(result, 0x00, size * sizeof(ResultType), queue));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // warmup launch
    kernel<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(values.data(), result, size);

    // execute the xtd function on the GPU
    BENCHMARK_ADVANCED(std::string(Catch::getResultCapture().getCurrentTestName()))(Catch::Benchmark::Chronometer meter) {
      meter.measure([&] {
        HIP_CHECK(hipEventRecord(start, queue));
        kernel<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(values.data(), result, size);
        HIP_CHECK(hipEventRecord(stop, queue));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        return std::chrono::duration<double, std::milli>(elapsed_ms);
      });
    };

    // free the GPU memory
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFreeAsync(result, queue));
    HIP_CHECK(hipStreamSynchronize(queue));
  }

  template <std::integral ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType)>
  inline void benchmark(const Device& device) {
    hipStream_t queue = device.queue();
    const Inputs& input = inputs(device);
    std::span<const InputType> values = input.values<InputType>();
    const size_t size = input.size();

    // allocate memory for the results and fill it with zeroes
    ResultType* result;
    HIP_CHECK(hipMallocAsync(&result, size * sizeof(ResultType), queue));
    // HIP_CHECK(hipMemsetAsync(result, 0x00, size * sizeof(ResultType), queue));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // warmup launch
    kernel<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(values.data(), result, size);

    // execute the xtd function on the GPU
    BENCHMARK_ADVANCED(std::string(Catch::getResultCapture().getCurrentTestName()))(Catch::Benchmark::Chronometer meter) {
      meter.measure([&] {
        HIP_CHECK(hipEventRecord(start, queue));
        kernel<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(values.data(), result, size);
        HIP_CHECK(hipEventRecord(stop, queue));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        return std::chrono::duration<double, std::milli>(elapsed_ms);
      });
    };

    // free the GPU memory
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFreeAsync(result, queue));
    HIP_CHECK(hipStreamSynchronize(queue));
  }

  template <xtd::floating_point ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType, InputType)>
  inline void benchmark(const Device& device, int ulps = 0) {
    hipStream_t queue = device.queue();
    const Inputs& input = inputs(device);
    std::span<const InputType> values = input.values<InputType>();
    const size_t size = input.size();

    std::vector<std::pair<InputType, InputType>> input_pairs;
    input_pairs.reserve(size);
    for (size_t t = 0; t < size; ++t) {
      auto [i, j] = halton<2>(t, size);
      input_pairs.emplace_back(values[i], values[j]);
    }

    std::pair<InputType, InputType>* d_input_pairs;
    ResultType* result;
    HIP_CHECK(hipMallocAsync(&d_input_pairs, size * sizeof(std::pair<InputType, InputType>), queue));
    HIP_CHECK(hipMallocAsync(&result, size * sizeof(ResultType), queue));

    HIP_CHECK(hipMemcpyAsync(
        d_input_pairs, input_pairs.data(), size * sizeof(std::pair<InputType, InputType>), hipMemcpyHostToDevice, queue));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // warmup launch
    kernel<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(d_input_pairs, result, size);

    // execute the xtd function on the GPU
    BENCHMARK_ADVANCED(std::string(Catch::getResultCapture().getCurrentTestName()))(Catch::Benchmark::Chronometer meter) {
      meter.measure([&] {
        HIP_CHECK(hipEventRecord(start, queue));
        kernel<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(d_input_pairs, result, size);
        HIP_CHECK(hipEventRecord(stop, queue));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        return std::chrono::duration<double, std::milli>(elapsed_ms);
      });
    };

    // free the GPU memory
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFreeAsync(d_input_pairs, queue));
    HIP_CHECK(hipFreeAsync(result, queue));
    HIP_CHECK(hipStreamSynchronize(queue));
  }

}  // namespace test::hip
