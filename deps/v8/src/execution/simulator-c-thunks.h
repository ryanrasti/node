// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// linux-workers: exact-signature calls from the ARM simulator into C++, on a
// wasm32 host.
//
// The simulator calls the C++ functions generated code reaches (runtime
// functions, C++ builtins, external references) through one generic type,
// with 20 int32 arguments: natively, a function just ignores the arguments
// it doesn't take. wasm checks an indirect call's signature against the
// function's, and traps on a mismatch. So each such function, where it's
// registered (its type known), also gets a thunk of the generic type, which
// unpacks the argument words as ARM's ABI laid them out (a 64-bit value in
// an even-aligned pair) and calls it; the simulator looks the thunk up by
// the function's address.

#ifndef V8_EXECUTION_SIMULATOR_C_THUNKS_H_
#define V8_EXECUTION_SIMULATOR_C_THUNKS_H_

#include "src/common/globals.h"

#if defined(USE_SIMULATOR) && V8_HOST_ARCH_WASM32

#include <array>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace v8::internal::sim_c_thunks {

using GenericCall = int64_t (*)(int32_t, int32_t, int32_t, int32_t, int32_t,
                                int32_t, int32_t, int32_t, int32_t, int32_t,
                                int32_t, int32_t, int32_t, int32_t, int32_t,
                                int32_t, int32_t, int32_t, int32_t, int32_t);
constexpr int kMaxWords = 20;

template <typename T>
constexpr int kWords = sizeof(T) == 8 ? 2 : 1;

// argument word offsets: 64-bit values start at an even word
template <typename... A>
constexpr std::array<int, sizeof...(A)> Offsets() {
  std::array<int, sizeof...(A)> offsets{};
  int word = 0, k = 0;
  ((word += (kWords<A> == 2 && word % 2) ? 1 : 0, offsets[k++] = word,
    word += kWords<A>),
   ...);
  return offsets;
}

template <typename T>
T FromWords(const int32_t* w) {
  static_assert(std::is_trivially_copyable_v<T> && sizeof(T) <= 8);
  if constexpr (std::is_pointer_v<T>) {
    return reinterpret_cast<T>(static_cast<uintptr_t>(static_cast<uint32_t>(w[0])));
  } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
    if constexpr (sizeof(T) == 8) {
      return static_cast<T>(static_cast<uint64_t>(static_cast<uint32_t>(w[0])) |
                            static_cast<uint64_t>(static_cast<uint32_t>(w[1]))
                                << 32);
    } else {
      return static_cast<T>(w[0]);
    }
  } else {  // doubles (in core registers: softfp), small structs
    T value;
    std::memcpy(&value, w, sizeof(T));
    return value;
  }
}

template <typename R>
int64_t ToWords(R result) {
  static_assert(std::is_trivially_copyable_v<R> && sizeof(R) <= 8);
  if constexpr (std::is_pointer_v<R>) {
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(result));
  } else if constexpr (std::is_integral_v<R> || std::is_enum_v<R>) {
    return static_cast<int64_t>(result);
  } else {
    int64_t words = 0;
    std::memcpy(&words, &result, sizeof(R));
    return words;
  }
}

template <typename F>
struct Traits;
template <typename R, typename... A>
struct Traits<R (*)(A...)> {
  template <auto f, size_t... I>
  static int64_t Call(const int32_t* w, std::index_sequence<I...>) {
    constexpr auto offsets = Offsets<A...>();
    if constexpr (sizeof...(A) > 0) {
      static_assert(offsets.back() + kWords<std::tuple_element_t<
                                         sizeof...(A) - 1, std::tuple<A...>>> <=
                    kMaxWords);
    }
    if constexpr (std::is_void_v<R>) {
      f(FromWords<A>(w + offsets[I])...);
      return 0;
    } else {
      return ToWords(f(FromWords<A>(w + offsets[I])...));
    }
  }
  static constexpr size_t kArity = sizeof...(A);
};
template <typename R, typename... A>
struct Traits<R (*)(A...) noexcept> : Traits<R (*)(A...)> {};

template <auto f>
int64_t Thunk(int32_t a0, int32_t a1, int32_t a2, int32_t a3, int32_t a4,
              int32_t a5, int32_t a6, int32_t a7, int32_t a8, int32_t a9,
              int32_t a10, int32_t a11, int32_t a12, int32_t a13, int32_t a14,
              int32_t a15, int32_t a16, int32_t a17, int32_t a18,
              int32_t a19) {
  const int32_t w[kMaxWords] = {a0,  a1,  a2,  a3,  a4,  a5,  a6,
                                a7,  a8,  a9,  a10, a11, a12, a13,
                                a14, a15, a16, a17, a18, a19};
  using T = Traits<decltype(f)>;
  return T::template Call<f>(w, std::make_index_sequence<T::kArity>());
}

// thunks, by their function's address
inline std::unordered_map<Address, GenericCall>& Table() {
  static auto* table = new std::unordered_map<Address, GenericCall>();
  return *table;
}
inline std::mutex& TableMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}
inline void Register(Address function, GenericCall thunk) {
  std::lock_guard<std::mutex> lock(TableMutex());
  Table().emplace(function, thunk);
}
inline GenericCall Lookup(Address function) {
  std::lock_guard<std::mutex> lock(TableMutex());
  auto it = Table().find(function);
  return it == Table().end() ? nullptr : it->second;
}

}  // namespace v8::internal::sim_c_thunks

// registers function f's thunk
#define V8_SIM_C_THUNK(f)                                           \
  ::v8::internal::sim_c_thunks::Register(                           \
      FUNCTION_ADDR(f), &::v8::internal::sim_c_thunks::Thunk<&f>)

#else
#define V8_SIM_C_THUNK(f) ((void)0)
#endif  // defined(USE_SIMULATOR) && V8_HOST_ARCH_WASM32

#endif  // V8_EXECUTION_SIMULATOR_C_THUNKS_H_
