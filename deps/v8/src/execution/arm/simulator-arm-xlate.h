// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// linux-workers: the embedded builtins as C, on a wasm32 host.
//
// V8 has no wasm target; on a wasm32 host it runs ARM code in its simulator.
// tools/arm-to-c.py translates each builtin of the embedded blob (the final
// ARM instructions, as mksnapshot wrote them in embedded.S) into a C++
// function over the simulator's own machine state: the registers and flags
// in locals while it runs, the stack and frames in memory, return addresses
// still addresses in the blob. The simulator's loop calls a builtin's
// function for a program counter at one of its entries, and simulates what
// the translation leaves out (calls into C++, which go through the
// simulator's redirection, and instructions it doesn't translate).
//
// This header is the runtime of the translated code, and the interface
// between it and the simulator: no V8 headers, so the translated code
// compiles on its own.

#ifndef V8_EXECUTION_ARM_SIMULATOR_ARM_XLATE_H_
#define V8_EXECUTION_ARM_SIMULATOR_ARM_XLATE_H_

#include <stdint.h>
#include <string.h>

#include <cmath>
#include <limits>

namespace v8_arm_xlate {

struct State;

// The translated builtins pass the core registers as arguments, from one
// builtin to the next: wasm tail calls, the registers in wasm locals
// throughout. (Just the interpreter's own, from one bytecode to the next,
// with the rest in the state: slower, the state's stores costing more than
// the arguments.) The flags stay in the state, each function loading and
// saving them if it uses them. (tools/arm-to-c.py's PARAM_REGS: the same.)
#define XL_PARAMS                                                            \
  uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, uint32_t r4,           \
      uint32_t r5, uint32_t r6, uint32_t r7, uint32_t r8, uint32_t r9,       \
      uint32_t r10, uint32_t r11, uint32_t r12, uint32_t r13, uint32_t r14
#define XL_TAIL extern "C" __attribute__((target("tail-call")))

// a translated builtin: runs from pc (an address in the blob) on, and goes
// on (tail-calls v8_arm_xlate_next) from where it leaves it
using Function = uint32_t (*)(State* s, uint32_t pc, XL_PARAMS);

// The simulator's machine state, by pointers into its fields, and the
// translated code's own.
struct State {
  int32_t* r;  // r0..r15
  bool *n, *z, *c, *v;
  uint32_t* vfp;  // s0..s63 (d<i> is s<2i>, s<2i+1>)
  bool *fn, *fz, *fc, *fv;  // FPSCR's NZCV
  bool* default_nan;
  int32_t* rounding_mode;  // FPSCR's RMode (bits 23:22, in place)
  bool *inv_op, *div_zero, *overflow, *underflow, *inexact;

  // each builtin's function, by index (null: simulate it)
  const Function* functions;
  // Calls (bl, blx) are native calls, as deep as kMaxDepth (deeper: jumps):
  // ret is the return address the innermost one waits for, where execution
  // returns to it. Its frames hold nothing but where to resume (the machine
  // state is saved across a call), so a native caller may also resume at
  // its return address from elsewhere.
  uint32_t ret;
  uint32_t depth;
  // set when execution needs the simulator (an instruction outside the blob,
  // as calls into C++ are, or untranslated): the translated code returns,
  // all of it, to the simulator's loop, which simulates the instruction.
  // So no translated frame is ever under a call into C++, nor under a fork
  // (Asyncify leaves them out: see the flake's wasm-ld-v8).
  bool unwind;
};
constexpr uint32_t kMaxDepth = 96;

struct Builtin {
  uint32_t start;  // offset in the blob
  uint32_t size;
  Function function;
  const char* name;
};

// the translated builtins, in blob order (tools/arm-to-c.py), and for each
// 32 bytes of the blob (builtins start 32-byte aligned), its builtin
extern "C" const Builtin v8_arm_xlate_builtins[];
extern "C" const uint32_t v8_arm_xlate_builtin_count;
extern "C" const uint16_t v8_arm_xlate_index[];
extern "C" const unsigned char v8_Default_embedded_blob_code_[];
extern "C" const uint32_t v8_arm_xlate_blob_size;
// each builtin's group: the builtins translated as one function
extern "C" const uint16_t v8_arm_xlate_group[];

// for each instruction of the blob, its entry number in the one group that
// has them (the interpreter's), or 0
extern "C" const uint16_t v8_arm_xlate_entry[];

// whether an offset in the blob is in a group's builtins
inline bool in_group(uint32_t offset, uint32_t group) {
  return offset < v8_arm_xlate_blob_size &&
         v8_arm_xlate_group[v8_arm_xlate_index[offset >> 5]] == group;
}

// Goes on from pc: tail-calls the function of the builtin there; or saves
// the registers to the state and returns, at the innermost native call's
// return address (s->ret), the simulator's end, or for the simulator
// (s->unwind).
XL_TAIL uint32_t v8_arm_xlate_next(State* s, uint32_t pc, XL_PARAMS);
// for the simulator: saves the registers, sets s->unwind, returns pc
XL_TAIL uint32_t v8_arm_xlate_unwind(State* s, uint32_t pc, XL_PARAMS);
// runs from pc on, the registers in the state, as v8_arm_xlate_next
extern "C" uint32_t v8_arm_xlate_enter(State* s, uint32_t pc);

// --- the translated code's runtime -----------------------------------------

#define XL_INLINE static inline __attribute__((always_inline))

XL_INLINE uint32_t ld32(uint32_t a) {
  uint32_t v;
  memcpy(&v, reinterpret_cast<const void*>(static_cast<uintptr_t>(a)), 4);
  return v;
}
XL_INLINE uint32_t ld16(uint32_t a) {
  uint16_t v;
  memcpy(&v, reinterpret_cast<const void*>(static_cast<uintptr_t>(a)), 2);
  return v;
}
XL_INLINE uint32_t ld16s(uint32_t a) {
  int16_t v;
  memcpy(&v, reinterpret_cast<const void*>(static_cast<uintptr_t>(a)), 2);
  return static_cast<uint32_t>(static_cast<int32_t>(v));
}
XL_INLINE uint32_t ld8(uint32_t a) {
  return *reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a));
}
XL_INLINE uint32_t ld8s(uint32_t a) {
  return static_cast<uint32_t>(static_cast<int32_t>(
      *reinterpret_cast<const int8_t*>(static_cast<uintptr_t>(a))));
}
XL_INLINE void st32(uint32_t a, uint32_t v) {
  memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(a)), &v, 4);
}
XL_INLINE void st16(uint32_t a, uint32_t v) {
  uint16_t h = static_cast<uint16_t>(v);
  memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(a)), &h, 2);
}
XL_INLINE void st8(uint32_t a, uint32_t v) {
  *reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(a)) =
      static_cast<uint8_t>(v);
}

XL_INLINE uint32_t ror(uint32_t x, uint32_t n) {
  n &= 31;
  return n == 0 ? x : (x >> n) | (x << (32 - n));
}

// flags of additions and subtractions, as the simulator computes them
// (CarryFrom, BorrowFrom, OverflowFrom)
XL_INLINE uint32_t carry_add(uint32_t l, uint32_t r, uint32_t c) {
  uint32_t rest = 0xFFFFFFFFu - l;
  return (r > rest) || (c && ((r + 1 > rest) || (r > rest - 1)));
}
XL_INLINE uint32_t borrow_sub(uint32_t l, uint32_t r, uint32_t c) {
  // BorrowFrom(l, r, c): r > l, or with no carry in, r + 1 > l
  return (r > l) || (!c && ((r + 1 > l) || (r > l - 1)));
}
XL_INLINE uint32_t overflow_add(uint32_t out, uint32_t l, uint32_t r) {
  return ((~(l ^ r)) & (l ^ out)) >> 31;
}
XL_INLINE uint32_t overflow_sub(uint32_t out, uint32_t l, uint32_t r) {
  return ((l ^ r) & (l ^ out)) >> 31;
}

// shifts by a register (GetShiftRm, by register): value, and the carry out
XL_INLINE uint32_t shift_reg(int type, uint32_t x, uint32_t amount,
                             uint32_t c_in, uint32_t* c_out) {
  amount &= 0xFF;
  if (amount == 0) {
    *c_out = c_in;
    return x;
  }
  switch (type) {
    case 0:  // LSL
      if (amount < 32) {
        *c_out = (x >> (32 - amount)) & 1;
        return x << amount;
      }
      *c_out = amount == 32 ? (x & 1) : 0;
      return 0;
    case 1:  // LSR
      if (amount < 32) {
        *c_out = (x >> (amount - 1)) & 1;
        return x >> amount;
      }
      *c_out = amount == 32 ? (x >> 31) : 0;
      return 0;
    case 2:  // ASR
      if (amount < 32) {
        *c_out = (x >> (amount - 1)) & 1;
        return static_cast<uint32_t>(static_cast<int32_t>(x) >>
                                     amount);
      }
      *c_out = x >> 31;
      return static_cast<uint32_t>(static_cast<int32_t>(x) >> 31);
    default: {  // ROR
      uint32_t result = ror(x, amount);
      *c_out = result >> 31;
      return result;
    }
  }
}

XL_INLINE uint32_t clz(uint32_t x) { return x == 0 ? 32 : __builtin_clz(x); }

XL_INLINE uint32_t sdiv(uint32_t l, uint32_t r) {
  int32_t a = static_cast<int32_t>(l), b = static_cast<int32_t>(r);
  if (b == 0) return 0;
  if (b == -1) return a == INT32_MIN ? l : static_cast<uint32_t>(-a);
  return static_cast<uint32_t>(a / b);
}
XL_INLINE uint32_t udiv(uint32_t l, uint32_t r) { return r == 0 ? 0 : l / r; }
XL_INLINE uint32_t smmul(uint32_t l, uint32_t r) {
  int64_t p = static_cast<int64_t>(static_cast<int32_t>(l)) *
              static_cast<int32_t>(r);
  return static_cast<uint32_t>(static_cast<uint64_t>(p) >> 32);
}

// --- VFP --------------------------------------------------------------------

XL_INLINE double getd(State* s, int d) {
  double x;
  memcpy(&x, &s->vfp[2 * d], 8);
  return x;
}
XL_INLINE void setd(State* s, int d, double x) { memcpy(&s->vfp[2 * d], &x, 8); }
XL_INLINE float getf(State* s, int n) {
  float x;
  memcpy(&x, &s->vfp[n], 4);
  return x;
}
XL_INLINE void setf(State* s, int n, float x) { memcpy(&s->vfp[n], &x, 4); }

XL_INLINE double canon(State* s, double x) {
  if (*s->default_nan && std::isnan(x)) {
    uint64_t bits = 0x7FF8000000000000ull;
    memcpy(&x, &bits, 8);
  }
  return x;
}
XL_INLINE float canonf(State* s, float x) {
  if (*s->default_nan && std::isnan(x)) {
    uint32_t bits = 0x7FC00000u;
    memcpy(&x, &bits, 4);
  }
  return x;
}

template <typename T>
XL_INLINE void fp_compare(State* s, T a, T b) {  // Compute_FPSCR_Flags
  if (std::isnan(a) || std::isnan(b)) {
    *s->fn = false, *s->fz = false, *s->fc = true, *s->fv = true;
  } else if (a == b) {
    *s->fn = false, *s->fz = true, *s->fc = true, *s->fv = false;
  } else if (a < b) {
    *s->fn = true, *s->fz = false, *s->fc = false, *s->fv = false;
  } else {
    *s->fn = false, *s->fz = false, *s->fc = true, *s->fv = false;
  }
}

constexpr int32_t kRN = 0 << 22, kRM = 2 << 22, kRZ = 3 << 22;

// Simulator::ConvertDoubleToInt, its flags too
static inline uint32_t fp_to_int(State* s, double val, bool unsigned_integer,
                                 bool round_to_zero) {
  int32_t mode = round_to_zero ? kRZ : *s->rounding_mode;
  int32_t result;
  if (unsigned_integer) {
    if (val > 4294967295.0) {
      result = static_cast<int32_t>(0xFFFFFFFFu);
    } else if (!(val >= 0)) {
      result = 0;
    } else {
      result = static_cast<int32_t>(static_cast<uint32_t>(val));
    }
  } else {
    if (!(val >= INT32_MIN)) {
      result = INT32_MIN;
    } else if (val > INT32_MAX) {
      result = INT32_MAX;
    } else {
      result = static_cast<int32_t>(val);
    }
  }
  const double max_uint = 4294967295.0, max_int = 2147483647.0,
               min_int = -2147483648.0;
  bool inv;
  if (val != val) {
    inv = true;
  } else if (mode == kRN) {
    inv = unsigned_integer
              ? (val >= (max_uint + 0.5)) || (val < -0.5)
              : (val >= (max_int + 0.5)) || (val < (min_int - 0.5));
  } else if (mode == kRM) {
    inv = unsigned_integer ? (val >= (max_uint + 1.0)) || (val < 0)
                           : (val >= (max_int + 1.0)) || (val < min_int);
  } else {
    inv = unsigned_integer
              ? (val >= (max_uint + 1.0)) || (val <= -1)
              : (val >= (max_int + 1.0)) || (val <= (min_int - 1.0));
  }
  *s->inv_op = inv;
  double abs_diff =
      unsigned_integer
          ? std::fabs(val - static_cast<double>(static_cast<uint32_t>(result)))
          : std::fabs(val - static_cast<double>(result));
  *s->inexact = (abs_diff != 0);
  if (inv) {
    if (val != val) {
      result = 0;
    } else if (unsigned_integer) {
      result = (val < 0) ? 0 : static_cast<int32_t>(0xFFFFFFFFu);
    } else {
      result = (val < 0) ? INT32_MIN : INT32_MAX;
    }
  } else if (mode == kRN) {
    int val_sign = (val > 0) ? 1 : -1;
    if (abs_diff > 0.5) {
      result += val_sign;
    } else if (abs_diff == 0.5) {
      result = ((result % 2) == 0)
                   ? result
                   : static_cast<int32_t>(static_cast<uint32_t>(result) +
                                          static_cast<uint32_t>(val_sign));
    }
  } else if (mode == kRM) {
    result = result > val ? result - 1 : result;
  }
  return static_cast<uint32_t>(result);
}

XL_INLINE uint32_t get_fpscr(State* s) {
  return (static_cast<uint32_t>(*s->fn) << 31) |
         (static_cast<uint32_t>(*s->fz) << 30) |
         (static_cast<uint32_t>(*s->fc) << 29) |
         (static_cast<uint32_t>(*s->fv) << 28) |
         (static_cast<uint32_t>(*s->default_nan) << 25) |
         (static_cast<uint32_t>(*s->inexact) << 4) |
         (static_cast<uint32_t>(*s->underflow) << 3) |
         (static_cast<uint32_t>(*s->overflow) << 2) |
         (static_cast<uint32_t>(*s->div_zero) << 1) |
         (static_cast<uint32_t>(*s->inv_op) << 0) |
         static_cast<uint32_t>(*s->rounding_mode);
}
XL_INLINE void set_fpscr(State* s, uint32_t x) {
  *s->fn = (x >> 31) & 1, *s->fz = (x >> 30) & 1, *s->fc = (x >> 29) & 1,
  *s->fv = (x >> 28) & 1;
  *s->default_nan = (x >> 25) & 1;
  *s->inexact = (x >> 4) & 1, *s->underflow = (x >> 3) & 1,
  *s->overflow = (x >> 2) & 1, *s->div_zero = (x >> 1) & 1,
  *s->inv_op = x & 1;
  *s->rounding_mode = static_cast<int32_t>(x & (3u << 22));
}

}  // namespace v8_arm_xlate

#endif  // V8_EXECUTION_ARM_SIMULATOR_ARM_XLATE_H_
