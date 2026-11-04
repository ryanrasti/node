// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// linux-workers: PushAllRegistersAndIterateStack for a wasm32 host. Other
// hosts push their callee-saved registers so that a conservative scan of
// the stack sees pointers held only in registers; wasm has no registers to
// push (its locals aren't addressable memory), so this only marks where the
// stack ends: a local's address, which is on clang's stack in linear
// memory. So the scan sees only what clang keeps there (address-taken
// locals, spills), never a pointer held only in a wasm local. V8's own heap
// doesn't depend on it (handles reach it precisely), but cppgc's (Oilpan)
// does: a cppgc object reached only by a raw pointer on the native stack
// across a GC is freed. Node's (ContextifyContext, ContextifyScript) are
// wrapped in their constructors, before anything can allocate on V8's
// heap, and from then on are reached through their JavaScript wrappers;
// code that holds a cppgc object only by a raw pointer across an
// allocation (as embedders may, relying on the scan) is unsafe here.

#include "src/heap/base/stack.h"

extern "C" void PushAllRegistersAndIterateStack(
    heap::base::Stack* stack, void* argument,
    heap::base::Stack::IterateStackCallback callback) {
  volatile int marker = 0;
  callback(stack, argument, const_cast<int*>(&marker));
}
