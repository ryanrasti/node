// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// linux-workers: PushAllRegistersAndIterateStack for a wasm32 host. Other
// hosts push their callee-saved registers so that a conservative scan of
// the stack sees pointers held only in registers; wasm has no registers to
// push (its locals aren't addressable memory), so this only marks where the
// stack ends: a local's address, which is on clang's stack in linear
// memory. V8's own heap is reached precisely (handles); a conservative scan
// would miss pointers held only in wasm locals.

#include "src/heap/base/stack.h"

extern "C" void PushAllRegistersAndIterateStack(
    heap::base::Stack* stack, void* argument,
    heap::base::Stack::IterateStackCallback callback) {
  volatile int marker = 0;
  callback(stack, argument, const_cast<int*>(&marker));
}
