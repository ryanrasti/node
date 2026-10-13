#!/usr/bin/env python3
# Copyright 2026 the V8 project authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
"""linux-workers: the embedded builtins, ARM code, as C++ for a wasm32 host.

usage: arm-to-c.py embedded.S out-prefix shards

Reads the embedded blob mksnapshot wrote (embedded.S, in its wasm dialect:
the bytes, a comment naming each builtin) and writes out-prefix-<i>.cc
(i < shards), each builtin a C++ function over the simulator's machine state
(src/execution/arm/simulator-arm-xlate.h), and out-prefix-table.cc, the
table the simulator finds them in.

The translation decodes each instruction as the simulator does
(Simulator::InstructionDecode, simulator-arm.cc) and does what it does, in
the same order. What it doesn't translate (calls into C++: svc; exclusive
loads and stores; NEON; what the simulator itself doesn't implement) ends
the function: the simulator runs that instruction, and the translated code
resumes after it. Constant pools are decoded too, harmlessly: nothing
branches to them.

A function's entries, where the simulator's loop may enter it: its start,
the instruction after each call (a return address), after each instruction
it leaves to the simulator, and each address its code takes of itself
(add rX, pc, #imm). Inside, branches are gotos; a computed jump (bx, ldr pc,
ldm {pc}, mov pc) goes back to the entry switch if it stays in the function.
"""

import re
import sys

AL = 14
COND = {
    0: "Z", 1: "!Z", 2: "C", 3: "!C", 4: "N", 5: "!N", 6: "V", 7: "!V",
    8: "C && !Z", 9: "!C || Z", 10: "N == V", 11: "N != V",
    12: "!Z && N == V", 13: "Z || N != V",
}


def bits(x, hi, lo):
  return (x >> lo) & ((1 << (hi - lo + 1)) - 1)


def bit(x, n):
  return (x >> n) & 1


def ror32(x, n):
  n &= 31
  return x if n == 0 else ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


def parse_embedded(path):
  """The blob's code bytes and its builtins: [(name, offset)]."""
  code = bytearray()
  builtins = []
  in_code = False
  directive = re.compile(r"\.(octa|quad|long|byte|balign)\s+(.*)")
  sizes = {"octa": 16, "quad": 8, "long": 4, "byte": 1}
  with open(path) as f:
    for line in f:
      s = line.strip()
      if not in_code:
        in_code = s == "v8_Default_embedded_blob_code_:"
        continue
      if s.startswith("// Builtins_"):
        builtins.append((s[len("// Builtins_"):], len(code)))
        continue
      if s.startswith(".Lend_v8_Default_embedded_blob_code_"):
        break
      m = directive.match(s)
      if not m:
        continue
      kind, args = m.groups()
      if kind == "balign":
        align = int(args.split(",")[0])
        while len(code) % align:
          code.append(0)
        continue
      for v in args.split(","):
        code += int(v, 0).to_bytes(sizes[kind], "little")
  assert in_code and builtins, "no embedded blob code in " + path
  return bytes(code), builtins


class Unsupported(Exception):
  pass


class Group:
  """Builtins translated as one function: jumps and calls between them are
  gotos (the bytecode handlers' dispatch, above all)."""

  def __init__(self, index, members):
    self.index = index  # its function: xg_<index>
    self.members = members
    self.labels = set()  # offsets that need a label
    self.entries = set()  # offsets the entry switch reaches
    for b in members:
      b.group = self
      self.labels.add(b.start)
      self.entries.add(b.start)

  def contains(self, off):
    return any(b.start <= off < b.end for b in self.members)

  def emit(self):
    out = []
    w = out.append
    ops = []
    for b in self.members:
      for a in range(b.start, b.end, 4):
        try:
          body = b.translate(a, b.word(a))
        except Unsupported:
          body = None
        ops.append((a, b.word(a), body))
      ops.append((b.end, None, None))  # falling off its end
    # the registers and flags it uses (the rest stay in the state)
    text = "".join(body for _, _, body in ops if body)
    calls = "goto call" in text
    regs = sorted({int(n) for n in re.findall(r"\br(\d+)\b", text)} |
                  ({14} if calls else set()))
    written = sorted({int(n) for n in re.findall(r"\br(\d+) = ", text)} |
                     ({14} if calls else set()))
    flags = [f for f in "NZCV" if re.search(r"\b%s\b" % f, text)]
    wflags = [f for f in "NZCV" if re.search(r"\b%s = " % f, text)]
    load = ("  " + " ".join("r%d = (uint32_t)R[%d];" % (n, n) for n in regs) +
            " " + " ".join("%s = *P%s;" % (f, f) for f in flags) + "\n")
    save = ("  " + " ".join("R[%d] = (int32_t)r%d;" % (n, n) for n in written) +
            " " + " ".join("*P%s = %s;" % (f, f) for f in wflags) + "\n")
    for b in self.members:
      w("// %s\n" % b.name)
    w("TAIL uint32_t xg_%d(State* s, uint32_t pc) {\n" % self.index)
    w("  const uint32_t B = (uint32_t)(uintptr_t)v8_Default_embedded_blob_code_;\n")
    # (the state's pointers in locals: V8 compiles without strict aliasing,
    # so through s they'd be reloaded after every store)
    w("  int32_t* const R = s->r;\n"
      "  bool *const PN = s->n, *const PZ = s->z, *const PC = s->c, "
      "*const PV = s->v;\n")
    w("  uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, "
      "r14, N, Z, C, V;\n")
    w(load)
    w("  uint32_t npc, ct, cr;\n")
    w("top:\n  switch (pc - B) {\n")
    for e in sorted(self.entries):
      w("    case 0x%x: goto L%x;\n" % (e, e))
    w("    default: npc = pc; goto fb;\n  }\n")
    for a, word, body in ops:
      if word is None:
        w("  npc = B + 0x%x; goto jmp;\n" % a)
        continue
      if a in self.labels:
        w("L%x:\n" % a)
      if body is None:
        w("  npc = B + 0x%x; goto fb;  // %08x\n" % (a, word))
      else:
        w(body)
    # a computed jump: back to the entry switch if it stays here
    if len(self.members) == 1:
      b = self.members[0]
      inside = "npc - B - 0x%x < 0x%x" % (b.start, b.end - b.start)
      inside_ct = "false"
    else:
      inside = "in_group(npc - B, %d)" % self.index
      inside_ct = "in_group(ct - B, %d)" % self.index
    w("jmp:\n  if (npc == s->ret) goto out;\n"
      "  if (%s) { pc = npc; goto top; }\n" % inside)
    # leaving: v8_arm_xlate_next goes on from npc
    w("out:\n" + save + "  return npc;\n")
    # an instruction for the simulator
    w("fb:\n" + save + "  npc = s->step(s, npc);\n" + load + "  goto jmp;\n")
    # a call (bl, blx; r14 set): a goto in the group; else native, unless
    # too deep
    w("call:\n  if (%s) { pc = ct; goto top; }\n" % inside_ct + save)
    w("  if (s->depth >= kMaxDepth) return ct;\n"
      "  {\n    uint32_t outer = s->ret;\n    s->ret = cr; s->depth++;\n"
      "    uint32_t r = v8_arm_xlate_next(s, ct);\n"
      "    s->ret = outer; s->depth--;\n    if (r != cr) return r;\n  }\n")
    w(load + "  npc = cr; goto jmp;\n}\n\n")
    return "".join(out)


class Builtin:
  """One builtin's translation."""

  def __init__(self, index, name, start, end, code):
    self.index, self.name, self.start, self.end = index, name, start, end
    self.code = code  # the whole blob
    self.group = None

  def word(self, off):
    return int.from_bytes(self.code[off:off + 4], "little")

  def inside(self, off):  # in its group
    return self.group.contains(off)

  # --- the first pass: labels and entries ---------------------------------

  def scan(self):
    for a in range(self.start, self.end, 4):
      i = self.word(a)
      cond = i >> 28
      if i == 0xCCCCCCCC:
        continue  # padding
      try:
        self.translate(a, i, dry=True)
        ok = True
      except Unsupported:
        ok = False
      if not ok:
        self.entry(a + 4)
        continue
      if cond == 0xF:
        continue
      t = bits(i, 27, 25)
      if t == 5:  # b, bl
        target = (a + 8 + (((i & 0xFFFFFF) ^ 0x800000) - 0x800000) * 4)
        if bit(i, 24):
          self.entry(a + 4)
        if self.inside(target):
          self.group.labels.add(target)
      elif (t == 0 and bits(i, 24, 20) == 0x12 and bits(i, 7, 4) == 3):
        self.entry(a + 4)  # blx rm
      elif t == 1 and bits(i, 15, 12) != 15 and bits(i, 19, 16) == 15:
        # rd = pc +/- imm: an address of this code, maybe a return address
        op = bits(i, 24, 21)
        imm = ror32(bits(i, 7, 0), bits(i, 11, 8) * 2)
        if op == 4:  # add
          self.entry(a + 8 + imm)
        elif op == 2:  # sub
          self.entry(a + 8 - imm)
      elif (t == 0 and bits(i, 24, 21) == 4 and bits(i, 19, 16) == 15 and
            bits(i, 15, 12) == 15 and not bit(i, 4)):
        # add pc, pc, rm, lsl #2: a table of branches follows
        k = a + 8
        while self.start <= k < self.end and self.word(k) >> 24 == 0xEA:
          self.entry(k)
          k += 4
      elif (t == 0 and bits(i, 24, 21) == 13 and bits(i, 3, 0) == 15 and
            bits(i, 11, 4) == 0 and bits(i, 15, 12) != 15):
        self.entry(a + 8)  # mov rd, pc

  def entry(self, off):
    if self.inside(off) and off % 4 == 0:
      self.group.entries.add(off)
      self.group.labels.add(off)

  # Translates one instruction, as the simulator decodes it: C++ statements,
  # or Unsupported. dry: just whether it's supported.
  def translate(self, a, i, dry=False):
    self.a = a
    self.jumps = False
    cond = i >> 28
    if cond == 0xF:
      body = self.special(i)
    else:
      t = bits(i, 27, 25)
      body = [self.type01, self.type01, self.type2, self.type3, self.type4,
              self.type5, self.type6, self.type7][t](i)
    if dry:
      return None
    if self.jumps:
      body += " goto jmp;"
    if cond != 0xF and cond != AL:
      body = "if (%s) { %s }" % (COND[cond], body)
    return "  %s  // %08x\n" % (body, i)

  # registers: reading pc is the instruction's address + 8
  def R(self, n):
    return "r%d" % n if n != 15 else "(B + 0x%x)" % (self.a + 8)

  def W(self, n, value):
    if n == 15:
      self.jumps = True
      return "npc = %s;" % value
    return "r%d = %s;" % (n, value)

  # --- data processing, multiplies, extra loads and stores ----------------

  def type01(self, i):
    t = bits(i, 27, 25)
    if t == 0 and bit(i, 7) and bit(i, 4):
      if bits(i, 7, 4) == 9:
        return self.multiply(i)
      return self.extra_load_store(i)
    if t == 0 and bit(i, 24) and not bit(i, 23) and not bit(i, 20) \
        and not bit(i, 7):
      return self.misc(i)
    if t == 1 and bits(i, 24, 8) == 0x120F0:
      if bits(i, 7, 0) in (0, 20):  # nop, csdb
        return ";"
      raise Unsupported
    return self.data_processing(i)

  def shifter(self, i, need_carry):
    """The shifter operand (GetShiftRm, GetImm): (statements, value, carry)."""
    if bits(i, 27, 25) == 1:
      rotate = bits(i, 11, 8) * 2
      imm = ror32(bits(i, 7, 0), rotate)
      carry = "C" if rotate == 0 else str(imm >> 31)
      return "", "0x%xu" % imm, carry
    shift = bits(i, 6, 5)
    rm = self.R(bits(i, 3, 0))
    if bit(i, 4):
      rs = self.R(bits(i, 11, 8))
      return ("uint32_t sc; uint32_t so = shift_reg(%d, %s, %s, C, &sc); " %
              (shift, rm, rs), "so", "sc")
    n = bits(i, 11, 7)
    if shift == 3 and n == 0:
      raise Unsupported  # rrx
    if shift in (1, 2) and n == 0:
      n = 32
    if shift == 0:
      if n == 0:
        return "", rm, "C"
      return "", "(%s << %d)" % (rm, n), "((%s >> %d) & 1)" % (rm, 32 - n)
    if shift == 1:
      if n == 32:
        return "", "0u", "(%s >> 31)" % rm
      return "", "(%s >> %d)" % (rm, n), "((%s >> %d) & 1)" % (rm, n - 1)
    if shift == 2:
      if n == 32:
        return "", "(uint32_t)((int32_t)%s >> 31)" % rm, "(%s >> 31)" % rm
      return ("", "(uint32_t)((int32_t)%s >> %d)" % (rm, n),
              "((%s >> %d) & 1)" % (rm, n - 1))
    return "", "ror(%s, %d)" % (rm, n), "(ror(%s, %d) >> 31)" % (rm, n)

  def data_processing(self, i):
    op = bits(i, 24, 21)
    S = bit(i, 20)
    rd, rn = bits(i, 15, 12), bits(i, 19, 16)
    pre, so, sc = self.shifter(i, S)
    rnv = self.R(rn)

    def nz(res):
      return "N = %s >> 31; Z = %s == 0;" % (res, res)

    logical = {0: "%s & %s", 1: "%s ^ %s", 12: "%s | %s", 14: "%s & ~%s"}
    if op in logical:
      e = logical[op] % (rnv, "(%s)" % so if op == 14 else so)
      s = pre + "uint32_t res = %s; " % e
      if S:
        s += "uint32_t c = %s; " % sc
      s += self.W(rd, "res")
      if S:
        s += " " + nz("res") + " C = c;"
      return "{ %s }" % s
    if op in (13, 15):  # mov, mvn
      e = so if op == 13 else "~%s" % so
      s = pre + "uint32_t res = %s; " % e
      if S:
        s += "uint32_t c = %s; " % sc
      s += self.W(rd, "res")
      if S:
        s += " " + nz("res") + " C = c;"
      return "{ %s }" % s
    if op in (2, 3, 4, 5, 6):  # sub, rsb, add, adc, sbc
      l, r = (rnv, "op") if op != 3 else ("op", rnv)
      s = pre + "uint32_t op = %s; uint32_t l = %s, r = %s; " % (so, l, r)
      if op == 2 or op == 3:
        s += "uint32_t res = l - r; "
        flags = "C = l >= r; V = overflow_sub(res, l, r);"
      elif op == 4:
        s += "uint32_t res = l + r; "
        flags = "C = carry_add(l, r, 0); V = overflow_add(res, l, r);"
      elif op == 5:
        s += "uint32_t res = l + r + C; "
        flags = "C = carry_add(l, r, C); V = overflow_add(res, l, r);"
      else:
        s += "uint32_t res = l - r - (C ? 0 : 1); "
        flags = "C = !borrow_sub(l, r, C); V = overflow_sub(res, l, r);"
      s += self.W(rd, "res")
      if S:
        s += " " + nz("res") + " " + flags
      return "{ %s }" % s
    if op == 7:  # rsc: the simulator does nothing (Format only)
      raise Unsupported
    if op == 8:
      if S:  # tst
        return "{ %suint32_t res = %s & %s; uint32_t c = %s; %s C = c; }" % (
            pre, rnv, so, sc, nz("res"))
      imm = (bits(i, 19, 16) << 12) | bits(i, 11, 0)  # movw
      return self.W(rd, "0x%xu" % imm)
    if op == 9:
      if S:  # teq
        return "{ %suint32_t res = %s ^ %s; uint32_t c = %s; %s C = c; }" % (
            pre, rnv, so, sc, nz("res"))
      raise Unsupported
    if op == 10:
      if S:  # cmp
        return ("{ %suint32_t l = %s, r = %s; uint32_t res = l - r; %s "
                "C = l >= r; V = overflow_sub(res, l, r); }" %
                (pre, rnv, so, nz("res")))
      imm = (bits(i, 19, 16) << 12) | bits(i, 11, 0)  # movt
      return self.W(rd, "(%s & 0xFFFFu) | 0x%xu" % (self.R(rd), imm << 16))
    if op == 11:
      if S:  # cmn
        return ("{ %suint32_t l = %s, r = %s; uint32_t res = l + r; %s "
                "C = carry_add(l, r, 0); V = overflow_add(res, l, r); }" %
                (pre, rnv, so, nz("res")))
      raise Unsupported
    raise Unsupported

  def multiply(self, i):
    rn, rm, rs, rd = bits(i, 19, 16), bits(i, 3, 0), bits(i, 11, 8), \
        bits(i, 15, 12)
    if bit(i, 24):
      raise Unsupported  # ldrex, strex
    if 15 in (rn, rm, rs, rd):
      raise Unsupported
    if not bit(i, 23):
      if not bit(i, 21):  # mul
        s = "{ uint32_t res = r%d * r%d; %s" % (rm, rs, self.W(rn, "res"))
        if bit(i, 20):
          s += " N = res >> 31; Z = res == 0;"
        return s + " }"
      if not bit(i, 22):  # mla
        return self.W(rn, "r%d + r%d * r%d" % (rd, rm, rs))
      return self.W(rn, "r%d - r%d * r%d" % (rd, rm, rs))  # mls
    if bit(i, 20) or bit(i, 21):
      raise Unsupported  # flags, or accumulate: the simulator's own way
    if bit(i, 22):  # smull
      p = "(uint64_t)((int64_t)(int32_t)r%d * (int32_t)r%d)" % (rm, rs)
    else:  # umull
      p = "((uint64_t)r%d * r%d)" % (rm, rs)
    return "{ uint64_t p = %s; r%d = (uint32_t)p; r%d = (uint32_t)(p >> 32); }" \
        % (p, rd, rn)

  def addressing(self, i, rn, offset, post_writes):
    """Statements computing addr, and writing rn back, as the simulator."""
    pu = bits(i, 24, 23)
    W = bit(i, 21)
    rnv = self.R(rn)
    if pu == 0:  # da: post-indexed, down
      return "uint32_t addr = %s; %s " % (rnv, self.W(rn, "%s - %s" % (rnv, offset)))
    if pu == 1:  # ia
      return "uint32_t addr = %s; %s " % (rnv, self.W(rn, "%s + %s" % (rnv, offset)))
    sign = "-" if pu == 2 else "+"
    s = "uint32_t addr = %s %s %s; " % (rnv, sign, offset)
    if W:
      s += self.W(rn, "addr") + " "
    return s

  def extra_load_store(self, i):
    rd, rn = bits(i, 15, 12), bits(i, 19, 16)
    if bit(i, 22):
      offset = "0x%xu" % ((bits(i, 11, 8) << 4) | bits(i, 3, 0))
    else:
      if bits(i, 3, 0) == 15:
        raise Unsupported
      offset = "r%d" % bits(i, 3, 0)
    s = "{ " + self.addressing(i, rn, offset, True)
    L = bit(i, 20)
    if (bits(i, 7, 4) & 0xD) == 0xD and not L:
      if rd % 2 or rd == 14:
        raise Unsupported
      if bit(i, 5):  # strd
        s += "uint32_t v0 = %s, v1 = %s; st32(addr, v0); st32(addr + 4, v1);" % (
            self.R(rd), self.R(rd + 1))
      else:  # ldrd
        s += "uint32_t v0 = ld32(addr), v1 = ld32(addr + 4); r%d = v0; r%d = v1;" % (
            rd, rd + 1)
    elif bit(i, 5):  # halfwords
      if L:
        s += self.W(rd, "%s(addr)" % ("ld16s" if bit(i, 6) else "ld16"))
      else:
        s += "st16(addr, %s);" % self.R(rd)
    else:
      if not L:
        raise Unsupported
      s += self.W(rd, "ld8s(addr)")
    return s + " }"

  def misc(self, i):
    rm = bits(i, 3, 0)
    if bits(i, 22, 21) == 1:
      op = bits(i, 7, 4)
      if op == 1:  # bx
        return self.W(15, self.R(rm))
      if op == 3:  # blx
        return "{ ct = %s; r14 = cr = B + 0x%x; goto call; }" % (
            self.R(rm), self.a + 4)
      raise Unsupported
    if bits(i, 22, 21) == 3 and bits(i, 7, 4) == 1:  # clz
      return self.W(bits(i, 15, 12), "clz(%s)" % self.R(rm))
    raise Unsupported

  # --- loads and stores ---------------------------------------------------

  def load_store(self, i, offset, pre=""):
    rd, rn = bits(i, 15, 12), bits(i, 19, 16)
    L, Bb = bit(i, 20), bit(i, 22)
    pu, W = bits(i, 24, 23), bit(i, 21)
    # a load from the constant pool: the blob is read-only, so its value
    if (rn == 15 and L and not Bb and pu in (2, 3) and not W and
        offset.startswith("0x")):
      off = int(offset[:-1], 16)
      at = self.a + 8 + (off if pu == 3 else -off)
      if 0 <= at and at + 4 <= len(self.code):
        return self.W(rd, "0x%xu" % self.word(at))
    s = "{ " + pre + self.addressing(i, rn, offset, True)
    if L:
      s += self.W(rd, "ld8(addr)" if Bb else "ld32(addr)")
    else:
      s += "%s(addr, %s);" % ("st8" if Bb else "st32", self.R(rd))
    return s + " }"

  def type2(self, i):
    return self.load_store(i, "0x%xu" % bits(i, 11, 0))

  def type3(self, i):
    pu = bits(i, 24, 23)
    rd, rn, rm = bits(i, 15, 12), bits(i, 19, 16), bits(i, 3, 0)
    if pu == 0:
      raise Unsupported
    if pu == 1:
      if not bit(i, 4):
        raise Unsupported  # the simulator's post-indexed load: from 0
      return self.media(i)
    if pu == 2:
      if bits(i, 22, 20) == 5 and bits(i, 7, 4) == 1:
        if 15 in (rm, bits(i, 11, 8), rn):
          raise Unsupported
        if rd == 15:  # smmul
          return self.W(rn, "smmul(r%d, r%d)" % (rm, bits(i, 11, 8)))
        return self.W(rn, "r%d + smmul(r%d, r%d)" % (rd, rm, bits(i, 11, 8)))
      if bits(i, 5, 4) == 1 and not bit(i, 22) and bit(i, 20):
        if 15 in (rm, bits(i, 11, 8), rn):
          raise Unsupported
        f = "udiv" if bit(i, 21) else "sdiv"
        return self.W(rn, "%s(r%d, r%d)" % (f, rm, bits(i, 11, 8)))
    if pu == 3:
      W = bit(i, 21)
      if W and bits(i, 6, 4) == 5:  # ubfx, sbfx
        width1, lsb = bits(i, 20, 16), bits(i, 11, 7)
        msb = width1 + lsb
        if msb > 31 or rd == 15:
          raise Unsupported
        if bit(i, 22):
          return self.W(rd, "(%s << %d) >> %d" % (self.R(rm), 31 - msb, 31 - width1))
        return self.W(rd, "(uint32_t)((int32_t)(%s << %d) >> %d)" % (
            self.R(rm), 31 - msb, 31 - width1))
      if not W and bits(i, 6, 4) == 1:  # bfc, bfi
        lsb, msb = bits(i, 11, 7), bits(i, 20, 16)
        if msb < lsb or rd == 15:
          raise Unsupported
        mask = (0xFFFFFFFF >> (32 - (msb - lsb + 1)))
        e = "(%s & 0x%xu)" % (self.R(rd), ~(mask << lsb) & 0xFFFFFFFF)
        if rm != 15:
          e += " | ((%s & 0x%xu) << %d)" % (self.R(rm), mask, lsb)
        return self.W(rd, e)
    # a load or store, the offset a shifted register (GetShiftRm, which the
    # simulator computes first)
    pre, so, sc = self.shifter(i, False)
    return self.load_store(i, "so", pre + "uint32_t so = %s; " % so
                           if so != "so" else pre)

  def media(self, i):
    rd, rn, rm = bits(i, 15, 12), bits(i, 19, 16), bits(i, 3, 0)
    if 15 in (rd, rm):
      raise Unsupported
    if not bit(i, 5):
      raise Unsupported  # pkhbt, pkhtb, usat
    op = bits(i, 22, 21)
    if bits(i, 9, 6) != 1 or op in (0, 2):
      if op == 1 and bits(i, 27, 16) == 0x6BF and bits(i, 11, 4) == 0xF3:
        return self.W(rd, "__builtin_bswap32(r%d)" % rm)  # rev
      raise Unsupported
    rot = bits(i, 11, 10) * 8
    v = "ror(r%d, %d)" % (rm, rot) if rot else "r%d" % rm
    if op == 1:
      e = "(uint32_t)(int32_t)(int16_t)%s" if bit(i, 20) else \
          "(uint32_t)(int32_t)(int8_t)%s"
    else:
      e = "(%s & 0xFFFFu)" if bit(i, 20) else "(%s & 0xFFu)"
    e = e % v
    if rn != 15:
      e = "r%d + %s" % (rn, e)
    return self.W(rd, e)

  def type4(self, i):
    if bit(i, 22):
      raise Unsupported
    rn, rlist = bits(i, 19, 16), bits(i, 15, 0)
    pu, W, L = bits(i, 24, 23), bit(i, 21), bit(i, 20)
    if pu == 0 or rn == 15 or rlist == 0:
      raise Unsupported
    regs = [r for r in range(16) if rlist >> r & 1]
    n = len(regs)
    start = {1: "base", 2: "base - %d" % (4 * n), 3: "base + 4"}[pu]
    wb = {1: "base + %d" % (4 * n), 2: "base - %d" % (4 * n),
          3: "base + %d" % (4 * n)}[pu]
    s = "{ uint32_t base = r%d; uint32_t addr = %s; " % (rn, start)
    for k, r in enumerate(regs):
      if L:
        s += self.W(r, "ld32(addr + %d)" % (4 * k)) + " "
      else:
        s += "st32(addr + %d, %s); " % (4 * k, self.R(r))
    if W:
      s += "r%d = %s; " % (rn, wb)
    return s + "}"

  def type5(self, i):
    off = ((i & 0xFFFFFF) ^ 0x800000) - 0x800000
    target = self.a + 8 + off * 4
    if not 0 <= target < len(self.code):
      raise Unsupported  # data, not a branch
    if bit(i, 24):  # bl
      if self.inside(target):
        return "{ r14 = B + 0x%x; goto L%x; }" % (self.a + 4, target)
      return "{ ct = B + 0x%x; r14 = cr = B + 0x%x; goto call; }" % (
          target, self.a + 4)
    if self.inside(target):
      return "goto L%x;" % target
    return "{ npc = B + 0x%x; goto out; }" % target

  # --- VFP ----------------------------------------------------------------

  @staticmethod
  def sreg(i, four, one):
    return (bits(i, four + 3, four) << 1) | bit(i, one)

  @staticmethod
  def dreg(i, four, one):
    return (bit(i, one) << 4) | bits(i, four + 3, four)

  def type6(self, i):
    cp, op = bits(i, 11, 8), bits(i, 24, 21)
    if cp not in (0xA, 0xB):
      raise Unsupported
    rn = bits(i, 19, 16)
    double = cp == 0xB
    if double and op == 2:  # vmov rt, rt2 <-> dm
      if bits(i, 7, 6) != 0 or not bit(i, 4):
        raise Unsupported
      rt, rt2, dm = bits(i, 15, 12), rn, self.dreg(i, 0, 5)
      if bit(i, 20):
        if 15 in (rt, rt2):
          raise Unsupported
        return "r%d = s->vfp[%d]; r%d = s->vfp[%d];" % (rt, 2 * dm, rt2, 2 * dm + 1)
      return "{ uint32_t a = %s, b = %s; s->vfp[%d] = a; s->vfp[%d] = b; }" % (
          self.R(rt), self.R(rt2), 2 * dm, 2 * dm + 1)
    if op in (8, 0xA, 0xC, 0xE):  # vldr, vstr
      off = bits(i, 7, 0) * 4
      if not bit(i, 23):
        off = -off
      d = self.dreg(i, 12, 22) if double else self.sreg(i, 12, 22)
      s = "{ uint32_t addr = %s + (uint32_t)%d; " % (self.R(rn), off)
      words = [2 * d, 2 * d + 1] if double else [d]
      for k, x in enumerate(words):
        if bit(i, 20):
          s += "s->vfp[%d] = ld32(addr + %d); " % (x, 4 * k)
        else:
          s += "st32(addr + %d, s->vfp[%d]); " % (4 * k, x)
      return s + "}"
    if op in (4, 5, 6, 7, 9, 0xB):  # vldm, vstm (HandleVList)
      pu, W, L = bits(i, 24, 23), bit(i, 21), bit(i, 20)
      if pu == 0 or rn == 15:
        raise Unsupported
      if double:
        d, n = self.dreg(i, 12, 22), bits(i, 7, 0) // 2
        words = [w for r in range(d, d + n) for w in (2 * r, 2 * r + 1)]
        size = 8 * n
      else:
        d, n = self.sreg(i, 12, 22), bits(i, 7, 0)
        words = list(range(d, d + n))
        size = 4 * n
      if any(x >= 64 for x in words):
        raise Unsupported
      start = {1: "base", 2: "base - %d" % size, 3: "base + %d" % (size // n if n else 0)}[pu]
      wb = {1: "base + %d", 2: "base - %d", 3: "base + %d"}[pu] % size
      s = "{ uint32_t base = r%d; uint32_t addr = %s; " % (rn, start)
      for k, x in enumerate(words):
        if L:
          s += "s->vfp[%d] = ld32(addr + %d); " % (x, 4 * k)
        else:
          s += "st32(addr + %d, s->vfp[%d]); " % (4 * k, x)
      if W:
        s += "r%d = %s; " % (rn, wb)
      return s + "}"
    raise Unsupported

  def type7(self, i):
    if bit(i, 24):
      raise Unsupported  # svc: calls into C++, stops
    cp = bits(i, 11, 8)
    if cp not in (10, 11):
      raise Unsupported
    return self.vfp(i)

  def vfp(self, i):
    m, d, n = self.sreg(i, 0, 5), self.sreg(i, 12, 22), self.sreg(i, 16, 7)
    vm, vd, vn = self.dreg(i, 0, 5), self.dreg(i, 12, 22), self.dreg(i, 16, 7)
    sz = bit(i, 8)
    opc1 = (bit(i, 23) << 2) | bits(i, 21, 20)
    opc2, opc3 = bits(i, 19, 16), bits(i, 7, 6)

    def un(expr_d, expr_f):  # dd = f(dm), or sd = f(sm)
      if sz:
        return "{ double x = getd(s, %d); setd(s, %d, %s); }" % (vm, vd, expr_d)
      return "{ float x = getf(s, %d); setf(s, %d, %s); }" % (m, d, expr_f)

    def bin_(op, extra=""):
      if sz:
        return ("{ double a = getd(s, %d), b = getd(s, %d); %s"
                "setd(s, %d, canon(s, a %s b)); }" % (vn, vm, extra, vd, op))
      return ("{ float a = getf(s, %d), b = getf(s, %d); %s"
              "setf(s, %d, canonf(s, a %s b)); }" % (n, m, extra, d, op))

    if bit(i, 4) == 0:
      if opc1 == 7:
        if opc2 == 0 and opc3 == 1:  # vmov
          if sz:
            return "s->vfp[%d] = s->vfp[%d]; s->vfp[%d] = s->vfp[%d];" % (
                2 * vd, 2 * vm, 2 * vd + 1, 2 * vm + 1) if vd != vm else ";"
          return "s->vfp[%d] = s->vfp[%d];" % (d, m)
        if opc2 == 0 and opc3 == 3:  # vabs
          return un("canon(s, std::fabs(x))", "canonf(s, std::fabs(x))")
        if opc2 == 1 and opc3 == 1:  # vneg
          return un("canon(s, -x)", "canonf(s, -x)")
        if opc2 == 7 and opc3 == 3:  # vcvt between double and single
          if sz:
            return "setf(s, %d, (float)getd(s, %d));" % (d, vm)
          return "setd(s, %d, (double)getf(s, %d));" % (vd, m)
        if (opc2 == 8 or (opc2 >> 1) == 6) and (opc3 & 1):
          return self.vcvt_int(i)
        if opc2 == 1 and opc3 == 3:  # vsqrt
          return un("canon(s, std::sqrt(x))", "canonf(s, std::sqrt(x))")
        if opc2 in (4, 5) and (opc3 & 1):  # vcmp, vcmpe
          if sz:
            a, b = "getd(s, %d)" % vd, ("getd(s, %d)" % vm if opc2 == 4 else "0.0")
            s = "{ double a = %s, b = %s; " % (a, b)
          else:
            a, b = "getf(s, %d)" % d, ("getf(s, %d)" % m if opc2 == 4 else "0.0f")
            s = "{ float a = %s, b = %s; " % (a, b)
          if bit(i, 7):
            s += "if (std::isnan(a)) *s->inv_op = true; "
          return s + "fp_compare(s, a, b); }"
        if opc3 == 0:  # vmov immediate
          high16 = (bits(i, 17, 16) << 4) | bits(i, 3, 0)
          high16 |= (0xFF * bit(i, 18)) << 6
          high16 |= (bit(i, 18) ^ 1) << 14
          high16 |= bit(i, 19) << 15
          imm = high16 << 48
          if sz:
            return "s->vfp[%d] = 0u; s->vfp[%d] = 0x%xu;" % (2 * vd, 2 * vd + 1, imm >> 32)
          return "{ uint64_t b = 0x%xull; double x; memcpy(&x, &b, 8); setf(s, %d, (float)x); }" % (imm, d)
        if opc2 == 6 and opc3 == 3:  # vrintz
          return un("canon(s, std::trunc(x))", "canonf(s, std::trunc(x))")
        raise Unsupported
      if opc1 == 3:
        return bin_("-" if opc3 & 1 else "+", "")
      if opc1 == 2 and not (opc3 & 1):
        return bin_("*", "")
      if opc1 == 4 and not (opc3 & 1):
        return bin_("/", "*s->div_zero = (b == 0); ")
      raise Unsupported  # vmla, vmls: rare
    # bit 4: transfers
    vc, va, vl = bit(i, 8), bits(i, 23, 21), bit(i, 20)
    rt = bits(i, 15, 12)
    if vc == 0 and va == 0:  # vmov between core and single
      if vl:
        if rt == 15:
          raise Unsupported
        return "r%d = s->vfp[%d];" % (rt, n)
      return "s->vfp[%d] = %s;" % (n, self.R(rt))
    if vl == 0 and vc == 1:
      if bit(i, 23) or rt == 15:
        raise Unsupported
      opc = (bits(i, 22, 21) << 2) | bits(i, 6, 5)
      if opc & 0xB:
        raise Unsupported
      return "s->vfp[%d] = r%d;" % (2 * vn + bit(i, 21), rt)
    if vl == 1 and vc == 1:
      opc = (bits(i, 22, 21) << 2) | bits(i, 6, 5)
      if opc & 0xB or rt == 15:
        raise Unsupported
      return "r%d = s->vfp[%d];" % (rt, 2 * vn + bit(i, 21))
    if vl == 1 and vc == 0 and va == 7 and bits(i, 19, 16) == 1:  # vmrs
      if rt == 15:
        return "N = *s->fn; Z = *s->fz; C = *s->fc; V = *s->fv;"
      return "r%d = get_fpscr(s);" % rt
    if vl == 0 and vc == 0 and va == 7 and bits(i, 19, 16) == 1:  # vmsr
      if rt == 15:
        raise Unsupported
      return "set_fpscr(s, r%d);" % rt
    raise Unsupported

  def vcvt_int(self, i):
    sz = bit(i, 8)
    if bit(i, 18):  # to an integer
      d = self.sreg(i, 12, 22)
      src = "getd(s, %d)" % self.dreg(i, 0, 5) if sz else \
          "(double)getf(s, %d)" % self.sreg(i, 0, 5)
      return "s->vfp[%d] = fp_to_int(s, %s, %s, %s);" % (
          d, src, "true" if not bit(i, 16) else "false",
          "true" if bit(i, 7) else "false")
    if bits(i, 19, 16) != 8:
      raise Unsupported
    m = self.sreg(i, 0, 5)
    v = "s->vfp[%d]" % m if not bit(i, 7) else "(int32_t)s->vfp[%d]" % m
    if sz:
      return "setd(s, %d, (double)%s);" % (self.dreg(i, 12, 22), v)
    return "setf(s, %d, (float)%s);" % (self.sreg(i, 12, 22), v)

  # --- the special condition: barriers ------------------------------------

  def special(self, i):
    if bit(i, 27) == 0 and bits(i, 26, 25) & 2 and bit(i, 20):
      sv = bits(i, 27, 23)
      if sv in (0xA, 0xB):
        if bits(i, 22, 20) == 5 and bits(i, 15, 12) == 0xF:
          return ";"  # pld
        if sv == 0xA and bits(i, 22, 20) == 7:
          return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"  # dmb, dsb, isb
    raise Unsupported


NEXT = """
TAIL uint32_t v8_arm_xlate_next(State* s, uint32_t pc) {
  const uint32_t B = (uint32_t)(uintptr_t)v8_Default_embedded_blob_code_;
  for (;;) {
    if (pc == s->ret) return pc;
    uint32_t offset = pc - B;
    if (offset < v8_arm_xlate_blob_size) {
      Function f = s->functions[v8_arm_xlate_index[offset >> 5]];
      if (f) {
        pc = f(s, pc);
        continue;
      }
    } else if (pc >= 0xFFFFFFFEu) {
      return pc;  // the simulator's end (end_sim_pc), or bad_lr
    }
    pc = s->step(s, pc);
  }
}
"""


def main():
  src, prefix, shards = sys.argv[1], sys.argv[2], int(sys.argv[3])
  code, named = parse_embedded(src)
  builtins = []
  for k, (name, start) in enumerate(named):
    end = named[k + 1][1] if k + 1 < len(named) else len(code)
    builtins.append(Builtin(k, name, start, end, code))
  header = ('// Autogenerated by tools/arm-to-c.py. Do not edit.\n'
            '#include "src/execution/arm/simulator-arm-xlate.h"\n'
            '#pragma clang diagnostic ignored "-Wunused-label"\n'
            '#pragma clang diagnostic ignored "-Wunused-variable"\n'
            '#pragma clang diagnostic ignored "-Wunused-but-set-variable"\n'
            '#pragma clang diagnostic ignored "-Wsometimes-uninitialized"\n'
            '#define TAIL extern "C"\n'
            'namespace v8_arm_xlate {\n')
  # each builtin alone (as one function, the bytecode handlers ran twice
  # as slow: too big a function, for clang and for the wasm engine both)
  groups = [Group(k, [b]) for k, b in enumerate(builtins)]
  # shards of about equal size
  per = (len(code) + shards - 1) // shards
  files = [[] for _ in range(shards)]
  for g in groups:
    files[min(g.members[0].start // per, shards - 1)].append(g)
  for k, gs in enumerate(files):
    with open("%s-%d.cc" % (prefix, k), "w") as f:
      f.write(header)
      for g in gs:
        for b in g.members:
          b.scan()
        f.write(g.emit())
      f.write("}  // namespace v8_arm_xlate\n")
  with open("%s-table.cc" % prefix, "w") as f:
    f.write(header)
    for g in groups:
      f.write("TAIL uint32_t xg_%d(State*, uint32_t);\n" % g.index)
    f.write('extern "C" const Builtin v8_arm_xlate_builtins[] = {\n')
    for b in builtins:
      f.write('  {0x%x, 0x%x, xg_%d, "%s"},\n' % (b.start, b.end - b.start,
                                                 b.group.index, b.name))
    f.write("};\n")
    f.write('extern "C" const uint16_t v8_arm_xlate_group[] = {\n')
    for k in range(0, len(builtins), 16):
      f.write("  %s,\n" % ", ".join(str(b.group.index)
                                    for b in builtins[k:k + 16]))
    f.write("};\n")
    f.write('extern "C" const uint32_t v8_arm_xlate_builtin_count = %d;\n' %
            len(builtins))
    f.write('extern "C" const uint32_t v8_arm_xlate_blob_size = 0x%x;\n' %
            len(code))
    index = []
    for b in builtins:
      assert b.start % 32 == 0 and b.end % 32 == 0, b.name
      index += [b.index] * ((b.end - b.start) // 32)
    f.write('extern "C" const uint16_t v8_arm_xlate_index[] = {\n')
    for k in range(0, len(index), 16):
      f.write("  %s,\n" % ", ".join(str(x) for x in index[k:k + 16]))
    f.write("};\n")
    f.write(NEXT)
    f.write("}  // namespace v8_arm_xlate\n")


if __name__ == "__main__":
  main()
