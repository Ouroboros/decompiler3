# ED9 Script VM Spec

Runtime semantics of the Kuro no Kiseki script interpreter, reverse engineered from `ed9_crack.exe`:

| Function | Address |
| --- | --- |
| `VMContext::execute` | `0x1403BD6B0` |
| `VMContext::Return` | `0x1403BE900` |
| `VMContext::Equal` | `0x1403BEA30` |
| `VMContext::NotEqual` | `0x1403BEB70` |
| `VMContext::CallModuleNoReturn` | `0x1403BECB0` |

Everything below is taken from these functions unless marked *convention* (observed in compiled scripts,
not enforced by the VM). "Halt" means `flags = 0`: the thread stops and is treated as finished.

The spec was written from the Hex-Rays pseudocode of these functions. The low-level details were then
checked against the disassembly: sign extension (`shl 2` / `sar 2`), tag stores (`bts 1Eh` / `bts 1Fh`),
float stores (`shr 2`), integer-to-float conversion (`cvtdq2ps`), float comparisons (`comiss` with
`setnbe` / `setnb`, `ucomiss` with `jp`), division (`cdq; idiv` without a zero check), branch tests
(`test 3FFFFFFFh`) and the `PUSH` instruction length. Float results assume the default MXCSR (round to
nearest, no flush-to-zero), which cannot be confirmed statically.

## 1. Execution Context

| Field | Meaning |
| --- | --- |
| `pc` | Byte offset of the next instruction in the script image |
| `script` | Current script: image bytes, image size, header, global table |
| `stack`, `stackTop`, `stackBottom` | Byte buffer; `stackTop` is the offset of the first free byte, `stackBottom` the push limit |
| `regs[16]` | 32-bit words, not cleared between calls |
| `funcIndex`, `funcName` | Current function (diagnostics) |
| `lineno` | Set by `DEBUG_SET_LINENO` |
| `flags` | See 1.1 |
| `syscall_cmd1/2/3` | Operands of the last `SYSCALL` |
| vtable slot 4 | Syscall handler `(ctx, cmd1, cmd2)` |
| vtable slot 5 | Module call `(ctx, module_name, func_name[, argc])`, returns success |
| vtable slot 6 | Thread finished |

### 1.1 Flags

| Bit | Meaning |
| --- | --- |
| `0x1` | Runnable; `execute` does nothing otherwise |
| `0x2` | Pause request, checked before every instruction |
| `0x4` | Blocking syscall in progress |
| `0x8` | Module call in progress (set by `CALL_SCRIPT`, cleared when it fails) |

### 1.2 Run Loop

```
execute(ctx, dt):
    if not (flags & 0x1): return
    loop:
        if pc >= image size: flags = 0; leave
        if flags & 0x2: leave
        if flags & 0x4:
            call syscall handler again with the saved cmd1, cmd2
            if flags & 0x4: leave
        opcode = image[pc]; pc += 1; execute instruction
        if flags & 0x4 or flags == 0: leave
    leave:
        if flags & 0x1: return              # suspended, resumes at pc
        vtable[6](ctx)                      # thread finished
```

An unknown opcode halts and returns at once, without calling vtable slot 6.

## 2. Values

Every stack slot, register and global holds a 32-bit word. The top two bits are a type tag, the low 30
bits the payload.

| Tag | Type | Word | Value |
| --- | --- | --- | --- |
| 0 | Raw | `payload` | Payload, read as a signed 30-bit integer when used as one |
| 1 | Integer | `0x40000000 \| (v & 0x3FFFFFFF)` | `(int32)(word << 2) >> 2`, range `[-2^29, 2^29 - 1]` |
| 2 | Float | `0x80000000 \| (float_bits >> 2)` | float32 with bits `word << 2` |
| 3 | String | `0xC0000000 \| offset` | NUL-terminated string at `image + offset` |

- Integer results are stored as `(r & 0x3FFFFFFF) | 0x40000000`: they wrap modulo `2^30`.
- Float results are stored as `(bits >> 2) | 0x80000000`: the two lowest bits of the float32 pattern are
  dropped (truncation, not rounding). A float carries 21 explicit mantissa bits.
- Comparison and logical results are Integers 0 or 1.
- Special words: `0xF0000000` marks a module-call frame (a tag-3 word), `0x0FFFFFFF` marks the bottom of
  a thread started by `CALL_SCRIPT_NO_RETURN`.
- A stack address is an Integer holding an absolute byte offset into the stack buffer (6).

## 3. Stack

- The stack grows upward. Every push writes 4 bytes, except the 8-byte script pointer of
  `PUSH_CALLER_FRAME`. A push halts unless `stackTop + size <= stackBottom`. Pops subtract from `stackTop`
  without an underflow check.
- Instruction offsets are signed byte offsets relative to `stackTop` (negative is below the top). An
  access halts unless `0 <= stackTop + offset <= stackBottom`.
- Binary operators pop `b` (top), then `a`, and push `a OP b`.

## 4. Instruction Set

An instruction is an opcode byte followed by little-endian operands. `word` is a tagged 32-bit value.

| Op | Mnemonic | Operands | Size | Stack | Effect |
| --- | --- | --- | --- | --- | --- |
| `00` | `PUSH` | size:u8, value:word | 6 | → v | Push `value`. `size` is always 4; `pc` advances by 6 regardless |
| `01` | `POP` | bytes:u8 | 2 | drop | `stackTop -= bytes` |
| `02` | `LOAD_STACK` | off:i32 | 5 | → v | Push the word at `stackTop + off` |
| `03` | `LOAD_STACK_DEREF` | off:i32 | 5 | → v | Push the word a stack address points to (6) |
| `04` | `PUSH_STACK_OFFSET` | off:i32 | 5 | → int | Push Integer `stackTop + off` |
| `05` | `POP_TO` | off:i32 | 5 | v → | Pop `v`, store it at `stackTop + off` (after the pop) |
| `06` | `POP_TO_DEREF` | off:i32 | 5 | v → | Pop `v`, store it where a stack address points (6) |
| `07` | `LOAD_GLOBAL` | index:u32 | 5 | → v | Push `globals[index]`; Integer 0 if `index >= global count` |
| `08` | `SET_GLOBAL` | index:u32 | 5 | v → | `globals[index] = v`; ignored if out of range |
| `09` | `GET_REG` | reg:u8 | 2 | → v | Push `regs[reg]`; halt if `reg >= 16` |
| `0A` | `SET_REG` | reg:u8 | 2 | v → | Pop into `regs[reg]`; halt if `reg >= 16` |
| `0B` | `JMP` | target:u32 | 5 | | `pc = target` (absolute image offset) |
| `0C` | `CALL` | func:u16 | 3 | | Call a function of this script (5.1) |
| `0D` | `RETURN` | | 1 | | Return (5.2) |
| `0E` | `POP_JMP_NOT_ZERO` | target:u32 | 5 | c → | Jump if the payload of `c` is nonzero (7.6) |
| `0F` | `POP_JMP_ZERO` | target:u32 | 5 | c → | Jump if the payload of `c` is zero (7.6) |
| `10`–`13` | `ADD` `SUB` `MUL` `DIV` | | 1 | a b → r | Arithmetic (7.1) |
| `14` | `MOD` | | 1 | a b → r | Integer remainder (7.2) |
| `15` `16` | `EQ` `NE` | | 1 | a b → int | Equality (7.3) |
| `17`–`1A` | `GT` `GE` `LT` `LE` | | 1 | a b → int | Ordering (7.3) |
| `1B` `1C` | `BITWISE_AND` `BITWISE_OR` | | 1 | a b → r | Bitwise (7.2) |
| `1D` `1E` | `LOGICAL_AND` `LOGICAL_OR` | | 1 | a b → int | Logical (7.4) |
| `1F` | `NEG` | | 1 | a → r | Negation (7.5) |
| `20` | `EZ` | | 1 | a → r | `a == 0` for Integers (7.5) |
| `21` | `NOT` | | 1 | a → r | 30-bit complement for Integers (7.5) |
| `22` | `CALL_SCRIPT` | module:word, func:word, argc:u8 | 10 | | Call another script (5.3) |
| `23` | `CALL_SCRIPT_NO_RETURN` | module:word, func:word, argc:u8 | 10 | | Replace the thread with another script (5.4) |
| `24` | `SYSCALL` | cmd1:u8, cmd2:u8, argc:u8 | 4 | | Engine call (5.5) |
| `25` | `PUSH_CALLER_FRAME` | retaddr:u32 | 5 | → frame | Push `funcIndex`, `retaddr`, script pointer (8 bytes), `0xF0000000` |
| `26` | `DEBUG_SET_LINENO` | line:u16 | 3 | | `lineno = line` |
| `27` | `POPN` | count:u8 | 2 | drop | `stackTop -= 4 * count` |
| `28` | `DEBUG_LOG` | value:i32 | 5 | | Log `debug : %d` |

The disassembler shows `PUSH` as `PUSH_INT`, `PUSH_FLOAT`, `PUSH_STR` or `PUSH_RAW` by tag, and as
`PUSH_CURRENT_FUNC_ID` / `PUSH_RET_ADDR` inside call sequences. These are not separate opcodes.

## 5. Calls

### 5.1 CALL

*Convention* for the caller:

```
PUSH raw <caller func index>      ; PUSH_CURRENT_FUNC_ID
PUSH raw <return address>         ; PUSH_RET_ADDR: offset of the instruction after CALL
PUSH argN ... PUSH arg1           ; parameter 1 is pushed last
CALL func
```

`CALL` halts if `func` is not below the function count, sets `pc` to the function's code offset and,
unless the function entry has `is_common_func` set, updates `funcIndex` / `funcName`. Nothing is pushed.

Inside the callee parameter `k` (1-based) is the word at `entry stackTop - 4 * k`. *Convention*: the
callee pops its locals and parameters before `RETURN`, leaving the return address on top, and passes its
result in `regs[0]` (`SET_REG 0`). The parameter order was confirmed on 404 game scripts: with parameter 1
pushed last, the declared parameter types match all 1,265,302 typed call-site arguments.

### 5.2 RETURN

```
w = pop
if w == 0xF0000000:                   # frame of PUSH_CALLER_FRAME
    script = pop 8 bytes; pc = pop; funcIndex = pop
    funcName = name of funcIndex in the restored script (halt if funcIndex >= function count)
elif w == 0x0FFFFFFF:                 # bottom of a CALL_SCRIPT_NO_RETURN thread
    halt; vtable[6](ctx)
else:
    pc = w; halt if w > image size
    f = pop; halt if f >= function count
    if not is_common_func(f): funcIndex = f; funcName = name of f
```

### 5.3 CALL_SCRIPT

```
PUSH_CALLER_FRAME <return address>
PUSH argN ... PUSH arg1
CALL_SCRIPT module, func, argc
```

`module` and `func` are String words naming the target script and function. The VM sets flag `0x8` and
calls vtable slot 5. On success the engine continues in the target function, whose `RETURN` finds
`0xF0000000` and restores the caller's script, `pc` and `funcIndex`. On failure the VM clears flag `0x8`,
pops `4 * argc` bytes and executes `RETURN`, resuming at the return address with `regs[0]` unchanged.

### 5.4 CALL_SCRIPT_NO_RETURN

Pops `argc` arguments (the first 32 are kept; extras are dropped, and with more than 32 the re-push reads
past a 32-entry buffer), sets `stackTop = 0`, pushes `0x0FFFFFFF`, pushes the kept arguments back in their
original order and calls vtable slot 5 with `argc`. All caller frames are discarded: when the target
returns, `RETURN` finds `0x0FFFFFFF` and the thread ends. On failure the arguments are popped and `RETURN`
ends the thread.

### 5.5 SYSCALL

Stores `cmd1`, `cmd2`, `argc` and calls vtable slot 4 with `cmd1`, `cmd2`. The handler reads `argc`
arguments in place from the top of the stack (argument 1 is the top word). *Convention*: the handler
leaves its result in `regs[0]` and the script pops the arguments afterwards. A handler that has to wait
sets flag `0x4`; `execute` then calls it again before every instruction fetch until the flag is cleared.

## 6. Stack Addresses

- `PUSH_STACK_OFFSET off` pushes Integer `stackTop + off`, the absolute byte offset of a stack slot.
  Callees receive such Integers as output parameters (`f(&a, &b)`).
- `LOAD_STACK_DEREF off` / `POP_TO_DEREF off` read the word at `stackTop + off` (after the pop for
  `POP_TO_DEREF`), take its signed payload `p`, then:
  1. halt unless `0 <= stackTop + p <= stackBottom`;
  2. if `p & 0x0F000000`: global reference with index `p & 0xF0FFFFFF` (reads give Integer 0 and writes
     are ignored when out of range);
  3. otherwise read or write the word at `stack + p`.
- Step 1 adds `stackTop` to an absolute offset. A global reference (`0x0F000000` and up) always fails it,
  so step 2 is unreachable, and a valid stack address fails it when the stack is deep enough.

## 7. Operators

Notation: `a` is the left operand (pushed first), `b` the right operand (top of stack).
`int(x)` is the signed 30-bit payload of `x`. `f(x)` is the float operand: an Integer is converted to
float32 with `cvtdq2ps` (rounds to nearest), any other word is reinterpreted as float32 bits `word << 2`. `trunc(r)`
stores a float result with its two lowest bits dropped. `wrap(r)` reduces an integer modulo `2^30`.
"Logs an error" means the message *script: operating on values of incompatible types*
(`script_value.cpp`).

### 7.1 ADD, SUB, MUL, DIV

- If `a` or `b` is a Float: `trunc(f(a) OP f(b))`, computed in float32. Division by `0.0` gives an
  infinity or NaN.
- Otherwise, in any mix of Integer, Raw and String: Integer `wrap(int(a) OP int(b))`. Division truncates
  toward zero. Integer division by zero is not checked: the host faults and the game crashes.

### 7.2 MOD, BITWISE_AND, BITWISE_OR

- If neither operand is an Integer: logs an error and pushes `a` unchanged.
- Otherwise: Integer `wrap(int(a) % int(b))` with the C remainder (sign of `a`; division by zero is not
  checked), or the bitwise AND / OR of both payloads. A Float operand contributes its payload bits.
  There is no float remainder.

### 7.3 Comparisons

- `EQ` / `NE`: two Strings compare by content (`strcmp`); else, if either is a Float, `f(a)` and `f(b)`
  compare as floats; else the payloads compare regardless of tag (Raw 0 equals Integer 0, a String equals
  the Integer whose value is its offset).
- `GT` / `GE` / `LT` / `LE`: if either is a Float, `f(a)` and `f(b)` compare as floats (false for NaN);
  else `int(a)` and `int(b)` compare as signed integers. Strings compare by offset, not by content.

### 7.4 LOGICAL_AND, LOGICAL_OR

Both operands are already evaluated (no short circuit). If either operand is a Float, an operand is true
when `f(x) != 0.0` (so `-0.0` is false and NaN true); otherwise when its payload is nonzero.

### 7.5 NEG, EZ, NOT

| Operand | `NEG` | `EZ` | `NOT` |
| --- | --- | --- | --- |
| Integer | `wrap(-int(a))` | Integer `int(a) == 0` | `wrap(~int(a))` |
| Float | `trunc(-f(a))` | logs an error, `a` unchanged | logs an error, `a` unchanged |
| Raw, String | logs an error, `a` unchanged | logs an error, `a` unchanged | logs an error, `a` unchanged |

### 7.6 Branches

`POP_JMP_ZERO` / `POP_JMP_NOT_ZERO` test `word & 0x3FFFFFFF`. Zero means Integer 0, Raw 0 or Float `+0.0`.
Float `-0.0` (payload `0x20000000`) and every String are nonzero.

## 8. Registers and Globals

- 16 registers. *Convention*: `regs[0]` carries return values; the compiler also uses it as the
  temporary of switch dispatch (`SET_REG 0` followed by `GET_REG 0` comparisons).
- The global table of a script has `global count` words shared by all its functions.

## 9. Script Image Fields Used by the VM

| Structure | Offset | Field |
| --- | --- | --- |
| Header | `0x04` | Function entry table offset |
| Header | `0x08` | Function count |
| Header | `0x0C` | Global table offset |
| Header | `0x10` | Global count |
| Function entry (`0x20` bytes) | `0x00` | Code offset |
| Function entry | `0x04` | Parameter count |
| Function entry | `0x05` | `is_common_func`: `CALL` / `RETURN` leave `funcIndex` and `funcName` unchanged |
| Function entry | `0x1C` | Name (String word) |

## 10. Consequences for Decompilation

- `POP_JMP_ZERO c` is `payload(c) == 0`, which differs from `c == 0` for Float `-0.0`.
- `EZ c` is `c == 0` only when `c` is an Integer; on a Float, Raw or String it is a no-op. `!f()` on a
  callee that returns Raw 0 (`PUSH_RAW 0; SET_REG 0`) evaluates to Raw 0, which is false.
- Raw words behave like Integers except in `NEG`, `EZ`, `NOT` and in `MOD` / `BITWISE_AND` /
  `BITWISE_OR` with two non-Integer operands. Local slots are commonly reserved with `PUSH_RAW 0`.
- Constant folding must wrap integers to 30 bits, truncate division and remainder toward zero, run float
  operations in float32 and drop the two lowest bits of every float result. `ED9ConstantFolder`
  (`falcom/ed9/constant_folder.py`) folds through `falcom/ed9/vm_values.py`.
- TypeScript numbers are doubles and its integers are not 30-bit, so emitted code only matches the VM
  where values stay in range.
