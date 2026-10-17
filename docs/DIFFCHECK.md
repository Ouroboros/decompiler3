# Differential Checker (`tools/diffcheck`)

Checks that decompiled HLIL behaves like the original ED9 bytecode, function by function.
It replaces the old structural validator (`tools/ir_semantic_validator.py`).

```bash
python -m tools.diffcheck tests/debug.dat
python -m tools.diffcheck tests/debug.dat --function FC_Event_PartySet --verbose
python -m tools.diffcheck tests/*.dat --json result.json
```

Exit code is 0 only when every function passes.

## Equivalence Contract

Two executions are equivalent when they produce the same **observable trace** for the same inputs.

**VM semantics (baseline).** `REGS[n]` and `GLOBALS[n]` are shared state that any instruction and any
callee may read or write.

**Calling convention of the stock scripts (assumed).**

- `REGS[0]` carries a callee's return value back to the caller.
- The compiler also uses `REGS[0]` as a function-local temporary for switch dispatch
  (`SET_REG(0)` followed by `GET_REG(0)` comparisons).
- Consequently no function reads `REGS` on entry, and callees do not read values the caller left in `REGS`.

**Observable trace.**

| Event | Compared fields |
| --- | --- |
| `call` | callee name, arguments, written `GLOBALS` at call time |
| `extern` | module, function, arguments, written `GLOBALS` at call time |
| `syscall` | subsystem, command, arguments, written `GLOBALS` at call time |
| `return` | return value, written `GLOBALS`, values written through pointer parameters |
| `fault` | fault kind (division by zero, undefined value in a condition, ...) |

`REGS` contents are not compared. Arguments that are pointers are compared by the value they point to.
Global writes between two calls may be reordered freely; only their state at the next call or return matters.

**Inputs.** Parameters, initial `GLOBALS`, initial `REGS[1..15]` (the VM never clears registers, so a read
before any write sees the caller's value), every call's return value, values written through pointer
arguments, and random `GLOBALS` / `REGS` mutations after calls are derived from `(seed, origin)`, so both
sides see the same world. Mutating them after calls exposes passes that reuse a global or register value
across a call; only registers the function reads are mutated.
Inputs are integers by default; an input becomes a float only when the function compares it with a float
literal. When a run reaches a logged VM type error (`EZ` / `NOT` / `NEG` / `MOD` / `BITWISE_AND` /
`BITWISE_OR` on unsupported operands), the bytecode side records a `type_error` fault and only the events
before it are compared: the rest is outside the contract (`falcom/ed9/spec/vm_spec.md` section 10).

## Reference Interpreter (bytecode)

Executes disassembled instructions directly; the lifter is not involved.

- Parameter `k` of an `n`-parameter function is the stack slot `n - k` counted from the deepest
  parameter slot, i.e. parameter 1 is pushed last. Evidence: `chr_set_talk_event(Value32, str)` is called
  as `PUSH_STR`, `PUSH_INT`, `CALL`; `item_add` declares a default of 1 for its second parameter, and
  callers push that count first.
- `CALL` only jumps; the callee pops its arguments and `RETURN` pops return address and func id.
  `CALL_SCRIPT` frames hold five slots (func id, return address, two-slot script pointer, script name
  marker `0xF0000000`). `SYSCALL` reads its arguments in place. Every external call leaves its result in
  `REGS[0]`.
- Binary operators compute `below OP top`.
- `LOAD_STACK_DEREF` / `POP_TO_DEREF` on a slot holding a parameter treat that parameter as a pointer into
  caller memory: reads come from the input plan, writes are reported in the `return` event.

## VM Value Semantics

[falcom/ed9/vm_values.py](../falcom/ed9/vm_values.py) implements the value semantics of
[falcom/ed9/spec/vm_spec.md](../falcom/ed9/spec/vm_spec.md) (30-bit integers, floats with two dropped bits,
type-tagged operators, payload-based branches), pinned by `tests/test_diffcheck_values.py`. The decompiler's
constant folding uses the same module. Checker-specific choices:

- Raw words are kept as a separate type; traces observe them by payload, because script output cannot
  tell them from integers. Floats are observed by bit pattern.
- `LOAD_GLOBAL` beyond the script's global table reads integer 0 and `SET_GLOBAL` there is ignored, on both
  sides.
- Integer division or remainder by zero is reported as a `div_by_zero` fault (the game crashes).

Not modeled: string offsets (arithmetic or ordering on strings faults on both sides), arithmetic on stack
addresses, global references through `*_DEREF` (the VM's bounds check halts before reaching that path),
raw-tagged return values from callees, and the `CALL_SCRIPT` failure path that leaves `REGS[0]` unchanged.

## HLIL Interpreter

Evaluates operators with the VM value semantics of the opcodes they were lifted from, and control flow as
the generated TypeScript does: `if`/`while` conditions and `!` test the 30-bit payload like `POP_JMP_ZERO`,
`&&`/`||` short-circuit, test operands like `LOGICAL_AND`/`LOGICAL_OR` and yield 1/0, call arguments are
evaluated left to right, switch cases never fall through and an unlabeled `break` inside a case leaves
the switch, labeled `break`/`continue` target the labeled loop, `return` without a value returns `REGS[0]`.

## TypeScript Check

The TypeScript emitted for the HLIL is parsed back into HLIL (`ts_parse.py`, which accepts only the subset
the generator writes and rejects undeclared names) and executed by the same interpreter on every run. A run
whose HLIL matches the bytecode but whose TypeScript does not fails with stage `TypeScript`; text that does
not parse is an `ERROR`. This covers what the generator adds on top of the HLIL: operator precedence and
parentheses, condition negation, boolean rewrites, constant folding, literals and statement layout. The
text is read with VM value semantics, so JavaScript's own number model (doubles, `/` on integers,
pass-by-value `addr_of`) is not part of the check (`falcom/ed9/spec/vm_spec.md` section 10).

## Search and Reporting

- Inputs are drawn from the function's constants (plus neighbors). When a run compares an input with a
  literal, a follow-up run replays all other inputs of that run and sets this input to the literal, so
  nested branches are reached one condition at a time.
- The reference interpreter also keeps, for every value derived from inputs, the expression that
  produced it (`symbolic.py`). For each conditional jump whose other side is still uncovered, the checker
  searches for the fewest input changes that flip the condition (candidate values: compared literals, the
  condition's own literals and their neighbors, `0, 1, -1, 2`) while every earlier branch of that run reading a changed input keeps its direction, and
  runs that input next. This reaches branches guarded by conjunctions of several call results, which
  single-input follow-ups cannot.
- The search stops when every block is covered, when no follow-ups remain and branch coverage has not
  grown for 64 runs, or when coverage has not grown for 1024 runs. Coverage is reported per function; uncovered blocks are not checked.
- Each run records at most 1024 events at first; both sides are compared on that prefix. When a truncated
  run still reached new blocks, the cap doubles (up to 16384) and the run is repeated, so long
  straight-line scripts are checked to the end while input-independent endless loops stay cheap.
- `FAIL` reports, per divergence site, the first divergent event of both traces and, unless
  `--no-localize`, the first HLIL stage whose output already diverges (`MLILToHLIL (or MLIL)` when the
  divergence predates the HLIL passes). `--max-mismatches N` keeps searching until N distinct sites are found.
- `ERROR` means the function could not be decompiled or the reference interpreter cannot model it.
