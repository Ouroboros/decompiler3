# Medium Level IL (MLIL) Design

> Status: Draft — this document captures the target architecture before implementation.

## Role in the Pipeline

```
SCP → Parser → Bytecode → Disassembler → LLIL → MLIL → HLIL → Decompiled Code
```

MLIL is the bridge between the Falcom‑specific stack‑based LLIL and the future HLIL /
decompiler passes. It must:

1. **Erase the explicit operand stack** — every temporary becomes a named variable.
2. **Expose control/data flow** suitable for SSA and optimizer passes.
3. **Retain Falcom semantics** (system call IDs, CALL_SCRIPT metadata, etc.) so that
   later passes still understand the VM behavior.

## Module Layout

| Path | Responsibility |
| --- | --- |
| `ir/mlil/mlil.py` | Core data structures (operations enum, expression & statement classes, variables, blocks, functions). |
| `ir/mlil/mlil_builder.py` | Builder helpers (variable creation, SSA versioning, convenience helpers for emitting statements). |
| `ir/mlil/mlil_formatter.py` | Pretty printer for MLIL functions (text dump used in tests and debugging). |
| `falcom/ed9/lifters/mlil_lifter.py` | LLIL → MLIL translator (Falcom specific). |
| `tests/test_mlil.py` | Unit / snapshot tests for the lifter and formatter. |
| `docs/MLIL_DESIGN.md` | This specification; should be kept consistent with implementation. |

## Operations & Node Types

- All MLIL nodes derive from:

  ```python
  class MediumLevelILInstruction(ILInstruction):
      operation: MediumLevelILOperation
      address: int  # optional source offset
      inst_index: int  # inherits LLIL inst_index for traceability
  ```

- Split into `MediumLevelILExpr` (produces a value) and `MediumLevelILStatement` (side effects only).
- Planned operation groups:

  | Category | Examples |
  | --- | --- |
  | Constants | `MLIL_CONST_INT`, `MLIL_CONST_FLOAT`, `MLIL_CONST_STR` |
  | Variable read/write | `MLIL_VAR`, `MLIL_SET_VAR`, `MLIL_PHI` |
  | Arithmetic / logical | `MLIL_ADD`, `MLIL_SUB`, `MLIL_MUL`, `MLIL_DIV`, `MLIL_MOD`, `MLIL_AND`, `MLIL_OR`, `MLIL_NOT`, `MLIL_CMP_*` |
  | Memory / stack artifacts | `MLIL_LOAD_STACK_SLOT`, `MLIL_STORE_STACK_SLOT` (only for explicitly referenced addresses) |
  | Control flow | `MLIL_GOTO`, `MLIL_IF`, `MLIL_RET` |
  | Calls | `MLIL_CALL`, `MLIL_CALL_SCRIPT`, `MLIL_SYSCALL` |
  | Falcom specific | Derived metadata on top of generic ops (no dedicated `PUSH_*` nodes; call setup is represented via variables/arguments). |

## Variable Model & SSA

- Introduce `MLILVar` (logical variable) and `MLILVarVersion` (SSA version).
- Stack slots become variables with canonical names:

  ```
  slot_0003        # absolute slot index
  arg_0 / arg_1    # parameters
  temp_nnn         # temporaries introduced during translation
  ```

- MLIL lifter emits `MLIL_SET_VAR` whenever LLIL stores to a slot, and `MLIL_VAR`
  whenever a value is consumed.
- Falcom‑specific helpers like `PUSH_CALLER_FRAME`, `PUSH_FUNC_ID`, `PUSH_RET_ADDR`
  are fully lowered to regular variables/arguments in MLIL; there is no dedicated
  MLIL opcode for them once the stack layout is eliminated.
- SSA form is optional but recommended:
  1. Initial pass builds non‑SSA MLIL while recording definitions/uses.
  2. Dominator‑based SSA construction inserts `MLIL_PHI` nodes per block entry.
  3. Later passes (constant prop, DCE) operate on SSA.

## Basic Blocks & Functions

```
class MediumLevelILBasicBlock:
    index: int
    instructions: list[MediumLevelILStatement]
    preds/succs: list[MediumLevelILBasicBlock]

class MediumLevelILFunction:
    name: str
    start_addr: int
    basic_blocks: list[MediumLevelILBasicBlock]
    variables: dict[str, MLILVar]
    llil_inst_to_mlil: dict[int, list[MediumLevelILInstruction]]
```

- Each MLIL block mirrors an LLIL block. Additional split/merge may occur once SSA
  is implemented.
- `llil_inst_to_mlil` uses the LLIL `inst_index` (already tracked globally) so that
  debugging tools can jump between layers.

## LLIL → MLIL Translation Pipeline

1. **Input**: `LowLevelILFunction`.
2. **Preprocessing**:
   - Build CFG (already available via `build_cfg()`).
   - Initialize block order and stack slot metadata.
3. **Stack elimination**:
   - Maintain `slot_index -> MLILVarVersion`.
   - When LLIL pushes/pops, map to writes/reads of the corresponding variable.
   - `LowLevelILStackAddr` becomes address expressions; pure stack pointers become
     `MLIL_ADDR_OF_SLOT`.
4. **Expression translation**:
   - Binary/unary ops map 1:1 (e.g., `LLIL_ADD` → `MLIL_ADD`), but operands now
     reference variables instead of stack loads.
5. **Control flow**:
   - `LLIL_IF` → `MLIL_IF` with condition expression, true/false block IDs.
   - `LLIL_JMP` → `MLIL_GOTO`.
   - `LLIL_CALL`/`LLIL_CALL_SCRIPT` carry callee + argument expressions.
6. **SSA conversion** (optional initial phase):
   - After linear translation, run a standard SSA construction pass.
   - Insert `MLIL_PHI` nodes at block entries for variables with multiple defs.
7. **Output**: `MediumLevelILFunction`.

## Testing Strategy

- Extend `tests/test_scp_parser.py` (or new `tests/test_mlil.py`) to:
  - Lift LLIL to MLIL for the existing `MayaEvent02_07_01` sample.
  - Dump MLIL as text to `{input_file}.mlil.py` for inspection (similar to LLIL dump).
  - Add unit tests for specific opcode translations (e.g., MOD, CALL_SCRIPT) once
    MLIL lifter is implemented.

## Future Work

- **Formatter**: Create `MLILFormatter` to pretty print SSA form (with variable names).
  - Formatter lives in `ir/mlil/mlil_formatter.py` and is used by tests/CLI dumps.
- **Analysis passes**: constant propagation, dead code elimination, condition simplification.
- **HLIL integration**: once MLIL is stable, HLIL will operate on SSA output to
  produce high‑level statements (if/else, loops, switch).

This document should be updated alongside implementation to reflect any changes in
operation naming or conversion flow.
