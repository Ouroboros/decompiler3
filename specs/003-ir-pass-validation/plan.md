# Implementation Plan: IR Semantic Consistency Validation

**Branch**: `003-ir-pass-validation` | **Date**: 2025-12-29 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-ir-pass-validation/spec.md`

**Note**: This plan defines a tool to validate semantic consistency across LLIL, MLIL, and HLIL layers by comparing actual decompiled output from SCP files.

## Summary

A CLI tool that loads SCP files, generates all three IR layers (LLIL, MLIL, HLIL), and compares them semantically to verify that transformations preserve program meaning. The tool produces detailed reports showing equivalence or differences, with source location mapping back to original bytecode.

## Technical Context

**Language/Version**: Python 3.14
**Primary Dependencies**: Existing IR modules (ir/llil, ir/mlil, ir/hlil), falcom/ed9 parser
**Storage**: N/A (in-memory analysis, text/JSON report output)
**Testing**: unittest (tests/test_ir_semantic_validator.py)
**Target Platform**: Cross-platform CLI tool
**Project Type**: Single project (validation script in tools/)
**Performance Goals**: < 2 seconds per function, < 30 seconds for batch validation
**Constraints**: Must work with any valid SCP file
**Scale/Scope**: Single SCP file analysis, all functions

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Language Separation | ✅ Pass | Chinese for user comm, English for code |
| II. Git Discipline | ✅ Pass | No auto-commit, explicit user confirmation |
| III. No Magic Numbers | ✅ Pass | Will use existing IR constants |
| IV. Import Structure | ✅ Pass | Imports at module top level |
| V. Code Formatting | ✅ Pass | Spaces around =, blank lines between conditionals |
| VI. Method Decorators | ✅ Pass | @classmethod used, no @staticmethod |
| VII. English-Only Comments | ✅ Pass | All comments in English |
| VIII. Concise Documentation | ✅ Pass | Brief, meaningful comments |

## Project Structure

### Documentation (this feature)

```text
specs/003-ir-pass-validation/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # IR comparison research
└── checklists/
    └── requirements.md  # Quality checklist
```

### Source Code (repository root)

```text
decompiler3/
├── tools/                           # Validation tools
│   ├── ir_consistency.py            # Existing - enum comparison (to be removed/replaced)
│   └── ir_semantic_validator.py     # NEW - semantic validation tool
├── ir/
│   ├── llil/                        # Existing - LLIL definitions
│   ├── mlil/                        # Existing - MLIL definitions
│   └── hlil/                        # Existing - HLIL definitions
├── falcom/ed9/
│   ├── parser/scp.py                # Existing - SCP parser
│   ├── llil_builder.py              # Existing - LLIL generation
│   ├── mlil_converter.py            # Existing - MLIL conversion
│   └── hlil_converter.py            # Existing - HLIL conversion
└── tests/
    └── test_ir_semantic_validator.py # NEW - unit tests
```

**Structure Decision**: Single-file tool in `tools/` directory, leveraging existing IR infrastructure.

## Architecture Overview

### Validation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  IR Semantic Validator                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐                                                   │
│  │ SCP File │                                                   │
│  └────┬─────┘                                                   │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │ ScpParser    │──▶│ LLIL Builder │──▶│ LLILFunction │        │
│  └──────────────┘   └──────────────┘   └──────┬───────┘        │
│                                                │                 │
│                     ┌──────────────┐   ┌──────▼───────┐        │
│                     │ MLIL Conv    │──▶│ MLILFunction │        │
│                     └──────────────┘   └──────┬───────┘        │
│                                                │                 │
│                     ┌──────────────┐   ┌──────▼───────┐        │
│                     │ HLIL Conv    │──▶│ HLILFunction │        │
│                     └──────────────┘   └──────────────┘        │
│                                                                  │
│       ┌────────────────────────────────────────┐                │
│       │           Semantic Comparator           │                │
│       │  ┌─────────────────────────────────┐   │                │
│       │  │ Normalize to SemanticOperations │   │                │
│       │  └─────────────────────────────────┘   │                │
│       │  ┌─────────────────────────────────┐   │                │
│       │  │ Compare Operations Pairwise     │   │                │
│       │  └─────────────────────────────────┘   │                │
│       │  ┌─────────────────────────────────┐   │                │
│       │  │ Track Variable Mappings         │   │                │
│       │  └─────────────────────────────────┘   │                │
│       │  ┌─────────────────────────────────┐   │                │
│       │  │ Generate Difference Report      │   │                │
│       │  └─────────────────────────────────┘   │                │
│       └────────────────────────────────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. SemanticOperation (Normalized Form)

All IR operations normalized to a common representation for comparison:

```
SemanticOperation:
  - kind: OperationKind (ARITHMETIC, COMPARISON, LOGICAL, CALL, BRANCH, ASSIGN, etc.)
  - operator: str (ADD, SUB, EQ, LT, AND, etc.)
  - operands: List[SemanticOperand]
  - result: Optional[SemanticOperand]
  - source_location: SourceLocation
```

#### 2. VariableMapping

Tracks correspondence across layers:

```
VariableMapping:
  - llil_storage: StackSlot | Register | FrameSlot
  - mlil_var: MLILVariable
  - hlil_var: HLILVariable
  - type_info: TypeInfo
```

#### 3. ComparisonResult

Result of comparing two operations:

```
ComparisonResult:
  - status: EQUIVALENT | TRANSFORMED | DIFFERENT
  - source_layer: IRLayer
  - target_layer: IRLayer
  - explanation: str
  - source_location: SourceLocation
```

### Comparison Strategy

#### Expected Transformations (Not Differences)

##### LLIL → MLIL Transformations

###### Storage Operations

| From | To | Description |
|------|-----|-------------|
| LLIL_STACK_LOAD(slot) | MLIL_VAR(var) | Stack slot → named variable |
| LLIL_STACK_STORE(slot, val) | MLIL_SET_VAR(var, val) | Stack store → variable assignment |
| LLIL_FRAME_LOAD(slot) | MLIL_VAR(param) | Frame slot → function parameter |
| LLIL_FRAME_STORE(slot, val) | MLIL_SET_VAR(param, val) | Frame store → parameter write |
| LLIL_REG_LOAD(reg) | MLIL_LOAD_REG(reg) | Register read |
| LLIL_REG_STORE(reg, val) | MLIL_STORE_REG(reg, val) | Register write |
| LLIL_STACK_ADDR(slot) | MLIL_ADDRESS_OF(var) | Stack address → address-of |
| LLIL_CONST(val) | MLIL_CONST(val) | Direct mapping |

###### Arithmetic Operations

| From | To | Description |
|------|-----|-------------|
| LLIL_ADD(a, b) | MLIL_ADD(a, b) | Addition |
| LLIL_SUB(a, b) | MLIL_SUB(a, b) | Subtraction |
| LLIL_MUL(a, b) | MLIL_MUL(a, b) | Multiplication |
| LLIL_DIV(a, b) | MLIL_DIV(a, b) | Division |
| LLIL_MOD(a, b) | MLIL_MOD(a, b) | Modulo |
| LLIL_NEG(x) | MLIL_NEG(x) | Negation |

###### Comparison Operations

| From | To | Description |
|------|-----|-------------|
| LLIL_EQ(a, b) | MLIL_EQ(a, b) | Equal |
| LLIL_NE(a, b) | MLIL_NE(a, b) | Not equal |
| LLIL_LT(a, b) | MLIL_LT(a, b) | Less than |
| LLIL_LE(a, b) | MLIL_LE(a, b) | Less or equal |
| LLIL_GT(a, b) | MLIL_GT(a, b) | Greater than |
| LLIL_GE(a, b) | MLIL_GE(a, b) | Greater or equal |
| LLIL_TEST_ZERO(x) | MLIL_EQ(x, 0) | Zero test → comparison |

###### Logical & Bitwise Operations

| From | To | Description |
|------|-----|-------------|
| LLIL_LOGICAL_AND(a, b) | MLIL_LOGICAL_AND(a, b) | Logical AND |
| LLIL_LOGICAL_OR(a, b) | MLIL_LOGICAL_OR(a, b) | Logical OR |
| LLIL_AND(a, b) | MLIL_AND(a, b) | Bitwise AND |
| LLIL_OR(a, b) | MLIL_OR(a, b) | Bitwise OR |
| LLIL_XOR(a, b) | MLIL_XOR(a, b) | Bitwise XOR (if exists) |
| LLIL_SHL(a, b) | MLIL_SHL(a, b) | Shift left (if exists) |
| LLIL_SHR(a, b) | MLIL_SHR(a, b) | Shift right (if exists) |
| LLIL_BITWISE_NOT(x) | MLIL_BITWISE_NOT(x) | Bitwise NOT |

###### Control Flow Operations

| From | To | Description |
|------|-----|-------------|
| LLIL_BRANCH(cond, target) | MLIL_IF(cond, bb1, bb2) | Conditional → structured IF |
| LLIL_JMP(target) | MLIL_GOTO(bb) | Unconditional jump |
| LLIL_CALL(target, args) | MLIL_CALL(target, args) | Function call |
| LLIL_SYSCALL(id, args) | MLIL_SYSCALL(id, args) | System call |
| LLIL_RET(val) | MLIL_RET(val) | Return |

###### Falcom VM-Specific Operations

| From | To | Description |
|------|-----|-------------|
| LLIL_PUSH_CALLER_FRAME | (eliminated) | Internal VM operation |
| LLIL_CALL_SCRIPT(script, func) | MLIL_CALL_SCRIPT(script, func) | Cross-script call |
| LLIL_PUSH_FUNC_ID | (folded into CALL) | Part of call sequence |
| LLIL_PUSH_RET_ADDR | (folded into CALL) | Part of call sequence |

###### Memory Operations

| From | To | Description |
|------|-----|-------------|
| LLIL_LOAD(addr) | MLIL_LOAD(addr) | Memory load |
| LLIL_STORE(addr, val) | MLIL_STORE(addr, val) | Memory store |
| LLIL_GLOBAL_LOAD(addr) | MLIL_GLOBAL_VAR(var) | Global variable read |
| LLIL_GLOBAL_STORE(addr, val) | MLIL_SET_GLOBAL_VAR(var, val) | Global variable write |

###### String Operations

| From | To | Description |
|------|-----|-------------|
| LLIL_STRING_CONST(idx) | MLIL_STRING_CONST(str) | String pool index → literal |
| LLIL_STRING_CONCAT(a, b) | MLIL_STRING_CONCAT(a, b) | String concatenation |

###### Internal/Debug Operations

| From | To | Description |
|------|-----|-------------|
| LLIL_SP_ADD(offset) | (eliminated) | Stack pointer adjustment |
| LLIL_LABEL(name) | (eliminated) | Label marker |
| LLIL_DEBUG(info) | MLIL_DEBUG(info) | Debug information |
| LLIL_NOP | MLIL_NOP | No operation |

##### MLIL → HLIL Transformations

###### Statement Transformations

| From | To | Description |
|------|-----|-------------|
| MLIL_SET_VAR(var, val) | HLILAssign(var, val) | Variable assignment |
| MLIL_VAR(var) | HLILVariable(var) | Variable reference |
| MLIL_CONST(val) | HLILConst(val) | Constant |
| MLIL_CALL(target, args) | HLILCall(target, args) | Function call |
| MLIL_SYSCALL(id, args) | HLILSyscall(id, args) | System call |
| MLIL_CALL_SCRIPT(s, f) | HLILExternCall(s, f) | Cross-script call |
| MLIL_RET(val) | HLILReturn(val) | Return statement |
| MLIL_ADDRESS_OF(var) | HLILAddressOf(var) | Address-of operator |

###### Control Flow Structuring

| From | To | Description |
|------|-----|-------------|
| MLIL_IF(cond, bb1, bb2) | HLILIf(cond, then, else) | Structured if-else |
| MLIL_GOTO (back-edge) | HLILWhile(cond, body) | While loop |
| MLIL_GOTO (post-test loop) | HLILDoWhile(cond, body) | Do-while loop |
| MLIL_GOTO (counted loop) | HLILFor(init, cond, step, body) | For loop |
| MLIL_GOTO (forward) | HLILBreak | Break statement |
| MLIL_GOTO (continue pattern) | HLILContinue | Continue statement |
| MLIL_GOTO (to merge point) | (eliminated) | Structured away |
| MLIL switch pattern | HLILSwitch(expr, cases) | Switch statement |

###### Expression Transformations

| From | To | Description |
|------|-----|-------------|
| Multiple MLIL_SET_VAR | Single HLILAssign | Expression inlining |
| MLIL binary ops | HLIL BinaryOp | Same operator, nested |
| MLIL unary ops | HLIL UnaryOp | Same operator |
| MLIL with temp vars | HLIL without temps | Dead temp elimination |
| MLIL_NEG(x) | HLILUnary(NEG, x) | Negation |
| MLIL_BITWISE_NOT(x) | HLILUnary(BIT_NOT, x) | Bitwise NOT |
| MLIL_LOGICAL_NOT(x) | HLILUnary(NOT, x) | Logical NOT |
| MLIL_TEST_ZERO(x) | HLILBinaryOp(EQ, x, 0) | Zero test → comparison |
| MLIL_UNDEF | (eliminated or error) | Undefined value |

###### Statement Wrappers

| From | To | Description |
|------|-----|-------------|
| MLIL_CALL (standalone) | HLILExprStmt(HLILCall) | Call as statement |
| MLIL_NOP | (eliminated) | No operation removed |
| MLIL_DEBUG(info) | HLILComment(info) | Debug → comment |
| Multiple MLIL statements | HLILBlock(stmts) | Statement grouping |

###### SSA-Related Transformations

| From | To | Description |
|------|-----|-------------|
| MLIL_PHI(vars...) | (resolved to single var) | PHI elimination |
| MLIL_VAR_SSA(var, version) | HLILVariable(var) | SSA version stripped |
| MLIL_SET_VAR_SSA | HLILAssign | SSA version stripped |

###### Type-Related Transformations

| From | To | Description |
|------|-----|-------------|
| MLIL_CAST(type, val) | HLILCast(type, val) | Type cast |
| MLIL with inferred types | HLIL with explicit types | Type annotation |
| Implicit type coercion | Explicit cast | Type safety |

##### Optimization Transformations

| From | To | Description |
|------|-----|-------------|
| var = const; use(var) | use(const) | Constant propagation |
| var = x; y = var | y = x | Copy propagation |
| Dead assignments | (eliminated) | Dead code elimination |
| x + 0, x * 1 | x | Identity simplification |
| x - x, x ^ x | 0 | Zero simplification |
| if (true) A else B | A | Constant condition folding |
| a && a | a | Boolean simplification |
| x * 2 | x << 1 | Strength reduction |
| Common subexpressions | Shared computation | CSE |

#### Semantic Equivalence Rules

##### Operations
1. **Arithmetic**: Same operator, semantically equivalent operands
2. **Comparison**: Same operator (or inverse with swapped operands: a < b ≡ b > a)
3. **Logical**: Boolean equivalence (a AND b ≡ b AND a for commutative)
4. **Bitwise**: Same operator, same operands

##### Data
5. **Variables**: Same storage location (after mapping resolution)
6. **Constants**: Same value (regardless of representation)
7. **Strings**: Same string content (from constant pool)
8. **Globals**: Same address/identifier

##### Control Flow
9. **Calls**: Same target, same arguments (in order)
10. **Syscalls**: Same syscall ID, same arguments
11. **Returns**: Same return value (or void)
12. **Branches**: Same reachable code paths

##### Types
13. **Type inference**: MLIL and HLIL types must be compatible
14. **Coercion**: Implicit type conversions must be semantically valid

### CLI Interface

```
usage: ir_semantic_validator.py [-h] [--compare LAYER LAYER] [--all]
                                [--variables] [--types] [--batch]
                                [--function NAME] [--json] [--no-color]
                                scp_file

positional arguments:
  scp_file              Path to SCP file to validate

options:
  --compare LAYER LAYER Compare two specific layers (llil/mlil/hlil)
  --all                 Three-way comparison of all layers
  --variables           Show variable mapping across layers
  --types               Validate type consistency
  --batch               Validate all functions in file
  --function NAME       Validate specific function only
  --json                Output results as JSON
  --no-color            Disable colored output
```

### Output Format

#### Text Report (default)

```
=== IR Semantic Validation Report ===
File: test.scp
Function: func_001

[LLIL → MLIL Comparison]
  ✓ 45 operations matched
  ~ 3 transformations (expected)
  ✗ 0 differences

  Transformations:
    0x0010: LLIL_STACK_LOAD(0) → MLIL_VAR(var_0)  [storage → variable]
    0x0014: LLIL_BRANCH → MLIL_IF                  [control flow]
    ...

[MLIL → HLIL Comparison]
  ✓ 42 operations matched
  ~ 5 transformations (expected)
  ✗ 0 differences

[Variable Mapping]
  Stack[0] → var_0 → local_count (int)
  Stack[4] → var_1 → local_index (int)
  Frame[0] → arg_0 → param_target (str)

Summary: PASS (0 semantic differences)
```

#### JSON Report (--json)

```json
{
  "file": "test.scp",
  "function": "func_001",
  "comparisons": {
    "llil_mlil": {
      "matched": 45,
      "transformed": 3,
      "different": 0,
      "transformations": [...]
    },
    "mlil_hlil": {...}
  },
  "variable_mapping": [...],
  "status": "PASS"
}
```

### Error Handling

| Error Type | Handling |
|------------|----------|
| SCP parse failure | Exit with error, show parse location |
| LLIL build failure | Report function, continue with next |
| MLIL conversion failure | Report failure, skip MLIL/HLIL comparison |
| HLIL conversion failure | Report failure, LLIL-MLIL only |
| Semantic difference found | Mark as DIFFERENT, include in report |
| Unknown operation type | Warn, treat as non-comparable |

## Complexity Tracking

> No constitution violations - feature uses straightforward architecture.

| Aspect | Complexity | Justification |
|--------|------------|---------------|
| Semantic Normalization | Medium | Different IR representations need mapping |
| Variable Tracking | Medium | Stack/register to variable resolution |
| Expression Comparison | Medium | Tree structure may differ |
| Report Generation | Low | Text output with structured format |

## Related Documents

| Document | Description |
|----------|-------------|
| [spec.md](spec.md) | Feature specification with user stories and requirements |
| [research.md](research.md) | Detailed IR structure analysis |
