# Research: IR Semantic Consistency Validation

**Feature**: 003-ir-pass-validation
**Date**: 2025-12-29

## Overview

This document analyzes how to compare semantic consistency across LLIL, MLIL, and HLIL layers when processing actual SCP files.

## 1. IR Layer Characteristics

### 1.1 LLIL (Low-Level IL)

**Source**: `ir/llil/llil.py`, `falcom/ed9/llil_builder.py`

- Stack-based representation
- Preserves original bytecode semantics
- Operations on stack slots, registers, frame slots
- No variable names, just storage locations
- Basic blocks with explicit jumps/branches

**Key Classes**:
- `LowLevelILFunction`: Function container
- `LowLevelILBasicBlock`: Basic block
- `LowLevelILInstruction`: Base instruction
- Operations: `LLIL_STACK_LOAD`, `LLIL_STACK_STORE`, `LLIL_ADD`, `LLIL_BRANCH`, etc.

### 1.2 MLIL (Medium-Level IL)

**Source**: `ir/mlil/mlil.py`, `ir/mlil/llil_to_mlil.py`

- Variable-based representation
- Stack operations eliminated
- Named variables instead of stack slots
- Optional SSA form for optimization
- Type inference available

**Key Classes**:
- `MediumLevelILFunction`: Function container
- `MediumLevelILBasicBlock`: Basic block
- `MLILVariable`: Variable representation
- Operations: `MLIL_SET_VAR`, `MLIL_VAR`, `MLIL_ADD`, `MLIL_IF`, etc.

### 1.3 HLIL (High-Level IL)

**Source**: `ir/hlil/hlil.py`, `ir/hlil/mlil_to_hlil.py`

- Structured control flow
- No goto/explicit jumps
- Expression trees (inlined operations)
- Named variables with types
- Statement-based (if/while/for)

**Key Classes**:
- `HLILFunction`: Function container
- `HLILStatement`: Base statement
- `HLILIf`, `HLILWhile`, `HLILFor`: Control flow
- `HLILAssign`, `HLILCall`, `HLILReturn`: Operations

---

## 2. Transformation Mappings

### 2.1 LLIL → MLIL Transformations

| LLIL | MLIL | Semantic Preservation |
|------|------|----------------------|
| `LLIL_STACK_LOAD(slot)` | `MLIL_VAR(var)` | slot → var mapping |
| `LLIL_STACK_STORE(slot, val)` | `MLIL_SET_VAR(var, val)` | slot → var mapping |
| `LLIL_FRAME_LOAD(slot)` | `MLIL_VAR(param)` | frame → parameter |
| `LLIL_ADD(a, b)` | `MLIL_ADD(a, b)` | Direct mapping |
| `LLIL_BRANCH(cond, target)` | `MLIL_IF(cond, true_bb, false_bb)` | CFG restructure |
| `LLIL_CALL(target, args)` | `MLIL_CALL(target, args)` | Direct mapping |
| `LLIL_SYSCALL(id, args)` | `MLIL_SYSCALL(id, args)` | Direct mapping |
| `LLIL_REG_LOAD(reg)` | `MLIL_LOAD_REG(reg)` | Direct mapping |
| `LLIL_CONST(val)` | `MLIL_CONST(val)` | Direct mapping |

**Variable Mapping Strategy**:
- Each LLIL stack slot at offset N → MLIL `var_N`
- Frame slots → Function parameters
- Registers → Register variables

### 2.2 MLIL → HLIL Transformations

| MLIL | HLIL | Semantic Preservation |
|------|------|----------------------|
| `MLIL_SET_VAR(var, val)` | `HLILAssign(var, val)` | Direct mapping |
| `MLIL_VAR(var)` | `HLILVariable(var)` | Direct mapping |
| `MLIL_IF + MLIL_GOTO` | `HLILIf(cond, then, else)` | Structured |
| Back-edge GOTO | `HLILWhile(cond, body)` | Loop detection |
| `MLIL_CALL(target, args)` | `HLILCall(target, args)` | Direct mapping |
| `MLIL_RET(val)` | `HLILReturn(val)` | Direct mapping |
| Multiple `MLIL_SET_VAR` | Single `HLILAssign` | Expression inlining |

**Control Flow Reconstruction**:
- Dominance analysis identifies loop headers
- Back edges → while/for loops
- Forward branches → if/else

---

## 3. Semantic Equivalence Definitions

### 3.1 Operation Equivalence

Two operations are semantically equivalent if:

1. **Same operator type** (or known transformation pair)
2. **Equivalent operands** (recursively)
3. **Same side effects** (assignments, calls)

### 3.2 Variable Equivalence

Two variables are equivalent if:

1. **Same storage origin** (same LLIL slot/register)
2. **Same value at point of use** (data flow analysis)
3. **Compatible types** (if type info available)

### 3.3 Expression Tree Equivalence

Two expression trees are equivalent if:

1. **Same operator** (or semantic equivalent)
2. **Equivalent sub-expressions** (order matters for non-commutative)
3. **For commutative ops**: operand sets equal (order doesn't matter)

### 3.4 Control Flow Equivalence

Two control flow structures are equivalent if:

1. **Same reachable code paths**
2. **Same conditions** (semantically)
3. **Same operations in each path**

---

## 4. Comparison Approach

### 4.1 Phase 1: IR Generation

```python
def generate_all_ir(scp_path: str) -> Tuple[LLILFunction, MLILFunction, HLILFunction]:
    # Parse SCP
    scp = ScpParser().parse(scp_path)

    # Generate LLIL
    llil_func = build_llil(scp.functions[0])

    # Convert to MLIL
    mlil_func = convert_llil_to_mlil(llil_func)

    # Convert to HLIL
    hlil_func = convert_mlil_to_hlil(mlil_func)

    return llil_func, mlil_func, hlil_func
```

### 4.2 Phase 2: Normalization

Convert each IR to normalized `SemanticOperation` list:

```python
def normalize_llil(llil: LLILFunction) -> List[SemanticOperation]:
    ops = []
    for block in llil.basic_blocks:
        for instr in block.instructions:
            ops.append(llil_to_semantic(instr))
    return ops
```

### 4.3 Phase 3: Comparison

Compare normalized operations pairwise:

```python
def compare_semantic_ops(
    source: List[SemanticOperation],
    target: List[SemanticOperation],
    var_mapping: VariableMapping
) -> List[ComparisonResult]:
    results = []
    # Match operations by execution order
    # Handle 1:N and N:1 mappings
    # Report differences
    return results
```

### 4.4 Phase 4: Report Generation

Generate human-readable report with:
- Summary (pass/fail counts)
- Per-operation comparison
- Variable mapping table
- Difference details with source locations

---

## 5. Handling Special Cases

### 5.1 Dead Code Elimination

HLIL may eliminate unreachable code. This is **expected**.

**Detection**: Operation in LLIL/MLIL has no corresponding HLIL.
**Status**: TRANSFORMED (not DIFFERENT)

### 5.2 Expression Inlining

HLIL inlines expressions that MLIL keeps separate.

**Example**:
```
MLIL: var_1 = a + b
      var_2 = var_1 * c
HLIL: var_2 = (a + b) * c
```

**Detection**: Multiple MLIL ops → single HLIL expression tree
**Status**: TRANSFORMED

### 5.3 Control Flow Restructuring

GOTO → structured loops.

**Example**:
```
MLIL: block_1: if (cond) goto block_3
      block_2: ... goto block_1
      block_3: ...
HLIL: while (cond) { ... }
```

**Detection**: Back-edge pattern detected
**Status**: TRANSFORMED

### 5.4 Constant Propagation

Variables may be replaced with constants.

**Example**:
```
MLIL: var_1 = 5
      var_2 = var_1 + 3
HLIL: var_2 = 8
```

**Detection**: Constant value matches computed value
**Status**: TRANSFORMED

---

## 6. Implementation Decisions

### 6.1 Comparison Granularity

**Decision**: Per-function comparison
**Rationale**: Functions are independent units. Cross-function validation is out of scope.

### 6.2 Variable Mapping Source

**Decision**: Use MLIL variable info + LLIL source tracking
**Rationale**: MLIL already tracks which LLIL slots became which variables

### 6.3 Type Comparison

**Decision**: Compare types only if both layers have type info
**Rationale**: LLIL often lacks types. Only MLIL→HLIL type conflicts are errors.

### 6.4 Report Format

**Decision**: Text with optional JSON export
**Rationale**: Text for human reading, JSON for automation/regression

---

## 7. Existing Infrastructure

### 7.1 Usable Components

| Component | Location | Purpose |
|-----------|----------|---------|
| ScpParser | `falcom/ed9/parser/scp.py` | Parse SCP files |
| FalcomVMBuilder | `falcom/ed9/llil_builder.py` | Generate LLIL |
| convert_falcom_llil_to_mlil | `falcom/ed9/mlil_converter.py` | LLIL→MLIL |
| convert_falcom_mlil_to_hlil | `falcom/ed9/hlil_converter.py` | MLIL→HLIL |
| LLILFormatter | `ir/llil/llil.py` | LLIL string repr |
| MLILFormatter | `ir/mlil/mlil_formatter.py` | MLIL string repr |
| HLILFormatter | `ir/hlil/hlil_formatter.py` | HLIL string repr |

### 7.2 Source Location Tracking

LLIL instructions have `address` field with original bytecode offset.
This propagates through MLIL and HLIL for error reporting.

---

## 8. Conclusion

The semantic validation approach is feasible:

1. **Normalization** to common representation allows comparison
2. **Expected transformations** can be whitelisted
3. **Variable mapping** is available from existing infrastructure
4. **Source locations** are tracked through the pipeline

The main complexity is handling:
- Expression tree comparison (different structure, same semantics)
- N:1 and 1:N operation mappings
- Control flow restructuring validation
