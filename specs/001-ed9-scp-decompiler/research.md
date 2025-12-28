# Technical Research: ED9 SCP Script Decompiler

**Date**: 2025-12-28
**Status**: Documented (post-implementation)

## 1. SCP File Format

### 1.1 File Structure

```
+------------------+
|    ScpHeader     |  0x18 bytes
+------------------+
|  Function Table  |  N × 0x20 bytes (ScpFunctionEntry)
+------------------+
|   Global Table   |  M × 4 bytes (ScpValue)
+------------------+
|    Bytecode      |  Variable length
+------------------+
|   String Pool    |  Variable length (null-terminated UTF-8)
+------------------+
```

### 1.2 ScpHeader (0x18 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 4 | magic | `#scp` (0x70637323) |
| 0x04 | 4 | func_offset | Offset to function table |
| 0x08 | 4 | func_count | Number of functions |
| 0x0C | 4 | global_offset | Offset to global variable table |
| 0x10 | 4 | global_count | Number of global variables |
| 0x14 | 4 | reserved | Reserved/unused |

### 1.3 ScpFunctionEntry (0x20 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 4 | code_offset | Offset to bytecode |
| 0x04 | 4 | param_count | Number of parameters |
| 0x08 | 4 | param_flags_offset | Offset to parameter flags array |
| 0x0C | 4 | default_values_offset | Offset to default values |
| 0x10 | 4 | debug_info_offset | Offset to debug info (optional) |
| 0x14 | 4 | name_hash | CRC32 hash of function name |
| 0x18 | 4 | name_offset | Offset to function name string |
| 0x1C | 4 | reserved | Reserved/unused |

### 1.4 ScpValue Encoding

Values are 32-bit with type tag in upper 2 bits:

```
31  30  29                                    0
+---+---+--------------------------------------+
|  Type |              Data (30 bits)          |
+---+---+--------------------------------------+

Type bits:
  00 = Raw     (0x00000000) - Raw 32-bit value
  01 = Integer (0x40000000) - Signed 30-bit integer
  10 = Float   (0x80000000) - 30-bit float encoding
  11 = String  (0xC0000000) - Offset to string pool
```

Integer encoding uses sign extension for negative values (30-bit signed).

### 1.5 Parameter Flags

Each parameter has a 4-byte flag:

| Flag | Value | Description |
|------|-------|-------------|
| Value32 | 0x00 | Regular 32-bit value |
| Nullable32 | 0x04 | Nullable 32-bit value |
| NullableStr | 0x08 | Nullable string |
| Offset | 0x0C | Code offset (branch target) |
| Pointer | 0x10 | Pointer value |

## 2. VM Architecture

### 2.1 Register Model

| Register | Purpose |
|----------|---------|
| SP | Stack pointer (grows downward) |
| FP | Frame pointer (parameter access) |
| PC | Program counter |
| REG[0-N] | General purpose registers |

### 2.2 Stack Frame Layout

```
High Address
+------------------+
|   Return Value   |  FP + (param_count + 1) * WORD_SIZE
+------------------+
|   Parameter N    |  FP + N * WORD_SIZE
+------------------+
|       ...        |
+------------------+
|   Parameter 1    |  FP + WORD_SIZE
+------------------+
|   Parameter 0    |  FP
+------------------+ <-- FP (Frame Pointer)
|  Saved FP/RA     |
+------------------+
|   Local var 0    |  SP - WORD_SIZE
+------------------+
|   Local var 1    |  SP - 2 * WORD_SIZE
+------------------+
|       ...        |
+------------------+ <-- SP (Stack Pointer)
Low Address
```

### 2.3 Opcode Categories

| Category | Examples | Description |
|----------|----------|-------------|
| Stack | PUSH, POP, DUP | Stack manipulation |
| Arithmetic | ADD, SUB, MUL, DIV, MOD | Math operations |
| Comparison | EQ, NE, LT, LE, GT, GE | Comparisons |
| Logical | AND, OR, NOT | Boolean logic |
| Control | JMP, JZ, JNZ, CALL, RET | Control flow |
| Memory | LOAD, STORE | Memory access |
| System | SYSCALL, DEBUG | System calls |

### 2.4 CALL Convention

The VM uses a specific pattern for function calls:

```
PUSH func_id      ; Function identifier
PUSH ret_addr     ; Return address (next instruction)
PUSH arg1         ; First argument
PUSH arg2         ; Second argument
...
PUSH argN         ; Last argument
CALL              ; Execute call
```

Stack simulation during disassembly recognizes this pattern for optimization.

## 3. Intermediate Representation Design

### 3.1 Three-Tier Architecture

```
VM Bytecode
    │
    ▼
┌─────────┐
│  LLIL   │  Low-Level IL - Stack-based, close to VM
└────┬────┘
     │
     ▼
┌─────────┐
│  MLIL   │  Medium-Level IL - Stack-free, SSA form
└────┬────┘
     │
     ▼
┌─────────┐
│  HLIL   │  High-Level IL - Structured control flow
└─────────┘
```

### 3.2 LLIL Design Rationale

**Goals:**
- Direct mapping from VM instructions
- Preserve stack semantics for analysis
- Enable stack slot identification

**Key decisions:**
- Stack operations use slot indices for tracking
- Frame-relative addressing for parameters
- Explicit control flow (no fall-through)

### 3.3 MLIL Design Rationale

**Goals:**
- Eliminate stack operations
- Enable standard compiler optimizations
- Support type inference

**Key decisions:**
- SSA form for optimization passes
- Variable naming from stack slots
- Type lattice with variant types for conflicts

### 3.4 HLIL Design Rationale

**Goals:**
- Recover structured control flow
- Generate readable code
- Support common patterns (if/while/switch)

**Key decisions:**
- Pattern matching for control structures
- Maintain expression trees
- Support break/continue semantics

## 4. Control Flow Recovery

### 4.1 Basic Block Identification

1. Start new block at:
   - Function entry
   - Branch targets
   - Instructions after branches

2. End block at:
   - Branch instructions (JMP, JZ, JNZ)
   - CALL instructions
   - RET instructions

### 4.2 Structure Recovery Patterns

**If-Then-Else:**
```
     ┌──────┐
     │ cond │
     └──┬───┘
    T/  \F
   ┌─┐   ┌─┐
   │T│   │F│
   └─┘   └─┘
     \   /
    ┌──┴──┐
    │merge│
    └─────┘
```

**While Loop:**
```
    ┌──────┐
    │header│◄───┐
    └──┬───┘    │
   T/  \F       │
  ┌─┐  ┌────┐   │
  │b│  │exit│   │
  │o│  └────┘   │
  │d│           │
  │y├───────────┘
  └─┘
```

**Switch:**
```
       ┌────────┐
       │dispatch│
       └───┬────┘
      /    |    \
   ┌──┐  ┌──┐  ┌──┐
   │c1│  │c2│  │cN│
   └──┘  └──┘  └──┘
      \    |    /
     ┌─────┴─────┐
     │   merge   │
     └───────────┘
```

## 5. Type Inference

### 5.1 Type Lattice

```
         UNKNOWN
            │
    ┌───────┼───────┐
    │       │       │
   INT    STRING  FLOAT
    │
   BOOL

VARIANT = Set of conflicting types
```

### 5.2 Inference Rules

| Operation | Result Type |
|-----------|-------------|
| ADD, SUB, MUL, DIV | INT or FLOAT |
| EQ, NE, LT, LE, GT, GE | BOOL |
| AND, OR | BOOL |
| String literal | STRING |
| Numeric literal | INT or FLOAT |

### 5.3 Type Unification

```python
def unify(t1, t2):
    if t1 == UNKNOWN: return t2
    if t2 == UNKNOWN: return t1
    if t1 == t2: return t1
    if t1 == BOOL and t2 == INT: return INT
    if t1 == INT and t2 == BOOL: return INT
    return VARIANT({t1, t2})
```

## 6. Signature Database Design

### 6.1 Requirements

- Map syscall (subsystem, cmd) to function names
- Define parameter names and types
- Support enum formatting
- Handle variadic parameters
- Support union types for polymorphic parameters

### 6.2 YAML Schema

```yaml
enums:
  EnumName:
    0: VALUE_A
    1: VALUE_B

functions:
  function_name:
    params:
      - name: param1
        type: number
        format: hex
      - name: param2
        type: string
    return:
      type: number

syscalls:
  syscall_name:
    id: [subsystem, cmd]
    params:
      - name: arg1
        type: "[string, number]"  # Union type
        variadic: true
    return:
      type: void
```

### 6.3 Format Hints

| Format | Description | Example |
|--------|-------------|---------|
| hex | Hexadecimal | 0x1234 |
| EnumName | Enum lookup | MY_ENUM_VALUE |
| (none) | Decimal | 1234 |

## 7. Code Generation Strategy

### 7.1 TypeScript Target

**Rationale:**
- Type annotations for readability
- Familiar syntax for game modders
- Optional static typing

### 7.2 Output Format

```typescript
function FunctionName(
    arg0: number = 0,
    arg1: string = ""
): number {
    // Function body
    if (condition) {
        // ...
    }
    return result;
}
```

### 7.3 Special Cases

- Default parameter values from SCP metadata
- Syscalls formatted using signature database
- Unknown syscalls output as `syscall(subsystem, cmd, ...args)`

## 8. Optimization Passes

### 8.1 MLIL Passes

| Pass | Purpose |
|------|---------|
| SSAConstruction | Convert to SSA form |
| ConstantPropagation | Propagate known constants |
| CopyPropagation | Eliminate redundant copies |
| DeadCodeElimination | Remove unused code |
| ExpressionSimplification | Simplify expressions |

### 8.2 HLIL Passes

| Pass | Purpose |
|------|---------|
| MLILToHLIL | Structure recovery |
| ExpressionSimplification | Simplify expressions |
| ControlFlowOptimization | Optimize control flow |
| CommonReturnExtraction | Extract common returns |
| CopyPropagation | Eliminate copies |
| DeadCodeElimination | Remove dead code |

## 9. Error Handling

### 9.1 Parse Errors

- Invalid magic number → Immediate exception with file info
- Truncated file → Exception with expected vs actual size
- Invalid offset → Exception with offset value

### 9.2 Disassembly Errors

- Unknown opcode → Immediate exception with address and opcode value
- Invalid operand → Exception with instruction context

### 9.3 Decompilation Errors

- Irreducible control flow → Fall back to goto statements
- Type conflicts → Use variant type
