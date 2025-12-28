# Data Model: ED9 SCP Script Decompiler

**Date**: 2025-12-28
**Status**: Documented (post-implementation)

## 1. Core Data Structures

### 1.1 ScpValue

Represents a typed value in SCP scripts.

```
┌─────────────────────────────────────────────┐
│                  ScpValue                   │
├─────────────────────────────────────────────┤
│ type: ScpValue.Type                         │
│   - Raw (0x00)                              │
│   - Integer (0x01)                          │
│   - Float (0x02)                            │
│   - String (0x03)                           │
├─────────────────────────────────────────────┤
│ value: int | float | str                    │
│   - Raw: 32-bit unsigned                    │
│   - Integer: 30-bit signed (sign-extended)  │
│   - Float: 30-bit float encoding            │
│   - String: offset → resolved string        │
├─────────────────────────────────────────────┤
│ Methods:                                    │
│   to_bytes() → bytes                        │
│   from_stream(fs) → ScpValue                │
└─────────────────────────────────────────────┘
```

**Location**: `falcom/ed9/parser/types_scp.py`

### 1.2 ScpFunction

Represents a parsed function from SCP file.

```
┌─────────────────────────────────────────────┐
│                ScpFunction                  │
├─────────────────────────────────────────────┤
│ name: str                                   │
│ name_hash: int (CRC32)                      │
│ code_offset: int                            │
│ param_count: int                            │
├─────────────────────────────────────────────┤
│ parameters: List[ScpParameter]              │
│   - name: str                               │
│   - flags: ScpParamFlags                    │
│   - default_value: Optional[ScpValue]       │
├─────────────────────────────────────────────┤
│ basic_blocks: List[BasicBlock]              │
│ instructions: List[Instruction]             │
└─────────────────────────────────────────────┘
```

**Location**: `falcom/ed9/parser/scp.py`

### 1.3 Instruction

Represents a disassembled VM instruction.

```
┌─────────────────────────────────────────────┐
│                Instruction                  │
├─────────────────────────────────────────────┤
│ address: int                                │
│ opcode: int                                 │
│ mnemonic: str                               │
│ operands: List[Operand]                     │
│ size: int (bytes)                           │
├─────────────────────────────────────────────┤
│ Operand types:                              │
│   - Immediate: int value                    │
│   - ScpValue: typed value                   │
│   - Offset: branch target                   │
└─────────────────────────────────────────────┘
```

**Location**: `falcom/ed9/disasm/instruction.py`

### 1.4 BasicBlock

Represents a control flow basic block.

```
┌─────────────────────────────────────────────┐
│                 BasicBlock                  │
├─────────────────────────────────────────────┤
│ start_addr: int                             │
│ end_addr: int                               │
│ instructions: List[Instruction]             │
├─────────────────────────────────────────────┤
│ predecessors: List[BasicBlock]              │
│ successors: List[BasicBlock]                │
├─────────────────────────────────────────────┤
│ is_entry: bool                              │
│ is_exit: bool                               │
└─────────────────────────────────────────────┘
```

**Location**: `falcom/ed9/disasm/basic_block.py`

## 2. IR Data Structures

### 2.1 LowLevelIL Hierarchy

```
ILInstruction (base)
    │
    ├── LowLevelILExpr (expressions - produce values)
    │   ├── LowLevelILConst          # Constant value
    │   ├── LowLevelILStackLoad      # Load from stack slot
    │   ├── LowLevelILStackAddr      # Address of stack slot
    │   ├── LowLevelILFrameLoad      # Load from frame (parameters)
    │   ├── LowLevelILRegLoad        # Load from register
    │   ├── LowLevelILGlobalLoad     # Load global variable
    │   ├── LowLevelILBinaryOp       # Binary operation
    │   ├── LowLevelILUnaryOp        # Unary operation
    │   └── LowLevelILCall           # Function call (returns value)
    │
    └── LowLevelILStatement (statements - side effects)
        ├── LowLevelILStackStore     # Store to stack slot
        ├── LowLevelILFrameStore     # Store to frame
        ├── LowLevelILRegStore       # Store to register
        ├── LowLevelILGlobalStore    # Store to global
        ├── LowLevelILSpAdd          # Adjust stack pointer
        ├── LowLevelILGoto           # Unconditional jump
        ├── LowLevelILBranch         # Conditional branch
        ├── LowLevelILReturn         # Function return
        ├── LowLevelILSyscall        # System call
        ├── LowLevelILDebug          # Debug instruction
        ├── LowLevelILLabel          # Label (branch target)
        └── LowLevelILNop            # No operation
```

**Key Classes**:

```
┌─────────────────────────────────────────────┐
│           LowLevelILFunction                │
├─────────────────────────────────────────────┤
│ name: str                                   │
│ parameters: List[IRParameter]               │
│ blocks: List[LowLevelILBasicBlock]          │
├─────────────────────────────────────────────┤
│ Methods:                                    │
│   add_basic_block(addr) → block             │
│   get_block_by_addr(addr) → block           │
│   build_cfg()                               │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│          LowLevelILBasicBlock               │
├─────────────────────────────────────────────┤
│ address: int                                │
│ instructions: List[LowLevelILInstruction]   │
│ sp_in: int                                  │
│ sp_out: int                                 │
├─────────────────────────────────────────────┤
│ incoming_edges: List[Edge]                  │
│ outgoing_edges: List[Edge]                  │
├─────────────────────────────────────────────┤
│ Properties:                                 │
│   has_terminal: bool                        │
│   terminal: LowLevelILInstruction           │
└─────────────────────────────────────────────┘
```

**Location**: `ir/llil/llil.py`

### 2.2 MediumLevelIL Hierarchy

```
ILInstruction (base)
    │
    ├── MediumLevelILExpr (expressions)
    │   ├── MLILConst                # Constant
    │   ├── MLILVar                  # Variable reference
    │   ├── MLILVarSSA               # SSA variable reference
    │   ├── MLILBinaryOp             # Binary operation
    │   ├── MLILUnaryOp              # Unary operation
    │   ├── MLILCall                 # Function call
    │   ├── MLILSyscall              # System call
    │   ├── MLILGlobalLoad           # Global variable load
    │   └── MLILRegLoad              # Register load
    │
    └── MediumLevelILStatement (statements)
        ├── MLILSetVar               # Variable assignment
        ├── MLILSetVarSSA            # SSA variable assignment
        ├── MLILPhi                  # SSA phi node
        ├── MLILGoto                 # Unconditional jump
        ├── MLILBranch               # Conditional branch
        ├── MLILReturn               # Function return
        ├── MLILGlobalStore          # Global store
        ├── MLILRegStore             # Register store
        └── MLILExprStmt             # Expression statement
```

**Variable System**:

```
┌─────────────────────────────────────────────┐
│               MLILVariable                  │
├─────────────────────────────────────────────┤
│ name: str                                   │
│   - "var_s0", "var_s1", ... (stack vars)    │
│   - "arg0", "arg1", ... (parameters)        │
│ slot_index: int (-1 if not from stack)      │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│             MLILVariableSSA                 │
├─────────────────────────────────────────────┤
│ base_var: MLILVariable                      │
│ version: int                                │
│   - Display: "var_s0#1", "var_s0#2"         │
└─────────────────────────────────────────────┘
```

**Type System**:

```
┌─────────────────────────────────────────────┐
│                 MLILType                    │
├─────────────────────────────────────────────┤
│ kind: MLILTypeKind                          │
│   - UNKNOWN                                 │
│   - INT                                     │
│   - BOOL                                    │
│   - STRING                                  │
│   - FLOAT                                   │
│   - POINTER                                 │
│   - VARIANT (multiple types)                │
│   - VOID                                    │
├─────────────────────────────────────────────┤
│ Methods:                                    │
│   is_unknown() → bool                       │
│   is_numeric() → bool                       │
│   is_pointer() → bool                       │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│             MLILVariantType                 │
├─────────────────────────────────────────────┤
│ types: Set[MLILType]                        │
│   - Represents conflicting type inferences  │
└─────────────────────────────────────────────┘
```

**Location**: `ir/mlil/mlil.py`, `ir/mlil/mlil_types.py`, `ir/mlil/mlil_ssa.py`

### 2.3 HighLevelIL Hierarchy

```
HLILNode (base)
    │
    ├── HLILExpression (expressions)
    │   ├── HLILConst                # Constant
    │   ├── HLILVariable             # Variable reference
    │   ├── HLILBinaryOp             # Binary operation
    │   ├── HLILUnaryOp              # Unary operation
    │   ├── HLILCall                 # Function call
    │   ├── HLILSyscall              # System call
    │   ├── HLILGlobalRef            # Global variable
    │   └── HLILRegRef               # Register reference
    │
    └── HLILStatement (statements)
        ├── HLILAssign               # Assignment
        ├── HLILExprStmt             # Expression statement
        ├── HLILReturn               # Return statement
        │
        ├── HLILIf                   # If-then-else
        ├── HLILWhile                # While loop
        ├── HLILDoWhile              # Do-while loop
        ├── HLILFor                  # For loop
        ├── HLILSwitch               # Switch statement
        │
        ├── HLILBreak                # Break statement
        ├── HLILContinue             # Continue statement
        │
        └── HLILBlock                # Statement block
```

**Control Flow Structures**:

```
┌─────────────────────────────────────────────┐
│                  HLILIf                     │
├─────────────────────────────────────────────┤
│ condition: HLILExpression                   │
│ then_body: HLILBlock                        │
│ else_body: Optional[HLILBlock]              │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│                 HLILWhile                   │
├─────────────────────────────────────────────┤
│ condition: HLILExpression                   │
│ body: HLILBlock                             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│                 HLILSwitch                  │
├─────────────────────────────────────────────┤
│ scrutinee: HLILExpression                   │
│ cases: List[HLILSwitchCase]                 │
│   - value: int                              │
│   - body: HLILBlock                         │
│ default_case: Optional[HLILBlock]           │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│                  HLILFor                    │
├─────────────────────────────────────────────┤
│ init: Optional[HLILStatement]               │
│ condition: Optional[HLILExpression]         │
│ update: Optional[HLILStatement]             │
│ body: HLILBlock                             │
└─────────────────────────────────────────────┘
```

**Variable Kinds**:

```
┌─────────────────────────────────────────────┐
│               HLILVariable                  │
├─────────────────────────────────────────────┤
│ name: str                                   │
│ type_hint: Optional[str]                    │
│ default_value: Optional[str]                │
│ kind: VariableKind                          │
│   - LOCAL                                   │
│   - PARAM                                   │
│   - GLOBAL                                  │
│   - REG                                     │
│ index: Optional[int]                        │
└─────────────────────────────────────────────┘
```

**Location**: `ir/hlil/hlil.py`

## 3. Signature Database Structures

### 3.1 FormatSignatureDB

```
┌─────────────────────────────────────────────┐
│            FormatSignatureDB                │
├─────────────────────────────────────────────┤
│ enums: Dict[str, Dict[int, str]]            │
│   - enum_name → {value → name}              │
│                                             │
│ functions: Dict[str, FunctionSig]           │
│   - function_name → signature               │
│                                             │
│ syscalls: Dict[Tuple[int,int], SyscallSig]  │
│   - (subsystem, cmd) → signature            │
├─────────────────────────────────────────────┤
│ Methods:                                    │
│   load_yaml(path)                           │
│   get_function(name) → FunctionSig          │
│   get_syscall(sub, cmd) → SyscallSig        │
│   get_enum_value(enum, val) → str           │
│   format_value(val, hint) → str             │
└─────────────────────────────────────────────┘
```

### 3.2 Signature Types

```
┌─────────────────────────────────────────────┐
│                ParamHint                    │
├─────────────────────────────────────────────┤
│ name: str                                   │
│ type: Optional[str]                         │
│   - "number", "string", "boolean"           │
│   - "[string, number]" (union)              │
│ format: Optional[str]                       │
│   - "hex", enum name, None                  │
│ variadic: bool                              │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│               FunctionSig                   │
├─────────────────────────────────────────────┤
│ params: List[ParamHint]                     │
│ return_hint: Optional[ReturnHint]           │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│                SyscallSig                   │
├─────────────────────────────────────────────┤
│ name: str                                   │
│ subsystem: int                              │
│ cmd: int                                    │
│ params: List[ParamHint]                     │
│ return_hint: Optional[ReturnHint]           │
└─────────────────────────────────────────────┘
```

**Location**: `falcom/ed9/format_signatures.py`

## 4. Analysis Structures

### 4.1 Dominance Analysis

```
┌─────────────────────────────────────────────┐
│            DominanceAnalysis                │
├─────────────────────────────────────────────┤
│ entry: BasicBlock                           │
│ blocks: List[BasicBlock]                    │
├─────────────────────────────────────────────┤
│ idom: Dict[Block, Block]                    │
│   - Immediate dominator for each block      │
│                                             │
│ dom_frontier: Dict[Block, Set[Block]]       │
│   - Dominance frontier for each block       │
├─────────────────────────────────────────────┤
│ Methods:                                    │
│   dominates(a, b) → bool                    │
│   get_idom(block) → Block                   │
│   get_dom_frontier(block) → Set[Block]      │
└─────────────────────────────────────────────┘
```

**Location**: `ir/mlil/mlil_ssa.py`

### 4.2 CFG Edge Types

```
┌─────────────────────────────────────────────┐
│                   Edge                      │
├─────────────────────────────────────────────┤
│ source: BasicBlock                          │
│ target: BasicBlock                          │
│ type: EdgeType                              │
│   - UNCONDITIONAL                           │
│   - TRUE_BRANCH                             │
│   - FALSE_BRANCH                            │
│   - SWITCH_CASE                             │
│   - FALLTHROUGH                             │
└─────────────────────────────────────────────┘
```

## 5. Relationships

```
ScpParser
    │
    ├──► ScpFunction ──► BasicBlock ──► Instruction
    │
    └──► LLILBuilder
            │
            ├──► LowLevelILFunction
            │       │
            │       └──► LowLevelILBasicBlock
            │               │
            │               └──► LowLevelILInstruction
            │
            └──► LLILToMLILTranslator
                    │
                    ├──► MediumLevelILFunction
                    │       │
                    │       ├──► MediumLevelILBasicBlock
                    │       │       │
                    │       │       └──► MediumLevelILInstruction
                    │       │
                    │       └──► MLILVariable / MLILVariableSSA
                    │
                    └──► MLILToHLILConverter
                            │
                            └──► HighLevelILFunction
                                    │
                                    └──► HLILStatement / HLILExpression
                                            │
                                            └──► TypeScriptGenerator
                                                    │
                                                    └──► .ts file
```

## 6. Constants

```python
# ir/llil/llil.py
WORD_SIZE = 4  # Bytes per stack slot

# Variable naming patterns
STACK_VAR_PREFIX = "var_s"    # var_s0, var_s1, ...
ARG_VAR_PREFIX = "arg"        # arg0, arg1, ...
GLOBAL_PREFIX = "GLOBAL"      # GLOBAL[0], GLOBAL[1], ...
REG_PREFIX = "REG"            # REG[0], REG[1], ...

# SSA version separator
SSA_VERSION_SEP = "#"         # var_s0#1, var_s0#2, ...
```
