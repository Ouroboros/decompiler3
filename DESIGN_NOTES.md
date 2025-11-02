# LLIL Design Notes

## Architecture Changes (2025-11-02)

### 1. Directory Structure
- **Change**: Moved `ir/falcom` → `falcom` (top-level)
- **Reason**: Falcom-specific builders should be at top level, not nested inside IR
- **Impact**: Import paths changed from `ir.falcom` to `falcom`

### 2. Terminal Instruction Inheritance (BinaryNinja Style)

Following BinaryNinja's design pattern, we now have proper control flow hierarchy:

```python
LowLevelILInstruction
  └── ControlFlow                # Base for all control flow
        ├── Terminal             # Instructions that end a block
        │     ├── LowLevelILGoto     # Unconditional jump
        │     ├── LowLevelILJmp      # Alias for Goto
        │     └── LowLevelILRet      # Return
        ├── LowLevelILIf         # Conditional branch (NOT terminal)
        ├── LowLevelILBranch     # Simplified branch
        └── LowLevelILCall       # Function call
```

**Key Points**:
- `Terminal` inherits from `ControlFlow`
- Only truly terminal instructions (goto, ret) inherit from `Terminal`
- Conditional branches (`LowLevelILIf`) are `ControlFlow` but NOT `Terminal`
- This matches BinaryNinja's design exactly

### 3. Control Flow Target Types

**Before**:
```python
class LowLevelILJmp:
    def __init__(self, target: Union[str, int]):  # ❌ Wrong
        self.target = target
```

**After**:
```python
class LowLevelILGoto(Terminal):
    def __init__(self, target: Union[LowLevelILLabel, InstructionIndex]):  # ✅ Correct
        self.target = target
```

**Changes**:
- Target should be `LowLevelILLabel` or `InstructionIndex`
- Not just string or int
- Follows BinaryNinja's type-safe design

### 4. LowLevelILLabel Design

We now have **two** label-related classes:

#### LowLevelILLabel (Control Flow Reference)
```python
class LowLevelILLabel:
    """Label for control flow targets (like BN)"""
    def __init__(self):
        self.resolved = False
        self.ref = False
        self.operand: Optional[InstructionIndex] = None
```
- Used as jump/branch targets
- Handles unresolved references
- Similar to BN's design

#### LowLevelILLabelInstr (Marking Instruction)
```python
class LowLevelILLabelInstr(LowLevelILInstruction):
    """Label instruction (for marking positions in code)"""
    def __init__(self, name: str):
        super().__init__(LowLevelILOperation.LLIL_LABEL)
        self.name = name
```
- Used to mark positions in code (like `label('my_label')`)
- Shows up in disassembly as `my_label:`
- Different purpose than `LowLevelILLabel`

### 5. BasicBlock Control Flow Graph

**New Features**:
```python
class LowLevelILBasicBlock:
    def __init__(self, start: int, index: int = 0):
        self.index = index  # Block index in function

        # Control flow edges (following BN design)
        self.outgoing_edges: List[LowLevelILBasicBlock] = []
        self.incoming_edges: List[LowLevelILBasicBlock] = []

    def add_outgoing_edge(self, target: LowLevelILBasicBlock):
        """Add bidirectional edge"""
        if target not in self.outgoing_edges:
            self.outgoing_edges.append(target)
        if self not in target.incoming_edges:
            target.incoming_edges.append(self)

    @property
    def has_terminal(self) -> bool:
        """Check if block ends with terminal instruction"""
        if not self.instructions:
            return False
        return isinstance(self.instructions[-1], Terminal)
```

**Benefits**:
- Explicit edge tracking
- CFG construction from terminal instructions
- Easy traversal (predecessors/successors)

### 6. Function-Level CFG Construction

```python
class LowLevelILFunction:
    def build_cfg(self):
        """Build control flow graph from terminal instructions"""
        for block in self.basic_blocks:
            last_instr = block.instructions[-1]

            if isinstance(last_instr, LowLevelILGoto):
                # Add edge to jump target
                ...
            elif isinstance(last_instr, LowLevelILIf):
                # Add edges to both true/false targets
                ...
            elif not isinstance(last_instr, Terminal):
                # Falls through to next block
                ...
```

## Comparison with BinaryNinja

| Feature | Our Design | BinaryNinja |
|---------|-----------|-------------|
| Terminal base class | ✅ `Terminal(ControlFlow)` | ✅ Same |
| Goto inherits Terminal | ✅ Yes | ✅ Yes |
| Ret inherits Terminal | ✅ Yes | ✅ Yes |
| If inherits Terminal | ❌ No (ControlFlow only) | ❌ No |
| Label type | ✅ `LowLevelILLabel` | ✅ `LowLevelILLabel` |
| Target types | ✅ `Union[Label, InstructionIndex]` | ✅ Same |
| BasicBlock edges | ✅ incoming/outgoing | ✅ Same |

## Testing

All features validated in `demos/cfg_demo.py`:
- ✅ Terminal inheritance hierarchy
- ✅ ControlFlow base class
- ✅ LowLevelILLabel resolution
- ✅ BasicBlock edge tracking
- ✅ Automatic CFG construction
- ✅ Terminal detection

## Migration Guide

### For Code Using Old Design

1. **Import Changes**:
   ```python
   # Old
   from ir.falcom import FalcomVMBuilder

   # New
   from falcom import FalcomVMBuilder
   ```

2. **Label Instructions**:
   ```python
   # Old (still works through builder.label())
   builder.label('my_label')

   # Direct usage changed
   # Old: LowLevelILLabel('my_label')
   # New: LowLevelILLabelInstr('my_label')
   ```

3. **Type Checking**:
   ```python
   # Check if instruction is terminal
   if isinstance(instr, Terminal):
       print("This instruction ends the block")

   # Check if block has terminal
   if block.has_terminal:
       print("Block properly terminated")
   ```

## Benefits

1. **Type Safety**: Proper types for jump targets
2. **BN Compatibility**: Matches industry-standard design
3. **Better Analysis**: CFG construction and traversal
4. **Clearer Semantics**: Terminal vs. non-terminal control flow
5. **Future-Proof**: Ready for advanced analyses (dominators, etc.)
