# Contract: Low-Level IL

**Module**: `ir/llil/llil.py`

## LowLevelILFunction

Container for LLIL representation of a function.

### Constructor

```python
LowLevelILFunction(name: str, parameters: List[IRParameter] = None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| name | str | Function name |
| parameters | List[IRParameter] | Function parameters |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| name | str | Function name |
| parameters | List[IRParameter] | Parameters with types/defaults |
| blocks | List[LowLevelILBasicBlock] | All basic blocks |
| entry_block | LowLevelILBasicBlock | Entry point block |

### Methods

#### add_basic_block()

Create a new basic block.

```python
def add_basic_block(self, address: int) -> LowLevelILBasicBlock
```

| Parameter | Type | Description |
|-----------|------|-------------|
| address | int | Block start address |
| **Returns** | LowLevelILBasicBlock | New block |

**Postconditions:**
- Block is added to `self.blocks`
- Block is registered in address lookup map

#### get_block_by_addr()

Find block by address.

```python
def get_block_by_addr(self, address: int) -> Optional[LowLevelILBasicBlock]
```

#### build_cfg()

Build control flow graph from terminal instructions.

```python
def build_cfg(self) -> None
```

**Preconditions:**
- All blocks have terminal instructions

**Postconditions:**
- Block edges are populated
- `incoming_edges` and `outgoing_edges` reflect control flow

## LowLevelILBasicBlock

A basic block in LLIL.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| address | int | Block start address |
| instructions | List[LowLevelILInstruction] | Ordered instructions |
| sp_in | int | Stack pointer at entry |
| sp_out | int | Stack pointer at exit |
| incoming_edges | List[Edge] | Predecessor edges |
| outgoing_edges | List[Edge] | Successor edges |
| has_terminal | bool | Has terminating instruction |
| terminal | LowLevelILInstruction | Terminal instruction |

### Methods

#### append()

Add instruction to block.

```python
def append(self, inst: LowLevelILInstruction) -> None
```

## LLIL Operations

### Stack Operations

| Class | Description | Display |
|-------|-------------|---------|
| LowLevelILStackLoad | Load from stack | `STACK[sp + N]<slot>` |
| LowLevelILStackStore | Store to stack | `STACK[sp + N]<slot> = expr` |
| LowLevelILStackAddr | Stack address | `&STACK[sp + N]<slot>` |
| LowLevelILSpAdd | Adjust SP | `sp += N` |

### Frame Operations

| Class | Description | Display |
|-------|-------------|---------|
| LowLevelILFrameLoad | Load parameter | `STACK[fp + N]` |
| LowLevelILFrameStore | Store to frame | `STACK[fp + N] = expr` |

### Control Flow

| Class | Description | Display |
|-------|-------------|---------|
| LowLevelILGoto | Unconditional jump | `goto label` |
| LowLevelILBranch | Conditional branch | `if cond goto T else F` |
| LowLevelILCall | Function call | `call(func, args...)` |
| LowLevelILReturn | Return | `return expr` |
| LowLevelILLabel | Branch target | `label:` |

### Expressions

| Class | Description | Display |
|-------|-------------|---------|
| LowLevelILConst | Constant | `123` |
| LowLevelILBinaryOp | Binary op | `a + b` |
| LowLevelILUnaryOp | Unary op | `-a` |
| LowLevelILSyscall | System call | `syscall(sub, cmd, args)` |

## Constants

```python
WORD_SIZE = 4  # Bytes per stack slot
```
