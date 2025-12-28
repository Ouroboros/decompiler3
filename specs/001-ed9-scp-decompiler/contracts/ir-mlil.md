# Contract: Medium-Level IL

**Module**: `ir/mlil/mlil.py`, `ir/mlil/mlil_types.py`, `ir/mlil/mlil_ssa.py`

## MediumLevelILFunction

Container for MLIL representation.

### Constructor

```python
MediumLevelILFunction(name: str, parameters: List[IRParameter] = None)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| name | str | Function name |
| parameters | List[IRParameter] | Parameters |
| blocks | List[MediumLevelILBasicBlock] | Basic blocks |
| variables | Dict[str, MLILVariable] | All variables |
| is_ssa | bool | Whether in SSA form |

### Methods

#### get_or_create_variable()

Get existing or create new variable.

```python
def get_or_create_variable(
    self,
    name: str,
    slot_index: int = -1
) -> MLILVariable
```

#### to_ssa()

Convert to SSA form.

```python
def to_ssa(self) -> None
```

**Postconditions:**
- All variables converted to SSA versions
- Phi nodes inserted at merge points
- `self.is_ssa` is True

#### from_ssa()

Convert out of SSA form.

```python
def from_ssa(self) -> None
```

## MLILVariable

Variable in MLIL.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| name | str | Variable name |
| slot_index | int | Original stack slot (-1 if none) |
| type | MLILType | Inferred type |

### Naming Conventions

```python
# Stack variables
mlil_stack_var_name(0)  # → "var_s0"
mlil_stack_var_name(1)  # → "var_s1"

# Parameters
mlil_arg_var_name(0)    # → "arg0"
mlil_arg_var_name(1)    # → "arg1"
```

## MLILVariableSSA

SSA-versioned variable.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| base_var | MLILVariable | Underlying variable |
| version | int | SSA version number |

### Display Format

```
var_s0#0  # First definition
var_s0#1  # Second definition
arg0#0    # Parameter (version 0)
```

## MLILType

Type representation.

### Type Kinds

| Kind | Description |
|------|-------------|
| UNKNOWN | Type not yet inferred |
| INT | Integer type |
| BOOL | Boolean type |
| STRING | String type |
| FLOAT | Floating point |
| POINTER | Pointer type |
| VARIANT | Multiple possible types |
| VOID | No value |

### Methods

#### is_unknown()

```python
def is_unknown(self) -> bool
```

#### is_numeric()

```python
def is_numeric(self) -> bool  # INT or FLOAT
```

### Type Unification

```python
def unify_types(t1: MLILType, t2: MLILType) -> MLILType
```

| t1 | t2 | Result |
|----|----|----|
| UNKNOWN | X | X |
| X | UNKNOWN | X |
| BOOL | INT | INT |
| INT | BOOL | INT |
| X | X | X |
| X | Y | VARIANT({X, Y}) |

## MLILPhi

SSA phi node for merging values.

### Constructor

```python
MLILPhi(dest: MLILVariableSSA, sources: List[Tuple[MLILVariableSSA, BasicBlock]])
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| dest | MLILVariableSSA | Target variable |
| sources | List[Tuple[...]] | (variable, predecessor) pairs |

## DominanceAnalysis

Dominance tree computation.

### Constructor

```python
DominanceAnalysis(entry: BasicBlock, blocks: List[BasicBlock])
```

### Methods

#### dominates()

Check if block A dominates block B.

```python
def dominates(self, a: BasicBlock, b: BasicBlock) -> bool
```

#### get_idom()

Get immediate dominator.

```python
def get_idom(self, block: BasicBlock) -> Optional[BasicBlock]
```

#### get_dom_frontier()

Get dominance frontier.

```python
def get_dom_frontier(self, block: BasicBlock) -> Set[BasicBlock]
```

## MLIL Operations

### Variables

| Class | Description | Display |
|-------|-------------|---------|
| MLILVar | Variable ref | `var_s0` |
| MLILVarSSA | SSA variable ref | `var_s0#1` |
| MLILSetVar | Assignment | `var_s0 = expr` |
| MLILSetVarSSA | SSA assignment | `var_s0#1 = expr` |
| MLILPhi | Phi node | `var_s0#2 = φ(var_s0#0, var_s0#1)` |

### Control Flow

| Class | Description |
|-------|-------------|
| MLILGoto | Unconditional jump |
| MLILBranch | Conditional branch |
| MLILReturn | Function return |

### Expressions

| Class | Description |
|-------|-------------|
| MLILConst | Constant value |
| MLILBinaryOp | Binary operation |
| MLILUnaryOp | Unary operation |
| MLILCall | Function call |
| MLILSyscall | System call |
