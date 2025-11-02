# Coding Style Guide

This document defines the coding style conventions for the Falcom decompiler project.

## General Principles

- **No hardcoding**: Use type-based logic and enums instead of string matching
- **No bare tuples**: Use NamedTuple for return values to ensure readability
- **Clear naming**: Use full, descriptive names (STACK not S, REG not R, sp not vsp)
- **Type safety**: Use proper types (e.g., LowLevelILInstruction) instead of strings

## Spacing Around Equals Sign

**ALL equals signs must have spaces on both sides:**

### Parameter Defaults
```python
# ✅ Correct
def func(param = None, size = 4, is_hex = False):
    pass

# ❌ Wrong
def func(param=None, size=4, is_hex=False):
    pass
```

### Keyword Arguments
```python
# ✅ Correct
result = func(key = value, size = 4)
self.add(lhs, rhs, push = True, size = 4)

# ❌ Wrong
result = func(key=value, size=4)
self.add(lhs, rhs, push=True, size=4)
```

### Variable Assignment
```python
# ✅ Correct (standard Python style)
x = 10
name = "test"

# ❌ Wrong
x=10
name="test"
```

## Naming Conventions

### Use Full Names
```python
# ✅ Correct
STACK[sp]
REG[0]
sp = 0

# ❌ Wrong
S[vsp]
R[0]
vsp = 0
```

### Function Names
```python
# ✅ Correct - specific, type-safe methods
def eq(self, lhs = None, rhs = None):
    pass

def add(self, lhs = None, rhs = None):
    pass

# ❌ Wrong - generic with string parameter
def compare(self, op_type: str):
    if op_type == "eq":  # hardcoded string
        pass
```

## Type Safety

### Use Instruction Types
```python
# ✅ Correct
def branch_if(self, condition: LowLevelILInstruction, target):
    pass

# ❌ Wrong
def branch_if(self, condition: str, target):  # "zero", "nonzero"
    pass
```

### Use Enums for Operations
```python
# ✅ Correct
op_expr_map = {
    LowLevelILOperation.LLIL_EQ: "(lhs == rhs) ? 1 : 0",
    LowLevelILOperation.LLIL_MUL: "lhs * rhs",
}

# ❌ Wrong
if op_name == "EQ":  # hardcoded string matching
    return "(lhs == rhs) ? 1 : 0"
```

## Return Values

### Use NamedTuple
```python
# ✅ Correct
class PatternMatch(NamedTuple):
    lines: List[str]
    skip_count: int

def try_pattern() -> Optional[PatternMatch]:
    return PatternMatch(lines = ["..."], skip_count = 2)

# ❌ Wrong
def try_pattern() -> Optional[Tuple[List[str], int]]:
    return (["..."], 2)  # unclear what each element means
```

## Separation of Concerns

### Don't Hardcode Formatting
```python
# ✅ Correct - caller controls indentation
def format_pattern() -> List[str]:
    return ["line1", "line2"]  # no indentation

# Usage
indent = "  "
for line in format_pattern():
    print(indent + line)

# ❌ Wrong - hardcoded indentation
def format_pattern() -> List[str]:
    return ["  line1", "  line2"]  # indentation baked in
```

## Architecture Principles

### Dual Mode Operations
Binary operations support both implicit and explicit modes:

```python
# Implicit mode: pop operands from vstack
result = builder.add()  # both lhs and rhs are None

# Explicit mode: provide operands directly
result = builder.add(lhs = x, rhs = y)  # both provided

# ❌ Wrong: mixed mode
result = builder.add(lhs = x)  # one None, one provided - INVALID
```

### Both-or-Neither Rule
For binary operations, operands must be:
- **Both None** (implicit mode - pop from vstack), OR
- **Both provided** (explicit mode - use given values)

Never allow one to be None while the other has a value.

## Documentation

### Real Examples Only
```python
# ✅ Correct - based on actual game code
def create_AV_04_0017():
    """
    Source: m4000.py from Kuro no Kiseki
    Original bytecode: ...
    """

# ❌ Wrong - abstract synthetic examples
def create_example_function():
    """
    Fictional example showing features
    """
```

## File Organization

### VM-Agnostic vs VM-Specific
- `ir/llil.py` and `ir/llil_builder.py`: VM-agnostic, generic LLIL
- `falcom/builder.py` and `falcom/constants.py`: Falcom VM-specific

Keep generic IR separate from VM-specific implementations.

## Emoji Usage

- **Do NOT use emojis** in code output unless explicitly requested by user
- Emojis in demo output are acceptable for clarity
