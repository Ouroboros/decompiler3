# Coding Style Guide

## Mandatory Rules

### 0. NEVER TOUCH binaryninja-api/

**ABSOLUTELY FORBIDDEN** to modify any files in `binaryninja-api/` directory.

- ✅ READ ONLY: Can read files for reference
- ❌ NO WRITES: Never modify, create, or delete files
- ❌ NO EDITS: Never edit any file under `binaryninja-api/`

This is third-party code. Do not touch it under any circumstances.

### 1. NO HARDCODED MAGIC NUMBERS

**NEVER** hardcode numeric constants. Always use named constants.

#### ❌ WRONG:
```python
word_offset = offset // 4
size = value * 4
```

#### ✅ CORRECT:
```python
from ir.llil import WORD_SIZE

word_offset = offset // WORD_SIZE
size = value * WORD_SIZE
```

**Common Constants:**
- `WORD_SIZE = 4` (defined in `ir/llil.py`)
  - Use for all byte/word conversions
  - Use for all stack offset calculations
  - Use for all size multiplications

**Why?**
1. **Maintainability**: Change once, update everywhere
2. **Readability**: `offset // WORD_SIZE` is clearer than `offset // 4`
3. **Flexibility**: Easy to port to different architectures

### 2. Import at Module Top Level

**ALWAYS** import at the top of the file. **NEVER** import inside functions unless absolutely necessary to avoid circular dependencies.

#### ✅ CORRECT:
```python
from ir.llil import WORD_SIZE, LowLevelILStackStore

class MyBuilder:
    def pop_to(self, offset: int):
        val = self.pop()
        self.add_instruction(LowLevelILStackStore(val, offset, WORD_SIZE))
```

#### ❌ WRONG:
```python
class MyBuilder:
    def pop_to(self, offset: int):
        from ir.llil import LowLevelILStackStore, WORD_SIZE  # Don't do this!
        val = self.pop()
        self.add_instruction(LowLevelILStackStore(val, offset, WORD_SIZE))
```

**Exception**: Only import inside functions when there is a proven circular dependency issue. Document the reason with a comment.

### 3. Comment Documentation

When documenting offsets or sizes, refer to constants:

#### ❌ WRONG:
```python
# offset is in bytes, converted to word index (offset // 4)
```

#### ✅ CORRECT:
```python
# offset is in bytes, converted to word index (offset // WORD_SIZE)
```

### 4. Assignment Spacing

**ALWAYS** add a space before and after the assignment operator `=`.

#### ❌ WRONG:
```python
x=5
result=calculate()
self.value=None
```

#### ✅ CORRECT:
```python
x = 5
result = calculate()
self.value = None
```

This applies to:
- Variable assignments: `var = value`
- Default parameters: `def func(arg = 0)`
- Keyword arguments: `func(key = value)`
- Annotated assignments: `x: int = 5`

### 5. English-Only Comments

**NEVER** use Chinese (or any non-English language) in code comments or documentation.

#### ❌ WRONG:
```python
rhs = stack_pop()  # 先弹出来的是右操作数（栈顶）
lhs = stack_pop()  # 再弹出来的是左操作数（下面那个）
```

#### ✅ CORRECT:
```python
rhs = stack_pop()  # First pop gets right operand (top of stack)
lhs = stack_pop()  # Second pop gets left operand (below it)
```

**Why?**
- Code consistency
- International collaboration
- Easier for tooling and static analysis

### 6. Concise Git Commit Messages

Keep commit messages **short and focused** (1-2 sentences).

#### ❌ WRONG:
```
Fix binary operation operand order in expand format

Issue: lhs/rhs were reversed in binary operation expansion

Based on VM C code analysis, the correct order is:
1. First pop → lhs
2. Second pop → rhs
3. Compute: rhs OP lhs (not lhs OP rhs!)

Example from GE operation in VM:
  lhs = *(_DWORD *)(v112 + stack_ptr_1);  // First pop
  rhs = *(_DWORD *)(v75 + stack_ptr_1);   // Second pop
  cmp_result = (int)(rhs) >= (int)(lhs);  // rhs >= lhs

Changes:
- Swap pop order: lhs first, then rhs
- Reverse expression: rhs OP lhs instead of lhs OP rhs
- Update all operations: ADD, SUB, MUL, DIV, EQ, NE, LT, LE, GT, GE
```

#### ✅ CORRECT:
```
Fix binary operation operand order

Corrected lhs/rhs pop order based on VM C code analysis.
```

## Enforcement

All code must follow these rules.

### Code Review Checklist:
- [ ] No hardcoded `4` for word size
- [ ] No hardcoded numeric constants without names
- [ ] All constants imported from appropriate modules
- [ ] Comments reference constant names, not values
- [ ] All assignments have spaces: `x = value` (not `x=value`)
- [ ] All comments are in English (no Chinese or other languages)
- [ ] Commit messages are concise (1-2 sentences)
