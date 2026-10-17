# AGENTS.md

## Scope

These instructions are **mandatory** and override your default behavior whenever you work in this repo. Read this file before starting any task to refresh the rules.

If any user request conflicts with this file, you must follow **this file** and, if needed, briefly explain why.

## Workflow

- **Language**: communicate with the user in Chinese. Everything else in English: code, commit messages, tool descriptions, and comments/documentation (Coding Style Guide rule 5).
- **Git commits**: never auto-commit; create commits only when the user explicitly requests (message format: Coding Style Guide rule 7).
- **Code modification**: never modify code during discussion; modify code only after the user explicitly confirms the approach.

## Environment

- Python 3.14.0: `D:\Dev\Python\python.exe`
- Shell: Git Bash (MINGW64) on Windows
- Codebase: existing IR modules (`ir/llil`, `ir/mlil`, `ir/hlil`) and `falcom/ed9` parser; in-memory analysis with text/JSON report output
- Equivalence check: `python -m tools.diffcheck tests/debug.dat` (contract in `docs/DIFFCHECK.md`)
- VM semantics: `falcom/ed9/spec/vm_spec.md` (from the game executable's interpreter; read it before touching value semantics)

---

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
- `WORD_SIZE = 4` (defined in `ir/llil/llil.py`)
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

### 6. NO @staticmethod - Use @classmethod

**NEVER** use `@staticmethod`. **ALWAYS** use `@classmethod` instead.

#### ❌ WRONG:
```python
class MyClass:
    @staticmethod
    def helper(value):
        return value * 2
```

#### ✅ CORRECT:
```python
class MyClass:
    @classmethod
    def helper(cls, value):
        return value * 2
```

**Why?**
- `@classmethod` allows subclass overriding and polymorphism
- `@staticmethod` breaks inheritance and is harder to extend
- `@classmethod` provides access to the class if needed later

### 7. Concise Git Commit Messages

**Requirements:**
- Keep messages **short and focused** (1-2 sentences)
- **NO signatures, attributions, or tool metadata**

#### ❌ WRONG (too verbose):
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

#### ❌ WRONG (with signatures):
```
Add disassembler framework

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

#### ✅ CORRECT:
```
Fix binary operation operand order
```

#### ✅ CORRECT:
```
Add disassembler framework
```

### 8. Blank Lines Between Conditional Blocks

Add a blank line between `if`/`elif`, `elif`/`elif`, `elif`/`else`, and `if`/`else`.

#### ✅ CORRECT:
```python
if cond1:
    action1()

elif cond2:
    action2()

else:
    action3()
```

### 9. Concise Comments

Keep comments brief and meaningful. Avoid redundant explanations.

- ✅ `# Skip self-assignment (var = var)`
- ❌ `# This code checks if the variable is equal to itself and if so we skip it`
- ✅ Use self-explanatory names instead of verbose comments
- ❌ Don't state the obvious: `x = 5  # Set x to 5`

## Enforcement

All code must follow these rules.

### Code Review Checklist:
- [ ] No hardcoded `4` for word size
- [ ] No hardcoded numeric constants without names
- [ ] All constants imported from appropriate modules
- [ ] Imports at module top level
- [ ] Comments reference constant names, not values
- [ ] All assignments have spaces: `x = value` (not `x=value`)
- [ ] All comments are in English (no Chinese or other languages)
- [ ] Comments are concise (no redundant explanations)
- [ ] Blank lines between conditional blocks
- [ ] No `@staticmethod` - use `@classmethod` instead
- [ ] Commit messages are concise (1-2 sentences)
- [ ] No Claude Code signatures or tool attributions in commits
