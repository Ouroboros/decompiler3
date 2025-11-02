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

### 2. Import Constants at Function/Method Level

For frequently used constants like `WORD_SIZE`, import at the function level to avoid circular dependencies:

```python
def some_method(self):
    from ir.llil import WORD_SIZE
    
    word_offset = offset // WORD_SIZE
```

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

## Enforcement

All code must follow these rules. Any hardcoded magic numbers will be rejected.

### Code Review Checklist:
- [ ] No hardcoded `4` for word size
- [ ] No hardcoded numeric constants without names
- [ ] All constants imported from appropriate modules
- [ ] Comments reference constant names, not values
