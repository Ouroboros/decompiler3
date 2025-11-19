# Coding Style Guide

## Rules

### -1. SPEAK CHINESE

### 0. NEVER TOUCH binaryninja-api/
READ ONLY - no writes, edits, or deletes.

### 1. NO HARDCODED MAGIC NUMBERS
Use named constants: `offset // WORD_SIZE` not `offset // 4`
Common: `WORD_SIZE = 4` (in `ir/llil.py`)

### 2. Import at Module Top Level
Import at file top, not inside functions (except circular dependency with comment).

### 3. Comment Documentation
Reference constants in comments: `offset // WORD_SIZE` not `offset // 4`

### 4. Assignment Spacing
Always space around `=`: `x = 5`

### 5. Blank Lines Between Conditional Blocks
Add blank line between `if`/`elif`, `elif`/`elif`, `elif`/`else`, `if`/`else`.
```python
if cond1:
    action1()

elif cond2:
    action2()

else:
    action3()
```

### 6. English-Only Comments
No Chinese or other languages.

### 7. NO @staticmethod
Use `@classmethod` for inheritance support.

### 8. Concise Git Commit Messages
Short (1-2 sentences), no signatures/metadata.

## Checklist
- [ ] No hardcoded numbers (use named constants)
- [ ] Imports at top level
- [ ] Spaces around `=`
- [ ] Blank lines between conditional blocks
- [ ] English-only comments
- [ ] `@classmethod` not `@staticmethod`
- [ ] Concise commits (no signatures)

## Environment

### Python
- Path: `D:\Dev\Python\python.exe`
- Version: 3.14.0
- Environment: Git Bash (MINGW64) on Windows
