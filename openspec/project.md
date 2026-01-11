# Project Context

## Purpose
Decompiler for Falcom game scripts (ED9/Trails series). Converts SCP bytecode into readable high-level code through a multi-level IR (Intermediate Representation) pipeline, similar to Binary Ninja's architecture.

Goals:
- Parse and disassemble SCP script files
- Lift bytecode to multi-level IR (LLIL → MLIL → HLIL)
- Generate readable decompiled output
- Validate semantic consistency across IR layers

## Tech Stack
- Python 3.14
- No external frameworks (pure Python)
- In-memory analysis with text/JSON output
- Git Bash (MINGW64) on Windows

## Project Conventions

### Code Style
- Named constants instead of magic numbers: `offset // WORD_SIZE` not `offset // 4`
- Imports at module top level (not inside functions)
- Spaces around `=` in assignments
- Blank lines between if/elif/else blocks
- English-only comments (concise, no redundancy)
- `@classmethod` instead of `@staticmethod`

### Architecture Patterns
Multi-level IR pipeline (inspired by Binary Ninja):
- **LLIL** (Low Level IL): Stack-based, close to bytecode
- **MLIL** (Medium Level IL): Variable-based, stack eliminated
- **HLIL** (High Level IL): Structured control flow, readable output

Key modules:
- `ir/llil/` - Low Level IL definitions and builder
- `ir/mlil/` - Medium Level IL, LLIL→MLIL translator
- `ir/hlil/` - High Level IL, MLIL→HLIL converter
- `falcom/ed9/` - ED9-specific parser and lifters
- `tools/` - Validation and analysis tools

### Testing Strategy
- `tools/ir_semantic_validator.py` for IR consistency validation
- Output to `tests/output/` directory
- Batch validation across SCP files

### Git Workflow
- Never auto-commit (wait for explicit user request)
- Concise commit messages (1-2 sentences, no signatures)
- Feature branches with numeric prefix: `NNN-feature-name`
- Specs stored in `specs/NNN-feature-name/`

## Domain Context
- **SCP files**: Falcom script bytecode format
- **Stack-based VM**: Original bytecode uses stack operations
- **LLIL lifting**: Converts bytecode to LLIL instructions
- **Stack elimination**: MLIL removes stack operations, introduces variables
- **Control flow recovery**: HLIL reconstructs if/while/switch structures
- **Address mapping**: SCP offset → LLIL → MLIL → HLIL tracking

## Important Constraints
- No dynamic attribute manipulation (`getattr`/`hasattr`/`setattr` for type fields)
- User communication in Chinese, code/comments in English
- No code changes during discussion (wait for user confirmation)

## External Dependencies
- None (pure Python, self-contained)
