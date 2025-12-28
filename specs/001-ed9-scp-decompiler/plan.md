# Implementation Plan: ED9 SCP Script Decompiler

**Branch**: `001-ed9-scp-decompiler` | **Date**: 2025-12-28 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-ed9-scp-decompiler/spec.md`

**Note**: This document describes the architecture of an already-implemented decompiler.

## Summary

A multi-phase decompiler that transforms ED9/ED61st engine SCP bytecode into readable TypeScript source code through a three-tier intermediate representation (LLIL → MLIL → HLIL) with optional signature-based output beautification.

## Technical Context

**Language/Version**: Python 3.14
**Primary Dependencies**: PyYAML (signature database), unittest (testing)
**Storage**: File-based I/O (.dat input, .ts output, optional .llil.asm/.mlil.asm/.hlil.ts debug output)
**Testing**: unittest (tests/test_scp_parser.py)
**Target Platform**: Cross-platform CLI tool (Windows primary, Linux/macOS compatible)
**Project Type**: Single project
**Performance Goals**: N/A (no specific requirements)
**Constraints**: N/A
**Scale/Scope**: Single game engine (ED9/ED61st), individual script files

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Language Separation | ✅ Pass | Chinese for user comm, English for code |
| II. Git Discipline | ✅ Pass | No auto-commit, explicit user confirmation |
| III. No Magic Numbers | ✅ Pass | WORD_SIZE = 4 used consistently |
| IV. Import Structure | ✅ Pass | Imports at module top level |
| V. Code Formatting | ✅ Pass | Spaces around =, blank lines between conditionals |
| VI. Method Decorators | ✅ Pass | @classmethod used, no @staticmethod |
| VII. English-Only Comments | ✅ Pass | All comments in English |
| VIII. Concise Documentation | ✅ Pass | Brief, meaningful comments |

## Project Structure

### Documentation (this feature)

```text
specs/001-ed9-scp-decompiler/
├── plan.md              # This file - architecture overview
├── spec.md              # Feature specification
├── research.md          # Technical research - file format, VM, IR design
├── data-model.md        # Data structures - all IL types and entities
├── quickstart.md        # Usage guide with examples
├── checklists/
│   └── requirements.md  # Quality checklist
└── contracts/           # Public API documentation
    ├── parser.md        # ScpParser, ScpValue API
    ├── ir-llil.md       # LowLevelIL API
    ├── ir-mlil.md       # MediumLevelIL API
    ├── ir-hlil.md       # HighLevelIL API
    ├── signatures.md    # FormatSignatureDB API
    └── codegen.md       # TypeScriptGenerator API
```

### Source Code (repository root)

```text
decompiler3/
├── falcom/                      # Game engine specific code
│   ├── base/                    # Base classes
│   └── ed9/                     # ED9 engine implementation
│       ├── parser/              # SCP file parsing
│       │   ├── scp.py           # Main SCP parser (ScpValue, ScpFunction)
│       │   ├── types_scp.py     # SCP type definitions
│       │   └── types_parser.py  # Parser type helpers
│       ├── disasm/              # Disassembler
│       │   ├── disassembler.py  # Main disassembler
│       │   ├── instruction.py   # Instruction definitions
│       │   ├── ed9_optable.py   # Opcode table
│       │   └── basic_block.py   # Basic block analysis
│       ├── lifters/             # LLIL lifter
│       │   └── vm_lifter.py     # VM bytecode → LLIL
│       ├── signatures/          # YAML signature files
│       ├── format_signatures.py # FormatSignatureDB implementation
│       ├── llil_builder.py      # LLIL construction
│       ├── mlil_converter.py    # LLIL → MLIL conversion
│       ├── mlil_passes.py       # MLIL optimization passes
│       ├── hlil_converter.py    # MLIL → HLIL conversion
│       └── hlil_passes.py       # HLIL optimization passes
│
├── ir/                          # Intermediate representation framework
│   ├── core/                    # Core IL infrastructure
│   │   ├── il_base.py           # Base IL classes
│   │   └── il_options.py        # IL options/config
│   ├── llil/                    # Low-Level IL
│   │   ├── llil.py              # LLIL definitions
│   │   ├── lifters/             # Architecture lifters
│   │   └── passes/              # LLIL passes
│   ├── mlil/                    # Medium-Level IL
│   │   ├── mlil.py              # MLIL definitions (SSA)
│   │   └── passes/              # MLIL passes
│   │       ├── pass_ssa_construction.py
│   │       └── pass_type_inference.py
│   ├── hlil/                    # High-Level IL
│   │   ├── hlil.py              # HLIL definitions (structured control flow)
│   │   ├── hlil_formatter.py    # HLIL formatting
│   │   ├── mlil_to_hlil.py      # MLIL → HLIL conversion
│   │   └── passes/              # HLIL passes
│   │       ├── pass_mlil_to_hlil.py
│   │       ├── pass_expression_simplification.py
│   │       ├── pass_control_flow_optimization.py
│   │       ├── pass_common_return_extraction.py
│   │       ├── pass_copy_propagation.py
│   │       └── pass_dead_code_elimination.py
│   └── pretty/                  # Pretty printing utilities
│
├── codegen/                     # Code generation
│   └── typescript.py            # TypeScript code generator
│
├── common/                      # Shared utilities
│   ├── config.py                # Configuration
│   ├── enum.py                  # Enum utilities
│   └── utils.py                 # General utilities
│
└── tests/                       # Unit tests
    └── test_scp_parser.py       # Main test file
```

**Structure Decision**: Single project structure. Game-specific code in `falcom/ed9/`, generic IR framework in `ir/`, code generation in `codegen/`.

## Architecture Overview

### Processing Pipeline

```
.dat file
    │
    ▼
┌─────────────────┐
│  SCP Parser     │  falcom/ed9/parser/scp.py
│  (ScpValue,     │  - Parse file header
│   ScpFunction)  │  - Extract functions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Disassembler   │  falcom/ed9/disasm/
│  (Instructions, │  - Decode opcodes
│   Basic Blocks) │  - Identify basic blocks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLIL Lifter    │  falcom/ed9/lifters/vm_lifter.py
│  (Low-Level IL) │  - Stack operations
└────────┬────────┘  - Direct instruction mapping
         │
         ▼
┌─────────────────┐
│  MLIL Converter │  falcom/ed9/mlil_converter.py
│  (Medium-Level  │  - SSA construction
│   IL + SSA)     │  - Type inference
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HLIL Converter │  falcom/ed9/hlil_converter.py
│  (High-Level IL)│  - Control flow structuring
│                 │  - if/while/switch recovery
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TypeScript     │  codegen/typescript.py
│  Code Generator │  - Generate readable code
│  + Signatures   │  - Apply format signatures
└────────┬────────┘
         │
         ▼
    .ts file
```

### Key Data Structures

| Entity | Location | Description |
|--------|----------|-------------|
| ScpValue | falcom/ed9/parser/types_scp.py | 4-byte value with 2-bit type tag (Raw/Int/Float/String) |
| ScpFunction | falcom/ed9/parser/scp.py | Function with name, offset, parameters, basic blocks |
| LowLevelIL | ir/llil/llil.py | Stack-based IL close to VM bytecode |
| MediumLevelIL | ir/mlil/mlil.py | SSA form with variable analysis |
| HighLevelIL | ir/hlil/hlil.py | Structured control flow (if/while/switch) |
| FormatSignatureDB | falcom/ed9/format_signatures.py | YAML-based syscall/function signatures |

## Complexity Tracking

> No constitution violations - feature uses straightforward architecture.

| Aspect | Complexity | Justification |
|--------|------------|---------------|
| 3-tier IR | Medium | Standard decompiler architecture (LLIL→MLIL→HLIL) |
| Signature DB | Low | Simple YAML loader with enum/function lookup |
| TypeScript output | Low | Direct HLIL → source code mapping |

## Related Documents

| Document | Description |
|----------|-------------|
| [spec.md](spec.md) | Feature specification with user stories and requirements |
| [research.md](research.md) | Technical research: SCP format, VM architecture, IR design |
| [data-model.md](data-model.md) | Data structures: ScpValue, IL hierarchies, type system |
| [quickstart.md](quickstart.md) | Usage guide with complete examples |
| [contracts/](contracts/) | Public API documentation for all modules |
