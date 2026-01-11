<!--
Sync Impact Report
==================
Version change: 1.0.0 → 1.1.0
Bump rationale: MINOR - removed one principle (backward compatible)

Modified principles: None
Added sections: None
Removed sections:
  - VII. Read-Only Boundaries (binaryninja-api/ restriction)

Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ No constitution references
  - .specify/templates/spec-template.md: ✅ No constitution references
  - .specify/templates/tasks-template.md: ✅ No constitution references

Follow-up TODOs: None
-->

# Falcom Decompiler Constitution

## Core Principles

### I. No Dynamic Attribute Manipulation
**NEVER** use `getattr`, `hasattr`, `setattr`, or `delattr` to add, remove, or manipulate class type fields.

- All class attributes must be explicitly declared in the class definition
- Use direct attribute access (`obj.attr`) instead of `getattr(obj, 'attr')`
- Use type annotations and explicit field definitions
- If an attribute might not exist, the class design is wrong - fix the class, not the access pattern

**Rationale**: Dynamic attribute manipulation hides type information, bypasses static analysis, and creates maintenance nightmares. If you need to check whether an attribute exists, the code architecture is fundamentally broken.

### II. Explicit Type Definitions
All class fields must be:
- Declared at class level or in `__init__`
- Have explicit type annotations
- Have default values or be guaranteed initialized

### III. No Hardcoded Magic Numbers
Use named constants instead of literal numbers in code.
- Define constants at module level: `WORD_SIZE = 4`
- Reference constants in expressions: `offset // WORD_SIZE`
- Document constants in comments when referenced

### IV. Import Organization
- All imports at module top level
- Exception: Circular dependency resolution with explanatory comment

### V. Code Style Consistency
- Space around assignment operators: `x = 5`
- Blank lines between conditional blocks (`if`/`elif`/`else`)
- English-only for code, comments, and commit messages
- Chinese for user communication

### VI. No Auto-Commit Policy
Never auto-commit changes. Only create commits when explicitly requested by user.

## Governance

This constitution supersedes all other practices. Violations must be corrected before code is merged.

All code modifications require explicit user confirmation before implementation.

**Version**: 1.1.0 | **Ratified**: 2026-01-09 | **Last Amended**: 2026-01-09
