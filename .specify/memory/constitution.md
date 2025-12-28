<!--
Sync Impact Report
==================
Version change: 1.0.0 → 1.1.0
Modified principles: Removed "Protected Directories", renumbered III-IX to III-VIII
Added sections: None
Removed sections: Protected Directories principle
Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ compatible
  - .specify/templates/spec-template.md: ✅ compatible
  - .specify/templates/tasks-template.md: ✅ compatible
Follow-up TODOs: None
-->

# ED9 Decompiler Constitution

## Core Principles

### I. Language Separation

- **User communication**: Chinese
- **Code, comments, commit messages, tool descriptions**: English

This ensures clear separation between user-facing communication and technical artifacts that may be shared or reviewed by international contributors.

### II. Git Discipline

- NEVER auto-commit; commits MUST be explicitly requested by user
- Commit messages MUST be concise (1-2 sentences), no signatures or metadata
- Code modifications MUST NOT occur during discussion; wait for explicit user confirmation

### III. No Magic Numbers

- All numeric constants MUST use named constants (e.g., `offset // WORD_SIZE` not `offset // 4`)
- Common constants defined in `ir/llil.py`: `WORD_SIZE = 4`
- Comments referencing values MUST also use constant names

### IV. Import Structure

- All imports MUST be at module top level
- Exception: circular dependency imports inside functions, with explanatory comment

### V. Code Formatting

- Assignment operators MUST have surrounding spaces: `x = 5`
- Blank lines MUST separate conditional blocks (`if`/`elif`, `elif`/`elif`, `elif`/`else`, `if`/`else`)

### VI. Method Decorators

- NEVER use `@staticmethod`
- Use `@classmethod` instead for inheritance support

### VII. English-Only Comments

- All code comments MUST be in English
- No Chinese or other languages in code comments

### VIII. Concise Documentation

- Comments MUST be brief and meaningful
- Avoid redundant explanations that restate obvious code behavior
- Prefer self-explanatory names over verbose comments

## Code Style

### Formatting Checklist

- [ ] Spaces around `=`
- [ ] Blank lines between conditional blocks
- [ ] Imports at top level
- [ ] Named constants (no magic numbers)
- [ ] `@classmethod` not `@staticmethod`
- [ ] English-only comments
- [ ] Concise comments

## Development Workflow

### Before Any Task

1. Read CLAUDE.md to refresh rules
2. Verify understanding of user requirements

### During Implementation

1. Wait for explicit user confirmation before modifying code
2. Use named constants for all numeric values
3. Follow formatting rules strictly

### Commit Process

1. Wait for explicit user request to commit
2. Write concise commit message (1-2 sentences)
3. No signatures or metadata in commit messages

## Governance

This constitution defines the non-negotiable coding standards for the ED9 Decompiler project. All contributions MUST comply with these principles.

### Amendment Process

1. Propose change with rationale
2. Document impact on existing code
3. Update version according to semantic versioning:
   - MAJOR: Backward incompatible changes to principles
   - MINOR: New principles or expanded guidance
   - PATCH: Clarifications, wording fixes

### Compliance

- All code reviews MUST verify compliance with these principles
- Violations MUST be corrected before merge
- Runtime guidance is maintained in `CLAUDE.md`

**Version**: 1.1.0 | **Ratified**: 2025-12-28 | **Last Amended**: 2025-12-28
