# Proposal: Simplify Validator Output

**Change ID**: `simplify-validator-output`
**Status**: Applied
**Created**: 2025-01-11

## Summary

Simplify IR semantic validator output to show only instruction names (opcodes) instead of full instruction text with operands.

## Current Behavior

```
0x0005E6A8   | call party_clear(STACK[sp<4>])           | goto loc_5E6AB                           | = OK
```

The output includes full instruction representation with operands, targets, and arguments.

## Proposed Behavior

```
0x0005E6A8   | call                                     | goto                                     | = OK
```

Only show the instruction opcode/name, without operands or arguments.

## Motivation

- Cleaner, more scannable output
- Focus on instruction type matching rather than full semantic details
- Reduced visual noise when validating large files
- Easier to spot instruction type mismatches at a glance

## Scope

- `tools/ir_semantic_validator.py`
  - `build_llil_mlil_address_table()` - LLIL-MLIL comparison
  - `AddressAligner._create_mapping_for_offset()` - MLIL-HLIL comparison
  - `_run_dual_output()` - file output mode

## Impact

- Low risk - output format change only
- No semantic logic changes
- Backward compatible (no API changes)
