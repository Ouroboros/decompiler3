# Proposal: fix-ssa-address-preservation

**Change ID**: `fix-ssa-address-preservation`
**Status**: Draft
**Created**: 2026-01-12

## Summary
Fix SSA conversion pass to preserve instruction addresses when creating new SSA instructions.

## Problem
When the SSA conversion pass (`ir/mlil/mlil_ssa.py`) creates new MLIL instructions (e.g., `MLILSetVarSSA`, `MLILCall`), it does not copy the `.address` field from the original instruction. This causes:

1. MLIL instructions to have `address = 0` after SSA conversion
2. Incorrect address mapping in the IR semantic validator
3. False positives in LLIL-MLIL comparison (e.g., `call` at 0x5B5C7 vs `call` at 0x0`)

## Root Cause
In `_rename_inst()` and `_rename_stmt()` methods, new instructions are created without copying the address:

```python
# Line 390 - address not copied
return MLILSetVarSSA(MLILVariableSSA(inst.var, new_ver), new_value)

# Line 484 - address not copied
return MLILCall(stmt.target, new_args)
```

## Solution
Copy `.address` field when creating replacement instructions in SSA conversion.

## Motivation

- **IR Validation Accuracy**: Address mapping is essential for validating semantic consistency between LLIL/MLIL/HLIL layers
- **Debugging Support**: Source addresses allow tracing MLIL instructions back to original SCP bytecode
- **Data Integrity**: SSA is a transformation, not a source change - addresses should be preserved through transformations
- **Validator False Positives**: Without addresses, validator shows LLIL `call` at 0x5B5C7 vs MLIL `call` at 0x0, incorrectly flagging as mismatch

## Scope
- Single file: `ir/mlil/mlil_ssa.py`
- Affects: `_rename_inst()` and `_rename_stmt()` methods

## Impact
- Low risk - preserves existing behavior, only adds address copying
- No API changes
- Improves validator accuracy
