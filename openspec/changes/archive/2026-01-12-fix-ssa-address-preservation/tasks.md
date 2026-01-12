# Tasks: fix-ssa-address-preservation

## Implementation Tasks

### 1. Fix `_rename_inst()` address preservation (mlil_ssa.py)
- [x] Copy address when creating `MLILSetVarSSA` (line ~390)
- [x] Copy address when creating pseudo-definitions for address-taken variables (line ~414)

### 2. Fix `_rename_stmt()` address preservation (mlil_ssa.py)
- [x] Copy address when creating `MLILIf` (line ~470)
- [x] Copy address when creating `MLILRet` (line ~477)
- [x] Copy address when creating `MLILCall` (line ~484)
- [x] Copy address when creating `MLILSyscall` (line ~487)
- [x] Copy address when creating `MLILCallScript` (line ~490)
- [x] Copy address for `MLILStoreGlobal` and `MLILStoreReg`

### 3. Fix `_apply_mapping_to_stmt()` address preservation (mlil_ssa.py)
- [x] Copy address when creating all replacement statements during SSA deconstruction

### 4. Fix SSA optimization passes
- [x] Fix `pass_ssa_constant_propagation.py` - all instruction reconstructions
- [x] Fix `pass_ssa_copy_propagation.py` - all instruction reconstructions
- [x] Fix `pass_ssa_expression_inlining.py` - all instruction reconstructions
- [x] Fix `pass_ssa_sccp.py` - all instruction reconstructions
- [x] Fix `pass_reg_global_propagation.py` - `_rebuild_call()` method

### 5. Validation
- [x] Run validator on test file: `python -m tools.ir_semantic_validator --batch tests/debug.dat`
- [x] Verify MLIL_CALL and MLIL_GOTO have same address (0x5B5C7, not 0x0)
- [x] Check validation pass rate: LLIL-MLIL 91%, MLIL-HLIL 90%

## Dependencies
None - self-contained fix

## Verification
After fix, this shows correct addresses:
```
>>> 0x0005B5C7: call                                     | call; goto
```

Both `call` and `goto` now share the same address 0x5B5C7 (was 0x00000000 for call before fix).
