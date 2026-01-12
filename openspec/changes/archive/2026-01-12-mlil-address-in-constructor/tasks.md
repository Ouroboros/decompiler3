# Tasks: mlil-address-in-constructor

## Implementation Tasks

### 1. Update base class (ir/mlil/mlil.py)
- [x] Add `address` keyword parameter to `MediumLevelILInstruction.__init__`

### 2. Update MLIL instruction classes (ir/mlil/mlil.py)
- [x] `MLILConst` - add `**kwargs`
- [x] `MLILUndef` - add `**kwargs`
- [x] `MLILVar` - add `**kwargs`
- [x] `MLILSetVar` - add `**kwargs`
- [x] `MLILBinaryOp` (base) - add `**kwargs`
- [x] All binary subclasses (MLILAdd, MLILSub, etc.) - add `**kwargs`
- [x] `MLILUnaryOp` (base) - add `**kwargs`
- [x] All unary subclasses (MLILNeg, MLILNot, etc.) - add `**kwargs`
- [x] `MLILGoto` - add `**kwargs`
- [x] `MLILIf` - add `**kwargs`
- [x] `MLILRet` - add `**kwargs`
- [x] `MLILCall` - add `**kwargs`
- [x] `MLILSyscall` - add `**kwargs`
- [x] `MLILCallScript` - add `**kwargs`
- [x] `MLILLoadGlobal` - add `**kwargs`
- [x] `MLILStoreGlobal` - add `**kwargs`
- [x] `MLILLoadReg` - add `**kwargs`
- [x] `MLILStoreReg` - add `**kwargs`
- [x] `MLILNop` - add `**kwargs`
- [x] `MLILDebug` - add `**kwargs`

### 3. Update SSA instruction classes (ir/mlil/mlil_ssa.py)
- [x] `MLILSetVarSSA` - add `**kwargs`
- [x] `MLILVarSSA` - add `**kwargs`
- [x] `MLILPhi` - add `**kwargs`

### 4. Update SSA conversion to use new API (ir/mlil/mlil_ssa.py)
- [x] `_rename_inst()` - use `address=` parameter
- [x] `_rename_stmt()` - use `address=` parameter
- [x] `_apply_mapping_to_stmt()` - use `address=` parameter

### 5. Update SSA optimization passes
- [x] `pass_ssa_constant_propagation.py` - use `address=` parameter
- [x] `pass_ssa_copy_propagation.py` - use `address=` parameter
- [x] `pass_ssa_expression_inlining.py` - use `address=` parameter
- [x] `pass_ssa_sccp.py` - use `address=` parameter
- [x] `pass_reg_global_propagation.py` - use `address=` parameter

### 6. Validation
- [x] Run validator: `python -m tools.ir_semantic_validator --output-dir tests/output tests/debug.dat`
- [x] Verify LLIL-MLIL pass rate unchanged (91%)

## Dependencies
None - self-contained refactoring

## Notes
- All changes are backward compatible (address defaults to 0)
- Binary operation subclasses (MLILAdd, MLILSub, etc.) inherit from MLILBinaryOp, still need `**kwargs` for proper propagation
- Unary operation subclasses (MLILNeg, MLILNot, etc.) inherit from MLILUnaryOp, still need `**kwargs` for proper propagation
