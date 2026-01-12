# Tasks: Simplify Validator Output

## Implementation Tasks

- [x] **T1**: Create `_get_instruction_name()` helper function
  - Extract opcode name from LLIL instruction (e.g., `call`, `goto`, `debug`)
  - Extract opcode name from MLIL instruction (e.g., `goto`, `debug`, `call`)
  - Extract opcode name from HLIL statement (e.g., `if`, `while`, `return`)

- [x] **T2**: Update `build_llil_mlil_address_table()` in LLIL-MLIL comparison
  - Replace `str(instr)` with instruction name extraction
  - Apply to both LLIL and MLIL columns

- [x] **T3**: Update `AddressAligner._create_mapping_for_offset()` in MLIL-HLIL comparison
  - Replace `str(instr)` / `str(stmt)` with instruction name extraction
  - Apply to both MLIL and HLIL columns

- [x] **T4**: Regenerate test output files
  - Run validator with `--output-dir tests/output`
  - Verify simplified format in output files

## Validation

- [x] Output shows only instruction names without operands
- [x] All existing validation modes still work
- [x] File output mode produces correct simplified format
