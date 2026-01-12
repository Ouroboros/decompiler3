# Spec: SSA Address Preservation

## Overview
SSA conversion must preserve instruction source addresses for debugging and IR validation.

## ADDED Requirements

### Requirement: SSA instruction address preservation
When SSA conversion creates a replacement instruction, the new instruction must inherit the `.address` field from the original instruction.

#### Scenario: MLILSetVarSSA preserves address
Given an MLIL_SET_VAR instruction at address 0x5B5C1
When SSA conversion creates MLILSetVarSSA to replace it
Then the new MLILSetVarSSA.address equals 0x5B5C1

#### Scenario: MLILCall preserves address after renaming
Given an MLIL_CALL instruction at address 0x5B5C7
When SSA conversion renames variables in the call arguments
And creates a new MLILCall with renamed arguments
Then the new MLILCall.address equals 0x5B5C7

#### Scenario: Pseudo-definitions have address 0
Given a call instruction that modifies an address-taken variable
When SSA conversion creates a pseudo-definition (MLILSetVarSSA with MLILUndef)
Then the pseudo-definition.address equals 0 (synthetic instruction)

## Technical Notes
- The `.address` field represents the source SCP offset
- Address 0 indicates a synthetic instruction (not derived from source)
- All non-synthetic SSA instructions must have non-zero addresses
