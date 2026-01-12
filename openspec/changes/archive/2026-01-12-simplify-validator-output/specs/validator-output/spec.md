# Spec: Validator Output Format

## MODIFIED Requirements

### Requirement: Instruction Display Format

The IR semantic validator shall display only instruction names (opcodes) in comparison output, without operands or arguments.

#### Scenario: LLIL-MLIL comparison output
- **Given**: A validator run comparing LLIL and MLIL
- **When**: Output is generated
- **Then**: LLIL column shows only instruction name (e.g., `call`, `goto`, `if`)
- **And**: MLIL column shows only instruction name (e.g., `goto`, `call`, `debug.line`)
- **And**: No operands, targets, or arguments are displayed

#### Scenario: MLIL-HLIL comparison output
- **Given**: A validator run comparing MLIL and HLIL
- **When**: Output is generated
- **Then**: MLIL column shows only instruction name
- **And**: HLIL column shows only statement type (e.g., `if`, `while`, `return`, `call`)
- **And**: No operands, conditions, or arguments are displayed

#### Scenario: Special markers preserved
- **Given**: An instruction that was eliminated or transformed
- **When**: Output is generated
- **Then**: Special markers like `(eliminated)`, `(none)` are preserved as-is
