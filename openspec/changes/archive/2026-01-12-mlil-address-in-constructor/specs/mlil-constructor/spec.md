# Spec: MLIL Constructor Address Parameter

## Overview
MLIL instruction constructors should accept an optional `address` keyword parameter to enable single-line instruction creation with address preservation.

## MODIFIED Requirements

### Requirement: MLIL instruction constructor signature
All MLIL instruction classes MUST accept `**kwargs` and pass them to the parent class constructor.

#### Scenario: Create instruction with address
Given a source instruction with address 0x5B5C7
When creating a new MLILCall with `address=0x5B5C7`
Then the new instruction's `.address` field equals 0x5B5C7

#### Scenario: Create instruction without address (backward compatibility)
Given no address parameter is provided
When creating a new MLILCall
Then the new instruction's `.address` field equals 0 (default)

#### Scenario: Address propagation through inheritance
Given MLILAdd inherits from MLILBinaryOp
When creating `MLILAdd(lhs, rhs, address=0x1234)`
Then the address is correctly propagated through the class hierarchy

## ADDED Requirements

### Requirement: Base class keyword parameter handling
`MediumLevelILInstruction.__init__` MUST accept `address` as a keyword-only parameter.

#### Scenario: Base class accepts address
Given `MediumLevelILInstruction.__init__(operation, *, address=0)`
When a subclass calls `super().__init__(op, address=addr)`
Then the address is set correctly

## Technical Notes
- Keyword-only parameter (`*`) prevents positional argument confusion
- `**kwargs` pattern allows future extensibility without signature changes
- Default value of 0 maintains backward compatibility
