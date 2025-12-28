# Feature Specification: IR Semantic Consistency Validation

**Feature Branch**: `003-ir-pass-validation`
**Created**: 2025-12-29
**Status**: Draft
**Input**: User description: "验证 LLIL MLIL HLIL 语义逻辑是否一致"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compare LLIL and MLIL Semantics (Priority: P1)

As a decompiler developer, I want to compare the semantic logic of LLIL and MLIL for the same function, so I can verify that the two representations express the same computation.

**Why this priority**: LLIL and MLIL are the closest in abstraction level. If they don't match semantically, there's a fundamental bug in the IR design or conversion.

**Independent Test**: Run `python tools/ir_semantic_validator.py --compare llil mlil test.scp` and see a side-by-side comparison showing whether each operation in LLIL has a semantically equivalent operation in MLIL.

**Acceptance Scenarios**:

1. **Given** an SCP function, **When** I compare LLIL and MLIL, **Then** I see that every arithmetic operation (ADD, SUB, MUL, DIV) in LLIL has a matching operation in MLIL with equivalent operands.

2. **Given** an SCP function with conditionals, **When** I compare LLIL and MLIL, **Then** I see that branch conditions are logically equivalent (same comparison operators, same operands).

3. **Given** an SCP function, **When** semantics differ, **Then** I see a clear report indicating which instruction differs and how.

---

### User Story 2 - Compare MLIL and HLIL Semantics (Priority: P2)

As a decompiler developer, I want to compare the semantic logic of MLIL and HLIL for the same function, so I can verify that structured control flow reconstruction preserves the original logic.

**Why this priority**: HLIL restructures control flow. This comparison ensures the restructuring doesn't change the program's meaning.

**Independent Test**: Run `python tools/ir_semantic_validator.py --compare mlil hlil test.scp` and see whether the HLIL structured code expresses the same logic as the MLIL basic blocks.

**Acceptance Scenarios**:

1. **Given** an SCP function with a loop, **When** I compare MLIL and HLIL, **Then** I see that the HLIL loop condition matches the MLIL branch condition and the loop body contains the same operations.

2. **Given** an SCP function with if-else, **When** I compare MLIL and HLIL, **Then** I see that both branches contain semantically equivalent operations.

3. **Given** an SCP function with function calls, **When** I compare MLIL and HLIL, **Then** I see that call targets and arguments are identical.

---

### User Story 3 - Full Three-Way Semantic Comparison (Priority: P3)

As a decompiler developer, I want to compare LLIL, MLIL, and HLIL simultaneously for the same function, so I can see the semantic equivalence across all three IR layers at once.

**Why this priority**: A unified view helps identify where semantics diverge in the pipeline.

**Independent Test**: Run `python tools/ir_semantic_validator.py --all test.scp` and see a three-column comparison of LLIL, MLIL, and HLIL with semantic equivalence indicators.

**Acceptance Scenarios**:

1. **Given** an SCP function, **When** I run three-way comparison, **Then** I see aligned rows showing corresponding operations in each IR layer.

2. **Given** a function where all layers are consistent, **When** I run comparison, **Then** I see all rows marked as "semantically equivalent".

3. **Given** a function with a semantic difference, **When** I run comparison, **Then** I see the differing row highlighted with an explanation.

---

### User Story 4 - Variable/Storage Tracking Validation (Priority: P1)

As a decompiler developer, I want to verify that variable storage is tracked consistently across IR layers, so I can ensure stack slots, registers, and named variables refer to the same logical values.

**Why this priority**: Variable tracking is fundamental. If a stack slot in LLIL doesn't map to the correct variable in MLIL/HLIL, the decompiled code will be incorrect.

**Independent Test**: Run `python tools/ir_semantic_validator.py --variables test.scp` and see a mapping showing how each LLIL stack slot/register maps to MLIL variables and HLIL named variables.

**Acceptance Scenarios**:

1. **Given** an SCP function with local variables, **When** I validate variable tracking, **Then** I see that each LLIL stack slot maps to exactly one MLIL variable.

2. **Given** an SCP function with parameters, **When** I validate variable tracking, **Then** I see that frame slots map correctly to function parameters in all layers.

3. **Given** an SCP function with register usage, **When** I validate variable tracking, **Then** I see that register reads/writes are consistent across layers.

---

### User Story 5 - Batch Validation (Priority: P2)

As a decompiler developer, I want to validate all functions in an SCP file at once, so I can quickly check for semantic consistency issues across the entire script.

**Why this priority**: Real SCP files have many functions. Batch validation saves time and catches issues systematically.

**Independent Test**: Run `python tools/ir_semantic_validator.py --batch test.scp` and see a summary report showing pass/fail status for each function.

**Acceptance Scenarios**:

1. **Given** an SCP file with multiple functions, **When** I run batch validation, **Then** I see a summary showing how many functions passed and how many have differences.

2. **Given** an SCP file where all functions are consistent, **When** I run batch validation, **Then** I see "All N functions passed" message.

3. **Given** an SCP file with some inconsistent functions, **When** I run batch validation, **Then** I see a list of functions with issues and can drill down into details.

---

### User Story 6 - Type Consistency Validation (Priority: P2)

As a decompiler developer, I want to verify that type information is consistent across IR layers, so I can ensure type inference produces correct results.

**Why this priority**: Types affect code generation and understanding. Inconsistent types can lead to incorrect decompiled code.

**Independent Test**: Run `python tools/ir_semantic_validator.py --types test.scp` and see type assignments for each variable across IR layers.

**Acceptance Scenarios**:

1. **Given** an SCP function with integer operations, **When** I validate types, **Then** I see that operand types are consistent (e.g., int + int = int in all layers).

2. **Given** an SCP function with string operations, **When** I validate types, **Then** I see that string variables are correctly typed in MLIL and HLIL.

3. **Given** an SCP function where type inference differs, **When** I validate types, **Then** I see a report explaining the type difference.

---

### User Story 7 - Source Location Mapping (Priority: P3)

As a decompiler developer, when I find a semantic inconsistency, I want to know the original SCP bytecode location, so I can debug the issue in the original script.

**Why this priority**: Finding the root cause requires knowing where in the original bytecode the issue originates.

**Independent Test**: When a difference is reported, see the original SCP instruction offset alongside the LLIL/MLIL/HLIL locations.

**Acceptance Scenarios**:

1. **Given** a semantic difference is detected, **When** I view the report, **Then** I see the original SCP bytecode offset (e.g., "SCP offset 0x1234").

2. **Given** a difference in a complex expression, **When** I view the report, **Then** I see which sub-expression differs and its original location.

---

### Edge Cases

- What happens when HLIL has fewer instructions due to optimization (e.g., dead code elimination)?
- How are syscalls compared across layers (same signature, different representation)?
- What happens when control flow is restructured (GOTO → while loop)?
- How are inlined expressions in HLIL compared to separate statements in MLIL?
- How are SSA phi nodes handled when comparing MLIL with SSA to non-SSA LLIL?
- What happens when a variable is optimized away (e.g., constant propagation)?
- How are compound expressions (a + b * c) compared when tree structure differs?
- How are global variable accesses compared (different naming conventions)?
- How are string constants from the constant pool validated?
- What happens when LLIL has no type info but MLIL/HLIL do?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load an SCP file and generate LLIL, MLIL, and HLIL for each function
- **FR-002**: System MUST compare arithmetic operations across IR layers for semantic equivalence
- **FR-003**: System MUST compare comparison operations (EQ, NE, LT, LE, GT, GE) across IR layers
- **FR-004**: System MUST compare logical operations (AND, OR, NOT) across IR layers
- **FR-005**: System MUST compare control flow structures (branches, loops, returns) for semantic equivalence
- **FR-006**: System MUST compare function/syscall arguments across IR layers
- **FR-007**: System MUST identify and report semantic differences with context
- **FR-008**: System MUST handle expected differences (e.g., stack → variable, GOTO → while) without false positives
- **FR-009**: System MUST output a human-readable comparison report
- **FR-010**: System MUST support comparing any two IR layers or all three simultaneously
- **FR-011**: System MUST track variable/storage mappings across layers (stack slot → variable → named variable)
- **FR-012**: System MUST verify constant values are preserved across layers
- **FR-013**: System MUST verify expression tree structures are semantically equivalent (even if tree shape differs)
- **FR-014**: System MUST verify side-effect ordering is preserved (assignments, calls)
- **FR-015**: System MUST support batch validation of all functions in an SCP file
- **FR-016**: System MUST generate summary statistics for batch validation
- **FR-017**: System MUST compare type information across layers where available
- **FR-018**: System MUST compare global variable access patterns across layers
- **FR-019**: System MUST validate string constant references across layers
- **FR-020**: System MUST report original SCP bytecode offset for each difference found
- **FR-021**: System MUST support exporting validation results for regression testing

### Key Entities

- **IRFunction**: Represents a function in any IR layer (LLIL, MLIL, or HLIL)
- **SemanticOperation**: Normalized representation of an operation for comparison
- **ComparisonResult**: Result of comparing two operations (equivalent, different, transformed)
- **DifferenceReport**: Details of a semantic difference including location and explanation
- **VariableMapping**: Tracks correspondence between storage locations across IR layers
- **BatchReport**: Aggregates validation results for all functions in an SCP file
- **SourceLocation**: Maps IR instruction to original SCP bytecode offset
- **TypeInfo**: Type assignment for a variable or expression in each IR layer

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Comparison report clearly shows semantic equivalence or differences for each operation
- **SC-002**: System correctly identifies 100% of arithmetic/logical operation matches
- **SC-003**: System correctly handles expected transformations (stack→variable, goto→loop) without false positives
- **SC-004**: Report generation completes within 2 seconds per function
- **SC-005**: Developers can use the report to identify semantic bugs in IR definitions or conversion code
- **SC-006**: Variable tracking correctly maps 100% of stack slots to their corresponding variables
- **SC-007**: Batch validation processes an entire SCP file and produces summary in under 30 seconds
- **SC-008**: Zero false negatives - all genuine semantic differences are reported
- **SC-009**: Every difference report includes original SCP bytecode location
- **SC-010**: Type consistency check identifies all type mismatches between layers

## Assumptions

- All three IR layers are generated from the same SCP file
- Semantic equivalence is defined as "produces the same result given the same inputs"
- Expected transformations (stack elimination, control flow structuring) are not semantic differences
- The tool compares the final output of each IR layer, not intermediate states
- Expression tree comparison uses semantic equivalence, not structural identity (a+b == b+a for commutative ops)
- Dead code elimination in HLIL is an expected optimization, not a semantic difference
- LLIL may lack type information; comparison only fails if MLIL and HLIL types conflict
- Global variable names may differ across layers; comparison uses resolved addresses
