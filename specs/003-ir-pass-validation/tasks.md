# Tasks: IR Semantic Consistency Validation

**Input**: Design documents from `/specs/003-ir-pass-validation/`
**Prerequisites**: plan.md (required), spec.md (required), research.md

**Tests**: Not explicitly requested - tests omitted per specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `tools/` for main script, `tests/` for test files
- Based on plan.md: Single-file tool in `tools/` directory

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the base structure and shared components

- [ ] T001 Create `tools/ir_semantic_validator.py` with CLI argument parsing (argparse)
- [ ] T002 [P] Define `OperationKind` enum (ARITHMETIC, COMPARISON, LOGICAL, CALL, BRANCH, ASSIGN, etc.) in `tools/ir_semantic_validator.py`
- [ ] T003 [P] Define `ComparisonStatus` enum (EQUIVALENT, TRANSFORMED, DIFFERENT) in `tools/ir_semantic_validator.py`
- [ ] T004 [P] Define `IRLayer` enum (LLIL, MLIL, HLIL) in `tools/ir_semantic_validator.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data structures that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T005 Implement `SemanticOperation` dataclass with kind, operator, operands, result, source_location in `tools/ir_semantic_validator.py`
- [ ] T006 Implement `SemanticOperand` dataclass for normalized operand representation in `tools/ir_semantic_validator.py`
- [ ] T007 Implement `SourceLocation` dataclass with scp_offset, llil_index, mlil_index, hlil_index in `tools/ir_semantic_validator.py`
- [ ] T008 Implement `ComparisonResult` dataclass with status, source_layer, target_layer, explanation, source_location in `tools/ir_semantic_validator.py`
- [ ] T009 Implement `VariableMapping` dataclass with llil_storage, mlil_var, hlil_var, type_info in `tools/ir_semantic_validator.py`
- [ ] T010 Implement IR generation pipeline: load SCP → generate LLIL → convert to MLIL → convert to HLIL in `tools/ir_semantic_validator.py`
- [ ] T011 Implement `normalize_llil_operation()` to convert LLIL instructions to `SemanticOperation` in `tools/ir_semantic_validator.py`
- [ ] T012 Implement `normalize_mlil_operation()` to convert MLIL instructions to `SemanticOperation` in `tools/ir_semantic_validator.py`
- [ ] T013 Implement `normalize_hlil_operation()` to convert HLIL instructions to `SemanticOperation` in `tools/ir_semantic_validator.py`

**Checkpoint**: Foundation ready - core normalization infrastructure complete

---

## Phase 3: User Story 1 - LLIL/MLIL Comparison (Priority: P1) 🎯 MVP

**Goal**: Compare semantic logic between LLIL and MLIL for the same function

**Independent Test**: `python tools/ir_semantic_validator.py --compare llil mlil test.scp`

### Implementation for User Story 1

- [ ] T014 [US1] Implement `LLILMLILComparator` class with `compare()` method in `tools/ir_semantic_validator.py`
- [ ] T015 [US1] Implement arithmetic operation comparison (ADD, SUB, MUL, DIV, MOD, NEG) in `LLILMLILComparator`
- [ ] T016 [US1] Implement comparison operation matching (EQ, NE, LT, LE, GT, GE) in `LLILMLILComparator`
- [ ] T017 [US1] Implement logical/bitwise operation comparison (AND, OR, NOT, XOR) in `LLILMLILComparator`
- [ ] T018 [US1] Implement control flow comparison (BRANCH→IF, JMP→GOTO, CALL, SYSCALL, RET) in `LLILMLILComparator`
- [ ] T019 [US1] Implement constant value verification across layers in `LLILMLILComparator`
- [ ] T020 [US1] Implement string constant validation (pool index → literal) in `LLILMLILComparator`
- [ ] T021 [US1] Implement global variable access comparison in `LLILMLILComparator`
- [ ] T022 [US1] Implement side-effect ordering verification (assignments, calls) in `LLILMLILComparator`
- [ ] T023 [US1] Implement expected transformation detection for storage operations (STACK_LOAD→VAR, STACK_STORE→SET_VAR) in `LLILMLILComparator`
- [ ] T024 [US1] Implement expected transformation detection for Falcom VM ops (PUSH_CALLER_FRAME elimination, CALL_SCRIPT) in `LLILMLILComparator`
- [ ] T025 [US1] Implement difference report generation for LLIL-MLIL comparison in `tools/ir_semantic_validator.py`
- [ ] T026 [US1] Wire `--compare llil mlil` CLI option to `LLILMLILComparator` in `tools/ir_semantic_validator.py`

**Checkpoint**: LLIL-MLIL comparison working with `--compare llil mlil` flag

---

## Phase 4: User Story 4 - Variable/Storage Tracking (Priority: P1)

**Goal**: Track variable correspondence across IR layers

**Independent Test**: `python tools/ir_semantic_validator.py --variables test.scp`

### Implementation for User Story 4

- [ ] T027 [US4] Implement `VariableTracker` class to build variable mappings in `tools/ir_semantic_validator.py`
- [ ] T028 [US4] Implement stack slot → MLIL variable mapping extraction in `VariableTracker`
- [ ] T029 [US4] Implement frame slot → function parameter mapping extraction in `VariableTracker`
- [ ] T030 [US4] Implement register → MLIL register variable mapping extraction in `VariableTracker`
- [ ] T031 [US4] Implement MLIL variable → HLIL named variable mapping in `VariableTracker`
- [ ] T032 [US4] Implement variable mapping report generation showing Stack[N] → var_N → name in `tools/ir_semantic_validator.py`
- [ ] T033 [US4] Wire `--variables` CLI option to `VariableTracker` in `tools/ir_semantic_validator.py`

**Checkpoint**: Variable tracking working with `--variables` flag

---

## Phase 5: User Story 2 - MLIL/HLIL Comparison (Priority: P2)

**Goal**: Compare semantic logic between MLIL and HLIL for structured control flow validation

**Independent Test**: `python tools/ir_semantic_validator.py --compare mlil hlil test.scp`

### Implementation for User Story 2

- [ ] T034 [US2] Implement `MLILHLILComparator` class with `compare()` method in `tools/ir_semantic_validator.py`
- [ ] T035 [US2] Implement statement transformation detection (SET_VAR→Assign, VAR→Variable, CONST→Const) in `MLILHLILComparator`
- [ ] T036 [US2] Implement control flow structuring detection (IF→HLILIf, GOTO→While/For/Break/Continue) in `MLILHLILComparator`
- [ ] T037 [US2] Implement expression inlining detection (multiple SET_VAR → single Assign) in `MLILHLILComparator`
- [ ] T038 [US2] Implement expression tree semantic equivalence checking in `MLILHLILComparator`
- [ ] T039 [US2] Implement SSA transformation detection (PHI elimination, version stripping) in `MLILHLILComparator`
- [ ] T040 [US2] Implement optimization detection (constant propagation, dead code elimination) in `MLILHLILComparator`
- [ ] T041 [US2] Wire `--compare mlil hlil` CLI option to `MLILHLILComparator` in `tools/ir_semantic_validator.py`

**Checkpoint**: MLIL-HLIL comparison working with `--compare mlil hlil` flag

---

## Phase 6: User Story 5 - Batch Validation (Priority: P2)

**Goal**: Validate all functions in an SCP file at once

**Independent Test**: `python tools/ir_semantic_validator.py --batch test.scp`

### Implementation for User Story 5

- [ ] T042 [US5] Implement `BatchValidator` class to iterate all functions in SCP in `tools/ir_semantic_validator.py`
- [ ] T043 [US5] Implement `BatchReport` dataclass with pass_count, fail_count, function_results in `tools/ir_semantic_validator.py`
- [ ] T044 [US5] Implement batch summary output (N functions passed, M functions failed) in `tools/ir_semantic_validator.py`
- [ ] T045 [US5] Implement drill-down output for failed functions in batch mode in `tools/ir_semantic_validator.py`
- [ ] T046 [US5] Wire `--batch` CLI option to `BatchValidator` in `tools/ir_semantic_validator.py`

**Checkpoint**: Batch validation working with `--batch` flag

---

## Phase 7: User Story 6 - Type Consistency (Priority: P2)

**Goal**: Verify type information consistency across IR layers

**Independent Test**: `python tools/ir_semantic_validator.py --types test.scp`

### Implementation for User Story 6

- [ ] T047 [US6] Implement `TypeValidator` class to compare type assignments in `tools/ir_semantic_validator.py`
- [ ] T048 [US6] Implement type extraction from MLIL variables in `TypeValidator`
- [ ] T049 [US6] Implement type extraction from HLIL expressions in `TypeValidator`
- [ ] T050 [US6] Implement type compatibility checking (int+int=int, string operations) in `TypeValidator`
- [ ] T051 [US6] Implement type mismatch reporting in `tools/ir_semantic_validator.py`
- [ ] T052 [US6] Wire `--types` CLI option to `TypeValidator` in `tools/ir_semantic_validator.py`

**Checkpoint**: Type validation working with `--types` flag

---

## Phase 8: User Story 3 - Three-Way Comparison (Priority: P3)

**Goal**: Compare all three IR layers simultaneously

**Independent Test**: `python tools/ir_semantic_validator.py --all test.scp`

### Implementation for User Story 3

- [ ] T053 [US3] Implement `ThreeWayComparator` class combining LLIL-MLIL and MLIL-HLIL in `tools/ir_semantic_validator.py`
- [ ] T054 [US3] Implement aligned row output showing corresponding operations in each layer in `tools/ir_semantic_validator.py`
- [ ] T055 [US3] Implement combined equivalence status (all match, partial match, difference) in `ThreeWayComparator`
- [ ] T056 [US3] Wire `--all` CLI option to `ThreeWayComparator` in `tools/ir_semantic_validator.py`

**Checkpoint**: Three-way comparison working with `--all` flag

---

## Phase 9: User Story 7 - Source Location Mapping (Priority: P3)

**Goal**: Map differences back to original SCP bytecode location

**Independent Test**: When difference reported, see SCP offset like "SCP offset 0x1234"

### Implementation for User Story 7

- [ ] T057 [US7] Implement `SourceLocationMapper` class to track SCP offsets through IR pipeline in `tools/ir_semantic_validator.py`
- [ ] T058 [US7] Extract and propagate SCP bytecode offset from LLIL instructions in `SourceLocationMapper`
- [ ] T059 [US7] Include SCP offset in all difference reports in `tools/ir_semantic_validator.py`
- [ ] T060 [US7] Implement sub-expression location tracking for complex expressions in `SourceLocationMapper`

**Checkpoint**: All difference reports include original SCP bytecode location

---

## Phase 10: Output Formats

**Purpose**: Text and JSON report output

- [ ] T061 Implement `TextReporter` class for human-readable output in `tools/ir_semantic_validator.py`
- [ ] T062 Implement colored output (✓ green, ~ yellow, ✗ red) with `--no-color` option in `TextReporter`
- [ ] T063 Implement `JSONReporter` class for machine-readable output in `tools/ir_semantic_validator.py`
- [ ] T064 Wire `--json` CLI option to `JSONReporter` in `tools/ir_semantic_validator.py`
- [ ] T065 Implement `--function NAME` filter to validate specific function only in `tools/ir_semantic_validator.py`

---

## Phase 11: Polish & Cross-Cutting Concerns

**Purpose**: Error handling, documentation, and final cleanup

- [ ] T066 Implement error handling for SCP parse failures (exit with error, show location) in `tools/ir_semantic_validator.py`
- [ ] T067 Implement error handling for LLIL/MLIL/HLIL conversion failures (report and continue) in `tools/ir_semantic_validator.py`
- [ ] T068 Implement unknown operation type warning in `tools/ir_semantic_validator.py`
- [ ] T069 Add module docstring and function docstrings in `tools/ir_semantic_validator.py`
- [ ] T070 Validate tool works with sample SCP files from existing test data

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational (Phase 2)
  - US1 (LLIL-MLIL) and US4 (Variables): Priority P1, can run in parallel
  - US2 (MLIL-HLIL), US5 (Batch), US6 (Types): Priority P2, can run in parallel after P1
  - US3 (Three-Way), US7 (Source Location): Priority P3, can run in parallel after P2
- **Output Formats (Phase 10)**: Can start after any comparison is working
- **Polish (Phase 11)**: Depends on core functionality being complete

### User Story Dependencies

| Story | Priority | Depends On | Can Start After |
|-------|----------|------------|-----------------|
| US1 - LLIL/MLIL Comparison | P1 | Phase 2 | T013 complete |
| US4 - Variable Tracking | P1 | Phase 2 | T013 complete |
| US2 - MLIL/HLIL Comparison | P2 | Phase 2 | T013 complete |
| US5 - Batch Validation | P2 | US1 or US2 | T026 or T041 complete |
| US6 - Type Consistency | P2 | Phase 2 | T013 complete |
| US3 - Three-Way Comparison | P3 | US1 + US2 | T026 + T041 complete |
| US7 - Source Location | P3 | Phase 2 | T013 complete |

### Parallel Opportunities

**Within Phase 1:**
- T002, T003, T004 can run in parallel (different enums)

**Within Phase 2:**
- T005, T006, T007, T008, T009 can run in parallel (different dataclasses)
- T011, T012, T013 can run in parallel (different IR layers)

**After Phase 2:**
- US1 and US4 can run in parallel (different features)
- US2, US5, US6 can run in parallel after US1 (different features)

---

## Parallel Example: Phase 2 Normalization

```bash
# Launch all normalization functions together:
Task: "Implement normalize_llil_operation() in tools/ir_semantic_validator.py"
Task: "Implement normalize_mlil_operation() in tools/ir_semantic_validator.py"
Task: "Implement normalize_hlil_operation() in tools/ir_semantic_validator.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T004)
2. Complete Phase 2: Foundational (T005-T013)
3. Complete Phase 3: User Story 1 - LLIL/MLIL Comparison (T014-T026)
4. **STOP and VALIDATE**: Test with `--compare llil mlil test.scp`
5. MVP is functional - basic semantic validation working

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 (LLIL-MLIL) → Test → MVP!
3. Add US4 (Variables) → Test → Variable tracking added
4. Add US2 (MLIL-HLIL) → Test → Full pipeline coverage
5. Add US5 (Batch) → Test → Scale validation
6. Add US3, US6, US7 → Test → Complete feature

---

## Completeness Check

### Spec.md Coverage

| User Story | Priority | Phase | Tasks | Status |
|------------|----------|-------|-------|--------|
| US1 - LLIL/MLIL Comparison | P1 | 3 | T014-T026 | ✅ |
| US2 - MLIL/HLIL Comparison | P2 | 5 | T034-T041 | ✅ |
| US3 - Three-Way Comparison | P3 | 8 | T053-T056 | ✅ |
| US4 - Variable Tracking | P1 | 4 | T027-T033 | ✅ |
| US5 - Batch Validation | P2 | 6 | T042-T046 | ✅ |
| US6 - Type Consistency | P2 | 7 | T047-T052 | ✅ |
| US7 - Source Location | P3 | 9 | T057-T060 | ✅ |

### Functional Requirements Coverage

| FR | Description | Task(s) |
|----|-------------|---------|
| FR-001 | Load SCP, generate IRs | T010 |
| FR-002 | Compare arithmetic ops | T015 |
| FR-003 | Compare comparison ops | T016 |
| FR-004 | Compare logical ops | T017 |
| FR-005 | Compare control flow | T018, T036 |
| FR-006 | Compare function/syscall args | T018, T035 |
| FR-007 | Report differences | T025, T061 |
| FR-008 | Handle expected transformations | T023-T024, T037-T040 |
| FR-009 | Human-readable report | T061 |
| FR-010 | Compare any two or all three | T026, T041, T056 |
| FR-011 | Track variable mappings | T027-T033 |
| FR-012 | Verify constants | T019 |
| FR-013 | Expression tree equivalence | T038 |
| FR-014 | Side-effect ordering | T022 |
| FR-015 | Batch validation | T042-T046 |
| FR-016 | Summary statistics | T044 |
| FR-017 | Compare types | T047-T052 |
| FR-018 | Compare global variables | T021 |
| FR-019 | Validate string constants | T020 |
| FR-020 | Report SCP offset | T057-T060 |
| FR-021 | Export for regression | T063-T064 |

### CLI Options Coverage

| Option | Task |
|--------|------|
| --compare LAYER LAYER | T026, T041 |
| --all | T056 |
| --variables | T033 |
| --types | T052 |
| --batch | T046 |
| --function NAME | T065 |
| --json | T064 |
| --no-color | T062 |

---

## Notes

- All code goes in single file `tools/ir_semantic_validator.py` per plan.md
- Uses existing IR infrastructure from `ir/llil`, `ir/mlil`, `ir/hlil`
- Uses existing converters from `falcom/ed9/`
- [P] tasks = can run in parallel within same phase
- [Story] label maps task to specific user story
- Commit after each phase or logical group of tasks
- **Total: 70 tasks**
