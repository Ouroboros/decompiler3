# Tasks: IR Semantic Consistency Validation

**Input**: Design documents from `/specs/003-ir-pass-validation/`
**Prerequisites**: plan.md (required), spec.md (required), research.md

**Core Concept**: IDA F5 style line mapping - establish **SCP bytecode offset** as the universal anchor for LLIL/MLIL/HLIL synchronization.

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story label (US1, US2, etc.)
- Include exact file paths in descriptions

---

## Phase 1: Setup - Core Data Structures (Key Entities from spec.md)

**Purpose**: Define all key entities and establish SCP bytecode offset as the universal anchor

### Address Tracking

- [x] T001 Add `scp_offset: int` field to `SemanticOperation` dataclass in `tools/ir_semantic_validator.py`
- [x] T002 [P] Add `AddressRange` dataclass for multi-byte instructions in `tools/ir_semantic_validator.py`
- [x] T003 [P] Define `AddressMapping` dataclass linking SCP offset to LLIL/MLIL/HLIL indices in `tools/ir_semantic_validator.py`

### Comparison Result Entities (from spec.md Key Entities)

- [x] T004 [P] Define `ComparisonResult` dataclass with status (equivalent/different/transformed), explanation, source_location in `tools/ir_semantic_validator.py`
- [x] T005 [P] Define `DifferenceReport` dataclass with location, explanation, layer info, severity in `tools/ir_semantic_validator.py`
- [x] T006 [P] Define `BatchReport` dataclass with function results list, summary stats, pass/fail counts in `tools/ir_semantic_validator.py`
- [x] T007 [P] Define `TypeInfo` dataclass with llil_type, mlil_type, hlil_type, compatibility status in `tools/ir_semantic_validator.py`

---

## Phase 2: Foundational - Line Correspondence Engine

**Purpose**: Build the core address-based mapping engine (IDA-style synchronization)

**CRITICAL**: All output features depend on this address-based mapping

### Address Extraction from Each IR Layer

- [x] T008 [P] Implement `extract_llil_addresses()` to get SCP offset for each LLIL instruction in `tools/ir_semantic_validator.py`
- [x] T009 [P] Implement `extract_mlil_addresses()` to propagate SCP offset from source LLIL in `tools/ir_semantic_validator.py`
- [x] T010 [P] Implement `extract_hlil_addresses()` to propagate SCP offset from source MLIL in `tools/ir_semantic_validator.py`

### Address-Based Alignment

- [x] T011 Implement `AddressAligner` class that groups instructions by SCP offset in `tools/ir_semantic_validator.py`
- [x] T012 Implement `build_address_table()` to create unified SCP -> (LLIL, MLIL, HLIL) mapping in `tools/ir_semantic_validator.py`
- [x] T013 [P] Handle N:1 mapping (multiple LLIL -> one MLIL expression tree) in `tools/ir_semantic_validator.py`
- [x] T014 [P] Handle 1:N mapping (one SCP -> multiple HLIL inlined expressions) in `tools/ir_semantic_validator.py`
- [x] T015 [P] Handle "eliminated" entries where LLIL/MLIL exists but HLIL optimized away in `tools/ir_semantic_validator.py`

### Output Formatting

- [x] T016 Implement `format_address_table()` with aligned columns (SCP | LLIL | MLIL | HLIL) in `tools/ir_semantic_validator.py`
- [x] T017 [P] Add color coding: green=equivalent, yellow=transformed, red=different, gray=eliminated in `tools/ir_semantic_validator.py`
- [x] T018 Add `--sync` CLI flag to enable IDA-style synchronized view in `tools/ir_semantic_validator.py`

**Checkpoint**: Address-based alignment infrastructure complete

---

## Phase 3: User Story 1 - LLIL/MLIL Comparison (Priority: P1) - MVP

**Goal**: Show exact SCP offset for each LLIL instruction and its MLIL counterpart

**Independent Test**: `python tools/ir_semantic_validator.py --compare llil mlil --sync tests/debug.dat --function FC_SelectArea`

**Expected Output**:
```
=== LLIL-MLIL Sync View: FC_SelectArea ===

SCP Offset   | LLIL Instruction                  | MLIL Instruction                  | Status
-------------|-----------------------------------|-----------------------------------|--------
0x000552A1   | push_caller_frame(ret_addr)       | (eliminated)                      | ~ TRANS
0x000552B5   | SYSCALL(menu_scene, ...)          | syscall(menu_scene, ...)          | = OK
0x000552E9   | BRANCH(0x55330)                   | if (reg0 == 0) goto BB_2          | ~ TRANS

Legend: ==equivalent, ~=transformed, !=different, o=eliminated
```

### Implementation for User Story 1

- [x] T019 [US1] Modify `LLILMLILComparator` to use `AddressAligner` for mapping in `tools/ir_semantic_validator.py`
- [x] T020 [US1] Extract SCP address from each `LowLevelILInstruction.address` field in `tools/ir_semantic_validator.py`
- [x] T021 [US1] Propagate address to MLIL via source instruction tracking in `tools/ir_semantic_validator.py`
- [x] T022 [US1] Create `build_llil_mlil_address_table()` with per-offset comparison in `tools/ir_semantic_validator.py`
- [x] T023 [P] [US1] Handle call sequences (push_caller_frame + args + CALL -> single MLIL CALL) in `tools/ir_semantic_validator.py`
- [x] T024 [P] [US1] Handle expression trees (multiple LLIL ops -> single MLIL expression) in `tools/ir_semantic_validator.py`
- [x] T025 [US1] Format synchronized LLIL-MLIL table with `format_address_table()` in `tools/ir_semantic_validator.py`
- [x] T026 [US1] Wire `--sync` output to `--compare llil mlil` command in `tools/ir_semantic_validator.py`

**Checkpoint**: LLIL-MLIL IDA-style sync view working (MVP complete)

---

## Phase 4: User Story 4 - Variable Tracking by Address (Priority: P1)

**Goal**: Show which SCP offset introduces each variable

**Independent Test**: `python tools/ir_semantic_validator.py --variables --sync tests/debug.dat --function FC_SelectArea`

**Expected Output**:
```
=== Variable Mapping: FC_SelectArea ===

SCP Offset   | Definition Site      | LLIL Storage | MLIL Var  | HLIL Var   | Type
-------------|----------------------|--------------|-----------|------------|------
0x000552C9   | syscall return       | REG[reg0]    | reg0      | result     | int
0x00055301   | STORE to stack       | STACK[sp+0]  | var_0     | index      | int
```

### Implementation for User Story 4

- [x] T027 [US4] Track variable definition site (first assignment SCP offset) in `tools/ir_semantic_validator.py`
- [x] T028 [P] [US4] Track variable use sites (all SCP offsets where variable is read) in `tools/ir_semantic_validator.py`
- [x] T029 [US4] Build `VariableLifetime` dataclass with def_offset, use_offsets, type in `tools/ir_semantic_validator.py`
- [x] T030 [US4] Format variable table with address context in `tools/ir_semantic_validator.py`
- [x] T031 [US4] Wire `--sync` output to `--variables` command in `tools/ir_semantic_validator.py`

**Checkpoint**: Variable tracking with address context working

---

## Phase 5: User Story 2 - MLIL/HLIL Comparison (Priority: P2)

**Goal**: Show how MLIL basic blocks map to HLIL structured code

**Independent Test**: `python tools/ir_semantic_validator.py --compare mlil hlil --sync tests/debug.dat --function FC_SelectArea`

**Expected Output**:
```
=== MLIL-HLIL Sync View: FC_SelectArea ===

SCP Offset   | MLIL Instruction              | HLIL Statement                   | Status
-------------|-------------------------------|----------------------------------|--------
0x000552B5   | reg0 = syscall(...)           | result = syscall(...)            | = OK
0x000552E9   | if (reg0 == 0) goto BB_2      | if (result == 0):                | ~ TRANS
0x00055315   | BB_1: if (var_0 < 10) ...     | while (index < 10):              | ~ TRANS
```

### Implementation for User Story 2

- [x] T032 [US2] Modify `MLILHLILComparator` to use `AddressAligner` in `tools/ir_semantic_validator.py`
- [x] T033 [US2] Propagate SCP offset from MLIL to HLIL statements in `tools/ir_semantic_validator.py`
- [x] T034 [P] [US2] Track control flow restructuring (GOTO -> while/if) with address ranges in `tools/ir_semantic_validator.py`
- [x] T035 [P] [US2] Handle HLIL expression inlining (multiple SCP offsets -> one HLIL line) in `tools/ir_semantic_validator.py`
- [x] T036 [US2] Create `build_mlil_hlil_address_table()` in `tools/ir_semantic_validator.py`
- [x] T037 [US2] Format MLIL-HLIL synchronized table in `tools/ir_semantic_validator.py`
- [x] T038 [US2] Wire `--sync` to `--compare mlil hlil` in `tools/ir_semantic_validator.py`

**Checkpoint**: MLIL-HLIL IDA-style sync view working

---

## Phase 6: User Story 5 - Batch Validation with Address Details (Priority: P2)

**Goal**: Batch validation showing address-level details for failed functions

**Independent Test**: `python tools/ir_semantic_validator.py --batch --sync tests/debug.dat`

**Expected Output**:
```
=== Batch Validation: debug.dat ===

Function 1/390: FC_SelectArea - PASS
Function 2/390: FC_MainMenu - PASS
Function 3/390: FC_Event_PartySet - FAIL (21 differences)
  First diff at SCP 0x0005DBFB: SWITCH case count mismatch

Summary: LLIL-MLIL: 358/390 passed (91.8%)
```

### Implementation for User Story 5

- [x] T039 [US5] Modify batch validation to show first-diff SCP offset for failed functions in `tools/ir_semantic_validator.py`
- [x] T040 [P] [US5] Implement `--failed-only --sync` to show full address table only for failures in `tools/ir_semantic_validator.py`
- [x] T041 [P] [US5] Add `--show-first N` to show first N differences per function in `tools/ir_semantic_validator.py`
- [x] T042 [US5] Wire batch options to CLI in `tools/ir_semantic_validator.py`
- [x] T043 [US5] Generate batch summary statistics (FR-016): total/pass/fail counts, percentage, breakdown by comparison type in `tools/ir_semantic_validator.py`

**Checkpoint**: Batch validation with address context

---

## Phase 7: User Story 6 - Type Consistency by Address (Priority: P2)

**Goal**: Show type at each SCP offset across layers

**Independent Test**: `python tools/ir_semantic_validator.py --types --sync tests/debug.dat --function FC_SelectArea`

**Expected Output**:
```
=== Type Tracking by Address: FC_SelectArea ===

SCP Offset   | Expression       | LLIL Type | MLIL Type | HLIL Type | Status
-------------|------------------|-----------|-----------|-----------|--------
0x000552B5   | syscall result   | (none)    | Value32   | int       | = OK
0x00055360   | flag variable    | (none)    | Bool      | int       | ! MISMATCH
```

### Implementation for User Story 6

- [x] T044 [US6] Track inferred type at each SCP offset in `tools/ir_semantic_validator.py`
- [x] T045 [P] [US6] Compare types across layers for each offset in `tools/ir_semantic_validator.py`
- [x] T046 [US6] Format type table with SCP offset column in `tools/ir_semantic_validator.py`
- [x] T047 [US6] Wire `--sync` to `--types` command in `tools/ir_semantic_validator.py`

**Checkpoint**: Type tracking with address context

---

## Phase 8: User Story 3 - Three-Way Address Synchronization (Priority: P3)

**Goal**: Full three-layer IDA-style synchronized view

**Independent Test**: `python tools/ir_semantic_validator.py --all --sync tests/debug.dat --function FC_SelectArea`

**Expected Output**:
```
=== Three-Way Sync View: FC_SelectArea ===

SCP Offset   | LLIL                     | MLIL                    | HLIL                     | Status
-------------|--------------------------|-------------------------|--------------------------|--------
0x000552A1   | push_caller_frame(...)   | (eliminated)            | (eliminated)             | o ELIM
0x000552B5   | SYSCALL(menu_scene, ...) | syscall(menu_scene, ...)| result = syscall(...)    | = OK
0x000552E9   | BRANCH(0x55330)          | if (...) goto BB_2      | (control flow)           | ~ CF
```

### Implementation for User Story 3

- [x] T048 [US3] Implement `ThreeWayAddressAligner` for LLIL-MLIL-HLIL by SCP offset in `tools/ir_semantic_validator.py`
- [x] T049 [P] [US3] Handle cascade inlining (LLIL->MLIL->HLIL) with visual indicators in `tools/ir_semantic_validator.py`
- [x] T050 [US3] Format three-column synchronized table in `tools/ir_semantic_validator.py`
- [x] T051 [US3] Wire `--sync` to `--all` command in `tools/ir_semantic_validator.py`

**Checkpoint**: Full three-way IDA-style view working

---

## Phase 9: User Story 7 - Source Location Mapping (Priority: P3)

**Goal**: Format output suitable for IDE integration (clickable offsets)

**Independent Test**: Output format can be parsed by external tools

**Expected Output Format**:
```
[SCP:0x000552A1] LLIL: push_caller_frame -> MLIL: (eliminated)
[SCP:0x000552B5] LLIL: SYSCALL -> MLIL: syscall -> HLIL: result = syscall()
```

### Implementation for User Story 7

- [x] T052 [US7] Implement `--format=ide` for parseable single-line output in `tools/ir_semantic_validator.py`
- [x] T053 [P] [US7] Implement `--format=grep` for grep-friendly output in `tools/ir_semantic_validator.py`
- [x] T054 [US7] Ensure hex offsets are consistent (0x%08X format) in `tools/ir_semantic_validator.py`

**Checkpoint**: IDE-friendly output format

---

## Phase 10: Output Formats

**Purpose**: JSON output with address-based structure

- [x] T055 Implement JSON output keyed by SCP offset in `tools/ir_semantic_validator.py`
- [x] T056 [P] Add `--output FILE` to write results to file in `tools/ir_semantic_validator.py`

---

## Phase 11: Raw SCP Bytecode Display

**Purpose**: Show original SCP instructions alongside IR layers (like IDA shows assembly)

- [x] T057 Add `scp_opcode: str` column to address table in `tools/ir_semantic_validator.py`
- [x] T058 [P] Extract SCP opcode name and operands from parser for each address in `tools/ir_semantic_validator.py`
- [x] T059 Format five-column table: SCP Offset | SCP Opcode | LLIL | MLIL | HLIL in `tools/ir_semantic_validator.py`

---

## Phase 12: Semantic Comparison Core (FR Coverage)

**Purpose**: Cover Functional Requirements from spec.md that need explicit implementation

### Constant and Expression Validation (FR-012, FR-013)

- [x] T060 [P] Implement constant value comparison across layers (FR-012) in `tools/ir_semantic_validator.py`
- [x] T061 [P] Implement expression tree semantic equivalence (FR-013: a+b == b+a for commutative ops) in `tools/ir_semantic_validator.py`
- [x] T062 Implement side-effect ordering validation for assignments and calls (FR-014) in `tools/ir_semantic_validator.py`

### Global and String Handling (FR-018, FR-019)

- [x] T063 [P] Implement global variable access pattern comparison (FR-018) in `tools/ir_semantic_validator.py`
- [x] T064 [P] Implement string constant pool reference validation (FR-019) in `tools/ir_semantic_validator.py`

### Regression Testing Support (FR-021)

- [x] T065 Implement `--export-regression FILE` to save validation results for regression testing in `tools/ir_semantic_validator.py`
- [x] T066 [P] Implement `--compare-regression FILE` to compare against baseline in `tools/ir_semantic_validator.py`

---

## Phase 13: Edge Cases (from spec.md)

**Purpose**: Handle edge cases identified in spec.md Edge Cases section

### Optimization-Related Cases

- [x] T067 Handle dead code elimination (HLIL has fewer instructions) - mark as ELIMINATED, not DIFFERENT in `tools/ir_semantic_validator.py`
- [x] T068 [P] Handle constant propagation (variable optimized away) - track original source in `tools/ir_semantic_validator.py`
- [x] T069 [P] Handle expression inlining (separate MLIL statements -> single HLIL expression) in `tools/ir_semantic_validator.py`

### SSA and Control Flow Cases

- [x] T070 Handle SSA phi nodes when comparing MLIL SSA to non-SSA LLIL in `tools/ir_semantic_validator.py`
- [x] T071 [P] Handle control flow restructuring (GOTO -> while/for/if) - classify as TRANSFORMED in `tools/ir_semantic_validator.py`

### Naming and Type Cases

- [x] T072 Handle global variable naming convention differences (compare by resolved address) in `tools/ir_semantic_validator.py`
- [x] T073 [P] Handle LLIL lacking type info (only fail if MLIL vs HLIL types conflict) in `tools/ir_semantic_validator.py`

### Complex Structure Cases

- [x] T074 Handle compound expression tree structure differences (a + b * c with different nesting) in `tools/ir_semantic_validator.py`
- [x] T075 [P] Handle syscall signature differences across layers in `tools/ir_semantic_validator.py`
- [x] T076 [P] Handle string constants from constant pool validation (Edge Case 9 from spec.md) in `tools/ir_semantic_validator.py`

---

## Phase 14: Error Handling (from plan.md)

**Purpose**: Implement error handling strategy from plan.md

- [x] T077 Handle SCP parse failure: exit with error, show parse location in `tools/ir_semantic_validator.py`
- [x] T078 [P] Handle LLIL build failure: report function name, continue with next function in `tools/ir_semantic_validator.py`
- [x] T079 [P] Handle MLIL conversion failure: report failure, skip MLIL/HLIL comparison for that function in `tools/ir_semantic_validator.py`
- [x] T080 [P] Handle HLIL conversion failure: report failure, perform LLIL-MLIL only for that function in `tools/ir_semantic_validator.py`
- [x] T081 Handle unknown operation type: warn, treat as non-comparable in `tools/ir_semantic_validator.py`

---

## Phase 15: Polish & Navigation

**Purpose**: Usability improvements and advanced navigation

- [x] T082 Handle SCP offsets that span multiple bytes (instruction length) in `tools/ir_semantic_validator.py`
- [x] T083 [P] Handle HLIL statements that merge multiple SCP offset ranges in `tools/ir_semantic_validator.py`
- [x] T084 Add `--range 0xXXXX-0xYYYY` to show only specific offset range in `tools/ir_semantic_validator.py`
- [x] T085 [P] Add `--export-map FILE` to export address mapping for external tools in `tools/ir_semantic_validator.py`
- [x] T086 Add `--goto 0xXXXX` to show context around specific SCP address in `tools/ir_semantic_validator.py`
- [x] T087 [P] Add `--context N` to show N lines before/after target address in `tools/ir_semantic_validator.py`
- [x] T088 [P] Add `--filter STATUS` to filter by status (different/transformed/eliminated/equivalent) in `tools/ir_semantic_validator.py`
- [x] T089 Add `--reverse-lookup "HLIL text"` to find SCP address from HLIL statement in `tools/ir_semantic_validator.py`

---

## Phase 16: Validation & Final Testing

**Purpose**: Validate complete implementation against complex real-world cases

- [x] T090 Validate with FC_SelectArea (basic function with branches) in `tools/ir_semantic_validator.py`
- [x] T091 [P] Validate with FC_Event_PartySet (complex SWITCH statements) in `tools/ir_semantic_validator.py`
- [x] T092 [P] Validate with large event function (EV_03_*) for performance testing in `tools/ir_semantic_validator.py`
- [x] T093 Performance test: verify <2s per function (SC-004) in `tools/ir_semantic_validator.py`
- [x] T094 [P] Performance test: verify <30s for batch validation (SC-007) in `tools/ir_semantic_validator.py`
- [x] T095 Verify all Success Criteria (SC-001 through SC-010) are met in `tools/ir_semantic_validator.py`
- [x] T096 Generate sample regression baseline for tests/debug.dat in `tools/ir_semantic_validator.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1: Setup**: Address tracking infrastructure - BLOCKS all
- **Phase 2: Foundational**: Address alignment engine - BLOCKS all user stories
- **Phase 3-9: User Stories**: All depend on Phase 2
- **Phase 10-11: Output Formats**: Can run after any user story
- **Phase 12-15: Core/Edge/Error**: Can run after Phase 2
- **Phase 16: Validation**: Depends on all features being implemented

### User Story Dependencies

| Story | Priority | Phase | Depends On | Can Start After |
|-------|----------|-------|------------|-----------------|
| US1 - LLIL/MLIL Sync | P1 | 3 | Phase 2 | T018 complete |
| US4 - Variable Tracking | P1 | 4 | Phase 2 | T018 complete |
| US2 - MLIL/HLIL Sync | P2 | 5 | Phase 2 | T018 complete |
| US5 - Batch with Address | P2 | 6 | US1 or US2 | T026 or T038 complete |
| US6 - Type by Address | P2 | 7 | Phase 2 | T018 complete |
| US3 - Three-Way Sync | P3 | 8 | US1 + US2 | T026 + T038 complete |
| US7 - IDE Format | P3 | 9 | Phase 2 | T018 complete |

### Parallel Opportunities

**Within Phase 1:**
- T002, T003 can run in parallel (different dataclass definitions)
- T004, T005, T006, T007 can run in parallel (independent dataclasses)

**Within Phase 2:**
- T008, T009, T010 can run in parallel (different IR layers)
- T013, T014, T015 can run in parallel (different mapping cases)

**After Phase 2:**
- US1, US4, US6, US7 can all start in parallel
- US2 can run parallel with US1

**After User Stories:**
- Phase 12, 13, 14 can run in parallel (FR/Edge/Error handling)

---

## Parallel Example: User Story 1

```bash
# After Phase 2 completes, launch parallel tasks:
Task: T023 "Handle call sequences in tools/ir_semantic_validator.py"
Task: T024 "Handle expression trees in tools/ir_semantic_validator.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T007)
2. Complete Phase 2: Foundational (T008-T018) - CRITICAL blocks all stories
3. Complete Phase 3: User Story 1 (T019-T026)
4. **STOP and VALIDATE**: Test LLIL-MLIL sync independently
5. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational -> Foundation ready
2. Add User Story 1 -> Test independently -> Deploy/Demo (MVP!)
3. Add User Story 4 -> Variable tracking with addresses
4. Add User Story 2 -> MLIL-HLIL sync
5. Add User Story 5 -> Batch validation
6. Add remaining stories (US6, US3, US7)
7. Add FR/Edge/Error handling (Phase 12-14)
8. Polish & Navigate (Phase 15)
9. Final Validation (Phase 16)

---

## Completeness Check

### Spec.md Key Entities Coverage

| Entity | Phase | Tasks |
|--------|-------|-------|
| SemanticOperation | 1 | T001 |
| AddressRange | 1 | T002 |
| AddressMapping | 1 | T003 |
| ComparisonResult | 1 | T004 |
| DifferenceReport | 1 | T005 |
| BatchReport | 1 | T006 |
| TypeInfo | 1 | T007 |
| VariableMapping | 4 | T029 (VariableLifetime) |

### Spec.md User Story Coverage

| User Story | Priority | Phase | Tasks | Key Feature |
|------------|----------|-------|-------|-------------|
| US1 - LLIL/MLIL | P1 | 3 | T019-T026 (8) | SCP offset synchronization |
| US4 - Variables | P1 | 4 | T027-T031 (5) | Definition/use by address |
| US2 - MLIL/HLIL | P2 | 5 | T032-T038 (7) | Control flow restructuring |
| US5 - Batch | P2 | 6 | T039-T043 (5) | First-diff location + statistics |
| US6 - Types | P2 | 7 | T044-T047 (4) | Type at each offset |
| US3 - Three-Way | P3 | 8 | T048-T051 (4) | Full IDA-style view |
| US7 - IDE Format | P3 | 9 | T052-T054 (3) | Parseable output |

### Spec.md Functional Requirements Coverage

| FR | Description | Phase | Tasks |
|----|-------------|-------|-------|
| FR-001 to FR-011 | Core IR loading, comparison, output | 2-5 | Covered by User Stories |
| FR-012 | Constant value verification | 12 | T060 |
| FR-013 | Expression tree equivalence | 12 | T061 |
| FR-014 | Side-effect ordering | 12 | T062 |
| FR-015 | Batch validation | 6 | T039-T042 |
| FR-016 | Batch summary statistics | 6 | T043 |
| FR-017 | Type comparison | 7 | T044-T047 |
| FR-018 | Global variable patterns | 12 | T063 |
| FR-019 | String constant validation | 12 | T064 |
| FR-020 | Report SCP bytecode offset | 2 | Core feature |
| FR-021 | Regression testing export | 12 | T065-T066 |

### Spec.md Edge Cases Coverage

| Edge Case | Phase | Tasks |
|-----------|-------|-------|
| Dead code elimination | 13 | T067 |
| Constant propagation | 13 | T068 |
| Expression inlining | 13 | T069 |
| SSA phi nodes | 13 | T070 |
| Control flow restructure | 13 | T071 |
| Global naming conventions | 13 | T072 |
| Type info missing in LLIL | 13 | T073 |
| Compound expression trees | 13 | T074 |
| Syscall signatures | 13 | T075 |
| String constants from pool | 13 | T076 |

### Plan.md Error Handling Coverage

| Error | Phase | Tasks |
|-------|-------|-------|
| SCP parse failure | 14 | T077 |
| LLIL build failure | 14 | T078 |
| MLIL conversion failure | 14 | T079 |
| HLIL conversion failure | 14 | T080 |
| Unknown operation type | 14 | T081 |

### Success Criteria Validation

| SC | Description | Validation Task |
|----|-------------|-----------------|
| SC-001 to SC-003 | Comparison accuracy | T095 |
| SC-004 | <2s per function | T093 |
| SC-005 | Developer usability | T095 |
| SC-006 | Variable mapping 100% | T095 |
| SC-007 | <30s batch validation | T094 |
| SC-008 | Zero false negatives | T095 |
| SC-009 | SCP offset in reports | Core feature |
| SC-010 | Type mismatch detection | T095 |

---

## Summary

| Category | Count |
|----------|-------|
| Total Tasks | 96 |
| Total Phases | 16 |
| Setup Phase (Key Entities) | 1 (Phase 1) |
| Foundational Phase | 1 (Phase 2) |
| User Story Phases | 7 (Phase 3-9) |
| Output Formats | 2 (Phase 10-11) |
| FR/Edge/Error Phases | 3 (Phase 12-14) |
| Polish/Validation | 2 (Phase 15-16) |
| Parallelizable Tasks | 44 (marked [P]) |

---

## Notes

- All code in `tools/ir_semantic_validator.py`
- `--sync` flag enables IDA-style synchronized view
- SCP offset is the universal anchor across all layers
- Addresses inherit: SCP -> LLIL -> MLIL -> HLIL
- Each task is specific enough for LLM execution without additional context
- Key Entities from spec.md fully covered in Phase 1
- **Total: 96 tasks across 16 phases**
