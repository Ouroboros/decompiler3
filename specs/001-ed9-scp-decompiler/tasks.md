# Tasks: ED9 SCP Script Decompiler

**Input**: Design documents from `/specs/001-ed9-scp-decompiler/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/
**Status**: Implemented - This task list documents the completed implementation

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: Source at repository root with `falcom/`, `ir/`, `codegen/`, `tests/`
- Paths based on plan.md structure

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project directory structure: `falcom/`, `ir/`, `codegen/`, `common/`, `tests/`
- [x] T002 Initialize Python 3.14 project with PyYAML dependency
- [x] T003 [P] Create common utilities in common/utils.py
- [x] T004 [P] Create enum utilities in common/enum.py
- [x] T005 [P] Create configuration module in common/config.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Create IL base classes in ir/core/il_base.py
- [x] T007 [P] Create IL options in ir/core/il_options.py
- [x] T008 Define WORD_SIZE constant in ir/llil/llil.py
- [x] T009 [P] Implement ScpValue type encoding in falcom/ed9/parser/types_scp.py
- [x] T010 [P] Define ScpParamFlags in falcom/ed9/parser/types_scp.py
- [x] T011 Create SCP header parsing in falcom/ed9/parser/scp.py
- [x] T012 [P] Create instruction definitions in falcom/ed9/disasm/instruction.py
- [x] T013 [P] Create opcode table in falcom/ed9/disasm/ed9_optable.py
- [x] T014 Create basic block analysis in falcom/ed9/disasm/basic_block.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - 反编译单个脚本文件 (Priority: P1) 🎯 MVP

**Goal**: Decompile a .dat script file to readable TypeScript code

**Independent Test**: Run `python tests/test_scp_parser.py`, input debug.dat, verify output debug.ts contains readable TypeScript code

### Implementation for User Story 1

#### Parsing Layer (FR-001, FR-002, FR-003)

- [x] T015 [P] [US1] Implement ScpValue.to_bytes() and from_stream() in falcom/ed9/parser/types_scp.py
- [x] T016 [P] [US1] Implement ScpParser.parse() in falcom/ed9/parser/scp.py
- [x] T017 [US1] Implement ScpParser.disasm_all_functions() with stack simulation in falcom/ed9/parser/scp.py

#### Disassembly Layer (FR-004, FR-005)

- [x] T018 [P] [US1] Implement disassembler in falcom/ed9/disasm/disassembler.py
- [x] T019 [US1] Implement CFG construction in falcom/ed9/disasm/basic_block.py

#### LLIL Layer (FR-006)

- [x] T020 [P] [US1] Define LLIL operations (stack, frame, control flow) in ir/llil/llil.py
- [x] T021 [P] [US1] Create LowLevelILFunction container in ir/llil/llil.py
- [x] T022 [P] [US1] Create LowLevelILBasicBlock in ir/llil/llil.py
- [x] T023 [US1] Implement ED9VMLifter in falcom/ed9/lifters/vm_lifter.py
- [x] T024 [P] [US1] Implement LLIL builder in falcom/ed9/llil_builder.py
- [x] T025 [P] [US1] Create LLIL formatter in falcom/ed9/llil_builder.py (FalcomLLILFormatter)

#### MLIL Layer (FR-007)

- [x] T026 [P] [US1] Define MLIL operations in ir/mlil/mlil.py
- [x] T027 [P] [US1] Create MLILVariable and MLILVariableSSA in ir/mlil/mlil_ssa.py
- [x] T028 [P] [US1] Implement type system in ir/mlil/mlil_types.py
- [x] T029 [US1] Implement LLIL to MLIL conversion in ir/mlil/llil_to_mlil.py
- [x] T030 [US1] Implement SSA construction pass in ir/mlil/passes/pass_ssa_construction.py
- [x] T031 [P] [US1] Implement DominanceAnalysis in ir/mlil/mlil_ssa.py
- [x] T032 [US1] Create MLIL converter in falcom/ed9/mlil_converter.py

#### HLIL Layer (FR-008)

- [x] T033 [P] [US1] Define HLIL statements (HLILIf, HLILWhile, HLILSwitch) in ir/hlil/hlil.py
- [x] T034 [P] [US1] Define HLIL expressions in ir/hlil/hlil.py
- [x] T035 [US1] Implement control flow structuring in ir/hlil/mlil_to_hlil.py
- [x] T036 [US1] Create MLIL to HLIL pass in ir/hlil/passes/pass_mlil_to_hlil.py
- [x] T037 [P] [US1] Implement expression simplification in ir/hlil/passes/pass_expression_simplification.py
- [x] T038 [P] [US1] Implement control flow optimization in ir/hlil/passes/pass_control_flow_optimization.py
- [x] T039 [P] [US1] Implement copy propagation in ir/hlil/passes/pass_copy_propagation.py
- [x] T040 [P] [US1] Implement dead code elimination in ir/hlil/passes/pass_dead_code_elimination.py
- [x] T041 [US1] Create HLIL converter in falcom/ed9/hlil_converter.py

#### Code Generation (FR-010, FR-011, FR-012)

- [x] T042 [US1] Implement TypeScriptGenerator in codegen/typescript.py
- [x] T043 [US1] Implement type mapping (HLIL types → TypeScript types) in codegen/typescript.py
- [x] T044 [US1] Implement operator precedence for parentheses in codegen/typescript.py
- [x] T045 [US1] Add function parameter default value generation in codegen/typescript.py

**Checkpoint**: At this point, User Story 1 should be fully functional - can decompile .dat to .ts

---

## Phase 4: User Story 2 - 查看中间表示 (Priority: P2)

**Goal**: View intermediate representations (LLIL/MLIL/HLIL) for debugging and understanding

**Independent Test**: Enable debug output option, verify .llil.asm, .mlil.asm, .hlil.ts files are generated

### Implementation for User Story 2

#### IR Formatters (FR-009)

- [x] T046 [P] [US2] Implement FalcomLLILFormatter.format_llil_function() in falcom/ed9/llil_builder.py
- [x] T047 [P] [US2] Implement MLILFormatter.format_function() in ir/mlil/mlil.py
- [x] T048 [P] [US2] Implement HLILFormatter.format_function() in ir/hlil/hlil_formatter.py

#### CFG Visualization

- [x] T049 [P] [US2] Implement FalcomLLILFormatter.to_dot() for GraphViz output in falcom/ed9/llil_builder.py
- [x] T050 [P] [US2] Implement MLILFormatter.to_dot() for GraphViz output in ir/mlil/mlil.py

#### Debug Output Integration

- [x] T051 [US2] Add debug file output (.llil.asm, .mlil.asm, .hlil.ts, .dot) in tests/test_scp_parser.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - 使用签名数据库美化输出 (Priority: P3)

**Goal**: Use signature database to replace syscalls with meaningful function names and format parameters

**Independent Test**: Configure signature YAML file, verify `syscall(5, 0, ...)` is replaced with `mes_message_talk(...)`

### Implementation for User Story 3

#### Signature Database (FR-013, FR-014, FR-015, FR-016, FR-017, FR-018)

- [x] T052 [P] [US3] Define ParamHint dataclass in falcom/ed9/format_signatures.py
- [x] T053 [P] [US3] Define FunctionSig dataclass in falcom/ed9/format_signatures.py
- [x] T054 [P] [US3] Define SyscallSig dataclass in falcom/ed9/format_signatures.py
- [x] T055 [US3] Implement FormatSignatureDB.load_yaml() in falcom/ed9/format_signatures.py
- [x] T056 [US3] Implement FormatSignatureDB.get_syscall() in falcom/ed9/format_signatures.py
- [x] T057 [US3] Implement FormatSignatureDB.get_function() in falcom/ed9/format_signatures.py
- [x] T058 [US3] Implement FormatSignatureDB.get_enum_value() in falcom/ed9/format_signatures.py
- [x] T059 [US3] Implement FormatSignatureDB.format_value() with hex/enum support in falcom/ed9/format_signatures.py
- [x] T060 [US3] Implement variadic parameter handling in falcom/ed9/format_signatures.py
- [x] T061 [US3] Implement union type parsing "[string, number]" in falcom/ed9/format_signatures.py

#### Code Generator Integration

- [x] T062 [US3] Add TypeScriptGenerator.set_signature_db() in codegen/typescript.py
- [x] T063 [US3] Implement syscall name replacement in codegen/typescript.py
- [x] T064 [US3] Implement parameter formatting with hints in codegen/typescript.py
- [x] T065 [US3] Implement enum value substitution in codegen/typescript.py

#### Signature Files

- [x] T066 [P] [US3] Create sample signature YAML in falcom/ed9/signatures/

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T067 [P] Implement common return extraction in ir/hlil/passes/pass_common_return_extraction.py
- [x] T068 [P] Implement type inference pass in ir/mlil/passes/pass_type_inference.py
- [x] T069 [P] Add edge case handling: empty functions, nested control flow
- [x] T070 [P] Add error handling: invalid magic, unknown opcode, truncated file
- [x] T071 Create main test file tests/test_scp_parser.py
- [x] T072 [P] Create project documentation specs/001-ed9-scp-decompiler/

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Uses components from US1 but independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Uses codegen from US1 but independently testable

### Within Each User Story

- Parsing/data structures before logic
- LLIL before MLIL before HLIL (pipeline order)
- IR definition before conversion logic
- Passes before converters that use them
- Core implementation before integration

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Within US1: T015-T016 (parsing), T020-T022 (LLIL defs), T026-T028 (MLIL defs), T033-T034 (HLIL defs)
- Within US2: T046-T050 (all formatters)
- Within US3: T052-T054 (data classes), T066 (signature files)

---

## Parallel Example: User Story 1 Foundation

```bash
# Launch LLIL definition tasks together:
Task: "Define LLIL operations in ir/llil/llil.py"
Task: "Create LowLevelILFunction in ir/llil/llil.py"
Task: "Create LowLevelILBasicBlock in ir/llil/llil.py"

# Launch MLIL definition tasks together:
Task: "Define MLIL operations in ir/mlil/mlil.py"
Task: "Create MLILVariable in ir/mlil/mlil_ssa.py"
Task: "Implement type system in ir/mlil/mlil_types.py"

# Launch HLIL pass tasks together:
Task: "Implement expression simplification in ir/hlil/passes/pass_expression_simplification.py"
Task: "Implement control flow optimization in ir/hlil/passes/pass_control_flow_optimization.py"
Task: "Implement copy propagation in ir/hlil/passes/pass_copy_propagation.py"
Task: "Implement dead code elimination in ir/hlil/passes/pass_dead_code_elimination.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently - decompile .dat to .ts
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!) - Basic decompilation
3. Add User Story 2 → Test independently → Deploy/Demo - Debug output
4. Add User Story 3 → Test independently → Deploy/Demo - Signature beautification
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (core decompilation)
   - Developer B: User Story 2 (formatters) after US1 IR definitions exist
   - Developer C: User Story 3 (signatures) after US1 codegen exists
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- All tasks marked [x] as this documents an already-implemented feature
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
