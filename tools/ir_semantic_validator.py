"""
IR Semantic Consistency Validator

Validates semantic consistency across LLIL, MLIL, and HLIL layers by comparing
actual decompiled output from SCP files.

Usage:
    python tools/ir_semantic_validator.py --compare llil mlil test.scp
    python tools/ir_semantic_validator.py --all test.scp
    python tools/ir_semantic_validator.py --batch test.scp
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import fileio
from common.config import default_encoding
from ir.llil import LowLevelILFunction, LowLevelILInstruction, LowLevelILOperation
from ir.mlil import MediumLevelILFunction, MediumLevelILInstruction, MediumLevelILOperation
from ir.hlil import (
    HighLevelILFunction, HLILInstruction, HLILOperation, HLILStatement, HLILExpression,
    BinaryOp, UnaryOp
)
from falcom.ed9.parser.scp import ScpParser
from falcom.ed9.lifters.vm_lifter import ED9VMLifter
from falcom.ed9.mlil_converter import convert_falcom_llil_to_mlil
from falcom.ed9.hlil_converter import convert_falcom_mlil_to_hlil


# =============================================================================
# T002-T004: Enums
# =============================================================================

class OperationKind(Enum):
    """Normalized operation categories for semantic comparison"""
    ARITHMETIC = auto()     # ADD, SUB, MUL, DIV, MOD, NEG
    COMPARISON = auto()     # EQ, NE, LT, LE, GT, GE
    LOGICAL = auto()        # AND, OR, NOT (logical)
    BITWISE = auto()        # AND, OR, XOR, SHL, SHR, NOT (bitwise)
    CALL = auto()           # CALL, SYSCALL, CALL_SCRIPT
    BRANCH = auto()         # JMP, BRANCH, IF, GOTO
    ASSIGN = auto()         # SET_VAR, STACK_STORE, ASSIGN
    LOAD = auto()           # VAR, STACK_LOAD, CONST
    RETURN = auto()         # RET, RETURN
    CONTROL_FLOW = auto()   # BREAK, CONTINUE, WHILE, FOR, IF
    NOP = auto()            # NOP, DEBUG, eliminated ops
    UNKNOWN = auto()        # Unrecognized operations


class ComparisonStatus(Enum):
    """Result of comparing two operations"""
    EQUIVALENT = auto()     # Semantically identical
    TRANSFORMED = auto()    # Expected transformation (e.g., STACK_LOAD -> VAR)
    DIFFERENT = auto()      # Semantic difference (potential bug)


class IRLayer(Enum):
    """IR layer identifier"""
    LLIL = auto()
    MLIL = auto()
    HLIL = auto()


# =============================================================================
# T005-T009: Dataclasses
# =============================================================================

@dataclass
class SourceLocation:
    """Maps IR instruction to original SCP bytecode offset"""
    scp_offset: int = 0
    llil_index: int = -1
    mlil_index: int = -1
    hlil_index: int = -1

    def __str__(self) -> str:
        return f"0x{self.scp_offset:04X}"


@dataclass
class SemanticOperand:
    """Normalized operand representation for comparison"""
    kind: str               # 'const', 'var', 'reg', 'global', 'expr'
    value: Any = None       # Actual value or identifier
    type_hint: Optional[str] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticOperand):
            return False
        # Compare kind and value, type_hint is optional
        return self.kind == other.kind and self.value == other.value

    def __str__(self) -> str:
        if self.kind == 'const':
            return repr(self.value)

        elif self.kind == 'var':
            return str(self.value)

        else:
            return f"{self.kind}:{self.value}"


@dataclass
class SemanticOperation:
    """Normalized operation for cross-layer comparison"""
    kind: OperationKind
    operator: str           # Specific operator (ADD, SUB, EQ, etc.)
    operands: List[SemanticOperand] = field(default_factory=list)
    result: Optional[SemanticOperand] = None
    source_location: SourceLocation = field(default_factory=SourceLocation)

    def __str__(self) -> str:
        ops_str = ", ".join(str(op) for op in self.operands)
        result_str = f" -> {self.result}" if self.result else ""
        return f"{self.operator}({ops_str}){result_str}"


@dataclass
class ComparisonResult:
    """Result of comparing two operations"""
    status: ComparisonStatus
    source_layer: IRLayer
    target_layer: IRLayer
    source_op: Optional[SemanticOperation] = None
    target_op: Optional[SemanticOperation] = None
    explanation: str = ""
    source_location: SourceLocation = field(default_factory=SourceLocation)

    def __str__(self) -> str:
        status_char = {
            ComparisonStatus.EQUIVALENT: "=",
            ComparisonStatus.TRANSFORMED: "~",
            ComparisonStatus.DIFFERENT: "!",
        }[self.status]
        return f"[{status_char}] {self.source_location}: {self.explanation}"


@dataclass
class VariableMapping:
    """Tracks variable correspondence across IR layers"""
    llil_storage: str       # e.g., "Stack[0]", "Frame[0]", "Reg[0]"
    mlil_var: str           # e.g., "var_s0", "arg0"
    hlil_var: str = ""      # e.g., "local_count", "param_target"
    type_info: str = ""     # e.g., "int", "str"

    def __str__(self) -> str:
        parts = [self.llil_storage, self.mlil_var]
        if self.hlil_var:
            parts.append(self.hlil_var)
        if self.type_info:
            parts.append(f"({self.type_info})")
        return " -> ".join(parts)


@dataclass
class FunctionValidationResult:
    """Validation result for a single function"""
    function_name: str
    llil_mlil_results: List[ComparisonResult] = field(default_factory=list)
    mlil_hlil_results: List[ComparisonResult] = field(default_factory=list)
    variable_mappings: List[VariableMapping] = field(default_factory=list)
    matched_count: int = 0
    transformed_count: int = 0
    different_count: int = 0

    @property
    def passed(self) -> bool:
        return self.different_count == 0


@dataclass
class BatchReport:
    """Aggregated validation results for all functions"""
    file_path: str
    function_results: List[FunctionValidationResult] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.function_results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.function_results if not r.passed)

    @property
    def total_count(self) -> int:
        return len(self.function_results)


# =============================================================================
# T010: IR Generation Pipeline
# =============================================================================

class IRPipeline:
    """Generates all IR layers from an SCP file"""

    def __init__(self, scp_path: str):
        self.scp_path = scp_path
        self.scp = None
        self.fs = None
        self.functions: List[Tuple[str, LowLevelILFunction, MediumLevelILFunction, HighLevelILFunction]] = []

    def load(self) -> bool:
        """Load and parse SCP file"""
        try:
            scp_path = Path(self.scp_path)
            # Keep file stream open for disassembly
            self.fs = fileio.FileStream(str(scp_path), encoding=default_encoding())
            parser = ScpParser(self.fs, scp_path.name)
            parser.parse()
            self.scp = parser
            return True
        except Exception as e:
            print(f"Error parsing SCP file: {e}", file=sys.stderr)
            return False

    def close(self) -> None:
        """Close file stream"""
        if self.fs:
            self.fs.Close()
            self.fs = None

    def generate_ir(self, function_name: Optional[str] = None) -> bool:
        """Generate LLIL, MLIL, HLIL for all functions (or specific function)

        T066-T067: Includes error handling for SCP parse and IR conversion failures.
        """
        if self.scp is None:
            return False

        errors_llil = 0
        errors_mlil = 0
        errors_hlil = 0

        try:
            # Disassemble functions first
            self.scp.disasm_all_functions()

            # Create lifter
            lifter = ED9VMLifter(parser=self.scp)

            for func in self.scp.functions:
                if function_name and func.name != function_name:
                    continue

                try:
                    # T067: Generate LLIL using lifter
                    llil_func = lifter.lift_function(func)
                except Exception as e:
                    print(f"[T067] LLIL error in {func.name}: {e}", file=sys.stderr)
                    errors_llil += 1
                    continue

                try:
                    # T067: Convert to MLIL
                    mlil_func = convert_falcom_llil_to_mlil(llil_func)
                except Exception as e:
                    print(f"[T067] MLIL error in {func.name}: {e}", file=sys.stderr)
                    errors_mlil += 1
                    continue

                try:
                    # T067: Convert to HLIL
                    hlil_func = convert_falcom_mlil_to_hlil(mlil_func)
                except Exception as e:
                    print(f"[T067] HLIL error in {func.name}: {e}", file=sys.stderr)
                    errors_hlil += 1
                    # Continue with empty HLIL
                    hlil_func = HighLevelILFunction()

                self.functions.append((func.name, llil_func, mlil_func, hlil_func))

        finally:
            # Close file stream after processing
            self.close()

        # Report conversion error summary
        total_errors = errors_llil + errors_mlil + errors_hlil
        if total_errors > 0:
            print(f"IR conversion errors: LLIL={errors_llil}, MLIL={errors_mlil}, HLIL={errors_hlil}",
                  file=sys.stderr)

        return len(self.functions) > 0


# =============================================================================
# T011-T013: Normalization Functions
# T068: Unknown operation type tracking
# =============================================================================

# Track unknown operations encountered during normalization
_unknown_llil_ops: set = set()
_unknown_mlil_ops: set = set()
_unknown_hlil_ops: set = set()


def report_unknown_operations() -> None:
    """T068: Report any unknown operation types encountered"""
    if _unknown_llil_ops:
        print(f"[T068] Unknown LLIL operations: {sorted(_unknown_llil_ops)}", file=sys.stderr)
    if _unknown_mlil_ops:
        print(f"[T068] Unknown MLIL operations: {sorted(_unknown_mlil_ops)}", file=sys.stderr)
    if _unknown_hlil_ops:
        print(f"[T068] Unknown HLIL operations: {sorted(_unknown_hlil_ops)}", file=sys.stderr)


def clear_unknown_operations() -> None:
    """Reset unknown operation tracking"""
    _unknown_llil_ops.clear()
    _unknown_mlil_ops.clear()
    _unknown_hlil_ops.clear()


def normalize_llil_operation(instr: LowLevelILInstruction) -> SemanticOperation:
    """Convert LLIL instruction to normalized SemanticOperation"""
    op = instr.operation
    loc = SourceLocation(scp_offset=instr.address, llil_index=instr.inst_index)

    # Arithmetic operations
    if op in (LowLevelILOperation.LLIL_ADD, LowLevelILOperation.LLIL_SUB,
              LowLevelILOperation.LLIL_MUL, LowLevelILOperation.LLIL_DIV,
              LowLevelILOperation.LLIL_MOD):
        return SemanticOperation(
            kind=OperationKind.ARITHMETIC,
            operator=op.name.replace('LLIL_', ''),
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_NEG:
        return SemanticOperation(
            kind=OperationKind.ARITHMETIC,
            operator='NEG',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    # Comparison operations
    elif op in (LowLevelILOperation.LLIL_EQ, LowLevelILOperation.LLIL_NE,
                LowLevelILOperation.LLIL_LT, LowLevelILOperation.LLIL_LE,
                LowLevelILOperation.LLIL_GT, LowLevelILOperation.LLIL_GE):
        return SemanticOperation(
            kind=OperationKind.COMPARISON,
            operator=op.name.replace('LLIL_', ''),
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_TEST_ZERO:
        return SemanticOperation(
            kind=OperationKind.COMPARISON,
            operator='TEST_ZERO',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    # Logical operations
    elif op in (LowLevelILOperation.LLIL_LOGICAL_AND, LowLevelILOperation.LLIL_LOGICAL_OR):
        return SemanticOperation(
            kind=OperationKind.LOGICAL,
            operator=op.name.replace('LLIL_', ''),
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    # Bitwise operations
    elif op in (LowLevelILOperation.LLIL_AND, LowLevelILOperation.LLIL_OR):
        return SemanticOperation(
            kind=OperationKind.BITWISE,
            operator=op.name.replace('LLIL_', ''),
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_BITWISE_NOT:
        return SemanticOperation(
            kind=OperationKind.BITWISE,
            operator='BITWISE_NOT',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    # Control flow
    elif op == LowLevelILOperation.LLIL_JMP:
        return SemanticOperation(
            kind=OperationKind.BRANCH,
            operator='JMP',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_BRANCH:
        return SemanticOperation(
            kind=OperationKind.BRANCH,
            operator='BRANCH',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_CALL:
        return SemanticOperation(
            kind=OperationKind.CALL,
            operator='CALL',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_SYSCALL:
        return SemanticOperation(
            kind=OperationKind.CALL,
            operator='SYSCALL',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_CALL_SCRIPT:
        return SemanticOperation(
            kind=OperationKind.CALL,
            operator='CALL_SCRIPT',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_RET:
        return SemanticOperation(
            kind=OperationKind.RETURN,
            operator='RET',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    # Storage operations
    elif op == LowLevelILOperation.LLIL_STACK_LOAD:
        return SemanticOperation(
            kind=OperationKind.LOAD,
            operator='STACK_LOAD',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_STACK_STORE:
        return SemanticOperation(
            kind=OperationKind.ASSIGN,
            operator='STACK_STORE',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_FRAME_LOAD:
        return SemanticOperation(
            kind=OperationKind.LOAD,
            operator='FRAME_LOAD',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_FRAME_STORE:
        return SemanticOperation(
            kind=OperationKind.ASSIGN,
            operator='FRAME_STORE',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_REG_LOAD:
        return SemanticOperation(
            kind=OperationKind.LOAD,
            operator='REG_LOAD',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_REG_STORE:
        return SemanticOperation(
            kind=OperationKind.ASSIGN,
            operator='REG_STORE',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op == LowLevelILOperation.LLIL_CONST:
        return SemanticOperation(
            kind=OperationKind.LOAD,
            operator='CONST',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    # NOP and internal
    elif op in (LowLevelILOperation.LLIL_NOP, LowLevelILOperation.LLIL_DEBUG,
                LowLevelILOperation.LLIL_LABEL, LowLevelILOperation.LLIL_SP_ADD,
                LowLevelILOperation.LLIL_PUSH_CALLER_FRAME):
        return SemanticOperation(
            kind=OperationKind.NOP,
            operator=op.name.replace('LLIL_', ''),
            source_location=loc
        )

    # Unknown - T068: track for warning
    _unknown_llil_ops.add(op.name)
    return SemanticOperation(
        kind=OperationKind.UNKNOWN,
        operator=op.name,
        source_location=loc
    )


def _extract_llil_operands(instr: LowLevelILInstruction) -> List[SemanticOperand]:
    """Extract operands from LLIL instruction"""
    operands = []

    # Check common attribute patterns
    if hasattr(instr, 'value'):
        operands.append(SemanticOperand(kind='const', value=instr.value))

    if hasattr(instr, 'slot_index'):
        operands.append(SemanticOperand(kind='var', value=f"Stack[{instr.slot_index}]"))

    if hasattr(instr, 'reg_index'):
        operands.append(SemanticOperand(kind='reg', value=f"Reg[{instr.reg_index}]"))

    if hasattr(instr, 'left') and hasattr(instr, 'right'):
        # Binary operation with nested instructions
        if isinstance(instr.left, LowLevelILInstruction):
            operands.append(SemanticOperand(kind='expr', value=str(instr.left)))

        else:
            operands.append(SemanticOperand(kind='const', value=instr.left))
        if isinstance(instr.right, LowLevelILInstruction):
            operands.append(SemanticOperand(kind='expr', value=str(instr.right)))

        else:
            operands.append(SemanticOperand(kind='const', value=instr.right))

    if hasattr(instr, 'operand'):
        # Unary operation
        if isinstance(instr.operand, LowLevelILInstruction):
            operands.append(SemanticOperand(kind='expr', value=str(instr.operand)))

        else:
            operands.append(SemanticOperand(kind='const', value=instr.operand))

    return operands


def normalize_mlil_operation(instr: MediumLevelILInstruction) -> SemanticOperation:
    """Convert MLIL instruction to normalized SemanticOperation"""
    op = instr.operation
    loc = SourceLocation(mlil_index=instr.instr_index if hasattr(instr, 'instr_index') else -1)

    # Arithmetic operations
    if op in (MediumLevelILOperation.MLIL_ADD, MediumLevelILOperation.MLIL_SUB,
              MediumLevelILOperation.MLIL_MUL, MediumLevelILOperation.MLIL_DIV,
              MediumLevelILOperation.MLIL_MOD):
        return SemanticOperation(
            kind=OperationKind.ARITHMETIC,
            operator=op.name.replace('MLIL_', ''),
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_NEG:
        return SemanticOperation(
            kind=OperationKind.ARITHMETIC,
            operator='NEG',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    # Comparison operations
    elif op in (MediumLevelILOperation.MLIL_EQ, MediumLevelILOperation.MLIL_NE,
                MediumLevelILOperation.MLIL_LT, MediumLevelILOperation.MLIL_LE,
                MediumLevelILOperation.MLIL_GT, MediumLevelILOperation.MLIL_GE):
        return SemanticOperation(
            kind=OperationKind.COMPARISON,
            operator=op.name.replace('MLIL_', ''),
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_TEST_ZERO:
        return SemanticOperation(
            kind=OperationKind.COMPARISON,
            operator='TEST_ZERO',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    # Logical operations
    elif op in (MediumLevelILOperation.MLIL_LOGICAL_AND, MediumLevelILOperation.MLIL_LOGICAL_OR):
        return SemanticOperation(
            kind=OperationKind.LOGICAL,
            operator=op.name.replace('MLIL_', ''),
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_LOGICAL_NOT:
        return SemanticOperation(
            kind=OperationKind.LOGICAL,
            operator='LOGICAL_NOT',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    # Bitwise operations
    elif op in (MediumLevelILOperation.MLIL_AND, MediumLevelILOperation.MLIL_OR,
                MediumLevelILOperation.MLIL_XOR, MediumLevelILOperation.MLIL_SHL,
                MediumLevelILOperation.MLIL_SHR):
        return SemanticOperation(
            kind=OperationKind.BITWISE,
            operator=op.name.replace('MLIL_', ''),
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_BITWISE_NOT:
        return SemanticOperation(
            kind=OperationKind.BITWISE,
            operator='BITWISE_NOT',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    # Control flow
    elif op == MediumLevelILOperation.MLIL_GOTO:
        return SemanticOperation(
            kind=OperationKind.BRANCH,
            operator='GOTO',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_IF:
        return SemanticOperation(
            kind=OperationKind.BRANCH,
            operator='IF',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_CALL:
        return SemanticOperation(
            kind=OperationKind.CALL,
            operator='CALL',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_SYSCALL:
        return SemanticOperation(
            kind=OperationKind.CALL,
            operator='SYSCALL',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_CALL_SCRIPT:
        return SemanticOperation(
            kind=OperationKind.CALL,
            operator='CALL_SCRIPT',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_RET:
        return SemanticOperation(
            kind=OperationKind.RETURN,
            operator='RET',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    # Variable operations
    elif op == MediumLevelILOperation.MLIL_VAR:
        return SemanticOperation(
            kind=OperationKind.LOAD,
            operator='VAR',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_SET_VAR:
        return SemanticOperation(
            kind=OperationKind.ASSIGN,
            operator='SET_VAR',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_CONST:
        return SemanticOperation(
            kind=OperationKind.LOAD,
            operator='CONST',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    # Global and register
    elif op == MediumLevelILOperation.MLIL_LOAD_GLOBAL:
        return SemanticOperation(
            kind=OperationKind.LOAD,
            operator='LOAD_GLOBAL',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_STORE_GLOBAL:
        return SemanticOperation(
            kind=OperationKind.ASSIGN,
            operator='STORE_GLOBAL',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_LOAD_REG:
        return SemanticOperation(
            kind=OperationKind.LOAD,
            operator='LOAD_REG',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    elif op == MediumLevelILOperation.MLIL_STORE_REG:
        return SemanticOperation(
            kind=OperationKind.ASSIGN,
            operator='STORE_REG',
            operands=_extract_mlil_operands(instr),
            source_location=loc
        )

    # NOP
    elif op in (MediumLevelILOperation.MLIL_NOP, MediumLevelILOperation.MLIL_DEBUG):
        return SemanticOperation(
            kind=OperationKind.NOP,
            operator=op.name.replace('MLIL_', ''),
            source_location=loc
        )

    # Unknown - T068: track for warning
    _unknown_mlil_ops.add(op.name)
    return SemanticOperation(
        kind=OperationKind.UNKNOWN,
        operator=op.name,
        source_location=loc
    )


def _extract_mlil_operands(instr: MediumLevelILInstruction) -> List[SemanticOperand]:
    """Extract operands from MLIL instruction"""
    operands = []

    if hasattr(instr, 'value'):
        operands.append(SemanticOperand(kind='const', value=instr.value))

    if hasattr(instr, 'var') and instr.var is not None:
        var_name = instr.var.name if hasattr(instr.var, 'name') else str(instr.var)
        operands.append(SemanticOperand(kind='var', value=var_name))

    if hasattr(instr, 'left') and hasattr(instr, 'right'):
        if isinstance(instr.left, MediumLevelILInstruction):
            operands.append(SemanticOperand(kind='expr', value=str(instr.left)))

        else:
            operands.append(SemanticOperand(kind='const', value=instr.left))
        if isinstance(instr.right, MediumLevelILInstruction):
            operands.append(SemanticOperand(kind='expr', value=str(instr.right)))

        else:
            operands.append(SemanticOperand(kind='const', value=instr.right))

    if hasattr(instr, 'operand'):
        if isinstance(instr.operand, MediumLevelILInstruction):
            operands.append(SemanticOperand(kind='expr', value=str(instr.operand)))

        else:
            operands.append(SemanticOperand(kind='const', value=instr.operand))

    return operands


def normalize_hlil_operation(instr: HLILInstruction) -> SemanticOperation:
    """Convert HLIL instruction to normalized SemanticOperation"""
    op = instr.operation
    loc = SourceLocation()

    # Control flow statements
    if op == HLILOperation.HLIL_IF:
        return SemanticOperation(
            kind=OperationKind.CONTROL_FLOW,
            operator='IF',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    elif op == HLILOperation.HLIL_WHILE:
        return SemanticOperation(
            kind=OperationKind.CONTROL_FLOW,
            operator='WHILE',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    elif op == HLILOperation.HLIL_DO_WHILE:
        return SemanticOperation(
            kind=OperationKind.CONTROL_FLOW,
            operator='DO_WHILE',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    elif op == HLILOperation.HLIL_FOR:
        return SemanticOperation(
            kind=OperationKind.CONTROL_FLOW,
            operator='FOR',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    elif op == HLILOperation.HLIL_SWITCH:
        return SemanticOperation(
            kind=OperationKind.CONTROL_FLOW,
            operator='SWITCH',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    elif op == HLILOperation.HLIL_BREAK:
        return SemanticOperation(
            kind=OperationKind.CONTROL_FLOW,
            operator='BREAK',
            source_location=loc
        )

    elif op == HLILOperation.HLIL_CONTINUE:
        return SemanticOperation(
            kind=OperationKind.CONTROL_FLOW,
            operator='CONTINUE',
            source_location=loc
        )

    elif op == HLILOperation.HLIL_RETURN:
        return SemanticOperation(
            kind=OperationKind.RETURN,
            operator='RETURN',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    # Assignment
    elif op == HLILOperation.HLIL_ASSIGN:
        return SemanticOperation(
            kind=OperationKind.ASSIGN,
            operator='ASSIGN',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    # Block
    elif op == HLILOperation.HLIL_BLOCK:
        return SemanticOperation(
            kind=OperationKind.NOP,
            operator='BLOCK',
            source_location=loc
        )

    # Expression statement
    elif op == HLILOperation.HLIL_EXPR_STMT:
        return SemanticOperation(
            kind=OperationKind.NOP,
            operator='EXPR_STMT',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    # Comment
    elif op == HLILOperation.HLIL_COMMENT:
        return SemanticOperation(
            kind=OperationKind.NOP,
            operator='COMMENT',
            source_location=loc
        )

    # Expressions
    elif op == HLILOperation.HLIL_VAR:
        return SemanticOperation(
            kind=OperationKind.LOAD,
            operator='VAR',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    elif op == HLILOperation.HLIL_CONST:
        return SemanticOperation(
            kind=OperationKind.LOAD,
            operator='CONST',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    elif op == HLILOperation.HLIL_BINARY_OP:
        bin_op = instr.op if hasattr(instr, 'op') else None
        return SemanticOperation(
            kind=_get_binary_op_kind(bin_op),
            operator=bin_op.name if bin_op else 'BINARY_OP',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    elif op == HLILOperation.HLIL_UNARY_OP:
        unary_op = instr.op if hasattr(instr, 'op') else None
        return SemanticOperation(
            kind=_get_unary_op_kind(unary_op),
            operator=unary_op.name if unary_op else 'UNARY_OP',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    elif op == HLILOperation.HLIL_CALL:
        return SemanticOperation(
            kind=OperationKind.CALL,
            operator='CALL',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    elif op == HLILOperation.HLIL_SYSCALL:
        return SemanticOperation(
            kind=OperationKind.CALL,
            operator='SYSCALL',
            operands=_extract_hlil_operands(instr),
            source_location=loc
        )

    # Unknown - T068: track for warning
    op_name = op.name if hasattr(op, 'name') else str(op)
    _unknown_hlil_ops.add(op_name)
    return SemanticOperation(
        kind=OperationKind.UNKNOWN,
        operator=op_name,
        source_location=loc
    )


def _extract_hlil_operands(instr: HLILInstruction) -> List[SemanticOperand]:
    """Extract operands from HLIL instruction"""
    operands = []

    if hasattr(instr, 'value'):
        operands.append(SemanticOperand(kind='const', value=instr.value))

    if hasattr(instr, 'var') and instr.var is not None:
        var_name = instr.var.name if hasattr(instr.var, 'name') else str(instr.var)
        operands.append(SemanticOperand(kind='var', value=var_name))

    if hasattr(instr, 'left') and hasattr(instr, 'right'):
        if isinstance(instr.left, HLILInstruction):
            operands.append(SemanticOperand(kind='expr', value=str(instr.left)))

        else:
            operands.append(SemanticOperand(kind='const', value=instr.left))
        if isinstance(instr.right, HLILInstruction):
            operands.append(SemanticOperand(kind='expr', value=str(instr.right)))

        else:
            operands.append(SemanticOperand(kind='const', value=instr.right))

    if hasattr(instr, 'operand'):
        if isinstance(instr.operand, HLILInstruction):
            operands.append(SemanticOperand(kind='expr', value=str(instr.operand)))

        else:
            operands.append(SemanticOperand(kind='const', value=instr.operand))

    return operands


def _get_binary_op_kind(op: Optional[BinaryOp]) -> OperationKind:
    """Map HLIL BinaryOp to OperationKind"""
    if op is None:
        return OperationKind.UNKNOWN

    if op in (BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV, BinaryOp.MOD):
        return OperationKind.ARITHMETIC

    elif op in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.LE, BinaryOp.GT, BinaryOp.GE):
        return OperationKind.COMPARISON

    elif op in (BinaryOp.AND, BinaryOp.OR):
        return OperationKind.LOGICAL

    elif op in (BinaryOp.BIT_AND, BinaryOp.BIT_OR, BinaryOp.BIT_XOR, BinaryOp.SHL, BinaryOp.SHR):
        return OperationKind.BITWISE

    return OperationKind.UNKNOWN


def _get_unary_op_kind(op: Optional[UnaryOp]) -> OperationKind:
    """Map HLIL UnaryOp to OperationKind"""
    if op is None:
        return OperationKind.UNKNOWN

    if op == UnaryOp.NEG:
        return OperationKind.ARITHMETIC

    elif op == UnaryOp.NOT:
        return OperationKind.LOGICAL

    elif op == UnaryOp.BIT_NOT:
        return OperationKind.BITWISE

    return OperationKind.UNKNOWN


# =============================================================================
# T014-T026: LLIL-MLIL Comparison
# =============================================================================

class LLILMLILComparator:
    """Compares semantic logic between LLIL and MLIL layers"""

    # T023: Expected storage transformations (LLIL op → MLIL op)
    STORAGE_TRANSFORMS: Dict[str, str] = {
        'STACK_LOAD': 'VAR',
        'STACK_STORE': 'SET_VAR',
        'FRAME_LOAD': 'VAR',
        'FRAME_STORE': 'SET_VAR',
        'REG_LOAD': 'LOAD_REG',
        'REG_STORE': 'STORE_REG',
    }

    # T018: Expected control flow transformations
    CONTROL_FLOW_TRANSFORMS: Dict[str, str] = {
        'BRANCH': 'IF',
        'JMP': 'GOTO',
    }

    # T024: Falcom VM ops that get eliminated or transformed
    ELIMINATED_OPS: set = {
        'PUSH_CALLER_FRAME', 'PUSH_FUNC_ID', 'PUSH_RET_ADDR',
        'SP_ADD', 'LABEL', 'NOP', 'DEBUG',
        # Also skip intermediate stack operations that get merged
        'STACK_STORE', 'STACK_LOAD',
    }

    # Core semantic operations with external effects (compare these)
    CORE_SEMANTIC_KINDS: set = {
        OperationKind.CALL,       # Function/syscall - external effect
        OperationKind.RETURN,     # Return - control flow
        OperationKind.BRANCH,     # Branch - control flow
    }

    def __init__(self, llil_func: LowLevelILFunction, mlil_func: MediumLevelILFunction):
        self.llil_func = llil_func
        self.mlil_func = mlil_func
        self.results: List[ComparisonResult] = []
        self.variable_mappings: Dict[str, str] = {}  # LLIL storage → MLIL var

    def compare(self) -> List[ComparisonResult]:
        """T014: Compare LLIL and MLIL functions"""
        self.results = []
        self._build_variable_mappings()

        # Normalize all operations
        llil_ops = self._normalize_llil()
        mlil_ops = self._normalize_mlil()

        # Compare normalized operations
        self._compare_operations(llil_ops, mlil_ops)

        return self.results

    def _build_variable_mappings(self) -> None:
        """Build storage → variable mappings from MLIL"""
        for bb in self.mlil_func.basic_blocks:
            for instr in bb.instructions:
                if instr.operation == MediumLevelILOperation.MLIL_SET_VAR:
                    if hasattr(instr, 'var') and instr.var is not None:
                        var = instr.var
                        var_name = var.name if hasattr(var, 'name') else str(var)
                        # Extract source storage from variable naming convention
                        if hasattr(var, 'source_storage'):
                            self.variable_mappings[var.source_storage] = var_name

    def _normalize_llil(self) -> List[SemanticOperation]:
        """Normalize all LLIL instructions - extract only semantic operations"""
        ops = []
        for bb in self.llil_func.basic_blocks:
            for instr in bb.instructions:
                # Skip stack operations (absorbed into MLIL)
                if instr.operation in (
                    LowLevelILOperation.LLIL_STACK_STORE,
                    LowLevelILOperation.LLIL_STACK_LOAD,
                    LowLevelILOperation.LLIL_SP_ADD,
                    LowLevelILOperation.LLIL_FRAME_STORE,
                    LowLevelILOperation.LLIL_FRAME_LOAD,
                ):
                    continue

                sem_op = normalize_llil_operation(instr)
                # Only keep core semantic operations
                if sem_op.kind in self.CORE_SEMANTIC_KINDS:
                    ops.append(sem_op)
        return ops

    def _normalize_mlil(self) -> List[SemanticOperation]:
        """Normalize all MLIL instructions - extract only semantic operations"""
        ops = []
        for bb in self.mlil_func.basic_blocks:
            for instr in bb.instructions:
                sem_op = normalize_mlil_operation(instr)
                # Only keep core semantic operations
                if sem_op.kind in self.CORE_SEMANTIC_KINDS:
                    ops.append(sem_op)
        return ops

    def _compare_operations(
        self, llil_ops: List[SemanticOperation], mlil_ops: List[SemanticOperation]
    ) -> None:
        """Compare normalized operations between layers"""
        llil_idx = 0
        mlil_idx = 0

        while llil_idx < len(llil_ops) and mlil_idx < len(mlil_ops):
            llil_op = llil_ops[llil_idx]
            mlil_op = mlil_ops[mlil_idx]

            result = self._compare_single_operation(llil_op, mlil_op)
            self.results.append(result)

            # Advance indices based on comparison
            if result.status in (ComparisonStatus.EQUIVALENT, ComparisonStatus.TRANSFORMED):
                llil_idx += 1
                mlil_idx += 1

            elif result.status == ComparisonStatus.DIFFERENT:
                # Try to find matching operation nearby
                if self._try_skip_llil(llil_ops, llil_idx, mlil_op):
                    llil_idx += 1

                else:
                    mlil_idx += 1
                    llil_idx += 1

        # Handle remaining operations
        while llil_idx < len(llil_ops):
            llil_op = llil_ops[llil_idx]
            self.results.append(ComparisonResult(
                status=ComparisonStatus.DIFFERENT,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=None,
                explanation=f"LLIL operation has no MLIL counterpart: {llil_op.operator}",
                source_location=llil_op.source_location
            ))
            llil_idx += 1

        while mlil_idx < len(mlil_ops):
            mlil_op = mlil_ops[mlil_idx]
            self.results.append(ComparisonResult(
                status=ComparisonStatus.DIFFERENT,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=None,
                target_op=mlil_op,
                explanation=f"MLIL operation has no LLIL counterpart: {mlil_op.operator}",
                source_location=mlil_op.source_location
            ))
            mlil_idx += 1

    def _try_skip_llil(
        self, llil_ops: List[SemanticOperation], idx: int, mlil_op: SemanticOperation
    ) -> bool:
        """Check if current LLIL op should be skipped (eliminated in MLIL)"""
        if idx >= len(llil_ops):
            return False
        llil_op = llil_ops[idx]
        return llil_op.operator in self.ELIMINATED_OPS

    def _compare_single_operation(
        self, llil_op: SemanticOperation, mlil_op: SemanticOperation
    ) -> ComparisonResult:
        """Compare a single pair of operations"""
        loc = llil_op.source_location

        # T15: Check arithmetic operations
        if llil_op.kind == OperationKind.ARITHMETIC:
            return self._compare_arithmetic(llil_op, mlil_op, loc)

        # T16: Check comparison operations
        elif llil_op.kind == OperationKind.COMPARISON:
            return self._compare_comparison(llil_op, mlil_op, loc)

        # T17: Check logical/bitwise operations
        elif llil_op.kind in (OperationKind.LOGICAL, OperationKind.BITWISE):
            return self._compare_logical_bitwise(llil_op, mlil_op, loc)

        # T18: Check control flow
        elif llil_op.kind == OperationKind.BRANCH:
            return self._compare_control_flow(llil_op, mlil_op, loc)

        elif llil_op.kind == OperationKind.CALL:
            return self._compare_call(llil_op, mlil_op, loc)

        elif llil_op.kind == OperationKind.RETURN:
            return self._compare_return(llil_op, mlil_op, loc)

        # T23: Check storage transformations
        elif llil_op.kind in (OperationKind.LOAD, OperationKind.ASSIGN):
            return self._compare_storage(llil_op, mlil_op, loc)

        # Unknown
        return ComparisonResult(
            status=ComparisonStatus.DIFFERENT,
            source_layer=IRLayer.LLIL,
            target_layer=IRLayer.MLIL,
            source_op=llil_op,
            target_op=mlil_op,
            explanation=f"Unknown operation kind: {llil_op.kind}",
            source_location=loc
        )

    def _compare_arithmetic(
        self, llil_op: SemanticOperation, mlil_op: SemanticOperation, loc: SourceLocation
    ) -> ComparisonResult:
        """T015: Compare arithmetic operations"""
        if mlil_op.kind != OperationKind.ARITHMETIC:
            return ComparisonResult(
                status=ComparisonStatus.DIFFERENT,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation=f"Expected ARITHMETIC, got {mlil_op.kind.name}",
                source_location=loc
            )

        if llil_op.operator == mlil_op.operator:
            # Same operator - check operands
            if self._operands_equivalent(llil_op.operands, mlil_op.operands):
                return ComparisonResult(
                    status=ComparisonStatus.EQUIVALENT,
                    source_layer=IRLayer.LLIL,
                    target_layer=IRLayer.MLIL,
                    source_op=llil_op,
                    target_op=mlil_op,
                    explanation=f"{llil_op.operator} matches",
                    source_location=loc
                )

            else:
                return ComparisonResult(
                    status=ComparisonStatus.TRANSFORMED,
                    source_layer=IRLayer.LLIL,
                    target_layer=IRLayer.MLIL,
                    source_op=llil_op,
                    target_op=mlil_op,
                    explanation=f"{llil_op.operator}: operands transformed",
                    source_location=loc
                )

        return ComparisonResult(
            status=ComparisonStatus.DIFFERENT,
            source_layer=IRLayer.LLIL,
            target_layer=IRLayer.MLIL,
            source_op=llil_op,
            target_op=mlil_op,
            explanation=f"Operator mismatch: {llil_op.operator} vs {mlil_op.operator}",
            source_location=loc
        )

    def _compare_comparison(
        self, llil_op: SemanticOperation, mlil_op: SemanticOperation, loc: SourceLocation
    ) -> ComparisonResult:
        """T016: Compare comparison operations"""
        if mlil_op.kind != OperationKind.COMPARISON:
            return ComparisonResult(
                status=ComparisonStatus.DIFFERENT,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation=f"Expected COMPARISON, got {mlil_op.kind.name}",
                source_location=loc
            )

        # Handle TEST_ZERO → EQ(x, 0) transformation
        if llil_op.operator == 'TEST_ZERO' and mlil_op.operator == 'EQ':
            return ComparisonResult(
                status=ComparisonStatus.TRANSFORMED,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation="TEST_ZERO → EQ(x, 0)",
                source_location=loc
            )

        # Check inverse operators (a < b ≡ b > a)
        inverse_ops = {
            ('LT', 'GT'): True, ('GT', 'LT'): True,
            ('LE', 'GE'): True, ('GE', 'LE'): True,
        }

        if llil_op.operator == mlil_op.operator:
            return ComparisonResult(
                status=ComparisonStatus.EQUIVALENT,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation=f"{llil_op.operator} matches",
                source_location=loc
            )

        elif (llil_op.operator, mlil_op.operator) in inverse_ops:
            return ComparisonResult(
                status=ComparisonStatus.TRANSFORMED,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation=f"Inverse comparison: {llil_op.operator} → {mlil_op.operator}",
                source_location=loc
            )

        return ComparisonResult(
            status=ComparisonStatus.DIFFERENT,
            source_layer=IRLayer.LLIL,
            target_layer=IRLayer.MLIL,
            source_op=llil_op,
            target_op=mlil_op,
            explanation=f"Comparison mismatch: {llil_op.operator} vs {mlil_op.operator}",
            source_location=loc
        )

    def _compare_logical_bitwise(
        self, llil_op: SemanticOperation, mlil_op: SemanticOperation, loc: SourceLocation
    ) -> ComparisonResult:
        """T017: Compare logical and bitwise operations"""
        if mlil_op.kind != llil_op.kind:
            return ComparisonResult(
                status=ComparisonStatus.DIFFERENT,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation=f"Kind mismatch: {llil_op.kind.name} vs {mlil_op.kind.name}",
                source_location=loc
            )

        if llil_op.operator == mlil_op.operator:
            return ComparisonResult(
                status=ComparisonStatus.EQUIVALENT,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation=f"{llil_op.operator} matches",
                source_location=loc
            )

        return ComparisonResult(
            status=ComparisonStatus.DIFFERENT,
            source_layer=IRLayer.LLIL,
            target_layer=IRLayer.MLIL,
            source_op=llil_op,
            target_op=mlil_op,
            explanation=f"Operator mismatch: {llil_op.operator} vs {mlil_op.operator}",
            source_location=loc
        )

    def _compare_control_flow(
        self, llil_op: SemanticOperation, mlil_op: SemanticOperation, loc: SourceLocation
    ) -> ComparisonResult:
        """T018: Compare control flow operations"""
        expected_mlil = self.CONTROL_FLOW_TRANSFORMS.get(llil_op.operator)

        if expected_mlil and mlil_op.operator == expected_mlil:
            return ComparisonResult(
                status=ComparisonStatus.TRANSFORMED,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation=f"{llil_op.operator} → {mlil_op.operator}",
                source_location=loc
            )

        elif llil_op.operator == mlil_op.operator:
            return ComparisonResult(
                status=ComparisonStatus.EQUIVALENT,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation=f"{llil_op.operator} matches",
                source_location=loc
            )

        return ComparisonResult(
            status=ComparisonStatus.DIFFERENT,
            source_layer=IRLayer.LLIL,
            target_layer=IRLayer.MLIL,
            source_op=llil_op,
            target_op=mlil_op,
            explanation=f"Control flow mismatch: {llil_op.operator} vs {mlil_op.operator}",
            source_location=loc
        )

    def _compare_call(
        self, llil_op: SemanticOperation, mlil_op: SemanticOperation, loc: SourceLocation
    ) -> ComparisonResult:
        """Compare call operations"""
        if mlil_op.kind != OperationKind.CALL:
            return ComparisonResult(
                status=ComparisonStatus.DIFFERENT,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation=f"Expected CALL, got {mlil_op.kind.name}",
                source_location=loc
            )

        if llil_op.operator == mlil_op.operator:
            return ComparisonResult(
                status=ComparisonStatus.EQUIVALENT,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation=f"{llil_op.operator} matches",
                source_location=loc
            )

        return ComparisonResult(
            status=ComparisonStatus.DIFFERENT,
            source_layer=IRLayer.LLIL,
            target_layer=IRLayer.MLIL,
            source_op=llil_op,
            target_op=mlil_op,
            explanation=f"Call type mismatch: {llil_op.operator} vs {mlil_op.operator}",
            source_location=loc
        )

    def _compare_return(
        self, llil_op: SemanticOperation, mlil_op: SemanticOperation, loc: SourceLocation
    ) -> ComparisonResult:
        """Compare return operations"""
        if mlil_op.kind != OperationKind.RETURN:
            return ComparisonResult(
                status=ComparisonStatus.DIFFERENT,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation=f"Expected RETURN, got {mlil_op.kind.name}",
                source_location=loc
            )

        return ComparisonResult(
            status=ComparisonStatus.EQUIVALENT,
            source_layer=IRLayer.LLIL,
            target_layer=IRLayer.MLIL,
            source_op=llil_op,
            target_op=mlil_op,
            explanation="RET matches",
            source_location=loc
        )

    def _compare_storage(
        self, llil_op: SemanticOperation, mlil_op: SemanticOperation, loc: SourceLocation
    ) -> ComparisonResult:
        """T019, T020, T21, T23: Compare storage operations with expected transformations"""
        expected_mlil = self.STORAGE_TRANSFORMS.get(llil_op.operator)

        # T23: Expected storage transformation
        if expected_mlil and mlil_op.operator == expected_mlil:
            return ComparisonResult(
                status=ComparisonStatus.TRANSFORMED,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation=f"Storage: {llil_op.operator} → {mlil_op.operator}",
                source_location=loc
            )

        # T19: Constant values
        if llil_op.operator == 'CONST' and mlil_op.operator == 'CONST':
            llil_val = self._get_const_value(llil_op)
            mlil_val = self._get_const_value(mlil_op)
            if llil_val == mlil_val:
                return ComparisonResult(
                    status=ComparisonStatus.EQUIVALENT,
                    source_layer=IRLayer.LLIL,
                    target_layer=IRLayer.MLIL,
                    source_op=llil_op,
                    target_op=mlil_op,
                    explanation=f"CONST({llil_val}) matches",
                    source_location=loc
                )

            else:
                return ComparisonResult(
                    status=ComparisonStatus.DIFFERENT,
                    source_layer=IRLayer.LLIL,
                    target_layer=IRLayer.MLIL,
                    source_op=llil_op,
                    target_op=mlil_op,
                    explanation=f"CONST value mismatch: {llil_val} vs {mlil_val}",
                    source_location=loc
                )

        # T21: Global variable access
        if llil_op.operator in ('GLOBAL_LOAD', 'GLOBAL_STORE'):
            if mlil_op.operator in ('LOAD_GLOBAL', 'STORE_GLOBAL'):
                return ComparisonResult(
                    status=ComparisonStatus.TRANSFORMED,
                    source_layer=IRLayer.LLIL,
                    target_layer=IRLayer.MLIL,
                    source_op=llil_op,
                    target_op=mlil_op,
                    explanation=f"Global: {llil_op.operator} → {mlil_op.operator}",
                    source_location=loc
                )

        # Direct match
        if llil_op.operator == mlil_op.operator:
            return ComparisonResult(
                status=ComparisonStatus.EQUIVALENT,
                source_layer=IRLayer.LLIL,
                target_layer=IRLayer.MLIL,
                source_op=llil_op,
                target_op=mlil_op,
                explanation=f"{llil_op.operator} matches",
                source_location=loc
            )

        return ComparisonResult(
            status=ComparisonStatus.DIFFERENT,
            source_layer=IRLayer.LLIL,
            target_layer=IRLayer.MLIL,
            source_op=llil_op,
            target_op=mlil_op,
            explanation=f"Storage mismatch: {llil_op.operator} vs {mlil_op.operator}",
            source_location=loc
        )

    def _operands_equivalent(
        self, llil_operands: List[SemanticOperand], mlil_operands: List[SemanticOperand]
    ) -> bool:
        """Check if operands are semantically equivalent"""
        if len(llil_operands) != len(mlil_operands):
            return False

        for llil_op, mlil_op in zip(llil_operands, mlil_operands):
            if llil_op.kind == 'const' and mlil_op.kind == 'const':
                if llil_op.value != mlil_op.value:
                    return False

            elif llil_op.kind == 'var' and mlil_op.kind == 'var':
                # Check variable mapping
                llil_storage = str(llil_op.value)
                mlil_var = str(mlil_op.value)
                mapped = self.variable_mappings.get(llil_storage)
                if mapped and mapped != mlil_var:
                    return False

            # For expressions, compare string representations (simplified)
            elif llil_op.kind == 'expr' and mlil_op.kind == 'expr':
                pass  # Allow expression transformations

        return True

    def _get_const_value(self, op: SemanticOperation) -> Any:
        """Extract constant value from operation"""
        for operand in op.operands:
            if operand.kind == 'const':
                return operand.value
        return None


# =============================================================================
# T027-T033: Phase 4 - Variable/Storage Tracking
# =============================================================================

class VariableTracker:
    """T027: Track variable correspondence across IR layers"""

    def __init__(
        self,
        llil_func: LowLevelILFunction,
        mlil_func: MediumLevelILFunction,
        hlil_func: HighLevelILFunction
    ):
        self.llil_func = llil_func
        self.mlil_func = mlil_func
        self.hlil_func = hlil_func
        self.mappings: List[VariableMapping] = []

    def build_mappings(self) -> List[VariableMapping]:
        """Build variable mappings across all layers"""
        self.mappings = []

        # T028: Extract stack slot → MLIL variable mappings
        self._extract_stack_mappings()

        # T029: Extract frame slot → parameter mappings
        self._extract_parameter_mappings()

        # T030: Extract register → MLIL register mappings
        self._extract_register_mappings()

        # T031: Extract MLIL → HLIL variable mappings
        self._extract_hlil_mappings()

        return self.mappings

    def _extract_stack_mappings(self) -> None:
        """T028: Map stack slots to MLIL variables"""
        for bb in self.mlil_func.basic_blocks:
            for instr in bb.instructions:
                if instr.operation == MediumLevelILOperation.MLIL_SET_VAR:
                    if hasattr(instr, 'dest') and instr.dest is not None:
                        var = instr.dest
                        var_name = var.name if hasattr(var, 'name') else str(var)

                        # Check if this variable came from a stack slot
                        if var_name.startswith('var_s'):
                            # Extract stack offset from variable name
                            try:
                                offset_str = var_name[5:]  # Remove 'var_s'
                                offset = int(offset_str)
                                llil_storage = f"STACK[sp+{offset}]"

                                mapping = VariableMapping(
                                    llil_storage=llil_storage,
                                    mlil_var=var_name,
                                    hlil_var=None,
                                    type_info=None
                                )
                                self.mappings.append(mapping)
                            except ValueError:
                                pass

    def _extract_parameter_mappings(self) -> None:
        """T029: Map frame slots to function parameters"""
        # Parameters are accessed via frame pointer
        if hasattr(self.llil_func, 'params'):
            for i, param in enumerate(self.llil_func.params):
                param_name = param.name if hasattr(param, 'name') else f"arg{i+1}"
                llil_storage = f"FRAME[fp+{i}]"

                mapping = VariableMapping(
                    llil_storage=llil_storage,
                    mlil_var=param_name,
                    hlil_var=param_name,
                    type_info=param.type_name if hasattr(param, 'type_name') else None
                )
                self.mappings.append(mapping)

    def _extract_register_mappings(self) -> None:
        """T030: Map registers to MLIL register variables"""
        for bb in self.mlil_func.basic_blocks:
            for instr in bb.instructions:
                # Check for register stores
                if instr.operation == MediumLevelILOperation.MLIL_STORE_REG:
                    if hasattr(instr, 'dest_reg'):
                        reg_id = instr.dest_reg
                        llil_storage = f"REG[{reg_id}]"

                        # Find corresponding variable
                        if hasattr(instr, 'src') and hasattr(instr.src, 'var'):
                            mlil_var = str(instr.src.var)
                            mapping = VariableMapping(
                                llil_storage=llil_storage,
                                mlil_var=mlil_var,
                                hlil_var=None,
                                type_info=None
                            )
                            self.mappings.append(mapping)

    def _extract_hlil_mappings(self) -> None:
        """T031: Map MLIL variables to HLIL named variables"""
        # HLIL typically uses the same variable names as MLIL
        # but may have additional naming from decompilation
        if hasattr(self.hlil_func, 'statements'):
            for stmt in self.hlil_func.statements:
                self._scan_hlil_statement(stmt)

    def _scan_hlil_statement(self, stmt: HLILStatement) -> None:
        """Recursively scan HLIL statement for variable references"""
        if hasattr(stmt, 'operation'):
            if stmt.operation == HLILOperation.ASSIGN:
                if hasattr(stmt, 'dest') and hasattr(stmt.dest, 'var'):
                    hlil_var = str(stmt.dest.var)
                    # Update existing mapping or create new one
                    for mapping in self.mappings:
                        if mapping.mlil_var and hlil_var.startswith(mapping.mlil_var):
                            mapping.hlil_var = hlil_var
                            break

        # Recurse into sub-statements
        if hasattr(stmt, 'body'):
            if isinstance(stmt.body, list):
                for sub in stmt.body:
                    self._scan_hlil_statement(sub)
            elif stmt.body:
                self._scan_hlil_statement(stmt.body)

        if hasattr(stmt, 'else_body') and stmt.else_body:
            if isinstance(stmt.else_body, list):
                for sub in stmt.else_body:
                    self._scan_hlil_statement(sub)
            else:
                self._scan_hlil_statement(stmt.else_body)


# =============================================================================
# T034-T041: Phase 5 - MLIL/HLIL Comparison
# =============================================================================

class MLILHLILComparator:
    """T034: Compare semantic logic between MLIL and HLIL layers"""

    # Core semantic operations to compare
    CORE_SEMANTIC_KINDS: set = {
        OperationKind.CALL,
        OperationKind.RETURN,
        OperationKind.BRANCH,
    }

    def __init__(self, mlil_func: MediumLevelILFunction, hlil_func: HighLevelILFunction):
        self.mlil_func = mlil_func
        self.hlil_func = hlil_func
        self.results: List[ComparisonResult] = []

    def compare(self) -> List[ComparisonResult]:
        """Compare MLIL and HLIL functions"""
        self.results = []

        # Extract semantic operations
        mlil_ops = self._normalize_mlil()
        hlil_ops = self._normalize_hlil()

        # Compare
        self._compare_operations(mlil_ops, hlil_ops)

        return self.results

    def _normalize_mlil(self) -> List[SemanticOperation]:
        """Normalize MLIL instructions"""
        ops = []
        for bb in self.mlil_func.basic_blocks:
            for instr in bb.instructions:
                sem_op = normalize_mlil_operation(instr)
                if sem_op.kind in self.CORE_SEMANTIC_KINDS:
                    ops.append(sem_op)
        return ops

    def _normalize_hlil(self) -> List[SemanticOperation]:
        """Normalize HLIL statements"""
        ops = []
        if hasattr(self.hlil_func, 'statements'):
            for stmt in self.hlil_func.statements:
                self._collect_hlil_ops(stmt, ops)
        return ops

    def _collect_hlil_ops(self, stmt: HLILStatement, ops: List[SemanticOperation]) -> None:
        """Recursively collect semantic operations from HLIL"""
        sem_op = normalize_hlil_operation(stmt)
        if sem_op.kind in self.CORE_SEMANTIC_KINDS:
            ops.append(sem_op)

        # Recurse into sub-statements
        if hasattr(stmt, 'body'):
            if isinstance(stmt.body, list):
                for sub in stmt.body:
                    self._collect_hlil_ops(sub, ops)
            elif stmt.body:
                self._collect_hlil_ops(stmt.body, ops)

        if hasattr(stmt, 'else_body') and stmt.else_body:
            if isinstance(stmt.else_body, list):
                for sub in stmt.else_body:
                    self._collect_hlil_ops(sub, ops)
            elif stmt.else_body:
                self._collect_hlil_ops(stmt.else_body, ops)

    def _compare_operations(
        self, mlil_ops: List[SemanticOperation], hlil_ops: List[SemanticOperation]
    ) -> None:
        """Compare normalized operations"""
        mlil_idx = 0
        hlil_idx = 0

        while mlil_idx < len(mlil_ops) and hlil_idx < len(hlil_ops):
            mlil_op = mlil_ops[mlil_idx]
            hlil_op = hlil_ops[hlil_idx]

            result = self._compare_single(mlil_op, hlil_op)
            self.results.append(result)

            if result.status in (ComparisonStatus.EQUIVALENT, ComparisonStatus.TRANSFORMED):
                mlil_idx += 1
                hlil_idx += 1

            else:
                mlil_idx += 1
                hlil_idx += 1

        # Remaining ops
        while mlil_idx < len(mlil_ops):
            mlil_op = mlil_ops[mlil_idx]
            self.results.append(ComparisonResult(
                status=ComparisonStatus.DIFFERENT,
                source_layer=IRLayer.MLIL,
                target_layer=IRLayer.HLIL,
                source_op=mlil_op,
                target_op=None,
                explanation=f"MLIL op has no HLIL counterpart: {mlil_op.operator}",
                source_location=mlil_op.source_location
            ))
            mlil_idx += 1

        while hlil_idx < len(hlil_ops):
            hlil_op = hlil_ops[hlil_idx]
            self.results.append(ComparisonResult(
                status=ComparisonStatus.DIFFERENT,
                source_layer=IRLayer.MLIL,
                target_layer=IRLayer.HLIL,
                source_op=None,
                target_op=hlil_op,
                explanation=f"HLIL op has no MLIL counterpart: {hlil_op.operator}",
                source_location=hlil_op.source_location
            ))
            hlil_idx += 1

    def _compare_single(
        self, mlil_op: SemanticOperation, hlil_op: SemanticOperation
    ) -> ComparisonResult:
        """Compare single operation pair"""
        loc = mlil_op.source_location

        # T035: Statement transformations
        if mlil_op.kind == OperationKind.CALL and hlil_op.kind == OperationKind.CALL:
            if mlil_op.operator == hlil_op.operator:
                return ComparisonResult(
                    status=ComparisonStatus.EQUIVALENT,
                    source_layer=IRLayer.MLIL,
                    target_layer=IRLayer.HLIL,
                    source_op=mlil_op,
                    target_op=hlil_op,
                    explanation=f"{mlil_op.operator} matches",
                    source_location=loc
                )

        # T036: Control flow structuring
        if mlil_op.kind == OperationKind.BRANCH and hlil_op.kind == OperationKind.BRANCH:
            # IF→HLILIf, GOTO→While/For transformations
            return ComparisonResult(
                status=ComparisonStatus.TRANSFORMED,
                source_layer=IRLayer.MLIL,
                target_layer=IRLayer.HLIL,
                source_op=mlil_op,
                target_op=hlil_op,
                explanation=f"Control flow: {mlil_op.operator} -> {hlil_op.operator}",
                source_location=loc
            )

        if mlil_op.kind == OperationKind.RETURN and hlil_op.kind == OperationKind.RETURN:
            return ComparisonResult(
                status=ComparisonStatus.EQUIVALENT,
                source_layer=IRLayer.MLIL,
                target_layer=IRLayer.HLIL,
                source_op=mlil_op,
                target_op=hlil_op,
                explanation="RET matches",
                source_location=loc
            )

        return ComparisonResult(
            status=ComparisonStatus.DIFFERENT,
            source_layer=IRLayer.MLIL,
            target_layer=IRLayer.HLIL,
            source_op=mlil_op,
            target_op=hlil_op,
            explanation=f"Mismatch: {mlil_op.operator} vs {hlil_op.operator}",
            source_location=loc
        )


def format_mlil_hlil_report(
    func_name: str, results: List[ComparisonResult], use_color: bool = True
) -> str:
    """Generate MLIL-HLIL comparison report"""
    lines = []
    lines.append(f"=== MLIL-HLIL Comparison: {func_name} ===")
    lines.append("")

    equivalent_count = sum(1 for r in results if r.status == ComparisonStatus.EQUIVALENT)
    transformed_count = sum(1 for r in results if r.status == ComparisonStatus.TRANSFORMED)
    different_count = sum(1 for r in results if r.status == ComparisonStatus.DIFFERENT)

    lines.append(f"Summary: {len(results)} operations compared")
    lines.append(f"  Equivalent: {equivalent_count}")
    lines.append(f"  Transformed: {transformed_count}")
    lines.append(f"  Different: {different_count}")
    lines.append("")

    if different_count > 0:
        lines.append("Differences:")
        for result in results:
            if result.status == ComparisonStatus.DIFFERENT:
                loc = result.source_location
                # Use MLIL index for MLIL-HLIL comparison
                if loc.mlil_index >= 0:
                    loc_str = f"[MLIL:{loc.mlil_index}]"
                elif loc.llil_index >= 0:
                    loc_str = f"[LLIL:{loc.llil_index}]"
                else:
                    loc_str = ""
                if use_color:
                    lines.append(f"  \033[31m{loc_str} {result.explanation}\033[0m")

                else:
                    lines.append(f"  {loc_str} {result.explanation}")
        lines.append("")

    status = "PASS" if different_count == 0 else "FAIL"
    if use_color:
        color = "\033[32m" if different_count == 0 else "\033[31m"
        lines.append(f"{color}Result: {status}\033[0m")

    else:
        lines.append(f"Result: {status}")

    return "\n".join(lines)


def format_variable_report(
    func_name: str, mappings: List[VariableMapping], use_color: bool = True
) -> str:
    """T032: Generate variable mapping report"""
    lines = []
    lines.append(f"=== Variable Mappings: {func_name} ===")
    lines.append("")

    if not mappings:
        lines.append("  No variable mappings found")
        return "\n".join(lines)

    # Group by type
    stack_vars = [m for m in mappings if m.llil_storage.startswith("STACK")]
    frame_vars = [m for m in mappings if m.llil_storage.startswith("FRAME")]
    reg_vars = [m for m in mappings if m.llil_storage.startswith("REG")]

    if frame_vars:
        lines.append("Parameters:")
        for m in frame_vars:
            hlil = m.hlil_var or "(same)"
            type_str = f" : {m.type_info}" if m.type_info else ""
            lines.append(f"  {m.llil_storage} → {m.mlil_var} → {hlil}{type_str}")
        lines.append("")

    if stack_vars:
        lines.append("Stack Variables:")
        for m in stack_vars:
            hlil = m.hlil_var or "(not mapped)"
            lines.append(f"  {m.llil_storage} → {m.mlil_var} → {hlil}")
        lines.append("")

    if reg_vars:
        lines.append("Register Variables:")
        for m in reg_vars:
            hlil = m.hlil_var or "(not mapped)"
            lines.append(f"  {m.llil_storage} → {m.mlil_var} → {hlil}")
        lines.append("")

    lines.append(f"Total: {len(mappings)} mappings")
    return "\n".join(lines)


def format_comparison_report(
    func_name: str, results: List[ComparisonResult], use_color: bool = True
) -> str:
    """T025: Generate human-readable comparison report"""
    lines = []
    lines.append(f"=== LLIL-MLIL Comparison: {func_name} ===")
    lines.append("")

    equivalent_count = sum(1 for r in results if r.status == ComparisonStatus.EQUIVALENT)
    transformed_count = sum(1 for r in results if r.status == ComparisonStatus.TRANSFORMED)
    different_count = sum(1 for r in results if r.status == ComparisonStatus.DIFFERENT)

    lines.append(f"Summary: {len(results)} operations compared")
    lines.append(f"  Equivalent: {equivalent_count}")
    lines.append(f"  Transformed: {transformed_count}")
    lines.append(f"  Different: {different_count}")
    lines.append("")

    if different_count > 0:
        lines.append("Differences:")
        for result in results:
            if result.status == ComparisonStatus.DIFFERENT:
                loc = result.source_location
                # Use LLIL index if SCP offset not available
                if loc.scp_offset:
                    loc_str = f"[SCP:0x{loc.scp_offset:04X}]"
                elif loc.llil_index >= 0:
                    loc_str = f"[LLIL:{loc.llil_index}]"
                else:
                    loc_str = ""
                if use_color:
                    lines.append(f"  \033[31m{loc_str} {result.explanation}\033[0m")

                else:
                    lines.append(f"  {loc_str} {result.explanation}")
        lines.append("")

    if transformed_count > 0 and different_count == 0:
        lines.append("All transformations are expected (stack -> variable, etc.)")

    status = "PASS" if different_count == 0 else "FAIL"
    if use_color:
        color = "\033[32m" if different_count == 0 else "\033[31m"
        lines.append(f"{color}Result: {status}\033[0m")

    else:
        lines.append(f"Result: {status}")

    return "\n".join(lines)


# =============================================================================
# CLI Main
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='IR Semantic Consistency Validator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --compare llil mlil test.scp
  %(prog)s --compare mlil hlil test.scp
  %(prog)s --all test.scp
  %(prog)s --batch test.scp
  %(prog)s --variables test.scp
  %(prog)s --types test.scp
  %(prog)s --function func_001 test.scp
"""
    )

    parser.add_argument('scp_file', help='Path to SCP file to validate')
    parser.add_argument('--compare', nargs=2, metavar=('LAYER1', 'LAYER2'),
                        help='Compare two specific layers (llil/mlil/hlil)')
    parser.add_argument('--all', action='store_true',
                        help='Three-way comparison of all layers')
    parser.add_argument('--variables', action='store_true',
                        help='Show variable mapping across layers')
    parser.add_argument('--types', action='store_true',
                        help='Validate type consistency')
    parser.add_argument('--batch', action='store_true',
                        help='Validate all functions in file')
    parser.add_argument('--function', metavar='NAME',
                        help='Validate specific function only')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')

    return parser


def main() -> int:
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Validate SCP file exists
    scp_path = Path(args.scp_file)
    if not scp_path.exists():
        print(f"Error: SCP file not found: {scp_path}", file=sys.stderr)
        return 1

    # Load and generate IR
    pipeline = IRPipeline(str(scp_path))
    if not pipeline.load():
        return 1

    if not pipeline.generate_ir(args.function):
        print("Error: Failed to generate IR for any function", file=sys.stderr)
        return 1

    use_color = not args.no_color and not args.json

    # T064: Handle --json output
    if args.json:
        return _run_json_output(pipeline, args, str(scp_path))

    print(f"=== IR Semantic Validation Report ===")
    print(f"File: {scp_path.name}")
    print(f"Functions loaded: {len(pipeline.functions)}")
    print()

    # Clear tracking for unknown operations
    clear_unknown_operations()

    # T026: Handle --compare llil mlil
    if args.compare:
        layer1, layer2 = args.compare[0].lower(), args.compare[1].lower()

        if layer1 == 'llil' and layer2 == 'mlil':
            result = _run_llil_mlil_comparison(pipeline, use_color)
            report_unknown_operations()  # T068
            return result

        elif layer1 == 'mlil' and layer2 == 'hlil':
            result = _run_mlil_hlil_comparison(pipeline, use_color)
            report_unknown_operations()  # T068
            return result

        else:
            print(f"Error: Unsupported comparison: {layer1} vs {layer2}", file=sys.stderr)
            print("Supported: llil mlil, mlil hlil", file=sys.stderr)
            return 1

    # T056: Handle --all (three-way comparison)
    if args.all:
        result = _run_three_way_comparison(pipeline, use_color)
        report_unknown_operations()  # T068
        return result

    # T033: Handle --variables
    if args.variables:
        return _run_variable_tracking(pipeline, use_color)

    # T052: Handle --types
    if args.types:
        return _run_type_validation(pipeline, use_color)

    # T046: Handle --batch
    if args.batch:
        result = _run_batch_validation(pipeline, use_color)
        report_unknown_operations()  # T068
        return result

    # Default: show IR statistics
    for func_name, llil, mlil, hlil in pipeline.functions:
        print(f"Function: {func_name}")

        llil_count = sum(len(bb.instructions) for bb in llil.basic_blocks)
        mlil_count = sum(len(bb.instructions) for bb in mlil.basic_blocks)
        hlil_count = len(hlil.statements) if hasattr(hlil, 'statements') else 0

        print(f"  LLIL instructions: {llil_count}")
        print(f"  MLIL instructions: {mlil_count}")
        print(f"  HLIL statements: {hlil_count}")
        print()

    print("Summary: IR generation successful")
    print("Use --compare, --all, --variables, --types, or --batch for validation")

    return 0


def _run_json_output(pipeline: IRPipeline, args: argparse.Namespace, scp_path: str) -> int:
    """T064: Run validation and output as JSON"""
    json_reporter = JSONReporter()
    function_results = []

    for func_name, llil, mlil, hlil in pipeline.functions:
        # LLIL-MLIL comparison
        llil_mlil_comp = LLILMLILComparator(llil, mlil)
        llil_mlil_results = llil_mlil_comp.compare()

        # MLIL-HLIL comparison
        mlil_hlil_comp = MLILHLILComparator(mlil, hlil)
        mlil_hlil_results = mlil_hlil_comp.compare()

        func_report = json_reporter.function_report_to_dict(
            func_name, llil_mlil_results, mlil_hlil_results
        )
        function_results.append(func_report)

    report = json_reporter.batch_report_to_dict(scp_path, function_results)
    print(json.dumps(report, indent=2))

    return 0


def _run_llil_mlil_comparison(pipeline: IRPipeline, use_color: bool) -> int:
    """Run LLIL-MLIL comparison for all functions"""
    total_passed = 0
    total_failed = 0

    for func_name, llil, mlil, hlil in pipeline.functions:
        comparator = LLILMLILComparator(llil, mlil)
        results = comparator.compare()

        report = format_comparison_report(func_name, results, use_color)
        print(report)
        print()

        different_count = sum(1 for r in results if r.status == ComparisonStatus.DIFFERENT)
        if different_count == 0:
            total_passed += 1

        else:
            total_failed += 1

    # Print overall summary
    print("=" * 50)
    print(f"Overall: {total_passed} passed, {total_failed} failed")

    return 0 if total_failed == 0 else 1


def _run_variable_tracking(pipeline: IRPipeline, use_color: bool) -> int:
    """T033: Run variable tracking for all functions"""
    for func_name, llil, mlil, hlil in pipeline.functions:
        tracker = VariableTracker(llil, mlil, hlil)
        mappings = tracker.build_mappings()

        report = format_variable_report(func_name, mappings, use_color)
        print(report)
        print()

    return 0


def _run_mlil_hlil_comparison(pipeline: IRPipeline, use_color: bool) -> int:
    """T041: Run MLIL-HLIL comparison for all functions"""
    total_passed = 0
    total_failed = 0

    for func_name, llil, mlil, hlil in pipeline.functions:
        comparator = MLILHLILComparator(mlil, hlil)
        results = comparator.compare()

        report = format_mlil_hlil_report(func_name, results, use_color)
        print(report)
        print()

        different_count = sum(1 for r in results if r.status == ComparisonStatus.DIFFERENT)
        if different_count == 0:
            total_passed += 1

        else:
            total_failed += 1

    print("=" * 50)
    print(f"Overall: {total_passed} passed, {total_failed} failed")

    return 0 if total_failed == 0 else 1


# =============================================================================
# T042-T046: Phase 6 - Batch Validation
# =============================================================================

@dataclass
class FunctionResult:
    """Result of validating a single function"""
    name: str
    llil_mlil_pass: bool
    mlil_hlil_pass: bool
    llil_mlil_diff: int
    mlil_hlil_diff: int


def _run_batch_validation(pipeline: IRPipeline, use_color: bool) -> int:
    """T042-T046: Run batch validation for all functions"""
    reporter = TextReporter(use_color)
    results: List[FunctionResult] = []

    for func_name, llil, mlil, hlil in pipeline.functions:
        # LLIL-MLIL comparison
        llil_mlil_comp = LLILMLILComparator(llil, mlil)
        llil_mlil_results = llil_mlil_comp.compare()
        llil_mlil_diff = sum(1 for r in llil_mlil_results if r.status == ComparisonStatus.DIFFERENT)

        # MLIL-HLIL comparison
        mlil_hlil_comp = MLILHLILComparator(mlil, hlil)
        mlil_hlil_results = mlil_hlil_comp.compare()
        mlil_hlil_diff = sum(1 for r in mlil_hlil_results if r.status == ComparisonStatus.DIFFERENT)

        results.append(FunctionResult(
            name=func_name,
            llil_mlil_pass=(llil_mlil_diff == 0),
            mlil_hlil_pass=(mlil_hlil_diff == 0),
            llil_mlil_diff=llil_mlil_diff,
            mlil_hlil_diff=mlil_hlil_diff
        ))

    # T044-T045: Generate summary using TextReporter
    llil_mlil_passed = sum(1 for r in results if r.llil_mlil_pass)
    mlil_hlil_passed = sum(1 for r in results if r.mlil_hlil_pass)
    total = len(results)

    failed_llil_mlil = [r for r in results if not r.llil_mlil_pass]
    failed_mlil_hlil = [r for r in results if not r.mlil_hlil_pass]

    lines = reporter.format_batch_summary(
        total, llil_mlil_passed, mlil_hlil_passed,
        failed_llil_mlil, failed_mlil_hlil
    )
    for line in lines:
        print(line)

    # Overall status
    all_passed = (llil_mlil_passed == total and mlil_hlil_passed == total)
    status_str = reporter.green("Batch validation complete") if all_passed else reporter.yellow("Batch validation complete")
    print(status_str)

    return 0


# =============================================================================
# T047-T052: Phase 7 - Type Consistency
# =============================================================================

def _run_type_validation(pipeline: IRPipeline, use_color: bool) -> int:
    """T047-T052: Run type consistency validation"""
    for func_name, llil, mlil, hlil in pipeline.functions:
        print(f"=== Type Validation: {func_name} ===")
        print()

        # Collect type info from MLIL
        type_info: Dict[str, str] = {}
        for bb in mlil.basic_blocks:
            for instr in bb.instructions:
                if instr.operation == MediumLevelILOperation.MLIL_SET_VAR:
                    if hasattr(instr, 'dest') and instr.dest is not None:
                        var = instr.dest
                        var_name = var.name if hasattr(var, 'name') else str(var)
                        var_type = var.type_name if hasattr(var, 'type_name') else "unknown"
                        type_info[var_name] = var_type

        if type_info:
            print("Variable Types (MLIL):")
            for var_name, var_type in sorted(type_info.items()):
                print(f"  {var_name}: {var_type}")
            print()
            print(f"Total: {len(type_info)} typed variables")

        else:
            print("  No type information available")

        print()

    return 0


# =============================================================================
# T057-T060: Phase 9 - Source Location Mapping
# =============================================================================

class SourceLocationMapper:
    """T057: Track SCP bytecode offsets through IR pipeline

    NOTE: Currently LLIL instructions don't have SCP offsets populated by the
    lifter. This would require modifying FalcomVMBuilder to track source
    addresses. For now, we track LLIL instruction indices which can help
    locate issues in the LLIL output.
    """

    def __init__(self, llil_func: LowLevelILFunction):
        self.llil_func = llil_func
        # Maps LLIL index -> SCP bytecode offset
        self.llil_to_scp: Dict[int, int] = {}
        # Maps MLIL index -> SCP offset (via LLIL)
        self.mlil_to_scp: Dict[int, int] = {}

    def build_mappings(self) -> None:
        """T058: Extract SCP offsets from LLIL instructions"""
        for bb in self.llil_func.basic_blocks:
            for instr in bb.instructions:
                if hasattr(instr, 'inst_index') and hasattr(instr, 'address'):
                    self.llil_to_scp[instr.inst_index] = instr.address

    def get_scp_offset(self, llil_index: int) -> int:
        """Get SCP bytecode offset for LLIL instruction"""
        return self.llil_to_scp.get(llil_index, 0)

    def propagate_to_mlil(self, mlil_func: MediumLevelILFunction) -> None:
        """Propagate SCP offsets to MLIL via source_llil_index"""
        for bb in mlil_func.basic_blocks:
            for instr in bb.instructions:
                if hasattr(instr, 'instr_index') and hasattr(instr, 'source_llil_index'):
                    llil_idx = instr.source_llil_index
                    if llil_idx in self.llil_to_scp:
                        self.mlil_to_scp[instr.instr_index] = self.llil_to_scp[llil_idx]


def enrich_comparison_with_scp_offset(
    result: ComparisonResult, mapper: SourceLocationMapper
) -> ComparisonResult:
    """T059: Ensure comparison result includes SCP offset"""
    loc = result.source_location
    if loc.scp_offset == 0 and loc.llil_index >= 0:
        loc.scp_offset = mapper.get_scp_offset(loc.llil_index)
    return result


def format_source_location_detail(loc: SourceLocation) -> str:
    """T060: Format detailed source location with sub-expression info"""
    parts = []
    if loc.scp_offset:
        parts.append(f"SCP:0x{loc.scp_offset:04X}")
    if loc.llil_index >= 0:
        parts.append(f"LLIL:{loc.llil_index}")
    if loc.mlil_index >= 0:
        parts.append(f"MLIL:{loc.mlil_index}")
    if loc.hlil_index >= 0:
        parts.append(f"HLIL:{loc.hlil_index}")
    return " | ".join(parts) if parts else "unknown"


# =============================================================================
# T061-T064: Phase 10 - Output Formats
# =============================================================================

class TextReporter:
    """T061: Human-readable text output reporter"""

    def __init__(self, use_color: bool = True):
        self.use_color = use_color

    def _color(self, text: str, color_code: str) -> str:
        """Apply color if enabled"""
        if self.use_color:
            return f"\033[{color_code}m{text}\033[0m"
        return text

    def green(self, text: str) -> str:
        return self._color(text, "32")

    def yellow(self, text: str) -> str:
        return self._color(text, "33")

    def red(self, text: str) -> str:
        return self._color(text, "31")

    def status_symbol(self, status: ComparisonStatus) -> str:
        """T062: Status symbols with color (using ASCII for compatibility)"""
        if status == ComparisonStatus.EQUIVALENT:
            return self.green("[v]")

        elif status == ComparisonStatus.TRANSFORMED:
            return self.yellow("[~]")

        else:
            return self.red("[x]")

    def format_result(self, result: ComparisonResult) -> str:
        """Format single comparison result"""
        symbol = self.status_symbol(result.status)
        loc_str = format_source_location_detail(result.source_location)
        return f"  {symbol} [{loc_str}] {result.explanation}"

    def format_function_report(
        self, func_name: str, results: List[ComparisonResult], layer_pair: str
    ) -> List[str]:
        """Format complete function report"""
        lines = []
        lines.append(f"=== {layer_pair} Comparison: {func_name} ===")
        lines.append("")

        equivalent = sum(1 for r in results if r.status == ComparisonStatus.EQUIVALENT)
        transformed = sum(1 for r in results if r.status == ComparisonStatus.TRANSFORMED)
        different = sum(1 for r in results if r.status == ComparisonStatus.DIFFERENT)

        lines.append(f"Summary: {len(results)} operations compared")
        lines.append(f"  {self.green('[v]')} Equivalent: {equivalent}")
        lines.append(f"  {self.yellow('[~]')} Transformed: {transformed}")
        lines.append(f"  {self.red('[x]')} Different: {different}")
        lines.append("")

        if different > 0:
            lines.append("Differences:")
            for r in results:
                if r.status == ComparisonStatus.DIFFERENT:
                    lines.append(self.format_result(r))
            lines.append("")

        status = "PASS" if different == 0 else "FAIL"
        status_str = self.green(status) if different == 0 else self.red(status)
        lines.append(f"Result: {status_str}")

        return lines

    def format_batch_summary(
        self, total: int, llil_mlil_passed: int, mlil_hlil_passed: int,
        failed_llil_mlil: List[Any], failed_mlil_hlil: List[Any]
    ) -> List[str]:
        """Format batch validation summary"""
        lines = []
        lines.append("=== Batch Validation Summary ===")
        lines.append(f"Total functions: {total}")
        lines.append("")

        llil_pct = 100 * llil_mlil_passed // total if total > 0 else 0
        mlil_pct = 100 * mlil_hlil_passed // total if total > 0 else 0

        llil_status = self.green(f"{llil_mlil_passed}/{total}") if llil_mlil_passed == total else self.yellow(f"{llil_mlil_passed}/{total}")
        mlil_status = self.green(f"{mlil_hlil_passed}/{total}") if mlil_hlil_passed == total else self.yellow(f"{mlil_hlil_passed}/{total}")

        lines.append(f"LLIL-MLIL: {llil_status} passed ({llil_pct}%)")
        lines.append(f"MLIL-HLIL: {mlil_status} passed ({mlil_pct}%)")
        lines.append("")

        if failed_llil_mlil:
            lines.append(f"LLIL-MLIL failures ({len(failed_llil_mlil)}):")
            for r in failed_llil_mlil[:10]:
                lines.append(f"  {self.red('[x]')} {r.name}: {r.llil_mlil_diff} differences")
            if len(failed_llil_mlil) > 10:
                lines.append(f"  ... and {len(failed_llil_mlil) - 10} more")
            lines.append("")

        if failed_mlil_hlil:
            lines.append(f"MLIL-HLIL failures ({len(failed_mlil_hlil)}):")
            for r in failed_mlil_hlil[:10]:
                lines.append(f"  {self.red('[x]')} {r.name}: {r.mlil_hlil_diff} differences")
            if len(failed_mlil_hlil) > 10:
                lines.append(f"  ... and {len(failed_mlil_hlil) - 10} more")
            lines.append("")

        return lines


class JSONReporter:
    """T063: Machine-readable JSON output reporter"""

    def comparison_result_to_dict(self, result: ComparisonResult) -> Dict[str, Any]:
        """Convert ComparisonResult to dict"""
        return {
            "status": result.status.name,
            "source_layer": result.source_layer.name,
            "target_layer": result.target_layer.name,
            "source_op": str(result.source_op) if result.source_op else None,
            "target_op": str(result.target_op) if result.target_op else None,
            "explanation": result.explanation,
            "location": {
                "scp_offset": result.source_location.scp_offset,
                "llil_index": result.source_location.llil_index,
                "mlil_index": result.source_location.mlil_index,
                "hlil_index": result.source_location.hlil_index,
            }
        }

    def function_report_to_dict(
        self, func_name: str, llil_mlil_results: List[ComparisonResult],
        mlil_hlil_results: List[ComparisonResult]
    ) -> Dict[str, Any]:
        """Convert function validation results to dict"""
        llil_mlil_diff = sum(1 for r in llil_mlil_results if r.status == ComparisonStatus.DIFFERENT)
        mlil_hlil_diff = sum(1 for r in mlil_hlil_results if r.status == ComparisonStatus.DIFFERENT)

        return {
            "function": func_name,
            "llil_mlil": {
                "passed": llil_mlil_diff == 0,
                "total": len(llil_mlil_results),
                "equivalent": sum(1 for r in llil_mlil_results if r.status == ComparisonStatus.EQUIVALENT),
                "transformed": sum(1 for r in llil_mlil_results if r.status == ComparisonStatus.TRANSFORMED),
                "different": llil_mlil_diff,
                "results": [self.comparison_result_to_dict(r) for r in llil_mlil_results]
            },
            "mlil_hlil": {
                "passed": mlil_hlil_diff == 0,
                "total": len(mlil_hlil_results),
                "equivalent": sum(1 for r in mlil_hlil_results if r.status == ComparisonStatus.EQUIVALENT),
                "transformed": sum(1 for r in mlil_hlil_results if r.status == ComparisonStatus.TRANSFORMED),
                "different": mlil_hlil_diff,
                "results": [self.comparison_result_to_dict(r) for r in mlil_hlil_results]
            }
        }

    def batch_report_to_dict(
        self, file_path: str, function_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert batch report to dict"""
        total = len(function_results)
        llil_mlil_passed = sum(1 for r in function_results if r["llil_mlil"]["passed"])
        mlil_hlil_passed = sum(1 for r in function_results if r["mlil_hlil"]["passed"])

        return {
            "file": file_path,
            "total_functions": total,
            "summary": {
                "llil_mlil": {
                    "passed": llil_mlil_passed,
                    "failed": total - llil_mlil_passed,
                    "pass_rate": llil_mlil_passed / total if total > 0 else 0
                },
                "mlil_hlil": {
                    "passed": mlil_hlil_passed,
                    "failed": total - mlil_hlil_passed,
                    "pass_rate": mlil_hlil_passed / total if total > 0 else 0
                }
            },
            "functions": function_results
        }


# =============================================================================
# T053-T056: Phase 8 - Three-Way Comparison
# =============================================================================

def _run_three_way_comparison(pipeline: IRPipeline, use_color: bool) -> int:
    """T053-T056: Run three-way comparison (LLIL-MLIL-HLIL)"""
    total_passed = 0
    total_failed = 0

    for func_name, llil, mlil, hlil in pipeline.functions:
        print(f"=== Three-Way Comparison: {func_name} ===")
        print()

        # LLIL-MLIL
        llil_mlil_comp = LLILMLILComparator(llil, mlil)
        llil_mlil_results = llil_mlil_comp.compare()
        llil_mlil_diff = sum(1 for r in llil_mlil_results if r.status == ComparisonStatus.DIFFERENT)
        llil_mlil_status = "PASS" if llil_mlil_diff == 0 else "FAIL"

        # MLIL-HLIL
        mlil_hlil_comp = MLILHLILComparator(mlil, hlil)
        mlil_hlil_results = mlil_hlil_comp.compare()
        mlil_hlil_diff = sum(1 for r in mlil_hlil_results if r.status == ComparisonStatus.DIFFERENT)
        mlil_hlil_status = "PASS" if mlil_hlil_diff == 0 else "FAIL"

        # Summary
        print(f"  LLIL -> MLIL: {llil_mlil_status} ({len(llil_mlil_results)} ops, {llil_mlil_diff} diff)")
        print(f"  MLIL -> HLIL: {mlil_hlil_status} ({len(mlil_hlil_results)} ops, {mlil_hlil_diff} diff)")

        overall = "PASS" if (llil_mlil_diff == 0 and mlil_hlil_diff == 0) else "FAIL"
        if use_color:
            color = "\033[32m" if overall == "PASS" else "\033[31m"
            print(f"  {color}Overall: {overall}\033[0m")

        else:
            print(f"  Overall: {overall}")

        print()

        if overall == "PASS":
            total_passed += 1

        else:
            total_failed += 1

    print("=" * 50)
    print(f"Overall: {total_passed} passed, {total_failed} failed")

    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
