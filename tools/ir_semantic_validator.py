"""
IR Semantic Consistency Validator

Validates semantic consistency across LLIL, MLIL, and HLIL layers by comparing
actual decompiled output from SCP files using CFG-based block matching.

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
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

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
# Section 1: Enums
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
    """Result of comparing two operations across IR layers"""
    EQUIVALENT = auto()     # Semantically identical
    TRANSFORMED = auto()    # Expected transformation (e.g., STACK_LOAD -> VAR)
    DIFFERENT = auto()      # Semantic difference (potential bug)
    ELIMINATED = auto()     # Source operation optimized away (no target match)
    INLINED = auto()        # Multiple source ops merged into one target op


class IRLayer(Enum):
    """IR layer identifier"""
    LLIL = auto()
    MLIL = auto()
    HLIL = auto()


# =============================================================================
# Section 2: Data Structures
# =============================================================================

# Status display symbols
STATUS_SYMBOLS = {
    ComparisonStatus.EQUIVALENT: ("=", "OK"),
    ComparisonStatus.TRANSFORMED: ("~", "TRANS"),
    ComparisonStatus.DIFFERENT: ("!", "DIFF"),
    ComparisonStatus.ELIMINATED: ("o", "ELIM"),
    ComparisonStatus.INLINED: (">", "INLINE"),
}


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
        symbol, _ = STATUS_SYMBOLS.get(self.status, ("?", "???"))
        return f"[{symbol}] {self.source_location}: {self.explanation}"


@dataclass
class MatchStatistics:
    """Match statistics for a comparison report"""
    total_source_blocks: int = 0
    matched_blocks: int = 0
    total_source_ops: int = 0
    aligned_ops: int = 0
    equivalent: int = 0
    transformed: int = 0
    different: int = 0
    eliminated: int = 0
    inlined: int = 0

    @property
    def block_coverage(self) -> float:
        if self.total_source_blocks == 0:
            return 100.0
        return 100.0 * self.matched_blocks / self.total_source_blocks

    @property
    def op_coverage(self) -> float:
        if self.total_source_ops == 0:
            return 100.0
        return 100.0 * self.aligned_ops / self.total_source_ops

    @property
    def total_classified(self) -> int:
        return self.equivalent + self.transformed + self.different + self.eliminated + self.inlined

    def status_percentages(self) -> Dict[str, float]:
        total = self.total_classified
        if total == 0:
            return {s.name: 0.0 for s in ComparisonStatus}
        return {
            ComparisonStatus.EQUIVALENT.name: 100.0 * self.equivalent / total,
            ComparisonStatus.TRANSFORMED.name: 100.0 * self.transformed / total,
            ComparisonStatus.DIFFERENT.name: 100.0 * self.different / total,
            ComparisonStatus.ELIMINATED.name: 100.0 * self.eliminated / total,
            ComparisonStatus.INLINED.name: 100.0 * self.inlined / total,
        }

    def accumulate(self, other: 'MatchStatistics') -> None:
        """Add another MatchStatistics into this one"""
        self.total_source_blocks += other.total_source_blocks
        self.matched_blocks += other.matched_blocks
        self.total_source_ops += other.total_source_ops
        self.aligned_ops += other.aligned_ops
        self.equivalent += other.equivalent
        self.transformed += other.transformed
        self.different += other.different
        self.eliminated += other.eliminated
        self.inlined += other.inlined


@dataclass
class BlockComparisonResult:
    """Result for a single matched block pair"""
    source_block_id: int
    target_block_id: int
    similarity: float
    source_offset: int = 0
    target_offset: int = 0
    results: List[ComparisonResult] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Full comparison report for a layer pair"""
    source_layer: IRLayer
    target_layer: IRLayer
    function_name: str = ""
    block_results: List[BlockComparisonResult] = field(default_factory=list)
    unmatched_source_blocks: List[int] = field(default_factory=list)
    unmatched_target_blocks: List[int] = field(default_factory=list)
    side_effect_violations: List[ComparisonResult] = field(default_factory=list)
    statistics: MatchStatistics = field(default_factory=MatchStatistics)
    passed: bool = True


@dataclass
class FunctionReport:
    """Report for all layer comparisons of a single function"""
    function_name: str
    llil_mlil: Optional[ComparisonReport] = None
    mlil_hlil: Optional[ComparisonReport] = None


@dataclass
class BatchReport:
    """Report for batch validation of all functions"""
    scp_path: str = ""
    function_reports: List[FunctionReport] = field(default_factory=list)
    total_functions: int = 0
    passed_functions: int = 0


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


# =============================================================================
# Section 3: Error Handling
# =============================================================================

class IRErrorKind(Enum):
    """Types of IR processing errors"""
    SCP_PARSE = auto()
    LLIL_BUILD = auto()
    MLIL_CONVERT = auto()
    HLIL_CONVERT = auto()
    UNKNOWN_OP = auto()


@dataclass
class IRError:
    """Error information for IR processing failures"""
    kind: IRErrorKind
    function_name: str
    message: str
    location: Optional[int] = None
    exception: Optional[str] = None

    def __str__(self) -> str:
        loc = f" at 0x{self.location:08X}" if self.location else ""
        return f"[{self.kind.name}] {self.function_name}{loc}: {self.message}"


class IRErrorHandler:
    """Centralized error handling for IR processing"""

    def __init__(self):
        self.errors: List[IRError] = []
        self.warnings: List[IRError] = []
        self.skipped_functions: Dict[str, IRErrorKind] = {}

    def handle_scp_parse_error(self, message: str, location: Optional[int] = None) -> IRError:
        error = IRError(
            kind=IRErrorKind.SCP_PARSE,
            function_name="<global>",
            message=message,
            location=location
        )
        self.errors.append(error)
        return error

    def handle_llil_build_error(
        self, func_name: str, message: str, exception: Optional[Exception] = None
    ) -> IRError:
        error = IRError(
            kind=IRErrorKind.LLIL_BUILD,
            function_name=func_name,
            message=message,
            exception=str(exception) if exception else None
        )
        self.errors.append(error)
        self.skipped_functions[func_name] = IRErrorKind.LLIL_BUILD
        return error

    def handle_mlil_convert_error(
        self, func_name: str, message: str, exception: Optional[Exception] = None
    ) -> IRError:
        error = IRError(
            kind=IRErrorKind.MLIL_CONVERT,
            function_name=func_name,
            message=message,
            exception=str(exception) if exception else None
        )
        self.errors.append(error)
        self.skipped_functions[func_name] = IRErrorKind.MLIL_CONVERT
        return error

    def handle_hlil_convert_error(
        self, func_name: str, message: str, exception: Optional[Exception] = None
    ) -> IRError:
        error = IRError(
            kind=IRErrorKind.HLIL_CONVERT,
            function_name=func_name,
            message=message,
            exception=str(exception) if exception else None
        )
        self.warnings.append(error)
        return error

    def handle_unknown_operation(
        self, func_name: str, operation: str, location: Optional[int] = None
    ) -> IRError:
        error = IRError(
            kind=IRErrorKind.UNKNOWN_OP,
            function_name=func_name,
            message=f"Unknown operation: {operation}",
            location=location
        )
        self.warnings.append(error)
        return error

    def should_skip_function(self, func_name: str) -> bool:
        return func_name in self.skipped_functions

    def can_do_llil_mlil(self, func_name: str) -> bool:
        skip_reason = self.skipped_functions.get(func_name)
        if skip_reason in (IRErrorKind.LLIL_BUILD, IRErrorKind.MLIL_CONVERT):
            return False
        return True

    def can_do_mlil_hlil(self, func_name: str) -> bool:
        skip_reason = self.skipped_functions.get(func_name)
        if skip_reason in (IRErrorKind.LLIL_BUILD, IRErrorKind.MLIL_CONVERT, IRErrorKind.HLIL_CONVERT):
            return False
        return True

    def get_error_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for error in self.errors:
            key = error.kind.name
            summary[key] = summary.get(key, 0) + 1
        return summary

    def get_warning_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for warning in self.warnings:
            key = warning.kind.name
            summary[key] = summary.get(key, 0) + 1
        return summary

    def print_summary(self, use_color: bool = False) -> None:
        if not self.errors and not self.warnings:
            return

        red = "\033[31m" if use_color else ""
        yellow = "\033[33m" if use_color else ""
        reset = "\033[0m" if use_color else ""

        if self.errors:
            print(f"\n{red}=== Errors ==={reset}", file=sys.stderr)
            for error in self.errors:
                print(f"  {error}", file=sys.stderr)

        if self.warnings:
            print(f"\n{yellow}=== Warnings ==={reset}", file=sys.stderr)
            for warning in self.warnings[:10]:
                print(f"  {warning}", file=sys.stderr)
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more", file=sys.stderr)

        err_summary = self.get_error_summary()
        warn_summary = self.get_warning_summary()

        if err_summary:
            print(f"\nError counts: {err_summary}", file=sys.stderr)
        if warn_summary:
            print(f"Warning counts: {warn_summary}", file=sys.stderr)


# Global error handler
_error_handler: Optional[IRErrorHandler] = None


def get_error_handler() -> IRErrorHandler:
    global _error_handler
    if _error_handler is None:
        _error_handler = IRErrorHandler()
    return _error_handler


def reset_error_handler() -> None:
    global _error_handler
    _error_handler = IRErrorHandler()


# =============================================================================
# Section 4: IR Pipeline
# =============================================================================

class IRPipeline:
    """Generates all IR layers from an SCP file"""

    def __init__(self, scp_path: str):
        self.scp_path = scp_path
        self.scp = None
        self.fs = None
        self.functions: List[Tuple[str, LowLevelILFunction, MediumLevelILFunction, HighLevelILFunction]] = []
        self.scp_instructions: Dict[str, Dict[int, str]] = {}
        self.error_handler: IRErrorHandler = get_error_handler()

    def load(self) -> bool:
        try:
            scp_path = Path(self.scp_path)
            self.fs = fileio.FileStream(str(scp_path), encoding=default_encoding())
            parser = ScpParser(self.fs, scp_path.name)
            parser.parse()
            self.scp = parser
            return True

        except Exception as e:
            self.error_handler.handle_scp_parse_error(str(e))
            print(f"Error parsing SCP file: {e}", file=sys.stderr)
            return False

    def close(self) -> None:
        if self.fs:
            self.fs.Close()
            self.fs = None

    def generate_ir(self, function_name: Optional[str] = None) -> bool:
        if self.scp is None:
            return False

        try:
            self.scp.disasm_all_functions()
            lifter = ED9VMLifter(parser=self.scp)

            for func in self.scp.functions:
                if function_name and func.name != function_name:
                    continue

                llil_func = None
                mlil_func = None
                hlil_func = None

                try:
                    llil_func = lifter.lift_function(func)

                except Exception as e:
                    self.error_handler.handle_llil_build_error(
                        func.name, "LLIL build failed", e
                    )
                    continue

                try:
                    mlil_func = convert_falcom_llil_to_mlil(llil_func)

                except Exception as e:
                    self.error_handler.handle_mlil_convert_error(
                        func.name, "MLIL conversion failed", e
                    )
                    continue

                try:
                    hlil_func = convert_falcom_mlil_to_hlil(mlil_func)

                except Exception as e:
                    self.error_handler.handle_hlil_convert_error(
                        func.name, "HLIL conversion failed", e
                    )
                    hlil_func = HighLevelILFunction(func.name)

                self.functions.append((func.name, llil_func, mlil_func, hlil_func))
                self._collect_scp_instructions(func)

        finally:
            self.close()

        return len(self.functions) > 0

    def can_compare_llil_mlil(self, func_name: str) -> bool:
        return self.error_handler.can_do_llil_mlil(func_name)

    def can_compare_mlil_hlil(self, func_name: str) -> bool:
        return self.error_handler.can_do_mlil_hlil(func_name)

    def _collect_scp_instructions(self, func) -> None:
        instructions: Dict[int, str] = {}

        if func.entry_block is None:
            return

        visited = set()
        stack = [func.entry_block]

        while stack:
            block = stack.pop()
            if block.offset in visited:
                continue
            visited.add(block.offset)

            for inst in block.instructions:
                instructions[inst.offset] = inst.mnemonic

            stack.extend(block.succs)

        self.scp_instructions[func.name] = instructions

    def get_scp_opcode(self, func_name: str, offset: int) -> str:
        if func_name not in self.scp_instructions:
            return ""
        return self.scp_instructions[func_name].get(offset, "")


# =============================================================================
# Helper: Extract instruction name
# =============================================================================

def _get_llil_name(instr: LowLevelILInstruction) -> str:
    """Get LLIL instruction name without operands"""
    return instr.operation.name.replace('LLIL_', '')


# =============================================================================
# Section 5: Normalization
# =============================================================================

# Track unknown operations encountered during normalization
_unknown_llil_ops: set = set()
_unknown_mlil_ops: set = set()
_unknown_hlil_ops: set = set()


def report_unknown_operations() -> None:
    if _unknown_llil_ops:
        print(f"Unknown LLIL operations: {sorted(_unknown_llil_ops)}", file=sys.stderr)
    if _unknown_mlil_ops:
        print(f"Unknown MLIL operations: {sorted(_unknown_mlil_ops)}", file=sys.stderr)
    if _unknown_hlil_ops:
        print(f"Unknown HLIL operations: {sorted(_unknown_hlil_ops)}", file=sys.stderr)


def clear_unknown_operations() -> None:
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

    # Falcom-specific GLOBAL operations
    elif op.name == 'LLIL_GLOBAL_STORE':
        return SemanticOperation(
            kind=OperationKind.ASSIGN,
            operator='GLOBAL_STORE',
            operands=_extract_llil_operands(instr),
            source_location=loc
        )

    elif op.name == 'LLIL_GLOBAL_LOAD':
        return SemanticOperation(
            kind=OperationKind.LOAD,
            operator='GLOBAL_LOAD',
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

    # Unknown
    _unknown_llil_ops.add(op.name)
    return SemanticOperation(
        kind=OperationKind.UNKNOWN,
        operator=op.name,
        source_location=loc
    )


def _extract_llil_operands(instr: LowLevelILInstruction) -> List[SemanticOperand]:
    """Extract operands from LLIL instruction"""
    operands = []

    if hasattr(instr, 'value'):
        operands.append(SemanticOperand(kind='const', value=instr.value))

    if hasattr(instr, 'slot_index'):
        operands.append(SemanticOperand(kind='var', value=f"Stack[{instr.slot_index}]"))

    if hasattr(instr, 'reg_index'):
        operands.append(SemanticOperand(kind='reg', value=f"Reg[{instr.reg_index}]"))

    if hasattr(instr, 'left') and hasattr(instr, 'right'):
        if isinstance(instr.left, LowLevelILInstruction):
            operands.append(SemanticOperand(kind='expr', value=str(instr.left)))

        else:
            operands.append(SemanticOperand(kind='const', value=instr.left))
        if isinstance(instr.right, LowLevelILInstruction):
            operands.append(SemanticOperand(kind='expr', value=str(instr.right)))

        else:
            operands.append(SemanticOperand(kind='const', value=instr.right))

    if hasattr(instr, 'operand'):
        if isinstance(instr.operand, LowLevelILInstruction):
            operands.append(SemanticOperand(kind='expr', value=str(instr.operand)))

        else:
            operands.append(SemanticOperand(kind='const', value=instr.operand))

    return operands


def normalize_mlil_operation(instr: MediumLevelILInstruction) -> SemanticOperation:
    """Convert MLIL instruction to normalized SemanticOperation"""
    op = instr.operation
    scp_offset = instr.address if hasattr(instr, 'address') else 0
    loc = SourceLocation(scp_offset=scp_offset, mlil_index=instr.instr_index if hasattr(instr, 'instr_index') else -1)

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

    # Unknown
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
    scp_offset = instr.address if hasattr(instr, 'address') else 0
    loc = SourceLocation(scp_offset=scp_offset)

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
        if hasattr(instr, 'src') and hasattr(instr.src, 'operation'):
            src_op = instr.src.operation
            if src_op == HLILOperation.HLIL_CALL:
                return SemanticOperation(
                    kind=OperationKind.CALL,
                    operator='CALL',
                    operands=_extract_hlil_operands(instr.src),
                    source_location=loc
                )

            elif src_op == HLILOperation.HLIL_SYSCALL:
                return SemanticOperation(
                    kind=OperationKind.CALL,
                    operator='SYSCALL',
                    operands=_extract_hlil_operands(instr.src),
                    source_location=loc
                )

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

    # Expression statement - unwrap
    elif op == HLILOperation.HLIL_EXPR_STMT:
        if hasattr(instr, 'expr') and instr.expr is not None:
            return normalize_hlil_operation(instr.expr)

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

    # Unknown
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
# Section 6: CFG Construction
# =============================================================================

@dataclass
class CFGNode:
    """Basic block node in CFG"""
    id: int
    operations: List[SemanticOperation] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    is_entry: bool = False
    is_exit: bool = False
    dominator: Optional[int] = None
    post_dominator: Optional[int] = None
    is_loop_header: bool = False
    loop_back_edges: List[int] = field(default_factory=list)


class CFG:
    """Control Flow Graph representation"""

    def __init__(self) -> None:
        self.nodes: Dict[int, CFGNode] = {}
        self.entry: Optional[int] = None
        self.exits: List[int] = []

    def add_node(self, node: CFGNode) -> None:
        self.nodes[node.id] = node
        if node.is_entry:
            self.entry = node.id
        if node.is_exit:
            self.exits.append(node.id)

    def add_edge(self, from_id: int, to_id: int) -> None:
        if from_id in self.nodes and to_id in self.nodes:
            if to_id not in self.nodes[from_id].successors:
                self.nodes[from_id].successors.append(to_id)
            if from_id not in self.nodes[to_id].predecessors:
                self.nodes[to_id].predecessors.append(from_id)

    def get_node(self, node_id: int) -> Optional[CFGNode]:
        return self.nodes.get(node_id)

    def __len__(self) -> int:
        return len(self.nodes)


def build_cfg_from_llil(llil_func: LowLevelILFunction) -> CFG:
    """Construct CFG from LLILFunction"""
    cfg = CFG()

    for i, bb in enumerate(llil_func.basic_blocks):
        ops = []
        for instr in bb.instructions:
            # Skip internal stack operations
            if instr.operation in (
                LowLevelILOperation.LLIL_STACK_STORE,
                LowLevelILOperation.LLIL_STACK_LOAD,
                LowLevelILOperation.LLIL_SP_ADD,
            ):
                continue
            ops.append(normalize_llil_operation(instr))

        node = CFGNode(
            id=i,
            operations=ops,
            is_entry=(i == 0),
        )
        cfg.add_node(node)

    # Build edges from basic block targets
    for i, bb in enumerate(llil_func.basic_blocks):
        if hasattr(bb, 'outgoing_edges'):
            for edge in bb.outgoing_edges:
                if hasattr(edge, 'target') and hasattr(edge.target, 'index'):
                    cfg.add_edge(i, edge.target.index)

        elif hasattr(bb, 'successors'):
            for succ in bb.successors:
                if hasattr(succ, 'index'):
                    cfg.add_edge(i, succ.index)

    # Mark exit nodes
    for node_id, node in cfg.nodes.items():
        if not node.successors:
            node.is_exit = True
            if node_id not in cfg.exits:
                cfg.exits.append(node_id)

    return cfg


def build_cfg_from_mlil(mlil_func: MediumLevelILFunction) -> CFG:
    """Construct CFG from MLILFunction"""
    cfg = CFG()

    for i, bb in enumerate(mlil_func.basic_blocks):
        ops = []
        for instr in bb.instructions:
            ops.append(normalize_mlil_operation(instr))

        node = CFGNode(
            id=i,
            operations=ops,
            is_entry=(i == 0),
        )
        cfg.add_node(node)

    # Build edges
    for i, bb in enumerate(mlil_func.basic_blocks):
        if hasattr(bb, 'outgoing_edges'):
            for edge in bb.outgoing_edges:
                if hasattr(edge, 'target') and hasattr(edge.target, 'index'):
                    cfg.add_edge(i, edge.target.index)

        elif hasattr(bb, 'successors'):
            for succ in bb.successors:
                if hasattr(succ, 'index'):
                    cfg.add_edge(i, succ.index)

    # Mark exit nodes
    for node_id, node in cfg.nodes.items():
        if not node.successors:
            node.is_exit = True
            if node_id not in cfg.exits:
                cfg.exits.append(node_id)

    return cfg


def build_cfg_from_hlil(hlil_func: HighLevelILFunction) -> CFG:
    """Construct CFG from HLILFunction (flatten structured control flow)"""
    cfg = CFG()

    def extract_statements(body) -> List[Any]:
        """Extract statement list from HLIL body"""
        if body is None:
            return []
        if isinstance(body, list):
            return body
        if hasattr(body, 'statements'):
            return list(body.statements)
        return [body]

    def flatten_statements(stmts: List[HLILStatement], block_id: int) -> Tuple[int, List[int]]:
        """Flatten HLIL statements into CFG blocks, return (next_id, exit_ids)"""
        nonlocal cfg
        if not stmts:
            return block_id, []

        current_ops: List[SemanticOperation] = []
        current_id = block_id
        exits: List[int] = []

        for stmt in stmts:
            op = normalize_hlil_operation(stmt)

            # Handle control flow statements
            if hasattr(stmt, 'operation'):
                if stmt.operation == HLILOperation.HLIL_IF:
                    # Save current block
                    if current_ops:
                        node = CFGNode(id=current_id, operations=current_ops)
                        cfg.add_node(node)
                        current_id += 1
                        current_ops = []

                    # Create condition block
                    cond_node = CFGNode(id=current_id, operations=[op])
                    cfg.add_node(cond_node)
                    cond_id = current_id
                    current_id += 1

                    # Process then branch
                    then_stmts = extract_statements(stmt.body if hasattr(stmt, 'body') else None)
                    if then_stmts:
                        then_start = current_id
                        then_id, then_exits = flatten_statements(then_stmts, current_id)
                        current_id = max(then_id, current_id)
                        cfg.add_edge(cond_id, then_start)

                    else:
                        then_exits = []

                    # Process else branch
                    else_exits = []
                    if hasattr(stmt, 'else_body') and stmt.else_body:
                        else_stmts = extract_statements(stmt.else_body)
                        if else_stmts:
                            else_start = current_id
                            else_id, else_exits = flatten_statements(else_stmts, current_id)
                            current_id = max(else_id, current_id)
                            cfg.add_edge(cond_id, else_start)

                    exits.extend(then_exits)
                    exits.extend(else_exits)
                    continue

                elif stmt.operation in (HLILOperation.HLIL_WHILE, HLILOperation.HLIL_FOR):
                    # Save current block
                    if current_ops:
                        node = CFGNode(id=current_id, operations=current_ops)
                        cfg.add_node(node)
                        prev_id = current_id
                        current_id += 1
                        current_ops = []
                        cfg.add_edge(prev_id, current_id)

                    # Create loop header
                    header_node = CFGNode(id=current_id, operations=[op], is_loop_header=True)
                    cfg.add_node(header_node)
                    header_id = current_id
                    current_id += 1

                    # Process loop body
                    body_stmts = extract_statements(stmt.body if hasattr(stmt, 'body') else None)
                    if body_stmts:
                        body_start = current_id
                        body_id, body_exits = flatten_statements(body_stmts, current_id)
                        current_id = max(body_id, current_id)
                        cfg.add_edge(header_id, body_start)
                        # Back edge
                        for exit_id in body_exits:
                            cfg.add_edge(exit_id, header_id)
                            cfg.nodes[header_id].loop_back_edges.append(exit_id)

                    exits.append(header_id)  # Loop exit
                    continue

            current_ops.append(op)

        # Save remaining ops
        if current_ops:
            node = CFGNode(id=current_id, operations=current_ops)
            cfg.add_node(node)
            exits.append(current_id)
            current_id += 1

        return current_id, exits

    # Flatten all statements
    stmts = None
    if hasattr(hlil_func, 'body') and hlil_func.body:
        body = hlil_func.body
        if hasattr(body, 'statements'):
            stmts = list(body.statements)

        elif hasattr(body, 'body'):
            stmts = [body]

    elif hasattr(hlil_func, 'statements') and hlil_func.statements:
        stmts = list(hlil_func.statements)

    if stmts:
        next_id, exit_ids = flatten_statements(stmts, 0)
        if cfg.nodes:
            cfg.nodes[0].is_entry = True
            cfg.entry = 0
            for eid in exit_ids:
                if eid in cfg.nodes:
                    cfg.nodes[eid].is_exit = True
                    cfg.exits.append(eid)

    return cfg


def compute_dominators(cfg: CFG) -> None:
    """Compute dominators using iterative data flow analysis"""
    if not cfg.entry or cfg.entry not in cfg.nodes:
        return

    all_nodes = set(cfg.nodes.keys())
    dom: Dict[int, set] = {n: all_nodes.copy() for n in cfg.nodes}
    dom[cfg.entry] = {cfg.entry}

    changed = True
    while changed:
        changed = False
        for node_id in cfg.nodes:
            if node_id == cfg.entry:
                continue

            node = cfg.nodes[node_id]
            if not node.predecessors:
                continue

            new_dom = all_nodes.copy()
            for pred in node.predecessors:
                new_dom &= dom[pred]
            new_dom.add(node_id)

            if new_dom != dom[node_id]:
                dom[node_id] = new_dom
                changed = True

    # Set immediate dominator
    for node_id in cfg.nodes:
        if node_id == cfg.entry:
            cfg.nodes[node_id].dominator = None
            continue

        doms = dom[node_id] - {node_id}
        if doms:
            for d in doms:
                if all(d in dom[other] or d == other for other in doms):
                    cfg.nodes[node_id].dominator = d
                    break


def compute_post_dominators(cfg: CFG) -> None:
    """Compute post-dominators (reverse dominance)"""
    if not cfg.exits:
        return

    all_nodes = set(cfg.nodes.keys())
    pdom: Dict[int, set] = {n: all_nodes.copy() for n in cfg.nodes}

    for exit_id in cfg.exits:
        pdom[exit_id] = {exit_id}

    changed = True
    while changed:
        changed = False
        for node_id in cfg.nodes:
            if node_id in cfg.exits:
                continue

            node = cfg.nodes[node_id]
            if not node.successors:
                continue

            new_pdom = all_nodes.copy()
            for succ in node.successors:
                new_pdom &= pdom[succ]
            new_pdom.add(node_id)

            if new_pdom != pdom[node_id]:
                pdom[node_id] = new_pdom
                changed = True

    # Set immediate post-dominator
    for node_id in cfg.nodes:
        if node_id in cfg.exits:
            cfg.nodes[node_id].post_dominator = None
            continue

        pdoms = pdom[node_id] - {node_id}
        if pdoms:
            for d in pdoms:
                if all(d in pdom[other] or d == other for other in pdoms):
                    cfg.nodes[node_id].post_dominator = d
                    break


def identify_loops(cfg: CFG) -> List[Tuple[int, List[int]]]:
    """Identify natural loops using back edges"""
    loops: List[Tuple[int, List[int]]] = []

    compute_dominators(cfg)

    for node_id, node in cfg.nodes.items():
        for succ in node.successors:
            succ_node = cfg.nodes.get(succ)
            if succ_node and node.dominator == succ:
                cfg.nodes[succ].is_loop_header = True
                cfg.nodes[succ].loop_back_edges.append(node_id)

                loop_body = _collect_loop_body(cfg, succ, node_id)
                loops.append((succ, loop_body))

    return loops


def _collect_loop_body(cfg: CFG, header: int, tail: int) -> List[int]:
    """Collect all nodes in a natural loop"""
    body = {header, tail}
    worklist = [tail]

    while worklist:
        node_id = worklist.pop()
        node = cfg.nodes.get(node_id)
        if not node:
            continue

        for pred in node.predecessors:
            if pred not in body:
                body.add(pred)
                worklist.append(pred)

    return list(body)


def compute_block_signature(node: CFGNode) -> str:
    """Create semantic fingerprint of basic block"""
    parts = []
    for op in node.operations:
        parts.append(f"{op.kind.name}:{op.operator}")
    return "|".join(parts) if parts else "EMPTY"


# =============================================================================
# Section 7: Block Matching
# =============================================================================

@dataclass
class BlockSimilarity:
    """Similarity score between two blocks"""
    source_block: int
    target_block: int
    score: float
    matched_ops: List[Tuple[int, int]] = field(default_factory=list)


def compute_operation_similarity(op1: SemanticOperation, op2: SemanticOperation) -> float:
    """Compare two SemanticOperations (0.0-1.0 score)"""
    score = 0.0

    # Kind match (40%)
    if op1.kind == op2.kind:
        score += 0.4

    # Operator match (40%)
    if op1.operator == op2.operator:
        score += 0.4

    elif _is_transform_pair(op1.operator, op2.operator):
        score += 0.3

    # Operand count match (20%)
    if len(op1.operands) == len(op2.operands):
        score += 0.1
        matching_operands = sum(
            1 for a, b in zip(op1.operands, op2.operands)
            if a.kind == b.kind
        )
        if op1.operands:
            score += 0.1 * (matching_operands / len(op1.operands))

    return min(1.0, score)


def _is_transform_pair(op1: str, op2: str) -> bool:
    """Check if two operators are known transformation pairs"""
    transforms = {
        ('STACK_LOAD', 'VAR'), ('STACK_STORE', 'SET_VAR'),
        ('FRAME_LOAD', 'VAR'), ('FRAME_STORE', 'SET_VAR'),
        ('BRANCH', 'IF'), ('JMP', 'GOTO'),
        ('TEST_ZERO', 'EQ'),
    }
    return (op1, op2) in transforms or (op2, op1) in transforms


def compute_block_content_similarity(
    node1: CFGNode, node2: CFGNode
) -> Tuple[float, List[Tuple[int, int]]]:
    """Compare block content using operation sequence alignment"""
    if not node1.operations or not node2.operations:
        if not node1.operations and not node2.operations:
            return 1.0, []
        return 0.0, []

    m, n = len(node1.operations), len(node2.operations)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sim = compute_operation_similarity(
                node1.operations[i - 1], node2.operations[j - 1]
            )
            dp[i][j] = max(
                dp[i - 1][j - 1] + sim,
                dp[i - 1][j],
                dp[i][j - 1]
            )

    # Backtrack to find matches
    matches: List[Tuple[int, int]] = []
    i, j = m, n
    while i > 0 and j > 0:
        sim = compute_operation_similarity(
            node1.operations[i - 1], node2.operations[j - 1]
        )
        if dp[i][j] == dp[i - 1][j - 1] + sim and sim > 0.5:
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1

        elif dp[i][j] == dp[i - 1][j]:
            i -= 1

        else:
            j -= 1

    matches.reverse()

    max_ops = max(m, n)
    total_sim = dp[m][n]
    normalized_score = total_sim / max_ops if max_ops > 0 else 1.0

    return min(1.0, normalized_score), matches


def compute_block_structural_similarity(node1: CFGNode, node2: CFGNode) -> float:
    """Compare structural similarity using degree and edge types"""
    in_diff = abs(len(node1.predecessors) - len(node2.predecessors))
    out_diff = abs(len(node1.successors) - len(node2.successors))

    max_in = max(len(node1.predecessors), len(node2.predecessors), 1)
    max_out = max(len(node1.successors), len(node2.successors), 1)

    in_sim = 1.0 - (in_diff / max_in)
    out_sim = 1.0 - (out_diff / max_out)

    entry_match = 1.0 if node1.is_entry == node2.is_entry else 0.5
    exit_match = 1.0 if node1.is_exit == node2.is_exit else 0.5

    return 0.4 * in_sim + 0.4 * out_sim + 0.1 * entry_match + 0.1 * exit_match


def compute_combined_block_similarity(
    node1: CFGNode, node2: CFGNode
) -> BlockSimilarity:
    """Combine content (70%) + structure (30%) similarity"""
    content_sim, matches = compute_block_content_similarity(node1, node2)
    struct_sim = compute_block_structural_similarity(node1, node2)

    combined = 0.7 * content_sim + 0.3 * struct_sim

    return BlockSimilarity(
        source_block=node1.id,
        target_block=node2.id,
        score=combined,
        matched_ops=matches
    )


class SimilarityMatrix:
    """Store pairwise block similarities"""

    def __init__(self, rows: int, cols: int):
        self.matrix: List[List[float]] = [
            [0.0] * cols for _ in range(rows)
        ]
        self.rows = rows
        self.cols = cols

    def set(self, i: int, j: int, value: float) -> None:
        if 0 <= i < self.rows and 0 <= j < self.cols:
            self.matrix[i][j] = value

    def get(self, i: int, j: int) -> float:
        if 0 <= i < self.rows and 0 <= j < self.cols:
            return self.matrix[i][j]
        return 0.0

    def get_row(self, i: int) -> List[float]:
        if 0 <= i < self.rows:
            return self.matrix[i]
        return []

    def get_col(self, j: int) -> List[float]:
        if 0 <= j < self.cols:
            return [self.matrix[i][j] for i in range(self.rows)]
        return []


def build_initial_similarity_matrix(
    source_cfg: CFG, target_cfg: CFG
) -> Tuple[SimilarityMatrix, Dict[Tuple[int, int], BlockSimilarity]]:
    """Build initial similarity matrix using block content similarity"""
    source_nodes = list(source_cfg.nodes.keys())
    target_nodes = list(target_cfg.nodes.keys())

    matrix = SimilarityMatrix(len(source_nodes), len(target_nodes))
    details: Dict[Tuple[int, int], BlockSimilarity] = {}

    for i, src_id in enumerate(source_nodes):
        for j, tgt_id in enumerate(target_nodes):
            src_node = source_cfg.nodes[src_id]
            tgt_node = target_cfg.nodes[tgt_id]

            sim = compute_combined_block_similarity(src_node, tgt_node)
            matrix.set(i, j, sim.score)
            details[(src_id, tgt_id)] = sim

    return matrix, details


def propagate_similarity(
    matrix: SimilarityMatrix,
    source_cfg: CFG,
    target_cfg: CFG,
    alpha: float = 0.3,
    max_iterations: int = 10,
    epsilon: float = 0.001
) -> SimilarityMatrix:
    """Iteratively propagate neighbor similarity until convergence"""
    source_nodes = list(source_cfg.nodes.keys())
    target_nodes = list(target_cfg.nodes.keys())

    src_idx = {nid: i for i, nid in enumerate(source_nodes)}
    tgt_idx = {nid: i for i, nid in enumerate(target_nodes)}

    for iteration in range(max_iterations):
        max_change = 0.0
        new_matrix = SimilarityMatrix(matrix.rows, matrix.cols)

        for i, src_id in enumerate(source_nodes):
            src_node = source_cfg.nodes[src_id]

            for j, tgt_id in enumerate(target_nodes):
                tgt_node = target_cfg.nodes[tgt_id]

                content_sim = matrix.get(i, j)

                neighbor_sim = 0.0
                neighbor_count = 0

                for src_succ in src_node.successors:
                    if src_succ in src_idx:
                        for tgt_succ in tgt_node.successors:
                            if tgt_succ in tgt_idx:
                                neighbor_sim += matrix.get(src_idx[src_succ], tgt_idx[tgt_succ])
                                neighbor_count += 1

                for src_pred in src_node.predecessors:
                    if src_pred in src_idx:
                        for tgt_pred in tgt_node.predecessors:
                            if tgt_pred in tgt_idx:
                                neighbor_sim += matrix.get(src_idx[src_pred], tgt_idx[tgt_pred])
                                neighbor_count += 1

                if neighbor_count > 0:
                    neighbor_sim /= neighbor_count
                    new_sim = (1 - alpha) * content_sim + alpha * neighbor_sim

                else:
                    new_sim = content_sim

                new_matrix.set(i, j, new_sim)

                max_change = max(max_change, abs(new_sim - content_sim))

        matrix = new_matrix

        if max_change < epsilon:
            break

    return matrix


def hungarian_matching(matrix: SimilarityMatrix) -> List[Tuple[int, int]]:
    """Find optimal block assignment using greedy Hungarian approximation"""
    n = max(matrix.rows, matrix.cols)
    cost = [[1.0] * n for _ in range(n)]

    for i in range(matrix.rows):
        for j in range(matrix.cols):
            cost[i][j] = 1.0 - matrix.get(i, j)

    matches: List[Tuple[int, int]] = []
    used_cols: set = set()

    for i in range(matrix.rows):
        best_j = -1
        best_cost = float('inf')

        for j in range(matrix.cols):
            if j not in used_cols and cost[i][j] < best_cost:
                best_cost = cost[i][j]
                best_j = j

        if best_j >= 0 and best_cost < 0.7:  # Only match if similarity > 0.3
            matches.append((i, best_j))
            used_cols.add(best_j)

    return matches


class CFGMatcher:
    """Match basic blocks between two CFGs"""

    def __init__(self, source_cfg: CFG, target_cfg: CFG):
        self.source_cfg = source_cfg
        self.target_cfg = target_cfg
        self.matches: List[Tuple[int, int]] = []
        self.similarity_details: Dict[Tuple[int, int], BlockSimilarity] = {}

    def match(self) -> List[Tuple[int, int]]:
        """Find optimal block matching between CFGs"""
        if not self.source_cfg.nodes or not self.target_cfg.nodes:
            return []

        matrix, self.similarity_details = build_initial_similarity_matrix(
            self.source_cfg, self.target_cfg
        )

        matrix = propagate_similarity(
            matrix, self.source_cfg, self.target_cfg
        )

        idx_matches = hungarian_matching(matrix)

        source_nodes = list(self.source_cfg.nodes.keys())
        target_nodes = list(self.target_cfg.nodes.keys())

        self.matches = []
        for i, j in idx_matches:
            if i < len(source_nodes) and j < len(target_nodes):
                self.matches.append((source_nodes[i], target_nodes[j]))

        return self.matches

    def get_similarity(self, src_id: int, tgt_id: int) -> Optional[BlockSimilarity]:
        return self.similarity_details.get((src_id, tgt_id))


# =============================================================================
# Section 8: Operation Comparison
# =============================================================================

class OperationMatcher:
    """Align operations within matched blocks"""

    def __init__(self, var_mapping: Dict[str, str]):
        self.var_mapping = var_mapping

    def align(
        self, source_ops: List[SemanticOperation], target_ops: List[SemanticOperation]
    ) -> List[Tuple[Optional[SemanticOperation], Optional[SemanticOperation]]]:
        """Align operations using Needleman-Wunsch"""
        aligned: List[Tuple[Optional[SemanticOperation], Optional[SemanticOperation]]] = []

        if not source_ops and not target_ops:
            return aligned

        if not source_ops:
            return [(None, op) for op in target_ops]

        if not target_ops:
            return [(op, None) for op in source_ops]

        m, n = len(source_ops), len(target_ops)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        GAP_PENALTY = -0.5

        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + GAP_PENALTY

        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + GAP_PENALTY

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match_score = compute_operation_similarity(source_ops[i - 1], target_ops[j - 1])
                dp[i][j] = max(
                    dp[i - 1][j - 1] + match_score,
                    dp[i - 1][j] + GAP_PENALTY,
                    dp[i][j - 1] + GAP_PENALTY
                )

        # Backtrack
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                match_score = compute_operation_similarity(source_ops[i - 1], target_ops[j - 1])
                if dp[i][j] == dp[i - 1][j - 1] + match_score:
                    aligned.append((source_ops[i - 1], target_ops[j - 1]))
                    i -= 1
                    j -= 1
                    continue

            if i > 0 and dp[i][j] == dp[i - 1][j] + GAP_PENALTY:
                aligned.append((source_ops[i - 1], None))
                i -= 1

            elif j > 0:
                aligned.append((None, target_ops[j - 1]))
                j -= 1

        aligned.reverse()
        return aligned


def compare_operands(
    op1: SemanticOperation,
    op2: SemanticOperation,
    var_mapping: Dict[str, str]
) -> bool:
    """Compare operands with variable mapping resolution"""
    if len(op1.operands) != len(op2.operands):
        return False

    for a, b in zip(op1.operands, op2.operands):
        if a.kind == 'const' and b.kind == 'const':
            if a.value != b.value:
                return False

        elif a.kind == 'var' and b.kind == 'var':
            mapped = var_mapping.get(str(a.value))
            if mapped and mapped != str(b.value):
                return False

        elif a.kind != b.kind:
            if not (a.kind == 'var' and b.kind == 'var'):
                return False

    return True


# Commutative operators
COMMUTATIVE_OPS = {'ADD', 'MUL', 'AND', 'OR', 'EQ', 'NE', 'LOGICAL_AND', 'LOGICAL_OR'}


def is_commutative(operator: str) -> bool:
    return operator in COMMUTATIVE_OPS


def compare_expression_trees(
    op1: SemanticOperation,
    op2: SemanticOperation,
    var_mapping: Dict[str, str]
) -> bool:
    """Compare expression trees with recursive semantic equivalence"""
    if op1.kind != op2.kind:
        return False

    if op1.operator != op2.operator and not _is_transform_pair(op1.operator, op2.operator):
        return False

    if is_commutative(op1.operator):
        if compare_operands(op1, op2, var_mapping):
            return True

        if len(op1.operands) == 2 and len(op2.operands) == 2:
            reversed_op1 = SemanticOperation(
                kind=op1.kind,
                operator=op1.operator,
                operands=[op1.operands[1], op1.operands[0]],
                result=op1.result,
                source_location=op1.source_location
            )
            return compare_operands(reversed_op1, op2, var_mapping)

    return compare_operands(op1, op2, var_mapping)


@dataclass
class TransformationRule:
    """Describes an expected transformation between IR layers"""
    source_pattern: str
    target_pattern: str
    description: str


# LLIL to MLIL expected transformations
LLIL_MLIL_TRANSFORMATIONS: List[TransformationRule] = [
    TransformationRule('STACK_LOAD', 'VAR', 'Stack slot to variable'),
    TransformationRule('STACK_STORE', 'SET_VAR', 'Stack store to assignment'),
    TransformationRule('FRAME_LOAD', 'VAR', 'Frame slot to parameter'),
    TransformationRule('FRAME_STORE', 'SET_VAR', 'Frame store to assignment'),
    TransformationRule('REG_LOAD', 'LOAD_REG', 'Register load'),
    TransformationRule('REG_STORE', 'STORE_REG', 'Register store'),
    TransformationRule('BRANCH', 'IF', 'Branch to conditional'),
    TransformationRule('JMP', 'GOTO', 'Jump to goto'),
    TransformationRule('TEST_ZERO', 'EQ', 'Zero test to equality'),
]

# MLIL to HLIL expected transformations
MLIL_HLIL_TRANSFORMATIONS: List[TransformationRule] = [
    TransformationRule('IF', 'IF', 'Conditional preserved'),
    TransformationRule('GOTO', 'WHILE', 'Back-edge goto to while loop'),
    TransformationRule('GOTO', 'FOR', 'Structured goto to for loop'),
    TransformationRule('SET_VAR', 'ASSIGN', 'Variable set to assignment'),
    TransformationRule('STORE_REG', 'ASSIGN', 'Register store to assignment'),
    TransformationRule('VAR', 'VAR', 'Variable reference'),
    TransformationRule('RET', 'RETURN', 'Return statement'),
    TransformationRule('RETURN', 'RET', 'Return statement'),
    TransformationRule('CALL', 'CALL', 'Function call preserved'),
    TransformationRule('SYSCALL', 'SYSCALL', 'System call preserved'),
]


def match_transformation(
    source_op: str, target_op: str, transformations: List[TransformationRule]
) -> Optional[TransformationRule]:
    """Detect if difference is an expected transformation"""
    for rule in transformations:
        if rule.source_pattern == source_op and rule.target_pattern == target_op:
            return rule
    return None


def classify_difference(
    source_op: SemanticOperation,
    target_op: SemanticOperation,
    var_mapping: Dict[str, str],
    layer_transforms: List[TransformationRule]
) -> ComparisonStatus:
    """Classify difference as EQUIVALENT, TRANSFORMED, or DIFFERENT"""
    # Check exact equivalence
    if source_op.kind == target_op.kind and source_op.operator == target_op.operator:
        if compare_operands(source_op, target_op, var_mapping):
            return ComparisonStatus.EQUIVALENT

    # Check expression tree equivalence
    if compare_expression_trees(source_op, target_op, var_mapping):
        return ComparisonStatus.EQUIVALENT

    # Check known transformations
    transform = match_transformation(source_op.operator, target_op.operator, layer_transforms)
    if transform:
        return ComparisonStatus.TRANSFORMED

    # Same kind but different operator might still be transformed
    if source_op.kind == target_op.kind:
        return ComparisonStatus.TRANSFORMED

    return ComparisonStatus.DIFFERENT


# =============================================================================
# Section 9: Side-Effect Validation
# =============================================================================

SIDE_EFFECT_OPS = {OperationKind.CALL, OperationKind.ASSIGN}


def has_side_effect(op: SemanticOperation) -> bool:
    return op.kind in SIDE_EFFECT_OPS


def extract_side_effect_sequence(
    ops: List[SemanticOperation]
) -> List[SemanticOperation]:
    return [op for op in ops if has_side_effect(op)]


def validate_side_effect_ordering(
    source_ops: List[SemanticOperation],
    target_ops: List[SemanticOperation],
    var_mapping: Dict[str, str]
) -> List[Tuple[str, SemanticOperation, SemanticOperation]]:
    """Validate side-effect ordering between IR layers"""
    violations: List[Tuple[str, SemanticOperation, SemanticOperation]] = []

    source_side_effects = extract_side_effect_sequence(source_ops)
    target_side_effects = extract_side_effect_sequence(target_ops)

    if not source_side_effects or not target_side_effects:
        return violations

    source_calls = [op for op in source_side_effects if op.kind == OperationKind.CALL]
    target_calls = [op for op in target_side_effects if op.kind == OperationKind.CALL]

    call_violations = _check_call_ordering(source_calls, target_calls, var_mapping)
    violations.extend(call_violations)

    assign_violations = _check_assignment_ordering(
        source_side_effects, target_side_effects, var_mapping
    )
    violations.extend(assign_violations)

    dep_violations = _check_call_assignment_dependencies(
        source_side_effects, target_side_effects, var_mapping
    )
    violations.extend(dep_violations)

    return violations


def _check_call_ordering(
    source_calls: List[SemanticOperation],
    target_calls: List[SemanticOperation],
    var_mapping: Dict[str, str]
) -> List[Tuple[str, SemanticOperation, SemanticOperation]]:
    """Verify call operations maintain relative order"""
    violations = []

    source_call_order: Dict[str, List[int]] = {}
    for idx, call in enumerate(source_calls):
        key = _call_signature(call)
        if key not in source_call_order:
            source_call_order[key] = []
        source_call_order[key].append(idx)

    target_call_order: Dict[str, List[int]] = {}
    for idx, call in enumerate(target_calls):
        key = _call_signature(call)
        if key not in target_call_order:
            target_call_order[key] = []
        target_call_order[key].append(idx)

    for key, source_indices in source_call_order.items():
        if key in target_call_order:
            target_indices = target_call_order[key]
            if len(source_indices) > 1 and len(target_indices) > 1:
                for i in range(min(len(source_indices), len(target_indices)) - 1):
                    si1, si2 = source_indices[i], source_indices[i + 1]
                    ti1, ti2 = target_indices[i], target_indices[i + 1]
                    if si1 < si2 and ti1 > ti2:
                        violations.append((
                            f"Call reordering: {key} calls swapped",
                            source_calls[si1],
                            target_calls[ti2]
                        ))

    return violations


def _call_signature(call: SemanticOperation) -> str:
    target = call.operands[0].value if call.operands else "unknown"
    return f"{call.operator}:{target}"


def _check_assignment_ordering(
    source_ops: List[SemanticOperation],
    target_ops: List[SemanticOperation],
    var_mapping: Dict[str, str]
) -> List[Tuple[str, SemanticOperation, SemanticOperation]]:
    """Check assignments to same variable maintain order"""
    violations = []

    source_assigns = [op for op in source_ops if op.kind == OperationKind.ASSIGN]
    target_assigns = [op for op in target_ops if op.kind == OperationKind.ASSIGN]

    source_by_dest: Dict[str, List[Tuple[int, SemanticOperation]]] = {}
    for idx, op in enumerate(source_assigns):
        dest = _get_assign_dest(op, var_mapping)
        if dest:
            if dest not in source_by_dest:
                source_by_dest[dest] = []
            source_by_dest[dest].append((idx, op))

    target_by_dest: Dict[str, List[Tuple[int, SemanticOperation]]] = {}
    for idx, op in enumerate(target_assigns):
        dest = _get_assign_dest(op, var_mapping)
        if dest:
            if dest not in target_by_dest:
                target_by_dest[dest] = []
            target_by_dest[dest].append((idx, op))

    for dest, source_writes in source_by_dest.items():
        if dest in target_by_dest and len(source_writes) > 1:
            target_writes = target_by_dest[dest]
            if len(target_writes) > 1:
                source_order = [idx for idx, _ in source_writes]
                target_order = [idx for idx, _ in target_writes]

                min_len = min(len(source_order), len(target_order))
                for i in range(min_len - 1):
                    if source_order[i] < source_order[i + 1]:
                        if i + 1 < len(target_order) and target_order[i] > target_order[i + 1]:
                            violations.append((
                                f"Assignment reordering: writes to {dest}",
                                source_writes[i][1],
                                target_writes[i + 1][1]
                            ))

    return violations


def _get_assign_dest(op: SemanticOperation, var_mapping: Dict[str, str]) -> Optional[str]:
    """Get normalized destination variable name"""
    if op.result:
        dest = str(op.result.value)
        return var_mapping.get(dest, dest)
    return None


def _check_call_assignment_dependencies(
    source_ops: List[SemanticOperation],
    target_ops: List[SemanticOperation],
    var_mapping: Dict[str, str]
) -> List[Tuple[str, SemanticOperation, SemanticOperation]]:
    """Check call-assignment dependency ordering"""
    violations = []

    source_pairs = _find_assign_call_pairs(source_ops, var_mapping)
    target_pairs = _find_assign_call_pairs(target_ops, var_mapping)

    for var, (assign_idx, call_idx) in source_pairs.items():
        if var in target_pairs:
            t_assign_idx, t_call_idx = target_pairs[var]
            if assign_idx < call_idx and t_assign_idx > t_call_idx:
                violations.append((
                    f"Dependency violation: call uses {var} before assignment",
                    source_ops[assign_idx] if assign_idx < len(source_ops) else source_ops[-1],
                    target_ops[t_call_idx] if t_call_idx < len(target_ops) else target_ops[-1]
                ))

    return violations


def _find_assign_call_pairs(
    ops: List[SemanticOperation],
    var_mapping: Dict[str, str]
) -> Dict[str, Tuple[int, int]]:
    """Find variable -> (last_assign_idx, first_call_using_var_idx) pairs"""
    pairs: Dict[str, Tuple[int, int]] = {}
    assigned_vars: Dict[str, int] = {}

    for idx, op in enumerate(ops):
        if op.kind == OperationKind.ASSIGN:
            dest = _get_assign_dest(op, var_mapping)
            if dest:
                assigned_vars[dest] = idx

        elif op.kind == OperationKind.CALL:
            for operand in op.operands:
                var = str(operand.value)
                mapped = var_mapping.get(var, var)
                if mapped in assigned_vars:
                    if mapped not in pairs:
                        pairs[mapped] = (assigned_vars[mapped], idx)

    return pairs


class SideEffectValidator:
    """Validator for side-effect ordering across IR layers"""

    def __init__(self, var_mapping: Dict[str, str]):
        self.var_mapping = var_mapping
        self.violations: List[ComparisonResult] = []

    def validate(
        self,
        source_ops: List[SemanticOperation],
        target_ops: List[SemanticOperation],
        source_layer: IRLayer,
        target_layer: IRLayer
    ) -> List[ComparisonResult]:
        self.violations = []

        ordering_issues = validate_side_effect_ordering(
            source_ops, target_ops, self.var_mapping
        )

        for msg, source_op, target_op in ordering_issues:
            self.violations.append(ComparisonResult(
                status=ComparisonStatus.DIFFERENT,
                source_layer=source_layer,
                target_layer=target_layer,
                source_op=source_op,
                target_op=target_op,
                explanation=f"Side-effect ordering: {msg}",
                source_location=source_op.source_location
            ))

        return self.violations


# =============================================================================
# Section 10: Unified IRLayerComparator
# =============================================================================

class IRLayerComparator:
    """Unified CFG-based comparator for any adjacent IR layer pair.

    Replaces both LLILMLILComparator and MLILHLILComparator with a single
    parameterized class that uses CFG block matching for all comparisons.
    """

    def __init__(
        self,
        source_cfg: CFG,
        target_cfg: CFG,
        source_layer: IRLayer,
        target_layer: IRLayer,
        transform_rules: List[TransformationRule],
        var_mapping: Dict[str, str],
    ):
        self.source_cfg = source_cfg
        self.target_cfg = target_cfg
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.transform_rules = transform_rules
        self.var_mapping = var_mapping

    def compare(self) -> ComparisonReport:
        """Run full CFG-based comparison and return a ComparisonReport."""
        report = ComparisonReport(
            source_layer=self.source_layer,
            target_layer=self.target_layer,
        )

        # Match blocks
        matcher = CFGMatcher(self.source_cfg, self.target_cfg)
        block_matches = matcher.match()

        matched_source = {m[0] for m in block_matches}
        matched_target = {m[1] for m in block_matches}

        # Statistics
        stats = report.statistics
        stats.total_source_blocks = len(self.source_cfg.nodes)
        stats.matched_blocks = len(block_matches)

        # Count total source ops
        for node in self.source_cfg.nodes.values():
            stats.total_source_ops += len(node.operations)

        # Compare matched blocks
        op_matcher = OperationMatcher(self.var_mapping)

        for src_id, tgt_id in block_matches:
            src_node = self.source_cfg.get_node(src_id)
            tgt_node = self.target_cfg.get_node(tgt_id)

            if not src_node or not tgt_node:
                continue

            similarity = matcher.get_similarity(src_id, tgt_id)
            sim_score = similarity.score if similarity else 0.0

            src_offset = src_node.operations[0].source_location.scp_offset if src_node.operations else 0
            tgt_offset = tgt_node.operations[0].source_location.scp_offset if tgt_node.operations else 0

            block_result = BlockComparisonResult(
                source_block_id=src_id,
                target_block_id=tgt_id,
                similarity=sim_score,
                source_offset=src_offset,
                target_offset=tgt_offset,
            )

            aligned = op_matcher.align(src_node.operations, tgt_node.operations)

            for src_op, tgt_op in aligned:
                if src_op is None and tgt_op is not None:
                    # Target-only operation (added by transformation)
                    result = ComparisonResult(
                        status=ComparisonStatus.TRANSFORMED,
                        source_layer=self.source_layer,
                        target_layer=self.target_layer,
                        source_op=None,
                        target_op=tgt_op,
                        explanation=f"Added: {tgt_op.operator}",
                        source_location=tgt_op.source_location,
                    )
                    stats.transformed += 1

                elif src_op is not None and tgt_op is None:
                    # Source-only operation (eliminated)
                    result = ComparisonResult(
                        status=ComparisonStatus.ELIMINATED,
                        source_layer=self.source_layer,
                        target_layer=self.target_layer,
                        source_op=src_op,
                        target_op=None,
                        explanation=f"Eliminated: {src_op.operator}",
                        source_location=src_op.source_location,
                    )
                    stats.eliminated += 1

                else:
                    # Both present - classify
                    status = classify_difference(
                        src_op, tgt_op, self.var_mapping, self.transform_rules
                    )
                    if status == ComparisonStatus.EQUIVALENT:
                        explanation = f"{src_op.operator} == {tgt_op.operator}"

                    elif status == ComparisonStatus.TRANSFORMED:
                        explanation = f"{src_op.operator} -> {tgt_op.operator}"

                    else:
                        explanation = f"Mismatch: {src_op.operator} vs {tgt_op.operator}"

                    result = ComparisonResult(
                        status=status,
                        source_layer=self.source_layer,
                        target_layer=self.target_layer,
                        source_op=src_op,
                        target_op=tgt_op,
                        explanation=explanation,
                        source_location=src_op.source_location,
                    )

                    if status == ComparisonStatus.EQUIVALENT:
                        stats.equivalent += 1

                    elif status == ComparisonStatus.TRANSFORMED:
                        stats.transformed += 1

                    else:
                        stats.different += 1

                    stats.aligned_ops += 1

                block_result.results.append(result)

            report.block_results.append(block_result)

        # Handle unmatched source blocks (ELIMINATED)
        for node_id, node in self.source_cfg.nodes.items():
            if node_id not in matched_source and node.operations:
                report.unmatched_source_blocks.append(node_id)
                for op in node.operations:
                    stats.eliminated += 1

        # Handle unmatched target blocks
        for node_id, node in self.target_cfg.nodes.items():
            if node_id not in matched_target and node.operations:
                report.unmatched_target_blocks.append(node_id)

        # Side-effect validation
        all_source_ops: List[SemanticOperation] = []
        all_target_ops: List[SemanticOperation] = []
        for node in self.source_cfg.nodes.values():
            all_source_ops.extend(node.operations)
        for node in self.target_cfg.nodes.values():
            all_target_ops.extend(node.operations)

        validator = SideEffectValidator(self.var_mapping)
        report.side_effect_violations = validator.validate(
            all_source_ops, all_target_ops, self.source_layer, self.target_layer
        )

        # Set pass/fail
        report.passed = (stats.different == 0 and len(report.side_effect_violations) == 0)

        return report


def _build_llil_mlil_var_mapping(mlil_func: MediumLevelILFunction) -> Dict[str, str]:
    """Build storage -> variable name mappings from MLIL SET_VAR instructions"""
    var_mapping: Dict[str, str] = {}
    for bb in mlil_func.basic_blocks:
        for instr in bb.instructions:
            if instr.operation == MediumLevelILOperation.MLIL_SET_VAR:
                if hasattr(instr, 'dest') and instr.dest is not None:
                    var = instr.dest
                    var_name = var.name if hasattr(var, 'name') else str(var)
                    if hasattr(var, 'source_storage'):
                        var_mapping[var.source_storage] = var_name
    return var_mapping


def _build_mlil_hlil_var_mapping(mlil_func: MediumLevelILFunction) -> Dict[str, str]:
    """Build MLIL var -> HLIL var mappings"""
    var_mapping: Dict[str, str] = {}
    for bb in mlil_func.basic_blocks:
        for instr in bb.instructions:
            if instr.operation == MediumLevelILOperation.MLIL_SET_VAR:
                if hasattr(instr, 'dest') and instr.dest is not None:
                    var = instr.dest
                    var_name = var.name if hasattr(var, 'name') else str(var)
                    var_mapping[var_name] = var_name
    return var_mapping


def compare_llil_mlil(
    llil_func: LowLevelILFunction, mlil_func: MediumLevelILFunction, func_name: str = ""
) -> ComparisonReport:
    """Convenience: compare LLIL and MLIL layers using CFG matching"""
    llil_cfg = build_cfg_from_llil(llil_func)
    mlil_cfg = build_cfg_from_mlil(mlil_func)
    var_mapping = _build_llil_mlil_var_mapping(mlil_func)

    comparator = IRLayerComparator(
        source_cfg=llil_cfg,
        target_cfg=mlil_cfg,
        source_layer=IRLayer.LLIL,
        target_layer=IRLayer.MLIL,
        transform_rules=LLIL_MLIL_TRANSFORMATIONS,
        var_mapping=var_mapping,
    )
    report = comparator.compare()
    report.function_name = func_name
    return report


def compare_mlil_hlil(
    mlil_func: MediumLevelILFunction, hlil_func: HighLevelILFunction, func_name: str = ""
) -> ComparisonReport:
    """Convenience: compare MLIL and HLIL layers using CFG matching"""
    mlil_cfg = build_cfg_from_mlil(mlil_func)
    hlil_cfg = build_cfg_from_hlil(hlil_func)
    var_mapping = _build_mlil_hlil_var_mapping(mlil_func)

    comparator = IRLayerComparator(
        source_cfg=mlil_cfg,
        target_cfg=hlil_cfg,
        source_layer=IRLayer.MLIL,
        target_layer=IRLayer.HLIL,
        transform_rules=MLIL_HLIL_TRANSFORMATIONS,
        var_mapping=var_mapping,
    )
    report = comparator.compare()
    report.function_name = func_name
    return report


# =============================================================================
# Section 11: Output Formatting + Validation Runner
# =============================================================================

PROGRESS_LINE_WIDTH = 70


class Progress:
    """Progress reporter for stderr"""

    def update(self, idx: int, total: int, label: str) -> None:
        print(f"\r  [{idx}/{total}] {label:<50s}", end="", file=sys.stderr, flush=True)

    def clear(self) -> None:
        print("\r" + " " * PROGRESS_LINE_WIDTH + "\r", end="", file=sys.stderr, flush=True)


class Formatter:
    """Encapsulates all output formatting with color support"""

    def __init__(self, use_color: bool = True):
        self.green = "\033[32m" if use_color else ""
        self.yellow = "\033[33m" if use_color else ""
        self.red = "\033[31m" if use_color else ""
        self.cyan = "\033[36m" if use_color else ""
        self.dim = "\033[2m" if use_color else ""
        self.reset = "\033[0m" if use_color else ""

    def _status_color(self, status: ComparisonStatus) -> str:
        if status == ComparisonStatus.EQUIVALENT:
            return self.green

        elif status == ComparisonStatus.TRANSFORMED:
            return self.yellow

        elif status == ComparisonStatus.ELIMINATED:
            return self.dim

        else:
            return self.red

    def comparison_report(self, report: ComparisonReport) -> str:
        """Format a comparison report as block-centric text output"""
        lines: List[str] = []

        src_name = report.source_layer.name
        tgt_name = report.target_layer.name

        if report.function_name:
            lines.append(f"{self.cyan}=== {report.function_name}: {src_name} -> {tgt_name} ==={self.reset}")

        else:
            lines.append(f"{self.cyan}=== {src_name} -> {tgt_name} ==={self.reset}")

        lines.append("")

        for block in report.block_results:
            lines.append(
                f"Block {block.source_block_id} [0x{block.source_offset:X}] <-> "
                f"Block {block.target_block_id} [0x{block.target_offset:X}] "
                f"(similarity: {block.similarity:.2f})"
            )

            for result in block.results:
                symbol, label = STATUS_SYMBOLS.get(result.status, ("?", "???"))
                scp_offset = f"[0x{result.source_location.scp_offset:03X}]"
                color = self._status_color(result.status)

                src_text = str(result.source_op) if result.source_op else "(none)"
                tgt_text = str(result.target_op) if result.target_op else "(eliminated)"

                if result.status == ComparisonStatus.ELIMINATED:
                    lines.append(f"  {color}{scp_offset} {symbol} {src_text} -> {tgt_text}{self.reset}")

                else:
                    lines.append(f"  {color}{scp_offset} {symbol} {src_text} <-> {tgt_text}{self.reset}")

            lines.append("")

        if report.unmatched_source_blocks:
            for block_id in report.unmatched_source_blocks:
                lines.append(f"{self.dim}Unmatched: {src_name} Block {block_id} -> (eliminated){self.reset}")
            lines.append("")

        if report.unmatched_target_blocks:
            for block_id in report.unmatched_target_blocks:
                lines.append(f"{self.yellow}Unmatched: {tgt_name} Block {block_id} (added){self.reset}")
            lines.append("")

        if report.side_effect_violations:
            lines.append(f"{self.red}Side-Effect Violations:{self.reset}")
            for v in report.side_effect_violations:
                lines.append(f"  {self.red}! {v.explanation}{self.reset}")
            lines.append("")

        lines.append(self.match_statistics(report.statistics))

        return "\n".join(lines)

    def match_statistics(self, stats: MatchStatistics) -> str:
        """Format match statistics block"""
        lines: List[str] = []

        lines.append("Match Statistics:")
        lines.append(
            f"  Block coverage:     {stats.matched_blocks}/{stats.total_source_blocks} "
            f"matched ({stats.block_coverage:.1f}%)"
        )
        lines.append(
            f"  Operation coverage: {stats.aligned_ops}/{stats.total_source_ops} "
            f"aligned ({stats.op_coverage:.1f}%)"
        )
        lines.append("")

        pcts = stats.status_percentages()

        lines.append(f"  {self.green}= Equivalent:   {stats.equivalent:4d} ({pcts['EQUIVALENT']:5.1f}%){self.reset}")
        lines.append(f"  {self.yellow}~ Transformed:  {stats.transformed:4d} ({pcts['TRANSFORMED']:5.1f}%){self.reset}")
        lines.append(f"  {self.red}! Different:    {stats.different:4d} ({pcts['DIFFERENT']:5.1f}%){self.reset}")
        lines.append(f"  {self.dim}o Eliminated:   {stats.eliminated:4d} ({pcts['ELIMINATED']:5.1f}%){self.reset}")
        lines.append(f"  {self.cyan}> Inlined:      {stats.inlined:4d} ({pcts['INLINED']:5.1f}%){self.reset}")

        return "\n".join(lines)

    def layer_summary(self, label: str, reports: List[ComparisonReport]) -> str:
        """Format aggregate summary for a list of comparison reports (one layer pair)"""
        aggregate = MatchStatistics()
        for r in reports:
            aggregate.accumulate(r.statistics)

        passed = sum(1 for r in reports if r.passed)
        total = len(reports)

        lines: List[str] = []
        lines.append(f"{label}: {passed}/{total} passed")
        lines.append(self.match_statistics(aggregate))
        return "\n".join(lines)

    def func_report(self, fr: FunctionReport) -> str:
        """Format a single function's batch result"""
        lines: List[str] = []
        lines.append(f"  {self.cyan}{fr.function_name}{self.reset}")

        if fr.llil_mlil:
            s = fr.llil_mlil.statistics
            status = f"{self.green}PASS{self.reset}" if fr.llil_mlil.passed else f"{self.red}FAIL{self.reset}"
            lines.append(
                f"    LLIL->MLIL: {status}  "
                f"blocks={s.matched_blocks}/{s.total_source_blocks}  "
                f"eq={s.equivalent} trans={s.transformed} diff={s.different} "
                f"elim={s.eliminated}"
            )

        if fr.mlil_hlil:
            s = fr.mlil_hlil.statistics
            status = f"{self.green}PASS{self.reset}" if fr.mlil_hlil.passed else f"{self.red}FAIL{self.reset}"
            lines.append(
                f"    MLIL->HLIL: {status}  "
                f"blocks={s.matched_blocks}/{s.total_source_blocks}  "
                f"eq={s.equivalent} trans={s.transformed} diff={s.different} "
                f"elim={s.eliminated}"
            )

        return "\n".join(lines)

    @classmethod
    def report_to_json(cls, report: ComparisonReport) -> Dict[str, Any]:
        """Convert ComparisonReport to JSON-serializable dict"""
        s = report.statistics
        return {
            "source_layer": report.source_layer.name,
            "target_layer": report.target_layer.name,
            "function": report.function_name,
            "passed": report.passed,
            "statistics": {
                "block_coverage": {
                    "matched": s.matched_blocks,
                    "total": s.total_source_blocks,
                    "percentage": round(s.block_coverage, 1),
                },
                "operation_coverage": {
                    "aligned": s.aligned_ops,
                    "total": s.total_source_ops,
                    "percentage": round(s.op_coverage, 1),
                },
                "status_counts": {
                    "equivalent": s.equivalent,
                    "transformed": s.transformed,
                    "different": s.different,
                    "eliminated": s.eliminated,
                    "inlined": s.inlined,
                },
            },
            "block_results": [
                {
                    "source_block": br.source_block_id,
                    "source_offset": f"0x{br.source_offset:X}",
                    "target_block": br.target_block_id,
                    "target_offset": f"0x{br.target_offset:X}",
                    "similarity": round(br.similarity, 3),
                    "results": [
                        {
                            "status": r.status.name,
                            "source_op": str(r.source_op) if r.source_op else None,
                            "target_op": str(r.target_op) if r.target_op else None,
                            "scp_offset": r.source_location.scp_offset,
                            "explanation": r.explanation,
                        }
                        for r in br.results
                    ],
                }
                for br in report.block_results
            ],
            "unmatched_source_blocks": report.unmatched_source_blocks,
            "unmatched_target_blocks": report.unmatched_target_blocks,
            "side_effect_violations": len(report.side_effect_violations),
        }

    @classmethod
    def function_report_to_json(cls, fr: FunctionReport) -> Dict[str, Any]:
        """Convert FunctionReport to JSON-serializable dict"""
        result: Dict[str, Any] = {"function": fr.function_name}

        if fr.llil_mlil:
            result["llil_mlil"] = cls.report_to_json(fr.llil_mlil)

        if fr.mlil_hlil:
            result["mlil_hlil"] = cls.report_to_json(fr.mlil_hlil)

        return result


class Validator:
    """Runs comparisons and outputs results"""

    def __init__(
        self,
        pipeline: IRPipeline,
        formatter: Formatter,
        progress: Progress,
        stream: TextIO = sys.stdout,
        as_json: bool = False,
    ):
        self.pipeline = pipeline
        self.fmt = formatter
        self.progress = progress
        self.stream = stream
        self.as_json = as_json

    def _print(self, text: str) -> None:
        print(text, file=self.stream, flush=True)

    def compare(self, layer1: str, layer2: str) -> int:
        """Compare a single layer pair across all functions"""
        layer1 = layer1.lower()
        layer2 = layer2.lower()

        all_reports: List[ComparisonReport] = []
        total = len(self.pipeline.functions)
        label = f"{layer1.upper()}->{layer2.upper()}"

        for idx, (func_name, llil, mlil, hlil) in enumerate(self.pipeline.functions, 1):
            self.progress.update(idx, total, f"{label} {func_name}")

            if layer1 == "llil" and layer2 == "mlil":
                if not self.pipeline.can_compare_llil_mlil(func_name):
                    continue
                report = compare_llil_mlil(llil, mlil, func_name)

            elif layer1 == "mlil" and layer2 == "hlil":
                if not self.pipeline.can_compare_mlil_hlil(func_name):
                    continue
                report = compare_mlil_hlil(mlil, hlil, func_name)

            else:
                print(f"Unsupported layer pair: {layer1} -> {layer2}", file=sys.stderr)
                return 1

            all_reports.append(report)

            if not self.as_json:
                self._print(self.fmt.comparison_report(report))
                self._print("")

        self.progress.clear()

        if self.as_json:
            json_output = [Formatter.report_to_json(r) for r in all_reports]
            self._print(json.dumps(json_output, indent=2))

        else:
            self._print(f"\n{self.fmt.layer_summary(label, all_reports)}")

        has_failures = any(not r.passed for r in all_reports)
        return 1 if has_failures else 0

    def batch(self, failed_only: bool = False) -> int:
        """Batch validate all functions across both layer pairs"""
        total = len(self.pipeline.functions)

        if not self.as_json:
            self._print(f"{self.fmt.cyan}=== Batch Validation: {self.pipeline.scp_path} ==={self.fmt.reset}")
            self._print(f"Functions: {total}\n")

        all_func_reports: List[FunctionReport] = []

        for idx, (func_name, llil, mlil, hlil) in enumerate(self.pipeline.functions, 1):
            self.progress.update(idx, total, f"Comparing {func_name}")

            func_report = FunctionReport(function_name=func_name)

            if self.pipeline.can_compare_llil_mlil(func_name):
                func_report.llil_mlil = compare_llil_mlil(llil, mlil, func_name)

            if self.pipeline.can_compare_mlil_hlil(func_name):
                func_report.mlil_hlil = compare_mlil_hlil(mlil, hlil, func_name)

            llil_ok = func_report.llil_mlil.passed if func_report.llil_mlil else True
            hlil_ok = func_report.mlil_hlil.passed if func_report.mlil_hlil else True

            all_func_reports.append(func_report)

            if not self.as_json:
                show_llil = not failed_only or not llil_ok
                show_hlil = not failed_only or not hlil_ok
                if show_llil or show_hlil:
                    self._print(self.fmt.func_report(func_report))
                    if func_report.llil_mlil and show_llil:
                        self._print(self.fmt.comparison_report(func_report.llil_mlil))
                    if func_report.mlil_hlil and show_hlil:
                        self._print(self.fmt.comparison_report(func_report.mlil_hlil))

        self.progress.clear()

        llil_mlil_reports = [fr.llil_mlil for fr in all_func_reports if fr.llil_mlil]
        mlil_hlil_reports = [fr.mlil_hlil for fr in all_func_reports if fr.mlil_hlil]

        if self.as_json:
            json_output = {
                "scp_path": self.pipeline.scp_path,
                "total_functions": total,
                "functions": [Formatter.function_report_to_json(fr) for fr in all_func_reports],
            }
            self._print(json.dumps(json_output, indent=2))

        else:
            self._print(f"\n{self.fmt.layer_summary('LLIL->MLIL', llil_mlil_reports)}")
            self._print(f"\n{self.fmt.layer_summary('MLIL->HLIL', mlil_hlil_reports)}")

        all_passed = all(r.passed for r in llil_mlil_reports) and all(r.passed for r in mlil_hlil_reports)
        return 0 if all_passed else 1

    def output_dir(self, dir_path: str) -> int:
        """Write LLIL-MLIL and MLIL-HLIL results to separate files"""
        from pathlib import Path

        out_path = Path(dir_path)
        out_path.mkdir(parents=True, exist_ok=True)

        scp_name = Path(self.pipeline.scp_path).stem
        no_color_fmt = Formatter(use_color=False)

        llil_mlil_file = out_path / f"{scp_name}_llil_mlil.txt"
        with open(llil_mlil_file, 'w', encoding='utf-8') as f:
            file_validator = Validator(self.pipeline, no_color_fmt, self.progress, f, self.as_json)
            file_validator.compare("llil", "mlil")
        print(f"LLIL->MLIL: {llil_mlil_file}", file=sys.stderr)

        mlil_hlil_file = out_path / f"{scp_name}_mlil_hlil.txt"
        with open(mlil_hlil_file, 'w', encoding='utf-8') as f:
            file_validator = Validator(self.pipeline, no_color_fmt, self.progress, f, self.as_json)
            file_validator.compare("mlil", "hlil")
        print(f"MLIL->HLIL: {mlil_hlil_file}", file=sys.stderr)

        return 0


# =============================================================================
# Section 12: CLI + Main
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IR Semantic Consistency Validator (CFG-based)"
    )
    parser.add_argument("scp_file", help="Path to SCP file")
    parser.add_argument(
        "--compare", nargs=2, metavar=("LAYER1", "LAYER2"),
        help="Compare two IR layers (e.g., llil mlil)"
    )
    parser.add_argument("--all", action="store_true", help="Compare all layer pairs")
    parser.add_argument("--batch", action="store_true", help="Batch validate all functions")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    parser.add_argument("--function", metavar="NAME", help="Filter specific function")
    parser.add_argument("--failed-only", action="store_true", help="Show only failed functions in batch mode")
    parser.add_argument("--export-regression", metavar="FILE", help="Export regression baseline")
    parser.add_argument("--compare-regression", metavar="FILE", help="Compare against regression baseline")
    parser.add_argument("--output-dir", metavar="DIR", help="Write results to separate files per layer pair")

    return parser


# =============================================================================
# Section 13: Regression Testing
# =============================================================================

class RegressionTester:
    """Regression testing support for validation results"""

    def export_baseline(self, pipeline: IRPipeline, output_file: str) -> bool:
        """Export validation results as regression baseline"""
        function_results = []

        for func_name, llil, mlil, hlil in pipeline.functions:
            func_report = FunctionReport(function_name=func_name)

            if pipeline.can_compare_llil_mlil(func_name):
                func_report.llil_mlil = compare_llil_mlil(llil, mlil, func_name)

            if pipeline.can_compare_mlil_hlil(func_name):
                func_report.mlil_hlil = compare_mlil_hlil(mlil, hlil, func_name)

            function_results.append(Formatter.function_report_to_json(func_report))

        baseline = {
            "version": "2.0",
            "file": pipeline.scp_path,
            "functions": function_results,
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(baseline, f, indent=2)
            return True

        except Exception as e:
            print(f"Error exporting baseline: {e}", file=sys.stderr)
            return False

    def compare_baseline(self, pipeline: IRPipeline, baseline_file: str) -> int:
        """Compare current results against baseline and report differences"""
        try:
            with open(baseline_file, 'r', encoding='utf-8') as f:
                baseline = json.load(f)

        except Exception as e:
            print(f"Error loading baseline: {e}", file=sys.stderr)
            return 1

        current_results: Dict[str, Dict] = {}
        for func_name, llil, mlil, hlil in pipeline.functions:
            func_report = FunctionReport(function_name=func_name)

            if pipeline.can_compare_llil_mlil(func_name):
                func_report.llil_mlil = compare_llil_mlil(llil, mlil, func_name)

            if pipeline.can_compare_mlil_hlil(func_name):
                func_report.mlil_hlil = compare_mlil_hlil(mlil, hlil, func_name)

            current_results[func_name] = Formatter.function_report_to_json(func_report)

        baseline_results = {f["function"]: f for f in baseline.get("functions", [])}

        regressions = []
        improvements = []

        for func_name, current in current_results.items():
            if func_name not in baseline_results:
                continue

            base = baseline_results[func_name]

            for layer_key in ("llil_mlil", "mlil_hlil"):
                if layer_key in current and layer_key in base:
                    cur_diff = current[layer_key]["statistics"]["status_counts"]["different"]
                    base_diff = base[layer_key]["statistics"]["status_counts"]["different"]
                    layer_label = layer_key.upper().replace("_", "-")

                    if cur_diff > base_diff:
                        regressions.append({
                            "function": func_name, "layer": layer_label,
                            "baseline_diff": base_diff, "current_diff": cur_diff,
                        })

                    elif cur_diff < base_diff:
                        improvements.append({
                            "function": func_name, "layer": layer_label,
                            "baseline_diff": base_diff, "current_diff": cur_diff,
                        })

        print("=== Regression Test Results ===")
        print(f"Baseline: {baseline_file}")
        print(f"Functions compared: {len(current_results)}")
        print()

        if regressions:
            print(f"REGRESSIONS FOUND: {len(regressions)}")
            for r in regressions:
                print(f"  {r['function']} ({r['layer']}): {r['baseline_diff']} -> {r['current_diff']} differences")
            print()

        if improvements:
            print(f"Improvements: {len(improvements)}")
            for i in improvements:
                print(f"  {i['function']} ({i['layer']}): {i['baseline_diff']} -> {i['current_diff']} differences")
            print()

        if not regressions and not improvements:
            print("No changes detected from baseline")

        return 1 if regressions else 0


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    # Initialize error handler
    reset_error_handler()

    # Load SCP
    pipeline = IRPipeline(args.scp_file)
    if not pipeline.load():
        return 1

    # Generate IR
    if not pipeline.generate_ir(function_name=args.function):
        print("No functions found", file=sys.stderr)
        return 1

    report_unknown_operations()

    # Regression testing (no formatter/progress needed)
    if args.export_regression:
        tester = RegressionTester()
        return 0 if tester.export_baseline(pipeline, args.export_regression) else 1

    if args.compare_regression:
        tester = RegressionTester()
        return tester.compare_baseline(pipeline, args.compare_regression)

    # Create validator
    use_color = not args.no_color and not args.output_dir
    formatter = Formatter(use_color)
    progress = Progress()
    validator = Validator(pipeline, formatter, progress, as_json=args.json)

    # Dispatch
    if args.output_dir:
        return validator.output_dir(args.output_dir)

    if args.compare:
        return validator.compare(args.compare[0], args.compare[1])

    if args.all:
        exit_code = 0
        code = validator.compare("llil", "mlil")
        if code != 0:
            exit_code = code

        code = validator.compare("mlil", "hlil")
        if code != 0:
            exit_code = code

        return exit_code

    if args.batch:
        return validator.batch(args.failed_only)

    # Default: batch
    return validator.batch()


if __name__ == "__main__":
    sys.exit(main())
