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


# =============================================================================
# T001-T003: Address Tracking Dataclasses (IDA-Style Synchronization)
# =============================================================================

@dataclass
class AddressRange:
    """T002: Address range for multi-byte instructions"""
    start: int
    end: int

    def __contains__(self, addr: int) -> bool:
        return self.start <= addr < self.end

    def __str__(self) -> str:
        if self.start == self.end:
            return f"0x{self.start:08X}"
        return f"0x{self.start:08X}-0x{self.end:08X}"


class MappingStatus(Enum):
    """Status of an address mapping entry"""
    EQUIVALENT = auto()     # All layers match semantically
    TRANSFORMED = auto()    # Expected transformation (stack->var, goto->while)
    DIFFERENT = auto()      # Semantic difference (potential bug)
    ELIMINATED = auto()     # Present in lower layer, optimized away in higher
    INLINED = auto()        # Multiple ops merged into one expression


@dataclass
class AddressMapping:
    """T003: Links SCP offset to all IR layer representations"""
    scp_offset: int
    address_range: Optional[AddressRange] = None
    scp_opcode: str = ""  # T057: Original SCP opcode mnemonic
    llil_text: str = ""
    llil_index: int = -1
    mlil_text: str = ""
    mlil_index: int = -1
    hlil_text: str = ""
    hlil_index: int = -1
    status: MappingStatus = MappingStatus.EQUIVALENT
    explanation: str = ""

    def __str__(self) -> str:
        return f"0x{self.scp_offset:08X}: {self.llil_text} | {self.mlil_text} | {self.hlil_text}"


@dataclass
class AddressTable:
    """Collection of address mappings for a function"""
    function_name: str
    mappings: List[AddressMapping] = field(default_factory=list)

    def add(self, mapping: AddressMapping) -> None:
        self.mappings.append(mapping)

    def get_by_offset(self, offset: int) -> Optional[AddressMapping]:
        for m in self.mappings:
            if m.scp_offset == offset:
                return m
        return None

    def sorted_by_offset(self) -> List[AddressMapping]:
        return sorted(self.mappings, key=lambda m: m.scp_offset)

    @property
    def equivalent_count(self) -> int:
        return sum(1 for m in self.mappings if m.status == MappingStatus.EQUIVALENT)

    @property
    def transformed_count(self) -> int:
        return sum(1 for m in self.mappings if m.status == MappingStatus.TRANSFORMED)

    @property
    def different_count(self) -> int:
        return sum(1 for m in self.mappings if m.status == MappingStatus.DIFFERENT)

    @property
    def eliminated_count(self) -> int:
        return sum(1 for m in self.mappings if m.status == MappingStatus.ELIMINATED)

    @property
    def inlined_count(self) -> int:
        return sum(1 for m in self.mappings if m.status == MappingStatus.INLINED)


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
# T005, T007, T029: Additional Key Entities from spec.md
# =============================================================================

class DifferenceSeverity(Enum):
    """Severity level for semantic differences"""
    INFO = auto()       # Expected transformation, no action needed
    WARNING = auto()    # Possible issue, may need review
    ERROR = auto()      # Definite semantic difference, likely bug


@dataclass
class DifferenceReport:
    """T005: Detailed report of a semantic difference with context"""
    scp_offset: int
    source_layer: IRLayer
    target_layer: IRLayer
    source_text: str
    target_text: str
    explanation: str
    severity: DifferenceSeverity = DifferenceSeverity.WARNING
    source_llil_index: int = -1
    source_mlil_index: int = -1
    source_hlil_index: int = -1

    def __str__(self) -> str:
        severity_char = {
            DifferenceSeverity.INFO: "i",
            DifferenceSeverity.WARNING: "!",
            DifferenceSeverity.ERROR: "X",
        }[self.severity]
        return f"[{severity_char}] 0x{self.scp_offset:08X}: {self.explanation}"


class TypeCompatibility(Enum):
    """Type compatibility status across IR layers"""
    COMPATIBLE = auto()     # Types are compatible (e.g., int == int)
    WIDENING = auto()       # Safe widening (e.g., int -> long)
    NARROWING = auto()      # Potential loss (e.g., long -> int)
    MISMATCH = auto()       # Incompatible types
    UNKNOWN = auto()        # Type not available in layer


@dataclass
class TypeInfo:
    """T007: Type information across IR layers for an expression/variable"""
    scp_offset: int
    expression_text: str = ""
    llil_type: str = "(none)"
    mlil_type: str = "(none)"
    hlil_type: str = "(none)"
    compatibility: TypeCompatibility = TypeCompatibility.UNKNOWN

    def __str__(self) -> str:
        status_char = {
            TypeCompatibility.COMPATIBLE: "=",
            TypeCompatibility.WIDENING: "~",
            TypeCompatibility.NARROWING: "!",
            TypeCompatibility.MISMATCH: "X",
            TypeCompatibility.UNKNOWN: "?",
        }[self.compatibility]
        return f"0x{self.scp_offset:08X}: {self.llil_type} -> {self.mlil_type} -> {self.hlil_type} [{status_char}]"


@dataclass
class VariableLifetime:
    """T029: Tracks variable definition and use sites by SCP offset"""
    variable_name: str
    def_offset: int                         # SCP offset of first assignment
    use_offsets: List[int] = field(default_factory=list)  # SCP offsets of reads
    llil_storage: str = ""                  # e.g., "STACK[sp+0]", "REG[reg0]"
    mlil_var: str = ""                      # e.g., "var_0", "arg0"
    hlil_var: str = ""                      # e.g., "index", "result"
    inferred_type: str = ""                 # e.g., "int", "str"

    def __str__(self) -> str:
        uses = ", ".join(f"0x{o:08X}" for o in self.use_offsets[:3])
        if len(self.use_offsets) > 3:
            uses += f"... (+{len(self.use_offsets) - 3})"
        return f"{self.variable_name}: def@0x{self.def_offset:08X}, uses=[{uses}]"


# =============================================================================
# T077-T081: Phase 14 - Error Handling (from plan.md)
# =============================================================================

class IRErrorKind(Enum):
    """T077-T081: Types of IR processing errors"""
    SCP_PARSE = auto()      # T077: SCP parse failure
    LLIL_BUILD = auto()     # T078: LLIL build failure
    MLIL_CONVERT = auto()   # T079: MLIL conversion failure
    HLIL_CONVERT = auto()   # T080: HLIL conversion failure
    UNKNOWN_OP = auto()     # T081: Unknown operation type


@dataclass
class IRError:
    """T077-T081: Error information for IR processing failures"""
    kind: IRErrorKind
    function_name: str
    message: str
    location: Optional[int] = None  # SCP offset if available
    exception: Optional[str] = None

    def __str__(self) -> str:
        loc = f" at 0x{self.location:08X}" if self.location else ""
        return f"[{self.kind.name}] {self.function_name}{loc}: {self.message}"


class IRErrorHandler:
    """T077-T081: Centralized error handling for IR processing

    From plan.md error handling strategy:
    - T077: SCP parse failure -> exit with error, show parse location
    - T078: LLIL build failure -> report function, continue with next
    - T079: MLIL conversion failure -> skip MLIL/HLIL comparison for function
    - T080: HLIL conversion failure -> perform LLIL-MLIL only
    - T081: Unknown operation type -> warn, treat as non-comparable
    """

    def __init__(self):
        self.errors: List[IRError] = []
        self.warnings: List[IRError] = []
        self.skipped_functions: Dict[str, IRErrorKind] = {}

    def handle_scp_parse_error(self, message: str, location: Optional[int] = None) -> IRError:
        """T077: Handle SCP parse failure - fatal error"""
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
        """T078: Handle LLIL build failure - skip function, continue"""
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
        """T079: Handle MLIL conversion failure - skip MLIL/HLIL comparison"""
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
        """T080: Handle HLIL conversion failure - perform LLIL-MLIL only"""
        error = IRError(
            kind=IRErrorKind.HLIL_CONVERT,
            function_name=func_name,
            message=message,
            exception=str(exception) if exception else None
        )
        self.warnings.append(error)  # Warning, not fatal - can still do LLIL-MLIL
        return error

    def handle_unknown_operation(
        self, func_name: str, operation: str, location: Optional[int] = None
    ) -> IRError:
        """T081: Handle unknown operation type - warn, non-comparable"""
        error = IRError(
            kind=IRErrorKind.UNKNOWN_OP,
            function_name=func_name,
            message=f"Unknown operation: {operation}",
            location=location
        )
        self.warnings.append(error)
        return error

    def should_skip_function(self, func_name: str) -> bool:
        """Check if function should be skipped due to previous errors"""
        return func_name in self.skipped_functions

    def can_do_llil_mlil(self, func_name: str) -> bool:
        """T079: Check if LLIL-MLIL comparison is possible"""
        skip_reason = self.skipped_functions.get(func_name)
        if skip_reason in (IRErrorKind.LLIL_BUILD, IRErrorKind.MLIL_CONVERT):
            return False
        return True

    def can_do_mlil_hlil(self, func_name: str) -> bool:
        """T080: Check if MLIL-HLIL comparison is possible"""
        skip_reason = self.skipped_functions.get(func_name)
        if skip_reason in (IRErrorKind.LLIL_BUILD, IRErrorKind.MLIL_CONVERT, IRErrorKind.HLIL_CONVERT):
            return False
        return True

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of errors by kind"""
        summary: Dict[str, int] = {}
        for error in self.errors:
            key = error.kind.name
            summary[key] = summary.get(key, 0) + 1
        return summary

    def get_warning_summary(self) -> Dict[str, int]:
        """Get summary of warnings by kind"""
        summary: Dict[str, int] = {}
        for warning in self.warnings:
            key = warning.kind.name
            summary[key] = summary.get(key, 0) + 1
        return summary

    def print_summary(self, use_color: bool = False) -> None:
        """Print error/warning summary to stderr"""
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
            for warning in self.warnings[:10]:  # Limit output
                print(f"  {warning}", file=sys.stderr)
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more", file=sys.stderr)

        # Summary
        err_summary = self.get_error_summary()
        warn_summary = self.get_warning_summary()

        if err_summary:
            print(f"\nError counts: {err_summary}", file=sys.stderr)
        if warn_summary:
            print(f"Warning counts: {warn_summary}", file=sys.stderr)


# Global error handler for the validation session
_error_handler: Optional[IRErrorHandler] = None


def get_error_handler() -> IRErrorHandler:
    """Get or create global error handler"""
    global _error_handler
    if _error_handler is None:
        _error_handler = IRErrorHandler()
    return _error_handler


def reset_error_handler() -> None:
    """Reset global error handler for new session"""
    global _error_handler
    _error_handler = IRErrorHandler()


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
        # T058: Map function_name -> {scp_offset -> instruction_mnemonic}
        self.scp_instructions: Dict[str, Dict[int, str]] = {}
        # T077-T081: Error handler for this pipeline
        self.error_handler: IRErrorHandler = get_error_handler()

    def load(self) -> bool:
        """Load and parse SCP file

        T077: Handle SCP parse failure - exit with error, show parse location
        """
        try:
            scp_path = Path(self.scp_path)
            # Keep file stream open for disassembly
            self.fs = fileio.FileStream(str(scp_path), encoding=default_encoding())
            parser = ScpParser(self.fs, scp_path.name)
            parser.parse()
            self.scp = parser
            return True

        except Exception as e:
            # T077: Report SCP parse failure with location if available
            self.error_handler.handle_scp_parse_error(str(e))
            print(f"Error parsing SCP file: {e}", file=sys.stderr)
            return False

    def close(self) -> None:
        """Close file stream"""
        if self.fs:
            self.fs.Close()
            self.fs = None

    def generate_ir(self, function_name: Optional[str] = None) -> bool:
        """Generate LLIL, MLIL, HLIL for all functions (or specific function)

        T077-T081: Enhanced error handling for all IR conversion stages:
        - T077: SCP parse failure handled in load()
        - T078: LLIL build failure - skip function, continue with next
        - T079: MLIL conversion failure - skip MLIL/HLIL comparison
        - T080: HLIL conversion failure - perform LLIL-MLIL only
        - T081: Unknown operation type handled during comparison
        """
        if self.scp is None:
            return False

        try:
            # Disassemble functions first
            self.scp.disasm_all_functions()

            # Create lifter
            lifter = ED9VMLifter(parser=self.scp)

            for func in self.scp.functions:
                if function_name and func.name != function_name:
                    continue

                llil_func = None
                mlil_func = None
                hlil_func = None

                try:
                    # T078: Generate LLIL using lifter
                    llil_func = lifter.lift_function(func)

                except Exception as e:
                    # T078: LLIL build failure - skip function, continue
                    self.error_handler.handle_llil_build_error(
                        func.name, "LLIL build failed", e
                    )
                    continue

                try:
                    # T079: Convert to MLIL
                    mlil_func = convert_falcom_llil_to_mlil(llil_func)

                except Exception as e:
                    # T079: MLIL conversion failure - skip MLIL/HLIL comparison
                    self.error_handler.handle_mlil_convert_error(
                        func.name, "MLIL conversion failed", e
                    )
                    continue

                try:
                    # T080: Convert to HLIL
                    hlil_func = convert_falcom_mlil_to_hlil(mlil_func)

                except Exception as e:
                    # T080: HLIL conversion failure - perform LLIL-MLIL only
                    self.error_handler.handle_hlil_convert_error(
                        func.name, "HLIL conversion failed", e
                    )
                    # Create empty HLIL so we can still do LLIL-MLIL comparison
                    hlil_func = HighLevelILFunction(func.name)

                self.functions.append((func.name, llil_func, mlil_func, hlil_func))

                # T058: Collect SCP instructions for this function
                self._collect_scp_instructions(func)

        finally:
            # Close file stream after processing
            self.close()

        return len(self.functions) > 0

    def can_compare_llil_mlil(self, func_name: str) -> bool:
        """T079: Check if LLIL-MLIL comparison is possible for this function"""
        return self.error_handler.can_do_llil_mlil(func_name)

    def can_compare_mlil_hlil(self, func_name: str) -> bool:
        """T080: Check if MLIL-HLIL comparison is possible for this function"""
        return self.error_handler.can_do_mlil_hlil(func_name)

    def _collect_scp_instructions(self, func) -> None:
        """T058: Collect SCP opcode names indexed by offset for a function"""
        instructions: Dict[int, str] = {}

        # Traverse all basic blocks to collect instructions
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
        """T058: Get SCP opcode mnemonic for a given function and offset"""
        if func_name not in self.scp_instructions:
            return ""
        return self.scp_instructions[func_name].get(offset, "")


# =============================================================================
# Helper: Extract instruction name (opcode only, no operands)
# =============================================================================

def _get_llil_name(instr: LowLevelILInstruction) -> str:
    """Extract LLIL instruction name without operands"""
    return instr.operation.name.replace('LLIL_', '').lower()


def _get_mlil_name(instr: MediumLevelILInstruction) -> str:
    """Extract MLIL instruction name without operands"""
    return instr.operation.name.replace('MLIL_', '').lower()


def _get_hlil_name(stmt: HLILStatement) -> str:
    """Extract HLIL statement name without operands"""
    return stmt.operation.name.replace('HLIL_', '').lower()


# =============================================================================
# T004-T011: Address Extraction and Alignment (IDA-Style Synchronization)
# =============================================================================

class AddressAligner:
    """T007: Groups instructions by SCP offset for IDA-style synchronized view"""

    def __init__(
        self,
        llil_func: LowLevelILFunction,
        mlil_func: MediumLevelILFunction,
        hlil_func: HighLevelILFunction
    ):
        self.llil_func = llil_func
        self.mlil_func = mlil_func
        self.hlil_func = hlil_func

        # Address -> instruction mappings
        self.llil_by_addr: Dict[int, List[Tuple[int, LowLevelILInstruction]]] = {}
        self.mlil_by_addr: Dict[int, List[Tuple[int, MediumLevelILInstruction]]] = {}
        self.hlil_by_addr: Dict[int, List[Tuple[int, HLILStatement]]] = {}

        # Track which MLIL/HLIL addresses come from which LLIL addresses
        self.mlil_source_addr: Dict[int, int] = {}  # mlil_index -> scp_offset
        self.hlil_source_addr: Dict[int, int] = {}  # hlil_index -> scp_offset

    def build_address_table(self) -> AddressTable:
        """T008: Create unified SCP -> (LLIL, MLIL, HLIL) mapping"""
        # Extract addresses from each layer
        self._extract_llil_addresses()
        self._extract_mlil_addresses()
        self._extract_hlil_addresses()

        # Build unified table
        table = AddressTable(function_name=self.llil_func.name)

        # Get all unique SCP offsets
        all_offsets = set(self.llil_by_addr.keys())
        all_offsets.update(self.mlil_by_addr.keys())
        all_offsets.update(self.hlil_by_addr.keys())

        for offset in sorted(all_offsets):
            mapping = self._create_mapping_for_offset(offset)
            table.add(mapping)

        return table

    def _extract_llil_addresses(self) -> None:
        """T004: Extract SCP offset for each LLIL instruction"""
        idx = 0
        for bb in self.llil_func.basic_blocks:
            for instr in bb.instructions:
                addr = instr.address
                if addr not in self.llil_by_addr:
                    self.llil_by_addr[addr] = []
                self.llil_by_addr[addr].append((idx, instr))
                idx += 1

    def _extract_mlil_addresses(self) -> None:
        """T005: Propagate SCP offset from LLIL to MLIL"""
        idx = 0
        for bb in self.mlil_func.basic_blocks:
            for instr in bb.instructions:
                # MLIL instructions may have source_addr from LLIL
                addr = self._get_mlil_source_address(instr, idx)
                if addr is not None:
                    if addr not in self.mlil_by_addr:
                        self.mlil_by_addr[addr] = []
                    self.mlil_by_addr[addr].append((idx, instr))
                    self.mlil_source_addr[idx] = addr
                idx += 1

    def _get_mlil_source_address(
        self, instr: MediumLevelILInstruction, idx: int
    ) -> Optional[int]:
        """Get source SCP address for MLIL instruction"""
        # MLIL instruction has address field (inherited from LLIL)
        if hasattr(instr, 'address') and instr.address != 0:
            return instr.address

        # Fallback: use llil_index to lookup LLIL address
        if hasattr(instr, 'llil_index') and instr.llil_index >= 0:
            # Look up LLIL address from index
            for addr, entries in self.llil_by_addr.items():
                for llil_idx, llil_instr in entries:
                    if llil_idx == instr.llil_index:
                        return addr

        return None

    def _extract_hlil_addresses(self) -> None:
        """T006: Propagate SCP offset from MLIL to HLIL"""
        # HLIL stores statements in body.statements
        statements = []
        if hasattr(self.hlil_func, 'body') and hasattr(self.hlil_func.body, 'statements'):
            statements = self.hlil_func.body.statements

        elif hasattr(self.hlil_func, 'statements'):
            statements = self.hlil_func.statements

        for idx, stmt in enumerate(statements):
            addr = self._get_hlil_source_address(stmt, idx)
            if addr is not None:
                if addr not in self.hlil_by_addr:
                    self.hlil_by_addr[addr] = []
                self.hlil_by_addr[addr].append((idx, stmt))
                self.hlil_source_addr[idx] = addr

    def _get_hlil_source_address(self, stmt: HLILStatement, idx: int) -> Optional[int]:
        """Get source SCP address for HLIL statement"""
        # Check if statement has address attribute (from MLIL)
        if hasattr(stmt, 'address') and stmt.address:
            return stmt.address

        # Check if statement has mlil_index and we can trace back to SCP address
        if hasattr(stmt, 'mlil_index') and stmt.mlil_index >= 0:
            mlil_idx = stmt.mlil_index
            if mlil_idx in self.mlil_source_addr:
                return self.mlil_source_addr[mlil_idx]

        return None

    def _create_mapping_for_offset(self, offset: int) -> AddressMapping:
        """Create an AddressMapping entry for a specific SCP offset"""
        mapping = AddressMapping(scp_offset=offset)

        # LLIL names (opcode only, no operands)
        if offset in self.llil_by_addr:
            llil_entries = self.llil_by_addr[offset]
            mapping.llil_index = llil_entries[0][0]
            llil_texts = [_get_llil_name(instr) for idx, instr in llil_entries]
            mapping.llil_text = "; ".join(llil_texts) if llil_texts else ""

        # MLIL names (opcode only, no operands)
        if offset in self.mlil_by_addr:
            mlil_entries = self.mlil_by_addr[offset]
            mapping.mlil_index = mlil_entries[0][0]
            mlil_texts = [_get_mlil_name(instr) for idx, instr in mlil_entries]
            mapping.mlil_text = "; ".join(mlil_texts) if mlil_texts else ""

        else:
            # T011: Handle eliminated entries
            if offset in self.llil_by_addr:
                mapping.mlil_text = "(eliminated)"
                mapping.status = MappingStatus.ELIMINATED

        # HLIL names (opcode only, no operands)
        if offset in self.hlil_by_addr:
            hlil_entries = self.hlil_by_addr[offset]
            mapping.hlil_index = hlil_entries[0][0]
            hlil_texts = [_get_hlil_name(stmt) for idx, stmt in hlil_entries]
            mapping.hlil_text = "; ".join(hlil_texts) if hlil_texts else ""

        else:
            # T011: Handle eliminated or inlined entries
            if offset in self.mlil_by_addr:
                mapping.hlil_text = "(eliminated)"
                if mapping.status == MappingStatus.EQUIVALENT:
                    mapping.status = MappingStatus.ELIMINATED

            elif offset in self.llil_by_addr:
                mapping.hlil_text = "(eliminated)"

        # Determine status based on content
        self._determine_mapping_status(mapping)

        return mapping

    def _determine_mapping_status(self, mapping: AddressMapping) -> None:
        """Determine the semantic status of a mapping"""
        # Already set to ELIMINATED
        if mapping.status == MappingStatus.ELIMINATED:
            return

        # Check for inlined content
        if ">" in mapping.hlil_text or "(inline" in mapping.hlil_text:
            mapping.status = MappingStatus.INLINED
            return

        # All present - check equivalence
        if mapping.llil_text and mapping.mlil_text and mapping.hlil_text:
            # Compare normalized operations
            if self._are_semantically_equivalent(mapping):
                mapping.status = MappingStatus.EQUIVALENT

            else:
                mapping.status = MappingStatus.TRANSFORMED

    def _are_semantically_equivalent(self, mapping: AddressMapping) -> bool:
        """Check if the operations at this offset are semantically equivalent"""
        # Simple heuristic: check if operators match
        llil_op = self._extract_operator(mapping.llil_text)
        mlil_op = self._extract_operator(mapping.mlil_text)
        hlil_op = self._extract_operator(mapping.hlil_text)

        # Known equivalent pairs
        equivalent_pairs = {
            ('SYSCALL', 'syscall', 'syscall'),
            ('CALL', 'call', 'call'),
            ('RET', 'return', 'return'),
        }

        ops = (llil_op.upper() if llil_op else '',
               mlil_op.lower() if mlil_op else '',
               hlil_op.lower() if hlil_op else '')

        return ops in equivalent_pairs or (llil_op == mlil_op == hlil_op)

    def _extract_operator(self, text: str) -> str:
        """Extract the main operator from instruction text"""
        if not text:
            return ""
        # Get first word/identifier
        text = text.strip()
        if text.startswith("("):
            return text
        parts = text.split("(")[0].split("=")[0].split()
        return parts[-1] if parts else ""

    def _format_llil_instruction(self, instr: LowLevelILInstruction) -> str:
        """Format LLIL instruction for display"""
        op_name = instr.operation.name.replace('LLIL_', '')

        # Build operands string
        operands = []
        if hasattr(instr, 'value'):
            operands.append(repr(instr.value))
        if hasattr(instr, 'slot_index'):
            operands.append(f"stack[{instr.slot_index}]")
        if hasattr(instr, 'reg_index'):
            operands.append(f"reg{instr.reg_index}")
        if hasattr(instr, 'target'):
            if hasattr(instr.target, 'name'):
                operands.append(instr.target.name)

            else:
                operands.append(str(instr.target))
        if hasattr(instr, 'args') and instr.args:
            operands.append(f"[{len(instr.args)} args]")

        if operands:
            return f"{op_name}({', '.join(operands)})"
        return op_name

    def _format_mlil_instruction(self, instr: MediumLevelILInstruction) -> str:
        """Format MLIL instruction for display"""
        op_name = instr.operation.name.replace('MLIL_', '').lower()

        # Build operands string
        operands = []
        if hasattr(instr, 'dest') and instr.dest:
            dest_name = instr.dest.name if hasattr(instr.dest, 'name') else str(instr.dest)
            return f"{dest_name} = {op_name}(...)"
        if hasattr(instr, 'target'):
            if hasattr(instr.target, 'name'):
                operands.append(instr.target.name)

            else:
                operands.append(str(instr.target))
        if hasattr(instr, 'args') and instr.args:
            operands.append(f"[{len(instr.args)} args]")

        if operands:
            return f"{op_name}({', '.join(operands)})"
        return op_name

    def _format_hlil_statement(self, stmt: HLILStatement) -> str:
        """Format HLIL statement for display"""
        if hasattr(stmt, 'operation'):
            op_name = stmt.operation.name.replace('HLIL_', '').lower()

            if stmt.operation == HLILOperation.ASSIGN:
                if hasattr(stmt, 'dest') and hasattr(stmt, 'src'):
                    dest = self._format_hlil_expr(stmt.dest)
                    src = self._format_hlil_expr(stmt.src)
                    return f"{dest} = {src}"

            elif stmt.operation == HLILOperation.IF:
                if hasattr(stmt, 'condition'):
                    cond = self._format_hlil_expr(stmt.condition)
                    return f"if ({cond})"

            elif stmt.operation == HLILOperation.WHILE:
                if hasattr(stmt, 'condition'):
                    cond = self._format_hlil_expr(stmt.condition)
                    return f"while ({cond})"

            elif stmt.operation == HLILOperation.RETURN:
                if hasattr(stmt, 'value') and stmt.value:
                    val = self._format_hlil_expr(stmt.value)
                    return f"return {val}"
                return "return"

            elif stmt.operation == HLILOperation.CALL:
                if hasattr(stmt, 'target'):
                    target = self._format_hlil_expr(stmt.target)
                    return f"{target}(...)"

            return op_name

        return str(stmt)[:50]

    def _format_hlil_expr(self, expr: Any) -> str:
        """Format HLIL expression for display"""
        if expr is None:
            return ""
        if hasattr(expr, 'name'):
            return expr.name
        if hasattr(expr, 'value'):
            return repr(expr.value)
        if hasattr(expr, 'var'):
            return str(expr.var)
        return str(expr)[:30]


def extract_llil_addresses(llil_func: LowLevelILFunction) -> Dict[int, List[LowLevelILInstruction]]:
    """T004: Get SCP offset for each LLIL instruction"""
    result: Dict[int, List[LowLevelILInstruction]] = {}
    for bb in llil_func.basic_blocks:
        for instr in bb.instructions:
            addr = instr.address
            if addr not in result:
                result[addr] = []
            result[addr].append(instr)
    return result


def extract_mlil_addresses(
    mlil_func: MediumLevelILFunction,
    llil_addr_map: Dict[int, List[LowLevelILInstruction]]
) -> Dict[int, List[MediumLevelILInstruction]]:
    """T005: Propagate SCP offset from LLIL to MLIL"""
    result: Dict[int, List[MediumLevelILInstruction]] = {}
    for bb in mlil_func.basic_blocks:
        for instr in bb.instructions:
            # Try to get source address from instruction
            addr = None
            if hasattr(instr, 'source_address') and instr.source_address:
                addr = instr.source_address

            elif hasattr(instr, 'llil_source') and instr.llil_source:
                if hasattr(instr.llil_source, 'address'):
                    addr = instr.llil_source.address

            if addr is not None:
                if addr not in result:
                    result[addr] = []
                result[addr].append(instr)
    return result


def extract_hlil_addresses(
    hlil_func: HighLevelILFunction,
    mlil_addr_map: Dict[int, List[MediumLevelILInstruction]]
) -> Dict[int, List[HLILStatement]]:
    """T006: Propagate SCP offset from MLIL to HLIL"""
    result: Dict[int, List[HLILStatement]] = {}
    if not hasattr(hlil_func, 'statements'):
        return result

    for stmt in hlil_func.statements:
        addr = None
        if hasattr(stmt, 'source_address') and stmt.source_address:
            addr = stmt.source_address

        if addr is not None:
            if addr not in result:
                result[addr] = []
            result[addr].append(stmt)
    return result


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

    # Falcom-specific GLOBAL operations (user-defined)
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
        # If src is a CALL, extract the CALL as the primary operation
        # This makes `var = call(...)` equivalent to `call(...)` in MLIL
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

    # Expression statement - unwrap to get the actual expression
    elif op == HLILOperation.HLIL_EXPR_STMT:
        if hasattr(instr, 'expr') and instr.expr is not None:
            # Recursively normalize the inner expression
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
# Phase 2: CFG Infrastructure (T005-T013)
# =============================================================================

@dataclass
class CFGNode:
    """T005: Basic block node in CFG"""
    id: int
    operations: List[SemanticOperation] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    is_entry: bool = False
    is_exit: bool = False
    # For dominance analysis
    dominator: Optional[int] = None
    post_dominator: Optional[int] = None
    # For loop detection
    is_loop_header: bool = False
    loop_back_edges: List[int] = field(default_factory=list)


class CFG:
    """T006: Control Flow Graph representation"""

    def __init__(self) -> None:
        self.nodes: Dict[int, CFGNode] = {}
        self.entry: Optional[int] = None
        self.exits: List[int] = []

    def add_node(self, node: CFGNode) -> None:
        """Add a node to the CFG"""
        self.nodes[node.id] = node
        if node.is_entry:
            self.entry = node.id
        if node.is_exit:
            self.exits.append(node.id)

    def add_edge(self, from_id: int, to_id: int) -> None:
        """Add an edge between two nodes"""
        if from_id in self.nodes and to_id in self.nodes:
            if to_id not in self.nodes[from_id].successors:
                self.nodes[from_id].successors.append(to_id)
            if from_id not in self.nodes[to_id].predecessors:
                self.nodes[to_id].predecessors.append(from_id)

    def get_node(self, node_id: int) -> Optional[CFGNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def __len__(self) -> int:
        return len(self.nodes)


def build_cfg_from_llil(llil_func: LowLevelILFunction) -> CFG:
    """T007: Construct CFG from LLILFunction"""
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
    """T008: Construct CFG from MLILFunction"""
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
    """T009: Construct CFG from HLILFunction (flatten structured control flow)"""
    cfg = CFG()

    def extract_statements(body) -> List[Any]:
        """Extract statement list from HLIL body (handles HLILBlock, list, or single stmt)"""
        if body is None:
            return []
        if isinstance(body, list):
            return body
        if hasattr(body, 'statements'):
            return list(body.statements)
        # Single statement
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
        # HLIL body is an HLILBlock with statements attribute
        body = hlil_func.body
        if hasattr(body, 'statements'):
            stmts = list(body.statements)

        elif hasattr(body, 'body'):
            # Nested body
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
    """T010: Compute dominators using iterative data flow analysis"""
    if not cfg.entry or cfg.entry not in cfg.nodes:
        return

    # Initialize: entry dominates itself, others dominated by all
    all_nodes = set(cfg.nodes.keys())
    dom: Dict[int, set] = {n: all_nodes.copy() for n in cfg.nodes}
    dom[cfg.entry] = {cfg.entry}

    # Iterate until convergence
    changed = True
    while changed:
        changed = False
        for node_id in cfg.nodes:
            if node_id == cfg.entry:
                continue

            node = cfg.nodes[node_id]
            if not node.predecessors:
                continue

            # dom(n) = {n} union intersection(dom(p) for p in preds)
            new_dom = all_nodes.copy()
            for pred in node.predecessors:
                new_dom &= dom[pred]
            new_dom.add(node_id)

            if new_dom != dom[node_id]:
                dom[node_id] = new_dom
                changed = True

    # Set immediate dominator (closest dominator)
    for node_id in cfg.nodes:
        if node_id == cfg.entry:
            cfg.nodes[node_id].dominator = None
            continue

        doms = dom[node_id] - {node_id}
        if doms:
            # Find immediate dominator (dominated by all others in doms)
            for d in doms:
                if all(d in dom[other] or d == other for other in doms):
                    cfg.nodes[node_id].dominator = d
                    break


def compute_post_dominators(cfg: CFG) -> None:
    """T011: Compute post-dominators (reverse dominance)"""
    if not cfg.exits:
        return

    # Create virtual exit node connecting all exits
    all_nodes = set(cfg.nodes.keys())
    pdom: Dict[int, set] = {n: all_nodes.copy() for n in cfg.nodes}

    for exit_id in cfg.exits:
        pdom[exit_id] = {exit_id}

    # Iterate until convergence (reverse direction)
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
    """T012: Identify natural loops using back edges"""
    loops: List[Tuple[int, List[int]]] = []

    # Ensure dominators are computed
    compute_dominators(cfg)

    # Find back edges: edge (n -> h) where h dominates n
    for node_id, node in cfg.nodes.items():
        for succ in node.successors:
            succ_node = cfg.nodes.get(succ)
            if succ_node and node.dominator == succ:
                # Found back edge to loop header
                cfg.nodes[succ].is_loop_header = True
                cfg.nodes[succ].loop_back_edges.append(node_id)

                # Collect loop body nodes
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
    """T013: Create semantic fingerprint of basic block"""
    # Signature = operation kinds and operators in sequence
    parts = []
    for op in node.operations:
        parts.append(f"{op.kind.name}:{op.operator}")
    return "|".join(parts) if parts else "EMPTY"


# =============================================================================
# Phase 3: Basic Block Matching Algorithm (T014-T023)
# =============================================================================

@dataclass
class BlockSimilarity:
    """T014: Similarity score between two blocks"""
    source_block: int
    target_block: int
    score: float
    matched_ops: List[Tuple[int, int]] = field(default_factory=list)


def compute_operation_similarity(op1: SemanticOperation, op2: SemanticOperation) -> float:
    """T015: Compare two SemanticOperations (0.0-1.0 score)"""
    score = 0.0

    # Kind match (40%)
    if op1.kind == op2.kind:
        score += 0.4

    # Operator match (40%)
    if op1.operator == op2.operator:
        score += 0.4

    elif _is_transform_pair(op1.operator, op2.operator):
        score += 0.3  # Transformed operator

    # Operand count match (20%)
    if len(op1.operands) == len(op2.operands):
        score += 0.1
        # Check operand types
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
    """T016: Compare block content using operation sequence alignment"""
    if not node1.operations or not node2.operations:
        if not node1.operations and not node2.operations:
            return 1.0, []
        return 0.0, []

    # Use dynamic programming for sequence alignment (Needleman-Wunsch variant)
    m, n = len(node1.operations), len(node2.operations)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sim = compute_operation_similarity(
                node1.operations[i - 1], node2.operations[j - 1]
            )
            dp[i][j] = max(
                dp[i - 1][j - 1] + sim,  # Match
                dp[i - 1][j],             # Skip op1
                dp[i][j - 1]              # Skip op2
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

    # Normalize score
    max_ops = max(m, n)
    total_sim = dp[m][n]
    normalized_score = total_sim / max_ops if max_ops > 0 else 1.0

    return min(1.0, normalized_score), matches


def compute_block_structural_similarity(node1: CFGNode, node2: CFGNode) -> float:
    """T017: Compare structural similarity using degree and edge types"""
    in_diff = abs(len(node1.predecessors) - len(node2.predecessors))
    out_diff = abs(len(node1.successors) - len(node2.successors))

    max_in = max(len(node1.predecessors), len(node2.predecessors), 1)
    max_out = max(len(node1.successors), len(node2.successors), 1)

    in_sim = 1.0 - (in_diff / max_in)
    out_sim = 1.0 - (out_diff / max_out)

    # Entry/exit bonus
    entry_match = 1.0 if node1.is_entry == node2.is_entry else 0.5
    exit_match = 1.0 if node1.is_exit == node2.is_exit else 0.5

    return 0.4 * in_sim + 0.4 * out_sim + 0.1 * entry_match + 0.1 * exit_match


def compute_combined_block_similarity(
    node1: CFGNode, node2: CFGNode
) -> BlockSimilarity:
    """T018: Combine content (70%) + structure (30%) similarity"""
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
    """T019: Store pairwise block similarities"""

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
    """T020: Build initial similarity matrix using block content similarity"""
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
    """T021: Iteratively propagate neighbor similarity until convergence"""
    source_nodes = list(source_cfg.nodes.keys())
    target_nodes = list(target_cfg.nodes.keys())

    # Create index mappings
    src_idx = {nid: i for i, nid in enumerate(source_nodes)}
    tgt_idx = {nid: i for i, nid in enumerate(target_nodes)}

    for iteration in range(max_iterations):
        max_change = 0.0
        new_matrix = SimilarityMatrix(matrix.rows, matrix.cols)

        for i, src_id in enumerate(source_nodes):
            src_node = source_cfg.nodes[src_id]

            for j, tgt_id in enumerate(target_nodes):
                tgt_node = target_cfg.nodes[tgt_id]

                # Content similarity (base)
                content_sim = matrix.get(i, j)

                # Neighbor similarity
                neighbor_sim = 0.0
                neighbor_count = 0

                # Successor similarity
                for src_succ in src_node.successors:
                    if src_succ in src_idx:
                        for tgt_succ in tgt_node.successors:
                            if tgt_succ in tgt_idx:
                                neighbor_sim += matrix.get(src_idx[src_succ], tgt_idx[tgt_succ])
                                neighbor_count += 1

                # Predecessor similarity
                for src_pred in src_node.predecessors:
                    if src_pred in src_idx:
                        for tgt_pred in tgt_node.predecessors:
                            if tgt_pred in tgt_idx:
                                neighbor_sim += matrix.get(src_idx[src_pred], tgt_idx[tgt_pred])
                                neighbor_count += 1

                if neighbor_count > 0:
                    neighbor_sim /= neighbor_count
                    # Combined similarity with neighbors
                    new_sim = (1 - alpha) * content_sim + alpha * neighbor_sim

                else:
                    # No neighbors - keep original similarity
                    new_sim = content_sim

                new_matrix.set(i, j, new_sim)

                max_change = max(max_change, abs(new_sim - content_sim))

        matrix = new_matrix

        if max_change < epsilon:
            break

    return matrix


def hungarian_matching(matrix: SimilarityMatrix) -> List[Tuple[int, int]]:
    """T022: Find optimal block assignment using Hungarian algorithm"""
    # Convert to cost matrix (1 - similarity for minimization)
    n = max(matrix.rows, matrix.cols)
    cost = [[1.0] * n for _ in range(n)]

    for i in range(matrix.rows):
        for j in range(matrix.cols):
            cost[i][j] = 1.0 - matrix.get(i, j)

    # Simple greedy matching (Hungarian algorithm approximation)
    # For production, use scipy.optimize.linear_sum_assignment
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
    """T023: Match basic blocks between two CFGs"""

    def __init__(self, source_cfg: CFG, target_cfg: CFG):
        self.source_cfg = source_cfg
        self.target_cfg = target_cfg
        self.matches: List[Tuple[int, int]] = []
        self.similarity_details: Dict[Tuple[int, int], BlockSimilarity] = {}

    def match(self) -> List[Tuple[int, int]]:
        """Find optimal block matching between CFGs"""
        if not self.source_cfg.nodes or not self.target_cfg.nodes:
            return []

        # Build initial similarity matrix
        matrix, self.similarity_details = build_initial_similarity_matrix(
            self.source_cfg, self.target_cfg
        )

        # Propagate similarity through neighbors
        matrix = propagate_similarity(
            matrix, self.source_cfg, self.target_cfg
        )

        # Find optimal matching
        idx_matches = hungarian_matching(matrix)

        # Convert indices back to node IDs
        source_nodes = list(self.source_cfg.nodes.keys())
        target_nodes = list(self.target_cfg.nodes.keys())

        self.matches = []
        for i, j in idx_matches:
            if i < len(source_nodes) and j < len(target_nodes):
                self.matches.append((source_nodes[i], target_nodes[j]))

        return self.matches

    def get_similarity(self, src_id: int, tgt_id: int) -> Optional[BlockSimilarity]:
        """Get similarity details for a block pair"""
        return self.similarity_details.get((src_id, tgt_id))


# =============================================================================
# Phase 4: Semantic Comparison Engine (T024-T033)
# =============================================================================

class OperationMatcher:
    """T024: Align operations within matched blocks"""

    def __init__(self, var_mapping: Dict[str, str]):
        self.var_mapping = var_mapping

    def align(
        self, source_ops: List[SemanticOperation], target_ops: List[SemanticOperation]
    ) -> List[Tuple[Optional[SemanticOperation], Optional[SemanticOperation]]]:
        """Align operations using sequence alignment"""
        aligned: List[Tuple[Optional[SemanticOperation], Optional[SemanticOperation]]] = []

        if not source_ops and not target_ops:
            return aligned

        if not source_ops:
            return [(None, op) for op in target_ops]

        if not target_ops:
            return [(op, None) for op in source_ops]

        # Use Needleman-Wunsch alignment
        m, n = len(source_ops), len(target_ops)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        GAP_PENALTY = -0.5

        # Initialize
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + GAP_PENALTY

        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + GAP_PENALTY

        # Fill
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


def sequence_alignment(
    source_ops: List[SemanticOperation],
    target_ops: List[SemanticOperation]
) -> List[Tuple[Optional[int], Optional[int]]]:
    """T025: Needleman-Wunsch algorithm for operation matching"""
    if not source_ops or not target_ops:
        return []

    m, n = len(source_ops), len(target_ops)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sim = compute_operation_similarity(source_ops[i - 1], target_ops[j - 1])
            dp[i][j] = max(dp[i - 1][j - 1] + sim, dp[i - 1][j], dp[i][j - 1])

    # Backtrack
    matches: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = m, n
    while i > 0 and j > 0:
        sim = compute_operation_similarity(source_ops[i - 1], target_ops[j - 1])
        if dp[i][j] == dp[i - 1][j - 1] + sim and sim > 0.3:
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1

        elif dp[i][j] == dp[i - 1][j]:
            i -= 1

        else:
            j -= 1

    matches.reverse()
    return matches


def compare_operands(
    op1: SemanticOperation,
    op2: SemanticOperation,
    var_mapping: Dict[str, str]
) -> bool:
    """T026: Compare operands with variable mapping resolution"""
    if len(op1.operands) != len(op2.operands):
        return False

    for a, b in zip(op1.operands, op2.operands):
        if a.kind == 'const' and b.kind == 'const':
            if a.value != b.value:
                return False

        elif a.kind == 'var' and b.kind == 'var':
            # Check variable mapping
            mapped = var_mapping.get(str(a.value))
            if mapped and mapped != str(b.value):
                return False

        elif a.kind != b.kind:
            # Allow some transformations
            if not (a.kind == 'var' and b.kind == 'var'):
                return False

    return True


# Commutative operators
COMMUTATIVE_OPS = {'ADD', 'MUL', 'AND', 'OR', 'EQ', 'NE', 'LOGICAL_AND', 'LOGICAL_OR'}


def is_commutative(operator: str) -> bool:
    """T027: Check if operator is commutative"""
    return operator in COMMUTATIVE_OPS


def compare_expression_trees(
    op1: SemanticOperation,
    op2: SemanticOperation,
    var_mapping: Dict[str, str]
) -> bool:
    """T028: Compare expression trees with recursive semantic equivalence"""
    # Same kind required
    if op1.kind != op2.kind:
        return False

    # Same operator required (or known transformation)
    if op1.operator != op2.operator and not _is_transform_pair(op1.operator, op2.operator):
        return False

    # Compare operands
    if is_commutative(op1.operator):
        # For commutative ops, try both orderings
        if compare_operands(op1, op2, var_mapping):
            return True

        # Try reversed operands
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
    """T029: Describes an expected transformation between IR layers"""
    source_pattern: str
    target_pattern: str
    description: str


# T030: LLIL to MLIL expected transformations
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

# T031: MLIL to HLIL expected transformations
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
    """T032: Detect if difference is an expected transformation"""
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
    """T033: Classify difference as EQUIVALENT, TRANSFORMED, or DIFFERENT"""
    # Check exact equivalence
    if source_op.kind == target_op.kind and source_op.operator == target_op.operator:
        if compare_operands(source_op, target_op, var_mapping):
            return ComparisonStatus.EQUIVALENT

    # Check expression tree equivalence (handles commutative ops)
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
# T062: Side-Effect Ordering Validation (FR-014)
# =============================================================================

# Operations with side effects that must preserve ordering
SIDE_EFFECT_OPS = {OperationKind.CALL, OperationKind.ASSIGN}


def has_side_effect(op: SemanticOperation) -> bool:
    """T062: Check if operation has side effects"""
    return op.kind in SIDE_EFFECT_OPS


def extract_side_effect_sequence(
    ops: List[SemanticOperation]
) -> List[SemanticOperation]:
    """T062: Extract ordered sequence of side-effecting operations"""
    return [op for op in ops if has_side_effect(op)]


def validate_side_effect_ordering(
    source_ops: List[SemanticOperation],
    target_ops: List[SemanticOperation],
    var_mapping: Dict[str, str]
) -> List[Tuple[str, SemanticOperation, SemanticOperation]]:
    """T062: Validate side-effect ordering between IR layers (FR-014)

    Returns list of (error_message, source_op, target_op) for ordering violations.

    Side effects that must be preserved:
    - Assignment order (writes to same variable)
    - Call order (calls may have observable effects)
    - Mixed assignment/call order (call may depend on assigned value)
    """
    violations: List[Tuple[str, SemanticOperation, SemanticOperation]] = []

    # Extract side-effecting operations in order
    source_side_effects = extract_side_effect_sequence(source_ops)
    target_side_effects = extract_side_effect_sequence(target_ops)

    if not source_side_effects or not target_side_effects:
        return violations

    # Check call ordering preservation
    source_calls = [op for op in source_side_effects if op.kind == OperationKind.CALL]
    target_calls = [op for op in target_side_effects if op.kind == OperationKind.CALL]

    # Verify calls appear in same relative order
    call_violations = _check_call_ordering(source_calls, target_calls, var_mapping)
    violations.extend(call_violations)

    # Check assignment ordering for same variables
    assign_violations = _check_assignment_ordering(
        source_side_effects, target_side_effects, var_mapping
    )
    violations.extend(assign_violations)

    # Check call-assignment dependencies
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
    """T062: Verify call operations maintain relative order"""
    violations = []

    # Map source calls by operator/target for matching
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

    # Check that calls to same target maintain order
    for key, source_indices in source_call_order.items():
        if key in target_call_order:
            target_indices = target_call_order[key]
            # Multiple calls to same target must maintain order
            if len(source_indices) > 1 and len(target_indices) > 1:
                # Check order preservation
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
    """T062: Generate signature for call matching"""
    # Use operator and first operand (target) if available
    target = call.operands[0].value if call.operands else "unknown"
    return f"{call.operator}:{target}"


def _check_assignment_ordering(
    source_ops: List[SemanticOperation],
    target_ops: List[SemanticOperation],
    var_mapping: Dict[str, str]
) -> List[Tuple[str, SemanticOperation, SemanticOperation]]:
    """T062: Check assignments to same variable maintain order"""
    violations = []

    # Extract assignments
    source_assigns = [op for op in source_ops if op.kind == OperationKind.ASSIGN]
    target_assigns = [op for op in target_ops if op.kind == OperationKind.ASSIGN]

    # Group by destination variable
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

    # Check ordering for variables with multiple writes
    for dest, source_writes in source_by_dest.items():
        if dest in target_by_dest and len(source_writes) > 1:
            target_writes = target_by_dest[dest]
            if len(target_writes) > 1:
                # Both have multiple writes - check order
                source_order = [idx for idx, _ in source_writes]
                target_order = [idx for idx, _ in target_writes]

                # Detect inversions
                min_len = min(len(source_order), len(target_order))
                for i in range(min_len - 1):
                    if source_order[i] < source_order[i + 1]:
                        # source has i before i+1
                        if i + 1 < len(target_order) and target_order[i] > target_order[i + 1]:
                            violations.append((
                                f"Assignment reordering: writes to {dest}",
                                source_writes[i][1],
                                target_writes[i + 1][1]
                            ))

    return violations


def _get_assign_dest(op: SemanticOperation, var_mapping: Dict[str, str]) -> Optional[str]:
    """T062: Get normalized destination variable name"""
    if op.result:
        dest = str(op.result.value)
        # Apply variable mapping
        return var_mapping.get(dest, dest)
    return None


def _check_call_assignment_dependencies(
    source_ops: List[SemanticOperation],
    target_ops: List[SemanticOperation],
    var_mapping: Dict[str, str]
) -> List[Tuple[str, SemanticOperation, SemanticOperation]]:
    """T062: Check call-assignment dependency ordering

    A call that reads a variable must come after assignments to that variable.
    """
    violations = []

    # Find assignment-call pairs in source
    source_pairs = _find_assign_call_pairs(source_ops, var_mapping)
    target_pairs = _find_assign_call_pairs(target_ops, var_mapping)

    # Check that dependencies are preserved
    for var, (assign_idx, call_idx) in source_pairs.items():
        if var in target_pairs:
            t_assign_idx, t_call_idx = target_pairs[var]
            # In source: assign before call (assign_idx < call_idx)
            # In target: should also be assign before call
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
    """T062: Find variable -> (last_assign_idx, first_call_using_var_idx) pairs"""
    pairs: Dict[str, Tuple[int, int]] = {}
    assigned_vars: Dict[str, int] = {}  # var -> last assign index

    for idx, op in enumerate(ops):
        if op.kind == OperationKind.ASSIGN:
            dest = _get_assign_dest(op, var_mapping)
            if dest:
                assigned_vars[dest] = idx

        elif op.kind == OperationKind.CALL:
            # Check if any operand references an assigned variable
            for operand in op.operands:
                var = str(operand.value)
                mapped = var_mapping.get(var, var)
                if mapped in assigned_vars:
                    if mapped not in pairs:
                        pairs[mapped] = (assigned_vars[mapped], idx)

    return pairs


class SideEffectValidator:
    """T062: Validator for side-effect ordering across IR layers"""

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
        """Validate side-effect ordering and return any violations as ComparisonResults"""
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
# Phase 5: LLIL-MLIL CFG Comparison (T034-T039)
# =============================================================================

class LLILMLILComparator:
    """T034: CFG-based LLIL-MLIL validator using block matching"""

    def __init__(self, llil_func: LowLevelILFunction, mlil_func: MediumLevelILFunction):
        self.llil_func = llil_func
        self.mlil_func = mlil_func
        self.results: List[ComparisonResult] = []
        self.variable_mappings: Dict[str, str] = {}
        self.llil_cfg: Optional[CFG] = None
        self.mlil_cfg: Optional[CFG] = None
        self.block_matches: List[Tuple[int, int]] = []

    def compare(self) -> List[ComparisonResult]:
        """T034-T039: Compare using CFG-based block matching"""
        self.results = []

        # T035: Build CFGs
        self.llil_cfg = build_cfg_from_llil(self.llil_func)
        self.mlil_cfg = build_cfg_from_mlil(self.mlil_func)

        # Build variable mappings
        self._build_variable_mappings()

        # T036: Match blocks using CFGMatcher
        matcher = CFGMatcher(self.llil_cfg, self.mlil_cfg)
        self.block_matches = matcher.match()

        # T036: Compare matched blocks
        self._compare_matched_blocks(matcher)

        # T037: Handle unmatched blocks
        self._handle_unmatched_blocks()

        # T062: Validate side-effect ordering
        self._validate_side_effect_ordering()

        return self.results

    def _validate_side_effect_ordering(self) -> None:
        """T062: Check side-effect ordering across all blocks"""
        # Collect all operations from all blocks
        all_llil_ops: List[SemanticOperation] = []
        all_mlil_ops: List[SemanticOperation] = []

        for node in self.llil_cfg.nodes.values():
            all_llil_ops.extend(node.operations)

        for node in self.mlil_cfg.nodes.values():
            all_mlil_ops.extend(node.operations)

        # Validate ordering
        validator = SideEffectValidator(self.variable_mappings)
        side_effect_results = validator.validate(
            all_llil_ops, all_mlil_ops, IRLayer.LLIL, IRLayer.MLIL
        )
        self.results.extend(side_effect_results)

    def _build_variable_mappings(self) -> None:
        """T035: Build storage -> variable mappings from MLIL"""
        for bb in self.mlil_func.basic_blocks:
            for instr in bb.instructions:
                if instr.operation == MediumLevelILOperation.MLIL_SET_VAR:
                    if hasattr(instr, 'dest') and instr.dest is not None:
                        var = instr.dest
                        var_name = var.name if hasattr(var, 'name') else str(var)
                        if hasattr(var, 'source_storage'):
                            self.variable_mappings[var.source_storage] = var_name

    def _compare_matched_blocks(self, matcher: CFGMatcher) -> None:
        """T036: Compare operations in matched block pairs"""
        op_matcher = OperationMatcher(self.variable_mappings)

        for llil_block_id, mlil_block_id in self.block_matches:
            llil_node = self.llil_cfg.get_node(llil_block_id)
            mlil_node = self.mlil_cfg.get_node(mlil_block_id)

            if not llil_node or not mlil_node:
                continue

            # Align operations within the block
            aligned = op_matcher.align(llil_node.operations, mlil_node.operations)

            for llil_op, mlil_op in aligned:
                if llil_op is None:
                    # MLIL has extra operation
                    self.results.append(ComparisonResult(
                        status=ComparisonStatus.TRANSFORMED,
                        source_layer=IRLayer.LLIL,
                        target_layer=IRLayer.MLIL,
                        source_op=None,
                        target_op=mlil_op,
                        explanation=f"MLIL operation added: {mlil_op.operator}",
                        source_location=mlil_op.source_location if mlil_op else SourceLocation()
                    ))

                elif mlil_op is None:
                    # LLIL operation eliminated
                    self.results.append(ComparisonResult(
                        status=ComparisonStatus.TRANSFORMED,
                        source_layer=IRLayer.LLIL,
                        target_layer=IRLayer.MLIL,
                        source_op=llil_op,
                        target_op=None,
                        explanation=f"LLIL operation eliminated: {llil_op.operator}",
                        source_location=llil_op.source_location
                    ))

                else:
                    # Both present - classify the difference
                    status = classify_difference(
                        llil_op, mlil_op, self.variable_mappings, LLIL_MLIL_TRANSFORMATIONS
                    )
                    if status == ComparisonStatus.EQUIVALENT:
                        explanation = f"{llil_op.operator} == {mlil_op.operator}"

                    elif status == ComparisonStatus.TRANSFORMED:
                        explanation = f"{llil_op.operator} -> {mlil_op.operator}"

                    else:
                        explanation = f"Mismatch: {llil_op.operator} vs {mlil_op.operator}"

                    self.results.append(ComparisonResult(
                        status=status,
                        source_layer=IRLayer.LLIL,
                        target_layer=IRLayer.MLIL,
                        source_op=llil_op,
                        target_op=mlil_op,
                        explanation=explanation,
                        source_location=llil_op.source_location
                    ))

    def _handle_unmatched_blocks(self) -> None:
        """T037: Report blocks without matches"""
        matched_llil = {m[0] for m in self.block_matches}
        matched_mlil = {m[1] for m in self.block_matches}

        # Unmatched LLIL blocks
        for node_id, node in self.llil_cfg.nodes.items():
            if node_id not in matched_llil and node.operations:
                for op in node.operations:
                    self.results.append(ComparisonResult(
                        status=ComparisonStatus.DIFFERENT,
                        source_layer=IRLayer.LLIL,
                        target_layer=IRLayer.MLIL,
                        source_op=op,
                        target_op=None,
                        explanation=f"LLIL block {node_id} unmatched: {op.operator}",
                        source_location=op.source_location
                    ))

        # Unmatched MLIL blocks
        for node_id, node in self.mlil_cfg.nodes.items():
            if node_id not in matched_mlil and node.operations:
                for op in node.operations:
                    self.results.append(ComparisonResult(
                        status=ComparisonStatus.DIFFERENT,
                        source_layer=IRLayer.LLIL,
                        target_layer=IRLayer.MLIL,
                        source_op=None,
                        target_op=op,
                        explanation=f"MLIL block {node_id} unmatched: {op.operator}",
                        source_location=op.source_location
                    ))


# =============================================================================
# Phase 6: Variable/Storage Tracking (keeping existing implementation)
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
# Phase 7: MLIL-HLIL CFG Comparison (T047-T052)
# =============================================================================

class MLILHLILComparator:
    """T047: Flat sequence MLIL-HLIL validator using operation alignment

    Uses flat operation sequences instead of block-matching because HLIL's
    structured control flow (IF/WHILE/FOR) creates fundamentally different
    CFG topology than MLIL's basic blocks with GOTO.
    """

    def __init__(self, mlil_func: MediumLevelILFunction, hlil_func: HighLevelILFunction):
        self.mlil_func = mlil_func
        self.hlil_func = hlil_func
        self.results: List[ComparisonResult] = []
        self.variable_mappings: Dict[str, str] = {}

    def compare(self) -> List[ComparisonResult]:
        """Compare by operation frequency matching (order-independent)

        HLIL has different traversal order than MLIL due to structured control flow.
        Instead of sequence alignment, we compare operation frequencies and match
        by normalized operator names.
        """
        self.results = []

        # Flatten all operations from both IRs
        mlil_ops = self._flatten_mlil_operations()
        hlil_ops = self._flatten_hlil_operations()

        # Filter out control flow operations
        mlil_ops = self._filter_operations(mlil_ops, is_mlil=True)
        hlil_ops = self._filter_operations(hlil_ops, is_mlil=False)

        # Normalize operators for comparison
        def normalize_op(op: str) -> str:
            # Map equivalent operators
            equiv = {
                'RET': 'RETURN', 'RETURN': 'RETURN',
                'SET_VAR': 'ASSIGN', 'STORE_REG': 'ASSIGN', 'ASSIGN': 'ASSIGN',
                'CALL_SCRIPT': 'CALL', 'CALL': 'CALL', 'SYSCALL': 'CALL',
            }
            return equiv.get(op, op)

        # Group operations by normalized operator
        mlil_by_op: Dict[str, List[SemanticOperation]] = {}
        for op in mlil_ops:
            key = normalize_op(op.operator)
            mlil_by_op.setdefault(key, []).append(op)

        hlil_by_op: Dict[str, List[SemanticOperation]] = {}
        for op in hlil_ops:
            key = normalize_op(op.operator)
            hlil_by_op.setdefault(key, []).append(op)

        # Compare by operator type
        all_ops = set(mlil_by_op.keys()) | set(hlil_by_op.keys())
        for op_type in all_ops:
            mlil_list = mlil_by_op.get(op_type, [])
            hlil_list = hlil_by_op.get(op_type, [])

            # Match as many as we can
            matched = min(len(mlil_list), len(hlil_list))
            for i in range(matched):
                self.results.append(ComparisonResult(
                    status=ComparisonStatus.EQUIVALENT,
                    source_layer=IRLayer.MLIL,
                    target_layer=IRLayer.HLIL,
                    source_op=mlil_list[i],
                    target_op=hlil_list[i],
                    explanation=f"{mlil_list[i].operator} == {hlil_list[i].operator}",
                    source_location=mlil_list[i].source_location
                ))

            # Unmatched MLIL ops
            for i in range(matched, len(mlil_list)):
                self.results.append(ComparisonResult(
                    status=ComparisonStatus.DIFFERENT,
                    source_layer=IRLayer.MLIL,
                    target_layer=IRLayer.HLIL,
                    source_op=mlil_list[i],
                    target_op=None,
                    explanation=f"MLIL has extra {op_type}",
                    source_location=mlil_list[i].source_location
                ))

            # Unmatched HLIL ops
            for i in range(matched, len(hlil_list)):
                self.results.append(ComparisonResult(
                    status=ComparisonStatus.DIFFERENT,
                    source_layer=IRLayer.MLIL,
                    target_layer=IRLayer.HLIL,
                    source_op=None,
                    target_op=hlil_list[i],
                    explanation=f"HLIL has extra {op_type}",
                    source_location=hlil_list[i].source_location
                ))

        return self.results

    def _flatten_mlil_operations(self) -> List[SemanticOperation]:
        """Extract all operations from MLIL in execution order"""
        ops: List[SemanticOperation] = []
        for bb in self.mlil_func.basic_blocks:
            for instr in bb.instructions:
                op = normalize_mlil_operation(instr)
                if op:
                    ops.append(op)
        return ops

    def _flatten_hlil_operations(self) -> List[SemanticOperation]:
        """Extract all operations from HLIL (recursively flatten structured stmts)"""
        ops: List[SemanticOperation] = []

        def flatten_stmt(stmt) -> None:
            op = normalize_hlil_operation(stmt)
            if op:
                ops.append(op)

            # Recursively flatten nested bodies
            if hasattr(stmt, 'operation'):
                if stmt.operation == HLILOperation.HLIL_IF:
                    # HLIL IF uses true_block/false_block instead of body/else_body
                    if hasattr(stmt, 'true_block') and stmt.true_block:
                        flatten_body(stmt.true_block)
                    if hasattr(stmt, 'false_block') and stmt.false_block:
                        flatten_body(stmt.false_block)
                    # Also check body/else_body for compatibility
                    if hasattr(stmt, 'body') and stmt.body:
                        flatten_body(stmt.body)
                    if hasattr(stmt, 'else_body') and stmt.else_body:
                        flatten_body(stmt.else_body)

                elif stmt.operation in (HLILOperation.HLIL_WHILE, HLILOperation.HLIL_FOR):
                    if hasattr(stmt, 'body') and stmt.body:
                        flatten_body(stmt.body)
                    # Also check loop_block for compatibility
                    if hasattr(stmt, 'loop_block') and stmt.loop_block:
                        flatten_body(stmt.loop_block)

                elif stmt.operation == HLILOperation.HLIL_SWITCH:
                    # Recursively flatten all switch cases
                    if hasattr(stmt, 'cases'):
                        for case in stmt.cases:
                            # Each case may be HLILCase with body, or direct statements
                            if hasattr(case, 'body') and case.body:
                                flatten_body(case.body)
                            elif hasattr(case, 'statements'):
                                for s in case.statements:
                                    flatten_stmt(s)
                            else:
                                # Case might be a direct statement
                                flatten_stmt(case)

                elif stmt.operation == HLILOperation.HLIL_BLOCK:
                    if hasattr(stmt, 'statements'):
                        for s in stmt.statements:
                            flatten_stmt(s)

        def flatten_body(body) -> None:
            if body is None:
                return
            if hasattr(body, 'statements'):
                for s in body.statements:
                    flatten_stmt(s)
            else:
                flatten_stmt(body)

        # Start from function body
        if hasattr(self.hlil_func, 'body') and self.hlil_func.body:
            flatten_body(self.hlil_func.body)
        elif hasattr(self.hlil_func, 'statements'):
            for s in self.hlil_func.statements:
                flatten_stmt(s)

        return ops

    def _filter_operations(self, ops: List[SemanticOperation], is_mlil: bool) -> List[SemanticOperation]:
        """Filter out operations that differ structurally between MLIL and HLIL"""
        filtered: List[SemanticOperation] = []
        for op in ops:
            # Skip GOTO (MLIL has these, HLIL uses structured control flow)
            if op.operator == 'GOTO':
                continue

            # Skip pure control flow markers
            if op.kind == OperationKind.BRANCH and op.operator in ('GOTO', 'JUMP', 'IF'):
                continue

            # Skip NOP/DEBUG/COMMENT operations
            if op.kind == OperationKind.NOP:
                continue

            # Skip control flow structure headers
            # MLIL IF/WHILE/FOR/SWITCH become HLIL structured statements
            if op.kind == OperationKind.CONTROL_FLOW and op.operator in ('WHILE', 'FOR', 'IF', 'SWITCH'):
                continue

            filtered.append(op)
        return filtered


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


# =============================================================================
# T012-T014: IDA-Style Synchronized Output Formatting
# =============================================================================

# Status symbols for display (ASCII-compatible for Windows console)
STATUS_SYMBOLS = {
    MappingStatus.EQUIVALENT: ("=", "OK"),
    MappingStatus.TRANSFORMED: ("~", "TRANS"),
    MappingStatus.DIFFERENT: ("!", "DIFF"),
    MappingStatus.ELIMINATED: ("o", "ELIM"),
    MappingStatus.INLINED: (">", "INLINE"),
}

# ANSI color codes
COLORS = {
    'green': '\033[32m',
    'yellow': '\033[33m',
    'red': '\033[31m',
    'gray': '\033[90m',
    'cyan': '\033[36m',
    'reset': '\033[0m',
    'bold': '\033[1m',
}


def format_sync_table(table: AddressTable, use_color: bool = True) -> str:
    """T012: Format address table with aligned columns (IDA-style)"""
    lines = []

    # Header
    lines.append(f"=== IDA-Style Sync View: {table.function_name} ===")
    lines.append("")

    # Calculate column widths
    addr_width = 12
    llil_width = 35
    mlil_width = 35
    hlil_width = 30
    status_width = 8

    # Table header
    header = (
        f"{'SCP Offset':<{addr_width}} | "
        f"{'LLIL':<{llil_width}} | "
        f"{'MLIL':<{mlil_width}} | "
        f"{'HLIL':<{hlil_width}} | "
        f"{'Status':<{status_width}}"
    )
    separator = "-" * len(header)

    lines.append(header)
    lines.append(separator)

    # Table rows (skip eliminated entries)
    for mapping in table.sorted_by_offset():
        if mapping.status == MappingStatus.ELIMINATED:
            continue

        row = format_sync_row(mapping, addr_width, llil_width, mlil_width, hlil_width, use_color)
        lines.append(row)

    lines.append(separator)

    # Legend
    lines.append("")
    lines.append("Legend: ==equivalent, ~=transformed, !=different, o=eliminated, >=inlined")
    lines.append("")

    # Summary
    total = len(table.mappings)
    lines.append(f"Summary: {total} SCP offsets mapped")
    if total > 0:
        eq_pct = table.equivalent_count / total * 100
        tr_pct = table.transformed_count / total * 100
        df_pct = table.different_count / total * 100
        el_pct = table.eliminated_count / total * 100
        in_pct = table.inlined_count / total * 100

        lines.append(f"  = Equivalent:  {table.equivalent_count:3d} ({eq_pct:5.1f}%)")
        lines.append(f"  ~ Transformed: {table.transformed_count:3d} ({tr_pct:5.1f}%)")
        lines.append(f"  ! Different:   {table.different_count:3d} ({df_pct:5.1f}%)")
        lines.append(f"  o Eliminated:  {table.eliminated_count:3d} ({el_pct:5.1f}%)")
        lines.append(f"  > Inlined:     {table.inlined_count:3d} ({in_pct:5.1f}%)")

    return "\n".join(lines)


def format_sync_row(
    mapping: AddressMapping,
    addr_width: int,
    llil_width: int,
    mlil_width: int,
    hlil_width: int,
    use_color: bool
) -> str:
    """Format a single row in the sync table"""
    # Get status symbol and color
    symbol, label = STATUS_SYMBOLS.get(mapping.status, ("?", "???"))

    # Determine color based on status
    if use_color:
        if mapping.status == MappingStatus.EQUIVALENT:
            color = COLORS['green']

        elif mapping.status == MappingStatus.TRANSFORMED:
            color = COLORS['yellow']

        elif mapping.status == MappingStatus.DIFFERENT:
            color = COLORS['red']

        elif mapping.status in (MappingStatus.ELIMINATED, MappingStatus.INLINED):
            color = COLORS['gray']

        else:
            color = COLORS['reset']
        reset = COLORS['reset']

    else:
        color = ""
        reset = ""

    # Truncate text to fit columns
    llil_text = truncate_text(mapping.llil_text, llil_width)
    mlil_text = truncate_text(mapping.mlil_text, mlil_width)
    hlil_text = truncate_text(mapping.hlil_text, hlil_width)

    # Format address
    addr_str = f"0x{mapping.scp_offset:08X}"

    # Build row
    row = (
        f"{color}{addr_str:<{addr_width}} | "
        f"{llil_text:<{llil_width}} | "
        f"{mlil_text:<{mlil_width}} | "
        f"{hlil_text:<{hlil_width}} | "
        f"{symbol} {label}{reset}"
    )

    return row


def truncate_text(text: str, max_width: int) -> str:
    """Truncate text to fit within max_width, adding ellipsis if needed"""
    if not text:
        return ""
    if len(text) <= max_width:
        return text
    return text[:max_width - 3] + "..."


def format_llil_mlil_sync_table(
    table: AddressTable,
    use_color: bool = True,
    show_scp: bool = False
) -> str:
    """T021: Format LLIL-MLIL synchronized table"""
    lines = []

    # Header
    lines.append(f"=== LLIL-MLIL Sync View: {table.function_name} ===")
    lines.append("")

    # Column widths
    addr_width = 12
    scp_op_width = 20 if show_scp else 0
    llil_width = 35 if show_scp else 40
    mlil_width = 35 if show_scp else 40
    status_width = 12

    # T059: Table header with optional SCP column
    if show_scp:
        header = (
            f"{'SCP Offset':<{addr_width}} | "
            f"{'SCP Opcode':<{scp_op_width}} | "
            f"{'LLIL Instruction':<{llil_width}} | "
            f"{'MLIL Instruction':<{mlil_width}} | "
            f"{'Status':<{status_width}}"
        )

    else:
        header = (
            f"{'SCP Offset':<{addr_width}} | "
            f"{'LLIL Instruction':<{llil_width}} | "
            f"{'MLIL Instruction':<{mlil_width}} | "
            f"{'Status':<{status_width}}"
        )
    separator = "-" * len(header)

    lines.append(header)
    lines.append(separator)

    # Table rows (only LLIL-MLIL, skip eliminated)
    for mapping in table.sorted_by_offset():
        if not mapping.llil_text and not mapping.mlil_text:
            continue

        if mapping.status == MappingStatus.ELIMINATED:
            continue

        symbol, label = STATUS_SYMBOLS.get(mapping.status, ("?", "???"))

        if use_color:
            if mapping.status == MappingStatus.EQUIVALENT:
                color = COLORS['green']

            elif mapping.status == MappingStatus.TRANSFORMED:
                color = COLORS['yellow']

            elif mapping.status == MappingStatus.DIFFERENT:
                color = COLORS['red']

            else:
                color = COLORS['gray']
            reset = COLORS['reset']

        else:
            color = ""
            reset = ""

        llil_text = truncate_text(mapping.llil_text or "(none)", llil_width)
        mlil_text = truncate_text(mapping.mlil_text or "(none)", mlil_width)
        addr_str = f"0x{mapping.scp_offset:08X}"

        # T059: Include SCP opcode column if requested
        if show_scp:
            scp_opcode = truncate_text(mapping.scp_opcode or "", scp_op_width)
            row = (
                f"{color}{addr_str:<{addr_width}} | "
                f"{scp_opcode:<{scp_op_width}} | "
                f"{llil_text:<{llil_width}} | "
                f"{mlil_text:<{mlil_width}} | "
                f"{symbol} {label}{reset}"
            )

        else:
            row = (
                f"{color}{addr_str:<{addr_width}} | "
                f"{llil_text:<{llil_width}} | "
                f"{mlil_text:<{mlil_width}} | "
                f"{symbol} {label}{reset}"
            )
        lines.append(row)

    lines.append(separator)

    # Legend and summary
    lines.append("")
    lines.append("Legend: ==equivalent, ~=transformed, !=different, o=eliminated, >=inlined")
    lines.append("")

    total = len([m for m in table.mappings if m.llil_text or m.mlil_text])
    if total > 0:
        eq_count = sum(1 for m in table.mappings if m.status == MappingStatus.EQUIVALENT and (m.llil_text or m.mlil_text))
        tr_count = sum(1 for m in table.mappings if m.status == MappingStatus.TRANSFORMED and (m.llil_text or m.mlil_text))
        df_count = sum(1 for m in table.mappings if m.status == MappingStatus.DIFFERENT and (m.llil_text or m.mlil_text))
        el_count = sum(1 for m in table.mappings if m.status == MappingStatus.ELIMINATED and (m.llil_text or m.mlil_text))
        in_count = sum(1 for m in table.mappings if m.status == MappingStatus.INLINED and (m.llil_text or m.mlil_text))

        lines.append(f"Summary: {total} SCP offsets mapped")
        lines.append(f"  = Equivalent:  {eq_count:3d} ({eq_count/total*100:5.1f}%)")
        lines.append(f"  ~ Transformed: {tr_count:3d} ({tr_count/total*100:5.1f}%)")
        lines.append(f"  ! Different:   {df_count:3d} ({df_count/total*100:5.1f}%)")
        lines.append(f"  o Eliminated:  {el_count:3d} ({el_count/total*100:5.1f}%)")
        lines.append(f"  > Inlined:     {in_count:3d} ({in_count/total*100:5.1f}%)")

    return "\n".join(lines)


def format_mlil_hlil_sync_table(
    table: AddressTable,
    use_color: bool = True
) -> str:
    """Format MLIL-HLIL synchronized table"""
    lines = []

    # Header
    lines.append(f"=== MLIL-HLIL Sync View: {table.function_name} ===")
    lines.append("")

    # Column widths
    addr_width = 12
    mlil_width = 40
    hlil_width = 40
    status_width = 12

    # Table header
    header = (
        f"{'SCP Offset':<{addr_width}} | "
        f"{'MLIL Instruction':<{mlil_width}} | "
        f"{'HLIL Statement':<{hlil_width}} | "
        f"{'Status':<{status_width}}"
    )
    separator = "-" * len(header)

    lines.append(header)
    lines.append(separator)

    # Table rows (only MLIL-HLIL, skip eliminated)
    for mapping in table.sorted_by_offset():
        if not mapping.mlil_text and not mapping.hlil_text:
            continue

        if mapping.status == MappingStatus.ELIMINATED:
            continue

        symbol, label = STATUS_SYMBOLS.get(mapping.status, ("?", "???"))

        if use_color:
            if mapping.status == MappingStatus.EQUIVALENT:
                color = COLORS['green']

            elif mapping.status == MappingStatus.TRANSFORMED:
                color = COLORS['yellow']

            elif mapping.status == MappingStatus.DIFFERENT:
                color = COLORS['red']

            else:
                color = COLORS['gray']
            reset = COLORS['reset']

        else:
            color = ""
            reset = ""

        mlil_text = truncate_text(mapping.mlil_text or "(none)", mlil_width)
        hlil_text = truncate_text(mapping.hlil_text or "(none)", hlil_width)
        addr_str = f"0x{mapping.scp_offset:08X}"

        row = (
            f"{color}{addr_str:<{addr_width}} | "
            f"{mlil_text:<{mlil_width}} | "
            f"{hlil_text:<{hlil_width}} | "
            f"{symbol} {label}{reset}"
        )
        lines.append(row)

    lines.append(separator)

    # Legend and summary
    lines.append("")
    lines.append("Legend: ==equivalent, ~=transformed, !=different, o=eliminated, >=inlined")
    lines.append("")

    total = len([m for m in table.mappings if m.mlil_text or m.hlil_text])
    if total > 0:
        eq_count = sum(1 for m in table.mappings if m.status == MappingStatus.EQUIVALENT and (m.mlil_text or m.hlil_text))
        tr_count = sum(1 for m in table.mappings if m.status == MappingStatus.TRANSFORMED and (m.mlil_text or m.hlil_text))
        df_count = sum(1 for m in table.mappings if m.status == MappingStatus.DIFFERENT and (m.mlil_text or m.hlil_text))
        el_count = sum(1 for m in table.mappings if m.status == MappingStatus.ELIMINATED and (m.mlil_text or m.hlil_text))
        in_count = sum(1 for m in table.mappings if m.status == MappingStatus.INLINED and (m.mlil_text or m.hlil_text))

        lines.append(f"Summary: {total} SCP offsets mapped")
        lines.append(f"  = Equivalent:  {eq_count:3d} ({eq_count/total*100:5.1f}%)")
        lines.append(f"  ~ Transformed: {tr_count:3d} ({tr_count/total*100:5.1f}%)")
        lines.append(f"  ! Different:   {df_count:3d} ({df_count/total*100:5.1f}%)")
        lines.append(f"  o Eliminated:  {el_count:3d} ({el_count/total*100:5.1f}%)")
        lines.append(f"  > Inlined:     {in_count:3d} ({in_count/total*100:5.1f}%)")

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
  %(prog)s --compare llil mlil --sync test.scp  # IDA-style view
  %(prog)s --all --sync test.scp                # Three-way IDA-style view
  %(prog)s --batch test.scp
  %(prog)s --variables test.scp
  %(prog)s --types test.scp
  %(prog)s --function func_001 --sync test.scp
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
    parser.add_argument('--sync', action='store_true',
                        help='T014: IDA-style synchronized view (address-based)')
    parser.add_argument('--format', choices=['table', 'ide', 'grep'],
                        default='table',
                        help='T052-T053: Output format (table=default, ide=single-line, grep=grep-friendly)')
    parser.add_argument('--output', metavar='FILE',
                        help='T056: Write results to file instead of stdout')
    parser.add_argument('--show-scp', action='store_true',
                        help='T059: Show SCP opcode column in sync view')
    parser.add_argument('--export-regression', metavar='FILE',
                        help='T065: Export validation results for regression baseline')
    parser.add_argument('--compare-regression', metavar='FILE',
                        help='T066: Compare current results against baseline')
    parser.add_argument('--failed-only', action='store_true',
                        help='T040: Show details only for failed functions in batch mode')
    parser.add_argument('--show-first', metavar='N', type=int, default=0,
                        help='T041: Show first N differences per function (0=all)')

    # T084-T089: Phase 15 - Navigation and filtering options
    parser.add_argument('--range', metavar='START-END',
                        help='T084: Show only specific SCP offset range (e.g., 0x1000-0x2000)')
    parser.add_argument('--export-map', metavar='FILE',
                        help='T085: Export address mapping as JSON for external tools')
    parser.add_argument('--goto', metavar='OFFSET',
                        help='T086: Show context around specific SCP address (e.g., 0x1234)')
    parser.add_argument('--context', metavar='N', type=int, default=3,
                        help='T087: Lines of context around --goto address (default: 3)')
    parser.add_argument('--filter', choices=['equivalent', 'transformed', 'different', 'eliminated'],
                        help='T088: Filter output by mapping status')
    parser.add_argument('--reverse-lookup', metavar='TEXT',
                        help='T089: Find SCP address from HLIL/MLIL text')
    parser.add_argument('--output-dir', metavar='DIR',
                        help='Output both LLIL-MLIL and MLIL-HLIL comparisons to separate files in DIR')

    return parser


def main() -> int:
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # T077-T081: Reset error handler for new session
    reset_error_handler()

    # Validate SCP file exists
    scp_path = Path(args.scp_file)
    if not scp_path.exists():
        print(f"Error: SCP file not found: {scp_path}", file=sys.stderr)
        return 1

    # Load and generate IR
    pipeline = IRPipeline(str(scp_path))
    if not pipeline.load():
        # T077: SCP parse failure - show error summary and exit
        error_handler = get_error_handler()
        error_handler.print_summary(use_color=not args.no_color)
        return 1

    if not pipeline.generate_ir(args.function):
        print("Error: Failed to generate IR for any function", file=sys.stderr)
        # T078-T080: Show error summary for conversion failures
        error_handler = get_error_handler()
        error_handler.print_summary(use_color=not args.no_color)
        return 1

    use_color = not args.no_color and not args.json

    # T064: Handle --json output
    if args.json:
        return _run_json_output(pipeline, args, str(scp_path))

    # T065-T066: Handle regression testing
    export_regression = getattr(args, 'export_regression', None)
    compare_regression = getattr(args, 'compare_regression', None)

    if export_regression:
        tester = RegressionTester(JSONReporter())
        success = tester.export_baseline(pipeline, str(scp_path), export_regression)
        if success:
            print(f"Regression baseline exported to: {export_regression}")
            return 0
        return 1

    if compare_regression:
        tester = RegressionTester(JSONReporter())
        return tester.compare_baseline(pipeline, str(scp_path), compare_regression)

    # T052-T054: Handle --format ide/grep
    output_format = getattr(args, 'format', 'table')
    output_path = getattr(args, 'output', None)

    if output_format == 'ide':
        compare_layers = None
        if args.compare:
            compare_layers = (args.compare[0].lower(), args.compare[1].lower())
        return _run_ide_format(pipeline, str(scp_path), output_path, compare_layers)

    if output_format == 'grep':
        compare_layers = None
        if args.compare:
            compare_layers = (args.compare[0].lower(), args.compare[1].lower())
        return _run_grep_format(pipeline, output_path, compare_layers)

    # Standard table format output
    print(f"=== IR Semantic Validation Report ===")
    print(f"File: {scp_path.name}")
    print(f"Functions loaded: {len(pipeline.functions)}")
    print()

    # Clear tracking for unknown operations
    clear_unknown_operations()

    # T026: Handle --compare llil mlil
    if args.compare:
        layer1, layer2 = args.compare[0].lower(), args.compare[1].lower()

        # T014: Use --sync mode for IDA-style output
        use_sync = getattr(args, 'sync', False)

        # T059: Get show_scp flag
        show_scp = getattr(args, 'show_scp', False)

        if layer1 == 'llil' and layer2 == 'mlil':
            if use_sync:
                result = _run_llil_mlil_sync(pipeline, use_color, show_scp)

            else:
                result = _run_llil_mlil_comparison(pipeline, use_color)
            report_unknown_operations()  # T068
            return result

        elif layer1 == 'mlil' and layer2 == 'hlil':
            if use_sync:
                result = _run_mlil_hlil_sync(pipeline, use_color, show_scp)

            else:
                result = _run_mlil_hlil_comparison(pipeline, use_color)
            report_unknown_operations()  # T068
            return result

        else:
            print(f"Error: Unsupported comparison: {layer1} vs {layer2}", file=sys.stderr)
            print("Supported: llil mlil, mlil hlil", file=sys.stderr)
            return 1

    # T056: Handle --all (three-way comparison)
    if args.all:
        use_sync = getattr(args, 'sync', False)
        if use_sync:
            result = _run_three_way_sync(pipeline, use_color)

        else:
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
        failed_only = getattr(args, 'failed_only', False)
        show_first = getattr(args, 'show_first', 0)
        show_scp = getattr(args, 'show_scp', False)
        result = _run_batch_validation(pipeline, use_color, failed_only, show_first, show_scp)
        report_unknown_operations()  # T068
        return result

    # T085: Handle --export-map
    export_map = getattr(args, 'export_map', None)
    if export_map:
        success = export_address_map(pipeline, export_map)
        if success:
            print(f"Address map exported to: {export_map}")
            return 0
        return 1

    # T086-T087: Handle --goto
    goto_offset_str = getattr(args, 'goto', None)
    if goto_offset_str:
        goto_offset = parse_offset(goto_offset_str)
        if goto_offset is None:
            print(f"Invalid offset format: {goto_offset_str}", file=sys.stderr)
            return 1
        context_lines = getattr(args, 'context', 3)
        return _run_goto_display(pipeline, goto_offset, context_lines, use_color)

    # T089: Handle --reverse-lookup
    reverse_lookup = getattr(args, 'reverse_lookup', None)
    if reverse_lookup:
        return _run_reverse_lookup(pipeline, reverse_lookup, use_color)

    # Handle --output-dir: output both comparisons to separate files
    output_dir = getattr(args, 'output_dir', None)
    if output_dir:
        return _run_dual_output(pipeline, output_dir, str(scp_path))

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

    # T077-T081: Show any error/warning summary at the end
    error_handler = get_error_handler()
    error_handler.print_summary(use_color=use_color)

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

    return 0


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

    return 0


def _run_ide_format(
    pipeline: IRPipeline, scp_path: str, output_path: Optional[str],
    compare_layers: Optional[Tuple[str, str]] = None
) -> int:
    """T052: Run validation with IDE-friendly output format"""
    formatter = IDEFormatter()
    total_different = 0

    with OutputWriter(output_path) as writer:
        for func_name, llil, mlil, hlil in pipeline.functions:
            results: List[ComparisonResult] = []

            # Determine which layers to compare
            if compare_layers:
                layer1, layer2 = compare_layers
                if layer1 == 'llil' and layer2 == 'mlil':
                    comp = LLILMLILComparator(llil, mlil)
                    results = comp.compare()

                elif layer1 == 'mlil' and layer2 == 'hlil':
                    comp = MLILHLILComparator(mlil, hlil)
                    results = comp.compare()

            else:
                # Default: LLIL-MLIL comparison
                comp = LLILMLILComparator(llil, mlil)
                results = comp.compare()

            # Count statistics
            equivalent = sum(1 for r in results if r.status == ComparisonStatus.EQUIVALENT)
            transformed = sum(1 for r in results if r.status == ComparisonStatus.TRANSFORMED)
            different = sum(1 for r in results if r.status == ComparisonStatus.DIFFERENT)
            total_different += different

            # Output non-equivalent results
            lines = formatter.format_results(scp_path, func_name, results)
            writer.writelines(lines)

            # Output summary line
            writer.write(formatter.format_summary(scp_path, func_name, equivalent, transformed, different))

    return 0


def _run_grep_format(
    pipeline: IRPipeline, output_path: Optional[str],
    compare_layers: Optional[Tuple[str, str]] = None,
    show_all: bool = False
) -> int:
    """T053: Run validation with grep-friendly output format"""
    formatter = GrepFormatter()
    total_different = 0

    with OutputWriter(output_path) as writer:
        for func_name, llil, mlil, hlil in pipeline.functions:
            results: List[ComparisonResult] = []

            # Determine which layers to compare
            if compare_layers:
                layer1, layer2 = compare_layers
                if layer1 == 'llil' and layer2 == 'mlil':
                    comp = LLILMLILComparator(llil, mlil)
                    results = comp.compare()

                elif layer1 == 'mlil' and layer2 == 'hlil':
                    comp = MLILHLILComparator(mlil, hlil)
                    results = comp.compare()

            else:
                # Default: LLIL-MLIL comparison
                comp = LLILMLILComparator(llil, mlil)
                results = comp.compare()

            # Count statistics
            equivalent = sum(1 for r in results if r.status == ComparisonStatus.EQUIVALENT)
            transformed = sum(1 for r in results if r.status == ComparisonStatus.TRANSFORMED)
            different = sum(1 for r in results if r.status == ComparisonStatus.DIFFERENT)
            total_different += different

            # Output results
            lines = formatter.format_results(func_name, results, show_all=show_all)
            writer.writelines(lines)

            # Output summary line
            writer.write(formatter.format_summary(func_name, equivalent, transformed, different))

    return 0


# =============================================================================
# T015-T022: IDA-Style Sync Functions
# =============================================================================

def _run_llil_mlil_sync(pipeline: IRPipeline, use_color: bool, show_scp: bool = False) -> int:
    """T022: Run LLIL-MLIL sync view for all functions (IDA-style)"""
    total_passed = 0
    total_failed = 0

    for func_name, llil, mlil, hlil in pipeline.functions:
        # Build address table from comparison results
        table = build_llil_mlil_address_table(func_name, llil, mlil, pipeline, show_scp)

        # Format and print
        report = format_llil_mlil_sync_table(table, use_color, show_scp)
        print(report)
        print()

        # Count differences
        if table.different_count == 0:
            total_passed += 1

        else:
            total_failed += 1

    print("=" * 50)
    print(f"Overall: {total_passed} passed, {total_failed} failed")

    return 0


def build_llil_mlil_address_table(
    func_name: str,
    llil_func: LowLevelILFunction,
    mlil_func: MediumLevelILFunction,
    pipeline: Optional[IRPipeline] = None,
    show_scp: bool = False
) -> AddressTable:
    """Build address table by matching LLIL and MLIL using CFG comparison"""
    table = AddressTable(function_name=func_name)

    # Build address-to-instruction maps for direct str() access
    llil_addr_to_instrs: Dict[int, List[LowLevelILInstruction]] = {}
    for bb in llil_func.basic_blocks:
        for instr in bb.instructions:
            addr = instr.address
            if addr not in llil_addr_to_instrs:
                llil_addr_to_instrs[addr] = []
            llil_addr_to_instrs[addr].append(instr)

    mlil_addr_to_instrs: Dict[int, List[MediumLevelILInstruction]] = {}
    for bb in mlil_func.basic_blocks:
        for instr in bb.instructions:
            addr = instr.address
            if addr not in mlil_addr_to_instrs:
                mlil_addr_to_instrs[addr] = []
            mlil_addr_to_instrs[addr].append(instr)

    # Get comparison results from existing comparator
    comparator = LLILMLILComparator(llil_func, mlil_func)
    results = comparator.compare()

    # Group results by SCP address for status tracking
    addr_to_results: Dict[int, List[ComparisonResult]] = {}
    for result in results:
        addr = result.source_location.scp_offset
        if addr not in addr_to_results:
            addr_to_results[addr] = []
        addr_to_results[addr].append(result)

    # Collect all addresses from both LLIL and MLIL
    all_addresses = set(llil_addr_to_instrs.keys()) | set(mlil_addr_to_instrs.keys())

    # Build address mappings
    for addr in sorted(all_addresses):
        mapping = AddressMapping(scp_offset=addr)

        # T058: Get SCP opcode if requested
        if show_scp and pipeline:
            mapping.scp_opcode = pipeline.get_scp_opcode(func_name, addr)

        # Get LLIL names (opcode only, no operands)
        llil_instrs = llil_addr_to_instrs.get(addr, [])
        llil_texts = [_get_llil_name(instr) for instr in llil_instrs]

        # Get MLIL names (opcode only, no operands)
        mlil_instrs = mlil_addr_to_instrs.get(addr, [])
        mlil_texts = [_get_mlil_name(instr) for instr in mlil_instrs]

        mapping.llil_text = "; ".join(llil_texts) if llil_texts else ""
        mapping.mlil_text = "; ".join(mlil_texts) if mlil_texts else "(eliminated)"

        # Determine status from comparison results
        has_different = False
        has_transformed = False
        for result in addr_to_results.get(addr, []):
            if result.status == ComparisonStatus.DIFFERENT:
                has_different = True

            elif result.status == ComparisonStatus.TRANSFORMED:
                has_transformed = True

        # Set status
        if has_different:
            mapping.status = MappingStatus.DIFFERENT

        elif has_transformed:
            mapping.status = MappingStatus.TRANSFORMED

        elif mapping.mlil_text == "(eliminated)":
            mapping.status = MappingStatus.ELIMINATED

        else:
            mapping.status = MappingStatus.EQUIVALENT

        table.add(mapping)

    return table


def _run_mlil_hlil_sync(pipeline: IRPipeline, use_color: bool, show_scp: bool = False) -> int:
    """Run MLIL-HLIL sync view for all functions (IDA-style)"""
    total_passed = 0
    total_failed = 0

    for func_name, llil, mlil, hlil in pipeline.functions:
        # Build address table
        aligner = AddressAligner(llil, mlil, hlil)
        table = aligner.build_address_table()

        # Format and print
        report = format_mlil_hlil_sync_table(table, use_color)
        print(report)
        print()

        # Count differences
        if table.different_count == 0:
            total_passed += 1

        else:
            total_failed += 1

    print("=" * 50)
    print(f"Overall: {total_passed} passed, {total_failed} failed")

    return 0


def _run_three_way_sync(pipeline: IRPipeline, use_color: bool) -> int:
    """T042: Run three-way sync view for all functions (IDA-style)"""
    total_passed = 0
    total_failed = 0

    for func_name, llil, mlil, hlil in pipeline.functions:
        # Build address table using comparators
        table = build_three_way_address_table(func_name, llil, mlil, hlil)

        # Format and print full three-way table
        report = format_sync_table(table, use_color)
        print(report)
        print()

        # Count differences
        if table.different_count == 0:
            total_passed += 1

        else:
            total_failed += 1

    print("=" * 50)
    print(f"Overall: {total_passed} passed, {total_failed} failed")

    return 0


def build_three_way_address_table(
    func_name: str,
    llil_func: LowLevelILFunction,
    mlil_func: MediumLevelILFunction,
    hlil_func: HighLevelILFunction
) -> AddressTable:
    """Build address table for three-way comparison"""
    table = AddressTable(function_name=func_name)

    # Get LLIL-MLIL comparison results
    llil_mlil_comp = LLILMLILComparator(llil_func, mlil_func)
    llil_mlil_results = llil_mlil_comp.compare()

    # Get MLIL-HLIL comparison results
    mlil_hlil_comp = MLILHLILComparator(mlil_func, hlil_func)
    mlil_hlil_results = mlil_hlil_comp.compare()

    # Collect all addresses from LLIL
    addr_to_data: Dict[int, Dict[str, Any]] = {}
    for bb in llil_func.basic_blocks:
        for instr in bb.instructions:
            addr = instr.address
            if addr not in addr_to_data:
                addr_to_data[addr] = {
                    'llil_ops': [],
                    'mlil_ops': [],
                    'hlil_ops': [],
                    'llil_mlil_status': None,
                    'mlil_hlil_status': None,
                }
            addr_to_data[addr]['llil_ops'].append(instr.operation.name.replace('LLIL_', ''))

    # Add LLIL-MLIL results
    for result in llil_mlil_results:
        addr = result.source_location.scp_offset
        if addr not in addr_to_data:
            addr_to_data[addr] = {
                'llil_ops': [],
                'mlil_ops': [],
                'hlil_ops': [],
                'llil_mlil_status': None,
                'mlil_hlil_status': None,
            }
        if result.source_op:
            addr_to_data[addr]['llil_ops'].append(str(result.source_op))
        if result.target_op:
            addr_to_data[addr]['mlil_ops'].append(str(result.target_op))
        addr_to_data[addr]['llil_mlil_status'] = result.status

    # Add MLIL-HLIL results (these may have different addresses)
    for result in mlil_hlil_results:
        addr = result.source_location.scp_offset
        if addr not in addr_to_data:
            addr_to_data[addr] = {
                'llil_ops': [],
                'mlil_ops': [],
                'hlil_ops': [],
                'llil_mlil_status': None,
                'mlil_hlil_status': None,
            }
        if result.source_op:
            addr_to_data[addr]['mlil_ops'].append(str(result.source_op))
        if result.target_op:
            addr_to_data[addr]['hlil_ops'].append(str(result.target_op))
        addr_to_data[addr]['mlil_hlil_status'] = result.status

    # Build address mappings
    for addr in sorted(addr_to_data.keys()):
        data = addr_to_data[addr]
        mapping = AddressMapping(scp_offset=addr)

        mapping.llil_text = "; ".join(data['llil_ops']) if data['llil_ops'] else ""
        mapping.mlil_text = "; ".join(data['mlil_ops']) if data['mlil_ops'] else "(eliminated)"
        mapping.hlil_text = "; ".join(data['hlil_ops']) if data['hlil_ops'] else "(eliminated)"

        # Determine status based on comparisons
        llil_mlil_stat = data['llil_mlil_status']
        mlil_hlil_stat = data['mlil_hlil_status']

        if llil_mlil_stat == ComparisonStatus.DIFFERENT or mlil_hlil_stat == ComparisonStatus.DIFFERENT:
            mapping.status = MappingStatus.DIFFERENT

        elif llil_mlil_stat == ComparisonStatus.TRANSFORMED or mlil_hlil_stat == ComparisonStatus.TRANSFORMED:
            mapping.status = MappingStatus.TRANSFORMED

        elif mapping.mlil_text == "(eliminated)" or mapping.hlil_text == "(eliminated)":
            mapping.status = MappingStatus.ELIMINATED

        else:
            mapping.status = MappingStatus.EQUIVALENT

        table.add(mapping)

    return table


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


def _run_batch_validation(
    pipeline: IRPipeline, use_color: bool,
    failed_only: bool = False, show_first: int = 0,
    show_scp: bool = False
) -> int:
    """T042-T046: Run batch validation for all functions

    T040: --failed-only shows details only for failed functions
    T041: --show-first N shows first N differences per function
    """
    reporter = TextReporter(use_color)
    results: List[FunctionResult] = []
    detailed_results: Dict[str, Tuple[List[ComparisonResult], List[ComparisonResult]]] = {}

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

        # Store detailed results for failed functions
        if failed_only and (llil_mlil_diff > 0 or mlil_hlil_diff > 0):
            detailed_results[func_name] = (llil_mlil_results, mlil_hlil_results)

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

    # T040: Show detailed results for failed functions
    if failed_only and detailed_results:
        print()
        print("=== Detailed Results for Failed Functions ===")

        for func_name, (llil_mlil_results, mlil_hlil_results) in detailed_results.items():
            print()
            print(f"--- {func_name} ---")

            # Show LLIL-MLIL differences
            llil_diff_results = [r for r in llil_mlil_results if r.status == ComparisonStatus.DIFFERENT]
            if llil_diff_results:
                print("LLIL-MLIL differences:")
                diff_to_show = llil_diff_results[:show_first] if show_first > 0 else llil_diff_results
                for r in diff_to_show:
                    loc = r.source_location
                    offset = f"0x{loc.scp_offset:08X}" if loc.scp_offset else "unknown"
                    print(f"  [{offset}] {r.explanation}")
                if show_first > 0 and len(llil_diff_results) > show_first:
                    print(f"  ... and {len(llil_diff_results) - show_first} more")

            # Show MLIL-HLIL differences
            mlil_diff_results = [r for r in mlil_hlil_results if r.status == ComparisonStatus.DIFFERENT]
            if mlil_diff_results:
                print("MLIL-HLIL differences:")
                diff_to_show = mlil_diff_results[:show_first] if show_first > 0 else mlil_diff_results
                for r in diff_to_show:
                    loc = r.source_location
                    offset = f"0x{loc.scp_offset:08X}" if loc.scp_offset else "unknown"
                    print(f"  [{offset}] {r.explanation}")
                if show_first > 0 and len(mlil_diff_results) > show_first:
                    print(f"  ... and {len(mlil_diff_results) - show_first} more")

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
# T067-T076: Phase 13 - Edge Case Handling (from spec.md)
# =============================================================================

class EdgeCaseClassifier:
    """T067-T076: Classifies and handles edge cases in IR comparison

    Edge cases from spec.md:
    - T067: Dead code elimination (HLIL has fewer instructions)
    - T068: Constant propagation (variable optimized away)
    - T069: Expression inlining (separate MLIL -> single HLIL)
    - T070: SSA phi nodes (MLIL SSA vs non-SSA LLIL)
    - T071: Control flow restructuring (GOTO -> while/for/if)
    - T072: Global variable naming (compare by resolved address)
    - T073: LLIL lacking type info (only fail if MLIL vs HLIL conflict)
    - T074: Compound expression trees (a + b * c with different nesting)
    - T075: Syscall signature differences
    - T076: String constants from pool
    """

    def __init__(self):
        self.edge_case_stats: Dict[str, int] = {
            'dead_code': 0,
            'constant_prop': 0,
            'inlined': 0,
            'phi_node': 0,
            'cf_restructure': 0,
            'global_naming': 0,
            'type_mismatch': 0,
            'expr_tree': 0,
            'syscall_sig': 0,
            'string_pool': 0
        }

    def classify_elimination(
        self,
        source_op: Optional[SemanticOperation],
        target_op: Optional[SemanticOperation]
    ) -> Tuple[MappingStatus, str]:
        """T067: Classify eliminated operations as ELIMINATED, not DIFFERENT"""
        if source_op and target_op is None:
            # Source exists, target eliminated
            if self._is_dead_code_candidate(source_op):
                self.edge_case_stats['dead_code'] += 1
                return MappingStatus.ELIMINATED, "Dead code eliminated"

            if self._is_constant_prop_candidate(source_op):
                self.edge_case_stats['constant_prop'] += 1
                return MappingStatus.ELIMINATED, "Constant propagation"

        return MappingStatus.DIFFERENT, ""

    def _is_dead_code_candidate(self, op: SemanticOperation) -> bool:
        """T067: Check if operation is likely dead code"""
        # NOP operations, debug statements, unreachable code
        if op.kind == OperationKind.NOP:
            return True

        # Stack frame setup that gets optimized
        if op.operator in ('PUSH_CALLER_FRAME', 'PUSH_RET_ADDR', 'POP'):
            return True

        return False

    def _is_constant_prop_candidate(self, op: SemanticOperation) -> bool:
        """T068: Check if operation is constant propagation candidate"""
        # Load of constant that gets inlined
        if op.kind == OperationKind.LOAD:
            return any(o.kind == 'const' for o in op.operands)
        return False

    def classify_inlining(
        self,
        source_ops: List[SemanticOperation],
        target_op: SemanticOperation
    ) -> Tuple[MappingStatus, str]:
        """T069: Classify multiple source ops merged into single target"""
        if len(source_ops) > 1:
            # Multiple operations merged into one expression
            self.edge_case_stats['inlined'] += 1
            source_descs = ', '.join(op.operator for op in source_ops)
            return MappingStatus.INLINED, f"Inlined: {source_descs} -> {target_op.operator}"

        return MappingStatus.TRANSFORMED, ""

    def is_phi_node(self, op: SemanticOperation) -> bool:
        """T070: Check if operation is an SSA phi node"""
        # Phi nodes in SSA form - specific to MLIL SSA
        return op.operator in ('PHI', 'MLIL_VAR_PHI', 'VAR_PHI')

    def handle_phi_node(
        self,
        source_op: SemanticOperation,
        target_op: SemanticOperation
    ) -> Tuple[ComparisonStatus, str]:
        """T070: Handle SSA phi nodes - they don't exist in non-SSA LLIL"""
        if self.is_phi_node(target_op) and not self.is_phi_node(source_op):
            self.edge_case_stats['phi_node'] += 1
            return ComparisonStatus.TRANSFORMED, "SSA phi node introduced"

        return ComparisonStatus.DIFFERENT, ""

    def classify_control_flow(
        self,
        source_op: SemanticOperation,
        target_op: SemanticOperation
    ) -> Tuple[ComparisonStatus, str]:
        """T071: Classify control flow restructuring as TRANSFORMED"""
        cf_transforms = {
            ('GOTO', 'WHILE'): "Back-edge goto -> while loop",
            ('GOTO', 'FOR'): "Structured goto -> for loop",
            ('BRANCH', 'IF'): "Branch -> if statement",
            ('JMP', 'GOTO'): "Jump -> goto",
            ('IF', 'WHILE'): "Conditional -> while loop",
            ('IF', 'FOR'): "Conditional -> for loop",
        }

        key = (source_op.operator, target_op.operator)
        if key in cf_transforms:
            self.edge_case_stats['cf_restructure'] += 1
            return ComparisonStatus.TRANSFORMED, cf_transforms[key]

        return ComparisonStatus.DIFFERENT, ""

    def compare_globals_by_address(
        self,
        source_op: SemanticOperation,
        target_op: SemanticOperation
    ) -> bool:
        """T072: Compare global variables by resolved address, not name"""
        # Extract global references
        source_globals = [o for o in source_op.operands if o.kind == 'global']
        target_globals = [o for o in target_op.operands if o.kind == 'global']

        if len(source_globals) != len(target_globals):
            return False

        for src, tgt in zip(source_globals, target_globals):
            # Compare by value (address), not by name
            if src.value != tgt.value:
                return False

        self.edge_case_stats['global_naming'] += 1
        return True

    def validate_types(
        self,
        llil_type: Optional[str],
        mlil_type: Optional[str],
        hlil_type: Optional[str]
    ) -> Tuple[bool, str]:
        """T073: Validate types - only fail if MLIL vs HLIL types conflict

        LLIL often lacks type info, so we only fail if MLIL and HLIL both
        have types but they conflict.
        """
        # LLIL lacking type is OK
        if llil_type is None:
            # Only check MLIL vs HLIL
            if mlil_type and hlil_type:
                if not self._types_compatible(mlil_type, hlil_type):
                    self.edge_case_stats['type_mismatch'] += 1
                    return False, f"Type mismatch: MLIL={mlil_type}, HLIL={hlil_type}"
            return True, ""

        # All three present - check chain
        if mlil_type and not self._types_compatible(llil_type, mlil_type):
            return False, f"LLIL-MLIL type mismatch: {llil_type} vs {mlil_type}"

        if mlil_type and hlil_type and not self._types_compatible(mlil_type, hlil_type):
            self.edge_case_stats['type_mismatch'] += 1
            return False, f"MLIL-HLIL type mismatch: {mlil_type} vs {hlil_type}"

        return True, ""

    def _types_compatible(self, type1: str, type2: str) -> bool:
        """T073: Check type compatibility with common equivalences"""
        # Normalize types
        type1 = type1.lower().strip()
        type2 = type2.lower().strip()

        if type1 == type2:
            return True

        # Common compatible type pairs
        compatible_pairs = {
            ('int', 'int32'), ('int', 'value32'), ('int32', 'value32'),
            ('str', 'string'), ('bool', 'int'), ('bool', 'int32'),
            ('float', 'float32'), ('float', 'value32'),
            ('void', 'none'), ('ptr', 'pointer'),
        }

        pair = (type1, type2)
        return pair in compatible_pairs or (type2, type1) in compatible_pairs

    def compare_expression_structure(
        self,
        source_op: SemanticOperation,
        target_op: SemanticOperation
    ) -> Tuple[bool, str]:
        """T074: Compare compound expressions with different tree structures

        Example: (a + b) * c vs a + (b * c) - semantically different
        But: (a + b) vs (b + a) - semantically equivalent for commutative ops
        """
        if source_op.kind != OperationKind.ARITHMETIC:
            return True, ""

        # For nested expressions, check semantic equivalence
        # This is handled by compare_expression_trees() for simple cases
        # For complex nested cases, we check operator precedence preservation

        source_depth = self._expression_depth(source_op)
        target_depth = self._expression_depth(target_op)

        if abs(source_depth - target_depth) > 1:
            self.edge_case_stats['expr_tree'] += 1
            return False, f"Expression tree depth mismatch: {source_depth} vs {target_depth}"

        return True, ""

    def _expression_depth(self, op: SemanticOperation) -> int:
        """T074: Calculate expression tree depth"""
        max_depth = 0
        for operand in op.operands:
            if operand.kind == 'expr':
                # Nested expression - would need recursive check
                max_depth = max(max_depth, 1)
        return max_depth + 1

    def compare_syscall_signatures(
        self,
        source_op: SemanticOperation,
        target_op: SemanticOperation
    ) -> Tuple[bool, str]:
        """T075: Handle syscall signature differences across layers"""
        if source_op.kind != OperationKind.CALL or target_op.kind != OperationKind.CALL:
            return True, ""

        if source_op.operator not in ('SYSCALL', 'CALL_SCRIPT'):
            return True, ""

        # Compare syscall target (subsystem, cmd) but allow arg count differences
        # as HLIL may inline constant arguments

        source_target = source_op.operands[0] if source_op.operands else None
        target_target = target_op.operands[0] if target_op.operands else None

        if source_target and target_target:
            if source_target.value != target_target.value:
                self.edge_case_stats['syscall_sig'] += 1
                return False, f"Syscall target mismatch: {source_target.value} vs {target_target.value}"

        return True, ""

    def validate_string_constants(
        self,
        source_op: SemanticOperation,
        target_op: SemanticOperation
    ) -> Tuple[bool, str]:
        """T076: Validate string constants from pool (Edge Case 9)"""
        source_strings = [o for o in source_op.operands if o.kind == 'const' and isinstance(o.value, str)]
        target_strings = [o for o in target_op.operands if o.kind == 'const' and isinstance(o.value, str)]

        if len(source_strings) != len(target_strings):
            self.edge_case_stats['string_pool'] += 1
            return False, "String constant count mismatch"

        for src, tgt in zip(source_strings, target_strings):
            if src.value != tgt.value:
                self.edge_case_stats['string_pool'] += 1
                return False, f"String mismatch: {repr(src.value)} vs {repr(tgt.value)}"

        return True, ""

    def get_stats(self) -> Dict[str, int]:
        """Get edge case statistics"""
        return dict(self.edge_case_stats)


def classify_with_edge_cases(
    source_op: Optional[SemanticOperation],
    target_op: Optional[SemanticOperation],
    classifier: EdgeCaseClassifier,
    var_mapping: Dict[str, str],
    layer_transforms: List[TransformationRule]
) -> Tuple[ComparisonStatus, str]:
    """T067-T076: Enhanced classification with edge case handling"""

    # Handle eliminated operations
    if target_op is None:
        status, explanation = classifier.classify_elimination(source_op, None)
        if status == MappingStatus.ELIMINATED:
            return ComparisonStatus.TRANSFORMED, explanation
        return ComparisonStatus.DIFFERENT, "Target operation missing"

    if source_op is None:
        return ComparisonStatus.TRANSFORMED, "Source operation missing (added in target)"

    # T070: Handle phi nodes
    if classifier.is_phi_node(source_op) or classifier.is_phi_node(target_op):
        status, explanation = classifier.handle_phi_node(source_op, target_op)
        if status != ComparisonStatus.DIFFERENT:
            return status, explanation

    # T071: Control flow restructuring
    if source_op.kind == OperationKind.BRANCH or target_op.kind == OperationKind.CONTROL_FLOW:
        status, explanation = classifier.classify_control_flow(source_op, target_op)
        if status != ComparisonStatus.DIFFERENT:
            return status, explanation

    # T072: Global variable comparison by address
    if any(o.kind == 'global' for o in source_op.operands):
        if classifier.compare_globals_by_address(source_op, target_op):
            return ComparisonStatus.EQUIVALENT, "Globals match by address"

    # T074: Expression structure comparison
    valid, msg = classifier.compare_expression_structure(source_op, target_op)
    if not valid:
        return ComparisonStatus.DIFFERENT, msg

    # T075: Syscall signature comparison
    valid, msg = classifier.compare_syscall_signatures(source_op, target_op)
    if not valid:
        return ComparisonStatus.DIFFERENT, msg

    # T076: String constant validation
    valid, msg = classifier.validate_string_constants(source_op, target_op)
    if not valid:
        return ComparisonStatus.DIFFERENT, msg

    # Fall back to standard classification
    return classify_difference(source_op, target_op, var_mapping, layer_transforms), ""


# =============================================================================
# T065-T066: Phase 12 - Regression Testing Support (FR-021)
# =============================================================================

class RegressionTester:
    """T065-T066: Regression testing support for validation results"""

    def __init__(self, json_reporter: JSONReporter):
        self.json_reporter = json_reporter

    def export_baseline(self, pipeline: IRPipeline, scp_path: str, output_file: str) -> bool:
        """T065: Export validation results as regression baseline"""
        function_results = []

        for func_name, llil, mlil, hlil in pipeline.functions:
            llil_mlil_comp = LLILMLILComparator(llil, mlil)
            llil_mlil_results = llil_mlil_comp.compare()

            mlil_hlil_comp = MLILHLILComparator(mlil, hlil)
            mlil_hlil_results = mlil_hlil_comp.compare()

            func_report = self.json_reporter.function_report_to_dict(
                func_name, llil_mlil_results, mlil_hlil_results
            )
            function_results.append(func_report)

        baseline = {
            "version": "1.0",
            "file": scp_path,
            "functions": function_results,
            "summary": self._compute_summary(function_results)
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(baseline, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting baseline: {e}", file=sys.stderr)
            return False

    def compare_baseline(self, pipeline: IRPipeline, scp_path: str, baseline_file: str) -> int:
        """T066: Compare current results against baseline and report differences"""
        try:
            with open(baseline_file, 'r', encoding='utf-8') as f:
                baseline = json.load(f)
        except Exception as e:
            print(f"Error loading baseline: {e}", file=sys.stderr)
            return 1

        # Generate current results
        current_results = {}
        for func_name, llil, mlil, hlil in pipeline.functions:
            llil_mlil_comp = LLILMLILComparator(llil, mlil)
            llil_mlil_results = llil_mlil_comp.compare()

            mlil_hlil_comp = MLILHLILComparator(mlil, hlil)
            mlil_hlil_results = mlil_hlil_comp.compare()

            func_report = self.json_reporter.function_report_to_dict(
                func_name, llil_mlil_results, mlil_hlil_results
            )
            current_results[func_name] = func_report

        # Build baseline map
        baseline_results = {f["function"]: f for f in baseline.get("functions", [])}

        # Compare
        regressions = []
        improvements = []

        for func_name, current in current_results.items():
            if func_name not in baseline_results:
                continue

            base = baseline_results[func_name]

            # Check LLIL-MLIL regression
            if current["llil_mlil"]["different"] > base["llil_mlil"]["different"]:
                regressions.append({
                    "function": func_name,
                    "layer": "LLIL-MLIL",
                    "baseline_diff": base["llil_mlil"]["different"],
                    "current_diff": current["llil_mlil"]["different"]
                })

            elif current["llil_mlil"]["different"] < base["llil_mlil"]["different"]:
                improvements.append({
                    "function": func_name,
                    "layer": "LLIL-MLIL",
                    "baseline_diff": base["llil_mlil"]["different"],
                    "current_diff": current["llil_mlil"]["different"]
                })

            # Check MLIL-HLIL regression
            if current["mlil_hlil"]["different"] > base["mlil_hlil"]["different"]:
                regressions.append({
                    "function": func_name,
                    "layer": "MLIL-HLIL",
                    "baseline_diff": base["mlil_hlil"]["different"],
                    "current_diff": current["mlil_hlil"]["different"]
                })

            elif current["mlil_hlil"]["different"] < base["mlil_hlil"]["different"]:
                improvements.append({
                    "function": func_name,
                    "layer": "MLIL-HLIL",
                    "baseline_diff": base["mlil_hlil"]["different"],
                    "current_diff": current["mlil_hlil"]["different"]
                })

        # Report
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

        return 0

    def _compute_summary(self, function_results: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics"""
        total = len(function_results)
        llil_mlil_passed = sum(1 for r in function_results if r["llil_mlil"]["passed"])
        mlil_hlil_passed = sum(1 for r in function_results if r["mlil_hlil"]["passed"])

        return {
            "total_functions": total,
            "llil_mlil_passed": llil_mlil_passed,
            "mlil_hlil_passed": mlil_hlil_passed
        }


# =============================================================================
# T052-T054: Phase 9 - IDE/Grep Format Output
# =============================================================================

class IDEFormatter:
    """T052: IDE-friendly single-line parseable output format.

    Output format: FILE:LINE:COL:STATUS:MESSAGE
    Example: test.scp:0x1234:0:DIFF:Expected ADD, got SUB

    This format is designed for IDE integration and can be parsed by
    most editors to jump directly to the source location.
    """

    def format_result(self, file_path: str, result: ComparisonResult) -> str:
        """Format single result as IDE-parseable line"""
        loc = result.source_location
        scp_offset = f"0x{loc.scp_offset:04X}" if loc.scp_offset else "0x0000"
        col = 0  # Column not applicable for bytecode

        # Status codes for IDE parsing
        status_map = {
            ComparisonStatus.EQUIVALENT: "OK",
            ComparisonStatus.TRANSFORMED: "WARN",
            ComparisonStatus.DIFFERENT: "ERR",
        }
        status = status_map.get(result.status, "INFO")

        # Clean message (remove newlines for single-line format)
        message = result.explanation.replace('\n', ' ').strip()

        return f"{file_path}:{scp_offset}:{col}:{status}:{message}"

    def format_results(
        self, file_path: str, func_name: str, results: List[ComparisonResult]
    ) -> List[str]:
        """Format all results for a function"""
        lines = []
        for result in results:
            if result.status != ComparisonStatus.EQUIVALENT:
                lines.append(self.format_result(file_path, result))
        return lines

    def format_summary(
        self, file_path: str, func_name: str,
        equivalent: int, transformed: int, different: int
    ) -> str:
        """Format summary line"""
        total = equivalent + transformed + different
        status = "PASS" if different == 0 else "FAIL"
        return f"{file_path}:SUMMARY:{func_name}:{status}:{total} ops, {different} diff"


class GrepFormatter:
    """T053: Grep-friendly output format for command-line filtering.

    Output format: STATUS\tLOCATION\tLAYERS\tMESSAGE
    Example: DIFF\t0x1234\tLLIL->MLIL\tBinary op mismatch: ADD vs SUB

    This format uses tabs as delimiters for easy parsing with
    cut, awk, or grep commands.
    """

    def format_result(self, result: ComparisonResult) -> str:
        """Format single result as grep-friendly line"""
        loc = result.source_location
        scp_offset = f"0x{loc.scp_offset:04X}" if loc.scp_offset else "0x0000"

        # Short status codes
        status_map = {
            ComparisonStatus.EQUIVALENT: "EQ",
            ComparisonStatus.TRANSFORMED: "XFORM",
            ComparisonStatus.DIFFERENT: "DIFF",
        }
        status = status_map.get(result.status, "UNK")

        layers = f"{result.source_layer.name}->{result.target_layer.name}"

        # Clean message
        message = result.explanation.replace('\n', ' ').replace('\t', ' ').strip()

        return f"{status}\t{scp_offset}\t{layers}\t{message}"

    def format_results(
        self, func_name: str, results: List[ComparisonResult],
        show_all: bool = False
    ) -> List[str]:
        """Format all results for a function"""
        lines = []
        lines.append(f"# Function: {func_name}")

        for result in results:
            if show_all or result.status != ComparisonStatus.EQUIVALENT:
                lines.append(self.format_result(result))

        return lines

    def format_summary(
        self, func_name: str,
        equivalent: int, transformed: int, different: int
    ) -> str:
        """Format summary line"""
        status = "PASS" if different == 0 else "FAIL"
        return f"SUMMARY\t{func_name}\t{status}\teq={equivalent},xform={transformed},diff={different}"


class OutputWriter:
    """T056: Unified output writer supporting stdout and file output."""

    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path
        self._file: Optional[TextIO] = None

    def __enter__(self) -> 'OutputWriter':
        if self.output_path:
            self._file = open(self.output_path, 'w', encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def write(self, text: str) -> None:
        """Write text to output (file or stdout)"""
        if self._file:
            self._file.write(text)
            self._file.write('\n')

        else:
            print(text)

    def writelines(self, lines: List[str]) -> None:
        """Write multiple lines to output"""
        for line in lines:
            self.write(line)


# =============================================================================
# T082-T089: Phase 15 - Navigation and Filtering
# =============================================================================

def parse_offset_range(range_str: str) -> Tuple[Optional[int], Optional[int]]:
    """T084: Parse offset range string (e.g., '0x1000-0x2000')"""
    if not range_str:
        return None, None

    parts = range_str.split('-')
    if len(parts) != 2:
        return None, None

    try:
        start = int(parts[0], 16) if parts[0].startswith('0x') else int(parts[0])
        end = int(parts[1], 16) if parts[1].startswith('0x') else int(parts[1])
        return start, end

    except ValueError:
        return None, None


def parse_offset(offset_str: str) -> Optional[int]:
    """T086: Parse single offset string (e.g., '0x1234')"""
    if not offset_str:
        return None

    try:
        return int(offset_str, 16) if offset_str.startswith('0x') else int(offset_str)

    except ValueError:
        return None


def filter_mappings_by_range(
    mappings: List[AddressMapping],
    start: Optional[int],
    end: Optional[int]
) -> List[AddressMapping]:
    """T084: Filter mappings to only include those within the specified range"""
    if start is None or end is None:
        return mappings

    return [m for m in mappings if start <= m.scp_offset <= end]


def filter_mappings_by_status(
    mappings: List[AddressMapping],
    status_filter: Optional[str]
) -> List[AddressMapping]:
    """T088: Filter mappings by status"""
    if not status_filter:
        return mappings

    status_map = {
        'equivalent': MappingStatus.EQUIVALENT,
        'transformed': MappingStatus.TRANSFORMED,
        'different': MappingStatus.DIFFERENT,
        'eliminated': MappingStatus.ELIMINATED,
    }

    target_status = status_map.get(status_filter.lower())
    if target_status is None:
        return mappings

    return [m for m in mappings if m.status == target_status]


def find_mapping_by_offset(
    mappings: List[AddressMapping],
    target_offset: int
) -> Optional[int]:
    """T086: Find index of mapping closest to target offset"""
    if not mappings:
        return None

    # Find exact match first
    for i, m in enumerate(mappings):
        if m.scp_offset == target_offset:
            return i

    # Find closest
    sorted_mappings = sorted(enumerate(mappings), key=lambda x: abs(x[1].scp_offset - target_offset))
    return sorted_mappings[0][0] if sorted_mappings else None


def get_context_range(
    mappings: List[AddressMapping],
    center_index: int,
    context_lines: int
) -> Tuple[int, int]:
    """T087: Get range of indices for context display"""
    start = max(0, center_index - context_lines)
    end = min(len(mappings), center_index + context_lines + 1)
    return start, end


def reverse_lookup_text(
    mappings: List[AddressMapping],
    search_text: str
) -> List[AddressMapping]:
    """T089: Find mappings containing the search text in HLIL/MLIL"""
    results = []
    search_lower = search_text.lower()

    for m in mappings:
        if search_lower in m.hlil_text.lower() or search_lower in m.mlil_text.lower():
            results.append(m)

    return results


def export_address_map(
    pipeline: IRPipeline,
    output_file: str
) -> bool:
    """T085: Export address mapping as JSON for external tools"""
    result = {
        "file": pipeline.scp_path,
        "functions": []
    }

    for func_name, llil, mlil, hlil in pipeline.functions:
        aligner = AddressAligner(llil, mlil, hlil)
        table = aligner.build_address_table()

        func_map = {
            "name": func_name,
            "mappings": []
        }

        for m in table.sorted_by_offset():
            func_map["mappings"].append({
                "scp_offset": f"0x{m.scp_offset:08X}",
                "scp_opcode": m.scp_opcode,
                "llil": m.llil_text,
                "mlil": m.mlil_text,
                "hlil": m.hlil_text,
                "status": m.status.name
            })

        result["functions"].append(func_map)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        return True

    except Exception as e:
        print(f"Error exporting address map: {e}", file=sys.stderr)
        return False


def _run_goto_display(
    pipeline: IRPipeline,
    goto_offset: int,
    context_lines: int,
    use_color: bool
) -> int:
    """T086-T087: Show context around specific SCP address"""
    target_offset = goto_offset

    for func_name, llil, mlil, hlil in pipeline.functions:
        aligner = AddressAligner(llil, mlil, hlil)
        table = aligner.build_address_table()
        mappings = table.sorted_by_offset()

        # Check if target is in this function's range
        if not mappings:
            continue

        min_offset = mappings[0].scp_offset
        max_offset = mappings[-1].scp_offset

        if target_offset < min_offset or target_offset > max_offset:
            continue

        # Found the function
        print(f"=== Goto 0x{target_offset:08X} in {func_name} ===")
        print()

        center_idx = find_mapping_by_offset(mappings, target_offset)
        if center_idx is None:
            print(f"Offset 0x{target_offset:08X} not found in mappings")
            return 0

        start_idx, end_idx = get_context_range(mappings, center_idx, context_lines)

        # Display with context
        for i in range(start_idx, end_idx):
            m = mappings[i]
            marker = ">>>" if i == center_idx else "   "

            if use_color:
                if i == center_idx:
                    color = "\033[33m"  # Yellow for target
                else:
                    color = ""
                reset = "\033[0m" if color else ""
            else:
                color = ""
                reset = ""

            print(f"{marker} {color}0x{m.scp_offset:08X}: {m.llil_text[:40]:<40} | {m.mlil_text[:40]}{reset}")

        print()
        return 0

    print(f"Offset 0x{target_offset:08X} not found in any function", file=sys.stderr)
    return 0


def _run_reverse_lookup(
    pipeline: IRPipeline,
    search_text: str,
    use_color: bool
) -> int:
    """T089: Find SCP address from HLIL/MLIL text"""
    total_found = 0

    for func_name, llil, mlil, hlil in pipeline.functions:
        aligner = AddressAligner(llil, mlil, hlil)
        table = aligner.build_address_table()
        mappings = table.sorted_by_offset()

        results = reverse_lookup_text(mappings, search_text)
        if results:
            print(f"=== Found in {func_name} ({len(results)} matches) ===")
            for m in results:
                print(f"  0x{m.scp_offset:08X}: {m.mlil_text[:50]} | {m.hlil_text[:50]}")
            total_found += len(results)
            print()

    if total_found == 0:
        print(f"No matches found for: {search_text}")

    else:
        print(f"Total matches: {total_found}")

    return 0


def _run_dual_output(pipeline: IRPipeline, output_dir: str, scp_path: str) -> int:
    """Output both LLIL-MLIL and MLIL-HLIL comparisons to separate files"""
    import os
    from pathlib import Path

    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scp_name = Path(scp_path).stem

    # Output LLIL-MLIL comparison
    llil_mlil_file = out_path / f"{scp_name}_llil_mlil.txt"
    with open(llil_mlil_file, 'w', encoding='utf-8') as f:
        f.write(f"=== LLIL-MLIL Comparison: {scp_name} ===\n\n")

        for func_name, llil, mlil, hlil in pipeline.functions:
            table = build_llil_mlil_address_table(func_name, llil, mlil, pipeline, show_scp=False)
            report = format_llil_mlil_sync_table(table, use_color=False, show_scp=False)
            f.write(report)
            f.write("\n\n")

    print(f"LLIL-MLIL: {llil_mlil_file}")

    # Output MLIL-HLIL comparison
    mlil_hlil_file = out_path / f"{scp_name}_mlil_hlil.txt"
    with open(mlil_hlil_file, 'w', encoding='utf-8') as f:
        f.write(f"=== MLIL-HLIL Comparison: {scp_name} ===\n\n")

        for func_name, llil, mlil, hlil in pipeline.functions:
            aligner = AddressAligner(llil, mlil, hlil)
            table = aligner.build_address_table()
            report = format_mlil_hlil_sync_table(table, use_color=False)
            f.write(report)
            f.write("\n\n")

    print(f"MLIL-HLIL: {mlil_hlil_file}")

    return 0


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

    return 0


if __name__ == '__main__':
    sys.exit(main())
