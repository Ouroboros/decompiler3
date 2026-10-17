'''Reference interpreter: executes disassembled ED9 bytecode directly

The decompiler is not consulted. Every stack effect below comes from the VM
calling convention, checked against VMContext::execute / VMContext::Return:

- CALL:        caller pushes func id, return address, then arguments with
               parameter 1 pushed last; CALL only jumps, the callee pops its
               arguments and RETURN pops return address and func id.
- CALL_SCRIPT: PUSH_CALLER_FRAME pushes func id, return address, script
               pointer (two slots) and script name before the arguments.
- SYSCALL:     arguments are read in place; the script pops them afterwards.
- Every external call leaves its result in REGS[0].
'''

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from falcom.ed9.disasm import BasicBlock, ED9Opcode, Instruction
from falcom.ed9.parser.types_scp import ScpValue
from falcom.ed9.vm_values import COMPARISON_OPS, EQUALITY_OPS, Op, Pointer, Raw, UNDEF, VMFault, binary_op, is_zero, normalize, unary_op
from ir.llil import WORD_SIZE

from .environment import (
    EVENT_CALL, EVENT_EXTERN, EVENT_SYSCALL, MAX_EVENTS, MAX_STEPS,
    ORIGIN_CELL, ORIGIN_OUT, ORIGIN_PARAM, ORIGIN_REG, ORIGIN_REG_MUTATION, ORIGIN_RETURN, InputPlan, Origin, RunStatus,
    Trace, WorldState, _RunEnd,
)
from .symbolic import Sym, SymInput, make_op


CALL_FRAME_SLOTS        = 2     # func id + return address
SCRIPT_FRAME_SLOTS      = 5     # func id + return address + script pointer (2 slots) + script name
RET_ADDR_FRAME_POSITION = 1     # return address is the second slot of every call frame
STACK_SPACE             = 'stack'
NEIGHBOR_DELTA          = 1

# Recorded branch conditions per run, first occurrence of each branch direction only
MAX_BRANCH_RECORDS      = 512


class FrameMarker:
    '''Non-value stack slot pushed as part of a call frame'''

    def __init__(self, kind: str, target: Optional[int] = None):
        self.kind = kind
        self.target = target

    def __repr__(self) -> str:
        return f'FrameMarker({self.kind}, {self.target})'


MARKER_FUNC_ID      = 'func_id'
MARKER_RET_ADDR     = 'ret_addr'
MARKER_SCRIPT       = 'script'
MARKER_SCRIPT_NAME  = 'script_name'

BINARY_OPCODES = {
    ED9Opcode.ADD:          Op.ADD,
    ED9Opcode.SUB:          Op.SUB,
    ED9Opcode.MUL:          Op.MUL,
    ED9Opcode.DIV:          Op.DIV,
    ED9Opcode.MOD:          Op.MOD,
    ED9Opcode.EQ:           Op.EQ,
    ED9Opcode.NE:           Op.NE,
    ED9Opcode.GT:           Op.GT,
    ED9Opcode.GE:           Op.GE,
    ED9Opcode.LT:           Op.LT,
    ED9Opcode.LE:           Op.LE,
    ED9Opcode.BITWISE_AND:  Op.BIT_AND,
    ED9Opcode.BITWISE_OR:   Op.BIT_OR,
    ED9Opcode.LOGICAL_AND:  Op.LOGICAL_AND,
    ED9Opcode.LOGICAL_OR:   Op.LOGICAL_OR,
}

UNARY_OPCODES = {
    ED9Opcode.NEG:  Op.NEG,
    ED9Opcode.NOT:  Op.BIT_NOT,
    ED9Opcode.EZ:   Op.TEST_ZERO,
}

# Every slot carries the expression deriving its value from run inputs; literals carry None
Slot = Tuple[Any, Optional[Sym]]


@dataclass
class BranchRecord:
    '''Conditional jump whose condition depends on run inputs'''
    block: int
    other: int
    condition: Sym
    observed_zero: bool


@dataclass
class BytecodeRun:
    trace: Trace
    covered_blocks: Set[int] = field(default_factory = set)
    covered_edges: Set[Tuple[int, int]] = field(default_factory = set)
    hints: Dict[Origin, Set[Any]] = field(default_factory = dict)
    branches: List[BranchRecord] = field(default_factory = list)


class _Returned(Exception):
    pass


def collect_blocks(entry: BasicBlock) -> Dict[int, BasicBlock]:
    blocks: Dict[int, BasicBlock] = {}
    pending = [entry]
    while pending:
        block = pending.pop()
        if block.offset in blocks:
            continue

        blocks[block.offset] = block
        pending.extend(block.succs)

    return blocks


def collect_constants(blocks: Dict[int, BasicBlock]) -> Set[Any]:
    constants: Set[Any] = set()
    for block in blocks.values():
        for inst in block.instructions:
            if inst.opcode in (ED9Opcode.PUSH_INT, ED9Opcode.PUSH_RAW, ED9Opcode.PUSH_FLOAT):
                constants.add(normalize(inst.operands[0].value))

    return constants


def collect_reg_indices(blocks: Dict[int, BasicBlock]) -> Set[int]:
    '''Registers other than REGS[0] that the function reads'''
    indices: Set[int] = set()
    for block in blocks.values():
        for inst in block.instructions:
            if inst.opcode == ED9Opcode.GET_REG and int(inst.operands[0].value) != 0:
                indices.add(int(inst.operands[0].value))

    return indices


def collect_global_indices(blocks: Dict[int, BasicBlock]) -> Set[int]:
    indices: Set[int] = set()
    for block in blocks.values():
        for inst in block.instructions:
            if inst.opcode in (ED9Opcode.LOAD_GLOBAL, ED9Opcode.SET_GLOBAL):
                indices.add(int(inst.operands[0].value))

    return indices


class BytecodeInterpreter:
    '''Executes one script function against an InputPlan'''

    def __init__(self, parser, func, plan: InputPlan, *,
                 blocks: Optional[Dict[int, BasicBlock]] = None,
                 max_events: int = MAX_EVENTS, max_steps: int = MAX_STEPS):
        self.parser = parser
        self.func = func
        self.plan = plan
        self.max_steps = max_steps
        self.blocks = blocks if blocks is not None else collect_blocks(func.entry_block)
        self.run_result = BytecodeRun(Trace(max_events = max_events))
        self.world = WorldState(plan, self.run_result.trace)
        self.reg_origins: Dict[int, Optional[Sym]] = {0: None}
        self._recorded_branches: Set[Tuple[int, int]] = set()
        self.stack: List[Slot] = []
        self.steps = 0

    # ------------------------------------------------------------------ run --

    def run(self) -> BytecodeRun:
        self._setup_frame()
        block = self.func.entry_block
        address = block.offset
        try:
            while True:
                self.run_result.covered_blocks.add(block.offset)
                next_block = None
                for inst in block.instructions:
                    address = inst.offset
                    self.steps += 1
                    if self.steps > self.max_steps:
                        raise _RunEnd(RunStatus.STEP_LIMIT)

                    next_block = self._execute(inst, block)
                    if next_block is not None:
                        break

                if next_block is None:
                    next_block = self._fallthrough(block)

                self.run_result.covered_edges.add((block.offset, next_block.offset))
                block = next_block

        except _Returned:
            pass

        except _RunEnd as end:
            self.run_result.trace.status = end.status

        except VMFault as fault:
            self.world.fault(fault, address)

        return self.run_result

    def _setup_frame(self):
        self.stack.append((FrameMarker(MARKER_FUNC_ID), None))
        self.stack.append((FrameMarker(MARKER_RET_ADDR), None))
        count = len(self.func.params)
        for slot in range(count):
            # Parameter 1 is pushed last
            number = count - slot
            self.stack.append((self.plan.param(number), SymInput((ORIGIN_PARAM, number))))

    def _fallthrough(self, block: BasicBlock) -> BasicBlock:
        if len(block.succs) != 1:
            raise VMFault('missing_fallthrough', block.name)

        return block.succs[0]

    def _block_at(self, offset: int) -> BasicBlock:
        block = self.blocks.get(int(offset))
        if block is None:
            raise VMFault('bad_branch_target', f'0x{int(offset):X}')

        return block

    # -------------------------------------------------------------- helpers --

    def _push(self, value: Any, sym: Optional[Sym] = None):
        self.stack.append((value, sym))

    def _pop(self) -> Slot:
        if not self.stack:
            raise VMFault('stack_underflow')

        return self.stack.pop()

    def _slot_index(self, byte_offset: int) -> int:
        index = len(self.stack) + int(byte_offset) // WORD_SIZE
        if index < 0 or index >= len(self.stack):
            raise VMFault('stack_out_of_range', str(byte_offset))

        return index

    def _deref(self, pointer: Pointer) -> Any:
        if pointer.space != STACK_SPACE or pointer.key >= len(self.stack):
            return UNDEF

        return self.stack[pointer.key][0]

    @classmethod
    def _as_sym(cls, origin: Any) -> Optional[Sym]:
        '''WorldState records either an origin tuple or the symbolic value a script stored'''
        if origin is None or isinstance(origin, Sym):
            return origin

        return SymInput(origin)

    def _hint(self, sym: Optional[Sym], values):
        for origin in sym.inputs if sym is not None else ():
            self.run_result.hints.setdefault(origin, set()).update(values)

    def _record_comparison(self, op: Op, lhs: Slot, rhs: Slot):
        # Only literal operands are useful targets; run-time values change every iteration
        for (value, sym), (other, other_sym) in ((lhs, rhs), (rhs, lhs)):
            if sym is None or other_sym is not None or not isinstance(other, (int, float)):
                continue

            if op in EQUALITY_OPS:
                self._hint(sym, {other})

            else:
                self._hint(sym, {other - NEIGHBOR_DELTA, other, other + NEIGHBOR_DELTA})

    def _record_branch(self, block: BasicBlock, taken: BasicBlock, other: BasicBlock, value: Any, sym: Optional[Sym]):
        key = (block.offset, taken.offset)
        if sym is None or key in self._recorded_branches or len(self._recorded_branches) >= MAX_BRANCH_RECORDS:
            return

        self._recorded_branches.add(key)
        self.run_result.branches.append(BranchRecord(block.offset, other.offset, sym, is_zero(value)))

    # ------------------------------------------------------------- execute --

    def _execute(self, inst: Instruction, block: BasicBlock) -> Optional[BasicBlock]:
        opcode = ED9Opcode(inst.opcode)
        operands = inst.operands

        match opcode:
            case ED9Opcode.PUSH_INT | ED9Opcode.PUSH_FLOAT:
                self._push(normalize(operands[0].value))

            case ED9Opcode.PUSH_RAW:
                self._push(normalize(Raw(int(operands[0].value))))

            case ED9Opcode.PUSH_STR:
                self._push(str(operands[0].value))

            case ED9Opcode.PUSH_CURRENT_FUNC_ID:
                self._push(FrameMarker(MARKER_FUNC_ID))

            case ED9Opcode.PUSH_RET_ADDR:
                self._push(FrameMarker(MARKER_RET_ADDR, int(operands[0].value)))

            case ED9Opcode.PUSH_CALLER_FRAME:
                self._push(FrameMarker(MARKER_FUNC_ID))
                self._push(FrameMarker(MARKER_RET_ADDR, int(operands[0].value)))
                self._push(FrameMarker(MARKER_SCRIPT))
                self._push(FrameMarker(MARKER_SCRIPT))
                self._push(FrameMarker(MARKER_SCRIPT_NAME))

            case ED9Opcode.LOAD_STACK:
                self._push(*self.stack[self._slot_index(operands[0].value)])

            case ED9Opcode.LOAD_STACK_DEREF:
                pointer, sym = self.stack[self._slot_index(operands[0].value)]
                if isinstance(pointer, Pointer):
                    self._push(*self.stack[pointer.key])

                else:
                    number = self._pointer_param(sym)
                    self._push(self.world.load_cell(number), SymInput((ORIGIN_CELL, number)))

            case ED9Opcode.PUSH_STACK_OFFSET:
                self._push(Pointer(STACK_SPACE, self._slot_index(operands[0].value)))

            case ED9Opcode.POP_TO:
                slot = self._pop()
                self.stack[self._slot_index(operands[0].value)] = slot

            case ED9Opcode.POP_TO_DEREF:
                slot = self._pop()
                pointer, sym = self.stack[self._slot_index(operands[0].value)]
                if isinstance(pointer, Pointer):
                    self.stack[pointer.key] = slot

                else:
                    self.world.store_cell(self._pointer_param(sym), slot[0])

            case ED9Opcode.POP:
                for _ in range(int(operands[0].value) // WORD_SIZE):
                    self._pop()

            case ED9Opcode.POPN:
                for _ in range(int(operands[0].value)):
                    self._pop()

            case ED9Opcode.GET_REG:
                index = int(operands[0].value)
                if index not in self.world.regs:
                    self.reg_origins[index] = SymInput((ORIGIN_REG, index))

                self._push(self.world.load_reg(index), self.reg_origins.get(index))

            case ED9Opcode.SET_REG:
                index = int(operands[0].value)
                value, sym = self._pop()
                self.world.store_reg(index, value)
                self.reg_origins[index] = sym

            case ED9Opcode.LOAD_GLOBAL:
                index = int(operands[0].value)
                value = self.world.load_global(index)
                self._push(value, self._as_sym(self.world.global_origins.get(index)))

            case ED9Opcode.SET_GLOBAL:
                value, sym = self._pop()
                self.world.store_global(int(operands[0].value), value, sym)

            case _ if opcode in BINARY_OPCODES:
                op = BINARY_OPCODES[opcode]
                rhs = self._pop()
                lhs = self._pop()
                if op in COMPARISON_OPS:
                    self._record_comparison(op, lhs, rhs)

                self._push(binary_op(op, lhs[0], rhs[0]), make_op(op, (lhs[0], rhs[0]), (lhs[1], rhs[1])))

            case _ if opcode in UNARY_OPCODES:
                op = UNARY_OPCODES[opcode]
                value, sym = self._pop()
                self._push(unary_op(op, value), make_op(op, (value,), (sym,)))

            case ED9Opcode.JMP:
                return self._block_at(operands[0].value)

            case ED9Opcode.POP_JMP_ZERO | ED9Opcode.POP_JMP_NOT_ZERO:
                value, sym = self._pop()
                self._hint(sym, {0, 1})
                jump_target = self._block_at(operands[0].value)
                next_block = block.false_succs[0] if block.false_succs else self._fallthrough(block)
                taken = is_zero(value) if opcode == ED9Opcode.POP_JMP_ZERO else not is_zero(value)
                chosen, other = (jump_target, next_block) if taken else (next_block, jump_target)
                self._record_branch(block, chosen, other, value, sym)
                return chosen

            case ED9Opcode.CALL:
                return self._call(inst)

            case ED9Opcode.CALL_SCRIPT:
                return self._call_script(inst)

            case ED9Opcode.SYSCALL:
                self._syscall(inst)

            case ED9Opcode.RETURN:
                self.world.finish(self.world.regs.get(0, UNDEF), self._deref, inst.offset)
                raise _Returned()

            case ED9Opcode.DEBUG_SET_LINENO | ED9Opcode.DEBUG_LOG:
                pass

            case _:
                raise VMFault('unsupported_opcode', opcode.name)

        return None

    # ---------------------------------------------------------------- calls --

    def _take_args(self, argc: int) -> List[Any]:
        if argc > len(self.stack):
            raise VMFault('stack_underflow')

        # Parameter 1 is the last pushed value
        return [self.stack[len(self.stack) - 1 - i][0] for i in range(argc)]

    def _return_target(self, argc: int, frame_slots: int) -> BasicBlock:
        frame_base = len(self.stack) - argc - frame_slots
        if frame_base < 0:
            raise VMFault('stack_underflow')

        func_id = self.stack[frame_base][0]
        ret_addr = self.stack[frame_base + RET_ADDR_FRAME_POSITION][0]
        if not (isinstance(func_id, FrameMarker) and func_id.kind == MARKER_FUNC_ID):
            raise VMFault('bad_call_frame', 'func id')

        if not (isinstance(ret_addr, FrameMarker) and ret_addr.kind == MARKER_RET_ADDR):
            raise VMFault('bad_call_frame', 'return address')

        return self._block_at(ret_addr.target)

    def _pointer_param(self, sym: Optional[Sym]) -> int:
        '''A dereferenced non-pointer must be a parameter holding a pointer into caller memory'''
        if not (isinstance(sym, SymInput) and sym.origin[0] == ORIGIN_PARAM):
            raise VMFault('deref_non_pointer')

        return sym.origin[1]

    def _apply_outcome(self, outcome, args: List[Any]):
        for pos, value in outcome.out_values.items():
            pointer = args[pos]
            if pointer.space == STACK_SPACE and pointer.key < len(self.stack):
                self.stack[pointer.key] = (value, SymInput((ORIGIN_OUT, outcome.call_index, pos)))

        for index in outcome.reg_writes:
            self.reg_origins[index] = SymInput((ORIGIN_REG_MUTATION, outcome.call_index, index))

        self.reg_origins[0] = SymInput((ORIGIN_RETURN, outcome.call_index))

    def _drop(self, count: int):
        if count > len(self.stack):
            raise VMFault('stack_underflow')

        del self.stack[len(self.stack) - count:]

    def _call(self, inst: Instruction) -> BasicBlock:
        func_id = int(inst.operands[0].value)
        argc = self.parser.get_func_argc(func_id)
        args = self._take_args(argc)
        target = self._return_target(argc, CALL_FRAME_SLOTS)
        outcome = self.world.external_call((EVENT_CALL, self.parser.get_func_name(func_id)), args, self._deref, inst.offset)
        self._apply_outcome(outcome, args)
        self._drop(argc + CALL_FRAME_SLOTS)
        return target

    def _call_script(self, inst: Instruction) -> BasicBlock:
        module, func_name, argc = (operand.value for operand in inst.operands)
        module = module.value if isinstance(module, ScpValue) else module
        func_name = func_name.value if isinstance(func_name, ScpValue) else func_name
        argc = int(argc)
        args = self._take_args(argc)
        target = self._return_target(argc, SCRIPT_FRAME_SLOTS)
        outcome = self.world.external_call((EVENT_EXTERN, str(module), str(func_name)), args, self._deref, inst.offset)
        self._apply_outcome(outcome, args)
        self._drop(argc + SCRIPT_FRAME_SLOTS)
        return target

    def _syscall(self, inst: Instruction):
        subsystem, cmd, argc = (int(operand.value) for operand in inst.operands)
        args = self._take_args(argc)
        outcome = self.world.external_call((EVENT_SYSCALL, subsystem, cmd), args, self._deref, inst.offset)
        self._apply_outcome(outcome, args)
