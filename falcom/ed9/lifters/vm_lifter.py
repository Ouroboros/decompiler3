"""
ED9 VM bytecode → Falcom LLIL lifter.

Reuses the disassembler output (basic blocks + instructions) and rebuilds the
same CFG inside `FalcomVMBuilder`, emitting `LowLevelILFunction` objects.
"""

from dataclasses import dataclass
from typing import *

from ir.llil.llil import *

from ..disasm import *
from ..llil_builder import *
from ..parser import *

if TYPE_CHECKING:
    from ..parser.scp import *


@dataclass
class LiftResult:
    """Helper container with both Function metadata and its lifted LLIL."""

    function: Function
    llil: LowLevelILFunction


class ED9VMLifter:
    """Lift ED9 VM bytecode (disassembled) into Falcom LLIL."""

    def __init__(
        self,
        *,
        parser: ScpParser,
    ) -> None:
        if parser is None:
            raise ValueError('ED9VMLifter requires a parser instance')
        self._parser = parser

    # ----------------------------------------------------------------- public --
    def lift_function(self, func: Function) -> LowLevelILFunction:
        if func.entry_block is None:
            raise ValueError(f'Function {func.name} has no entry block')

        builder = FalcomVMBuilder()
        builder.create_function(func.name, func.offset, len(func.params))

        blocks = self._collect_blocks(func.entry_block)
        blocks.sort(key = lambda b: b.offset)

        llil_blocks: Dict[int, LowLevelILBasicBlock] = {}
        for block in blocks:
            llil_blocks[block.offset] = builder.create_basic_block(block.offset, block.name)

        block_map = {block.offset: block for block in blocks}

        for block in blocks:
            builder.set_current_block(llil_blocks[block.offset])
            for inst in block.instructions:
                self._translate_instruction(builder, inst, block, block_map, llil_blocks)

        return builder.finalize()

    def lift_functions(self, functions: Iterable[Function]) -> list[LiftResult]:
        results: list[LiftResult] = []
        for func in functions:
            results.append(LiftResult(function = func, llil = self.lift_function(func)))
        return results

    # --------------------------------------------------------------- helpers --
    def _collect_blocks(self, entry: BasicBlock) -> list[BasicBlock]:
        blocks: list[BasicBlock] = []
        visited: set[int] = set()
        stack = [entry]

        while stack:
            block = stack.pop()
            if block.offset in visited:
                continue
            visited.add(block.offset)
            blocks.append(block)
            stack.extend(block.succs)

        return blocks

    def _translate_instruction(
        self,
        builder: FalcomVMBuilder,
        inst: Instruction,
        block: BasicBlock,
        block_map: Dict[int, BasicBlock],
        llil_blocks: Dict[int, LowLevelILBasicBlock],
    ) -> None:
        opcode = ED9Opcode(inst.opcode)

        match opcode:
            case ED9Opcode.PUSH_INT:
                builder.push_int(int(inst.operands[0].value))

            case ED9Opcode.PUSH_FLOAT:
                builder.push(builder.const_float(inst.operands[0].value))

            case ED9Opcode.PUSH_STR:
                builder.push_str(str(inst.operands[0].value))

            case ED9Opcode.PUSH_RAW:
                builder.push_raw(int(inst.operands[0].value))

            case ED9Opcode.PUSH_CURRENT_FUNC_ID:
                builder.push_func_id()

            case ED9Opcode.PUSH_RET_ADDR:
                target = self._require_block(inst.operands[0].value, block_map, llil_blocks)
                builder.push_ret_addr(target)

            case ED9Opcode.PUSH_CALLER_FRAME:
                target = self._require_block(inst.operands[0].value, block_map, llil_blocks)
                builder.push_caller_frame(target)

            case ED9Opcode.LOAD_STACK | ED9Opcode.LOAD_STACK_DEREF:
                builder.load_stack(int(inst.operands[0].value))

            case ED9Opcode.PUSH_STACK_OFFSET:
                builder.push_stack_addr(int(inst.operands[0].value))

            case ED9Opcode.POP_TO | ED9Opcode.POP_TO_DEREF:
                builder.pop_to(int(inst.operands[0].value))

            case ED9Opcode.POP:
                pop_size = int(inst.operands[0].value) if inst.operands else 0
                self._pop_words(builder, pop_size // WORD_SIZE if pop_size > 0 else 0)

            case ED9Opcode.POPN:
                count = int(inst.operands[0].value) if inst.operands else 0
                self._pop_words(builder, count)

            case ED9Opcode.GET_REG:
                builder.get_reg(int(inst.operands[0].value))

            case ED9Opcode.SET_REG:
                builder.set_reg(int(inst.operands[0].value))

            case ED9Opcode.LOAD_GLOBAL:
                builder.load_global(int(inst.operands[0].value))

            case ED9Opcode.SET_GLOBAL:
                builder.set_global(int(inst.operands[0].value))

            case (
                ED9Opcode.ADD |
                ED9Opcode.SUB |
                ED9Opcode.MUL |
                ED9Opcode.DIV |
                ED9Opcode.EQ |
                ED9Opcode.NE |
                ED9Opcode.GT |
                ED9Opcode.GE |
                ED9Opcode.LT |
                ED9Opcode.LE |
                ED9Opcode.BITWISE_AND |
                ED9Opcode.BITWISE_OR |
                ED9Opcode.LOGICAL_AND |
                ED9Opcode.LOGICAL_OR
            ):
                self._emit_binary_op(builder, opcode)

            case ED9Opcode.NEG:
                builder.neg()

            case ED9Opcode.NOT:
                builder.logical_not()

            case ED9Opcode.EZ:
                builder.test_zero()

            case ED9Opcode.JMP:
                target = self._require_block(inst.operands[0].value, block_map, llil_blocks)
                builder.jmp(target)

            case ED9Opcode.POP_JMP_ZERO:
                true_block = self._branch_target(block.true_succs, llil_blocks, block_map, current = block)
                false_block = self._branch_target(block.false_succs, llil_blocks, block_map, prefer_fallthrough = True, current = block)
                builder.pop_jmp_zero(true_block, false_block)

            case ED9Opcode.POP_JMP_NOT_ZERO:
                true_block = self._branch_target(block.true_succs, llil_blocks, block_map, current = block)
                false_block = self._branch_target(block.false_succs, llil_blocks, block_map, prefer_fallthrough = True, current = block)
                builder.pop_jmp_not_zero(true_block, false_block)

            case ED9Opcode.CALL:
                func_id = int(inst.operands[0].value)
                builder.call(self._resolve_call_target(func_id))

            case ED9Opcode.CALL_SCRIPT:
                module = self._scpvalue_to_str(inst.operands[0].value)
                func_name = self._scpvalue_to_str(inst.operands[1].value)
                argc = int(inst.operands[2].value)
                builder.call_script(module, func_name, argc)

            case ED9Opcode.SYSCALL:
                subsystem = int(inst.operands[0].value)
                cmd = int(inst.operands[1].value)
                argc = int(inst.operands[2].value)
                builder.syscall(subsystem, cmd, argc)

            case ED9Opcode.DEBUG_SET_LINENO:
                builder.debug_line(int(inst.operands[0].value))

            case ED9Opcode.DEBUG_LOG:
                pass

            case ED9Opcode.RETURN:
                builder.ret()

            case ED9Opcode.MOD:
                raise NotImplementedError('MOD opcode not supported in lifter yet')

            case _:
                raise NotImplementedError(f'Unhandled opcode: {opcode.name} (0x{opcode.value:02X})')

    def _emit_binary_op(self, builder: FalcomVMBuilder, opcode: ED9Opcode) -> None:
        mapping = {
            ED9Opcode.ADD: builder.add,
            ED9Opcode.SUB: builder.sub,
            ED9Opcode.MUL: builder.mul,
            ED9Opcode.DIV: builder.div,
            ED9Opcode.EQ: builder.eq,
            ED9Opcode.NE: builder.ne,
            ED9Opcode.GT: builder.gt,
            ED9Opcode.GE: builder.ge,
            ED9Opcode.LT: builder.lt,
            ED9Opcode.LE: builder.le,
            ED9Opcode.BITWISE_AND: builder.bitwise_and,
            ED9Opcode.BITWISE_OR: builder.bitwise_or,
            ED9Opcode.LOGICAL_AND: builder.logical_and,
            ED9Opcode.LOGICAL_OR: builder.logical_or,
        }
        mapping[opcode]()

    def _require_block(
        self,
        offset: int,
        block_map: Dict[int, BasicBlock],
        llil_blocks: Dict[int, LowLevelILBasicBlock],
    ) -> LowLevelILBasicBlock:
        block = block_map.get(offset)
        if block is None:
            raise KeyError(f'No basic block at offset 0x{offset:08X}')
        return llil_blocks[block.offset]

    def _branch_target(
        self,
        candidates: list[BasicBlock],
        llil_blocks: Dict[int, LowLevelILBasicBlock],
        block_map: Dict[int, BasicBlock],
        *,
        prefer_fallthrough: bool = False,
        current: BasicBlock | None = None,
    ) -> LowLevelILFunction.BasicBlock:
        if candidates:
            return llil_blocks[candidates[0].offset]

        if prefer_fallthrough and current is not None:
            for succ in current.succs:
                if candidates and succ.offset == candidates[0].offset:
                    continue
                return llil_blocks[succ.offset]

        raise RuntimeError('Missing branch target while lifting')

    def _resolve_call_target(self, func_id: int) -> str:
        return self._parser.get_func_name(func_id)

    def _scpvalue_to_str(self, value: ScpValue) -> str:
        if not isinstance(value, ScpValue):
            raise TypeError(f'Expected ScpValue, got {type(value)}')

        if value.type != ScpValue.Type.String:
            raise TypeError(f'CALL_SCRIPT expects string ScpValue, got {value.type}')

        return str(value.value)

    def _pop_words(self, builder: FalcomVMBuilder, words: int) -> None:
        available = builder.sp_get()
        consume = min(words, available)
        builder.pop_bytes(consume * WORD_SIZE)
        # Remaining words correspond to frame cleanup below current sp; skip emitting IL
