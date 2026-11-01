'''TVM bytecode -> LLIL -> MLIL -> HLIL with the unchanged decompiler core'''

import contextlib
import io
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from ir.core import PARAM_TYPE_INT, PARAM_TYPE_STR, ILModule, IRParameter
from ir.hlil import HighLevelILFunction
from ir.hlil.hlil_passes import (
    CommonReturnExtractionPass, ControlFlowOptimizationPass, DeadCodeEliminationPass, ExpressionSimplificationPass,
    MLILToHLILPass,
)
from ir.llil import (
    LowLevelILBuilder, LowLevelILEq, LowLevelILExpr, LowLevelILFunction, LowLevelILOperation, LowLevelILRegLoad,
    LowLevelILRegStore, LowLevelILStackStore, LowLevelILStatement,
)
from ir.mlil import MediumLevelILFunction
from ir.mlil.llil_to_mlil import LLILToMLILTranslator
from ir.mlil.passes import (
    LLILToMLILPass, RegGlobalValuePropagationPass, SSAConversionPass, SSADeconstructionPass, SSAOptimizationPass,
    SSATypeInferencePass,
)
from ir.pipeline import PassObserver, Pipeline

from .image import PARAM_INT, PARAM_STR, Function, Image
from .isa import BLOCK_END_OPS, JUMP_OPS, WORD_SIZE, Instruction, Op
from .target import TVMHost, TVMMetadata, TVMTarget


PARAM_TYPE_NAMES = {PARAM_INT: PARAM_TYPE_INT, PARAM_STR: PARAM_TYPE_STR}


class TVMOperation(IntEnum):
    '''TVM extensions of LLIL'''
    LLIL_GLOBAL_LOAD    = LowLevelILOperation.LLIL_USER_DEFINED
    LLIL_GLOBAL_STORE   = LowLevelILOperation.LLIL_USER_DEFINED + 1
    LLIL_HOST_CALL      = LowLevelILOperation.LLIL_USER_DEFINED + 2


class TVMGlobalLoad(LowLevelILExpr):
    def __init__(self, index: int):
        super().__init__(TVMOperation.LLIL_GLOBAL_LOAD)
        self.index = index

    def __str__(self) -> str:
        return f'G[{self.index}]'


class TVMGlobalStore(LowLevelILStatement):
    def __init__(self, index: int, value: LowLevelILExpr):
        super().__init__(TVMOperation.LLIL_GLOBAL_STORE)
        self.index = index
        self.value = value

    def __str__(self) -> str:
        return f'G[{self.index}] = {self.value}'


class TVMHostCall(LowLevelILStatement):
    '''HOST: inline call, the result lands in the result register'''

    def __init__(self, target: TVMHost, args: List[LowLevelILExpr]):
        super().__init__(TVMOperation.LLIL_HOST_CALL)
        self.target = target
        self.args = args

    def __str__(self) -> str:
        return self.target.format_call([str(arg) for arg in self.args])


BINARY_BUILDERS = {
    Op.ADD: 'add', Op.SUB: 'sub', Op.MUL: 'mul', Op.DIV: 'div', Op.MOD: 'mod',
    Op.EQ: 'eq', Op.NE: 'ne', Op.LT: 'lt', Op.LE: 'le', Op.GT: 'gt', Op.GE: 'ge',
    Op.AND: 'bitwise_and', Op.OR: 'bitwise_or',
}
UNARY_BUILDERS = {Op.NEG: 'neg', Op.NOT: 'bitwise_not'}


class TVMLifter:
    '''TVM function -> LLIL; stack operands are byte offsets, the IR counts slots'''

    def __init__(self, image: Image):
        self.image = image

    def lift(self, func: Function) -> LowLevelILFunction:
        instructions = func.instructions
        leaders = {0} | {inst.operands[0] for inst in instructions if inst.op in JUMP_OPS}
        leaders |= {inst.offset + inst.size for inst in instructions if inst.op in BLOCK_END_OPS}
        leaders = sorted(leader for leader in leaders if leader in {inst.offset for inst in instructions})

        builder = LowLevelILBuilder()
        params = [IRParameter(f'arg{number}', PARAM_TYPE_NAMES[flag], None, flags = flag)
                  for number, flag in enumerate(func.param_flags, start = 1)]
        builder.create_function(func.name, func.offset, params, target = TVMTarget)
        blocks = {leader: builder.create_basic_block(func.offset + leader) for leader in leaders}
        self.builder, self.blocks, self.func = builder, blocks, func

        for index, leader in enumerate(leaders):
            following = leaders[index + 1] if index + 1 < len(leaders) else None
            builder.restore_stack_for_offset(leader)
            builder.set_current_block(blocks[leader])
            body = [inst for inst in instructions if leader <= inst.offset and (following is None or inst.offset < following)]
            for inst in body:
                builder.set_current_address(func.offset + inst.offset)
                self._lift(inst, following)

            if body[-1].op not in BLOCK_END_OPS:
                self._goto(following)

        builder.function.build_cfg()
        return builder.function

    def _goto(self, offset: int):
        self.builder.save_stack_for_offset(offset)
        self.builder.jmp(self.blocks[offset])

    def _args(self, argc: int) -> List[LowLevelILExpr]:
        # Parameter 1 was pushed first, so it pops last
        return list(reversed([self.builder.pop() for _ in range(argc)]))

    def _lift(self, inst: Instruction, following: Optional[int]):
        builder, op, operands = self.builder, inst.op, inst.operands
        if op == Op.PUSH:
            builder.push(builder.const_int(operands[0]))

        elif op == Op.PUSHS:
            builder.push(builder.const_str(self.image.strings[operands[0]]))

        elif op == Op.LDL:
            builder.load_stack(operands[0] // WORD_SIZE)

        elif op == Op.STL:
            value = builder.pop()
            slot = builder.sp_get() + operands[0] // WORD_SIZE
            if 0 <= slot < builder.function.num_params:
                builder.frame_store(value, slot)

            else:
                builder.add_instruction(LowLevelILStackStore(value, offset = operands[0] // WORD_SIZE, slot_index = slot))

        elif op == Op.LDG:
            builder.push(TVMGlobalLoad(operands[0]))

        elif op == Op.STG:
            builder.add_instruction(TVMGlobalStore(operands[0], builder.pop()))

        elif op == Op.LDR:
            builder.push(builder.reg_load(operands[0]))

        elif op == Op.STR:
            builder.reg_store(operands[0], builder.pop())

        elif op == Op.DROP:
            for _ in range(operands[0]):
                builder.pop()

        elif op in BINARY_BUILDERS:
            getattr(builder, BINARY_BUILDERS[op])()

        elif op in UNARY_BUILDERS:
            getattr(builder, UNARY_BUILDERS[op])()

        elif op == Op.JMP:
            self._goto(operands[0])

        elif op in (Op.JZ, Op.JNZ):
            condition = builder.pop()
            if op == Op.JZ:
                condition = LowLevelILEq(condition, builder.const_int(0))

            builder.save_stack_for_offset(operands[0])
            builder.save_stack_for_offset(following)
            builder.branch_if(condition, self.blocks[operands[0]], self.blocks[following])

        elif op == Op.CALL:
            args = self._args(operands[1])
            builder.save_stack_for_offset(following)
            builder.call(self.image.functions[operands[0]].name, self.blocks[following], args)

        elif op == Op.HOST:
            args = self._args(operands[1])
            builder.add_instruction(TVMHostCall(TVMHost(self.image.strings[operands[0]]), args))

        elif op == Op.RET:
            builder.ret()

        elif op == Op.LINE:
            builder.debug_line(operands[0])

        elif op == Op.LOG:
            builder.debug_log(operands[0])

        else:
            raise ValueError(f'unknown TVM op {op!r}')


class TVMTranslator(LLILToMLILTranslator):
    '''Translates the TVM LLIL extensions and registers; everything else is the core translator'''

    def _translate_instruction(self, llil_inst):
        if isinstance(llil_inst, TVMGlobalStore):
            self.builder.store_global(llil_inst.index, self._translate_expr(llil_inst.value))

        elif isinstance(llil_inst, LowLevelILRegStore):
            self.builder.store_reg(llil_inst.reg_index, self._translate_expr(llil_inst.value))

        elif isinstance(llil_inst, TVMHostCall):
            self.builder.call(llil_inst.target, [self._translate_expr(arg) for arg in llil_inst.args])

        else:
            super()._translate_instruction(llil_inst)

    def _translate_expr(self, llil_expr):
        if isinstance(llil_expr, TVMGlobalLoad):
            return self.builder.load_global(llil_expr.index)

        if isinstance(llil_expr, LowLevelILRegLoad):
            return self.builder.load_reg(llil_expr.reg_index)

        return super()._translate_expr(llil_expr)


def mlil_pipeline() -> Pipeline:
    return Pipeline([
        LLILToMLILPass(TVMTranslator), SSAConversionPass(), SSAOptimizationPass(folder = TVMTarget.values),
        SSATypeInferencePass(None), SSADeconstructionPass(), RegGlobalValuePropagationPass(),
    ])


def hlil_pipeline() -> Pipeline:
    return Pipeline([
        MLILToHLILPass(), ExpressionSimplificationPass(), ControlFlowOptimizationPass(), CommonReturnExtractionPass(),
        DeadCodeEliminationPass(),
    ])


class TVMModuleDecompiler:
    '''One TVM image as LLIL, MLIL and HLIL modules sharing the image metadata'''

    def __init__(self, image: Image, name: str = 'tvm'):
        self.image = image
        self.lifter = TVMLifter(image)
        self.llil = ILModule(name, TVMMetadata.from_image(image))
        self.mlil = self.llil.derive()
        self.hlil = self.llil.derive()

    def decompile(self, func: Function, observer: Optional[PassObserver] = None) -> Tuple[LowLevelILFunction, MediumLevelILFunction, HighLevelILFunction]:
        with contextlib.redirect_stdout(io.StringIO()):
            llil = self.llil.add_function(self.lifter.lift(func))
            mlil = self.mlil.add_function(mlil_pipeline().run(llil, observer = observer))
            hlil = self.hlil.add_function(hlil_pipeline().run(mlil, observer = observer))

        return llil, mlil, hlil
