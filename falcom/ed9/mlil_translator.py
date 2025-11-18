'''
Falcom ED9 VM LLIL to MLIL Translator

Extends the generic LLILToMLILTranslator to handle Falcom-specific instructions:
- Global variables (LowLevelILGlobalStore/Load)
- Registers (LowLevelILRegStore/Load)
- Script calls (LowLevelILCallScript)
- Syscalls (LowLevelILSyscall)
- Falcom-specific constants (LowLevelILConstFuncId, LowLevelILConstRetAddrBlock, etc.)
'''

from ir.llil import *
from ir.mlil import *
from .llil_ext import *
from .constants import *


class FalcomLLILToMLILTranslator(LLILToMLILTranslator):
    '''Falcom-specific LLIL to MLIL translator'''

    def _translate_instruction(self, llil_inst: LowLevelILInstruction):
        '''Translate LLIL instruction, handling Falcom-specific operations'''

        # Skip caller frame preparation (implementation detail, not needed in MLIL)
        if isinstance(llil_inst, LowLevelILPushCallerFrame):
            return  # Omit from MLIL - calling convention detail

        # Skip stack stores for caller frame values
        if isinstance(llil_inst, LowLevelILStackStore):
            if self._is_caller_frame_value(llil_inst.value):
                return  # Omit caller frame preparation from MLIL

        # Falcom-specific instructions
        if isinstance(llil_inst, LowLevelILGlobalStore):
            value = self._translate_expr(llil_inst.value)
            self.builder.store_global(llil_inst.index, value)

        elif isinstance(llil_inst, LowLevelILRegStore):
            value = self._translate_expr(llil_inst.value)
            self.builder.store_reg(llil_inst.reg_index, value)

        elif isinstance(llil_inst, LowLevelILCallScript):
            self._translate_call_script(llil_inst)

        elif isinstance(llil_inst, LowLevelILSyscall):
            self._translate_syscall(llil_inst)

        else:
            # Fall back to generic handling
            super()._translate_instruction(llil_inst)

    def _is_caller_frame_value(self, expr: LowLevelILInstruction) -> bool:
        '''Check if expression is a caller frame value (should be omitted in MLIL)'''
        return isinstance(expr, (
            LowLevelILConstFuncId,
            LowLevelILConstRetAddrBlock,
            LowLevelILConstScript,
            LowLevelILConstScriptName,
        ))

    def _translate_call_script(self, llil_inst: LowLevelILCallScript):
        '''Translate Falcom script call

        Like regular Call, CallScript is terminal in LLIL. In MLIL:
        1. CallScript instruction (non-terminal)
        2. Goto to return target (terminal)
        '''
        # Translate arguments
        mlil_args = [self._translate_expr(arg) for arg in llil_inst.args]

        # Generate MLIL CallScript
        self.builder.call_script(llil_inst.module, llil_inst.func, mlil_args)

        # Add goto to return target (always a LowLevelILBasicBlock in Falcom)
        return_block = self.block_map[llil_inst.return_target]
        self.builder.goto(return_block)

    def _translate_syscall(self, llil_inst: LowLevelILSyscall):
        '''Translate Falcom syscall'''
        # Verify args match argc
        if llil_inst.argc > 0 and not llil_inst.args:
            raise ValueError(f'Syscall argc={llil_inst.argc} but args is empty')

        if len(llil_inst.args) != llil_inst.argc:
            raise ValueError(f'Syscall argc={llil_inst.argc} but got {len(llil_inst.args)} args')

        # Translate arguments from LLIL
        mlil_args = [self._translate_expr(arg) for arg in llil_inst.args]

        # Generate MLIL Syscall
        self.builder.syscall(llil_inst.subsystem, llil_inst.cmd, mlil_args)

    def _translate_expr(self, llil_expr: LowLevelILInstruction) -> MediumLevelILInstruction:
        '''Translate LLIL expression, handling Falcom-specific types'''

        # Falcom-specific expressions
        if isinstance(llil_expr, LowLevelILGlobalLoad):
            return self.builder.load_global(llil_expr.index)

        elif isinstance(llil_expr, LowLevelILRegLoad):
            return self.builder.load_reg(llil_expr.reg_index)

        # Falcom-specific constants
        elif isinstance(llil_expr, LowLevelILConstFuncId):
            return MLILConst('<current_func_id>', is_hex = False)

        elif isinstance(llil_expr, LowLevelILConstRetAddrBlock):
            return MLILConst(f'<ret_addr:{llil_expr.block.label}>', is_hex = False)

        elif isinstance(llil_expr, LowLevelILConstScript):
            return MLILConst('<current_script>', is_hex = False)

        else:
            # Fall back to generic handling
            return super()._translate_expr(llil_expr)


def translate_falcom_llil_to_mlil(llil_func: LowLevelILFunction) -> MediumLevelILFunction:
    '''Convenience function to translate Falcom LLIL to MLIL with optimization'''

    print(f'Translating {llil_func.name} @ 0x{llil_func.start_addr:08X}')

    translator = FalcomLLILToMLILTranslator()
    mlil_func = translator.translate(llil_func)

    # Apply optimizations
    mlil_func = optimize_mlil(mlil_func)

    return mlil_func
