'''ED9 external calls (ir/core/target.py CallTarget and CallModel)'''

from dataclasses import dataclass
from typing import List

from ir.core import CallModel, CallTarget


# Every call leaves its result in REGS[0] (falcom/ed9/spec/vm_spec.md section 8)
RETURN_REGISTER = 0


@dataclass(frozen = True)
class ED9Syscall(CallTarget):
    '''SYSCALL subsystem, cmd'''
    subsystem: int
    cmd: int

    def format_call(self, args: List[str]) -> str:
        return f'syscall({", ".join([str(self.subsystem), str(self.cmd), *args])})'


@dataclass(frozen = True)
class ED9ScriptCall(CallTarget):
    '''CALL_SCRIPT of function func in script module

    returns is False for CALL_SCRIPT_NO_RETURN: every caller frame is discarded and the thread ends when the
    callee returns (falcom/ed9/spec/vm_spec.md section 5.4), so the calling function never continues.
    '''
    module: str
    func: str
    returns: bool = True

    @property
    def qualified_name(self) -> str:
        return f'{self.module}:{self.func}'

    def format_call(self, args: List[str]) -> str:
        call = f'{self.qualified_name}({", ".join(args)})'
        return call if self.returns else f'noreturn {call}'


class ED9Calls(CallModel):
    return_register = RETURN_REGISTER

    @classmethod
    def parameter_number(cls, frame_slot: int, param_count: int) -> int:
        '''Callers push parameter 1 last, so it sits in the highest frame slot and arguments pop in order'''
        return param_count - frame_slot
