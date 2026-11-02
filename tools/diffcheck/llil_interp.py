'''LLIL interpreter: executes the lifted stack IR against the same world as the bytecode

- STACK slots are absolute (slot_index); frame slot f holds parameter num_params - f, so parameter 1
  sits in the highest frame slot as callers push it last
- sp follows SpAdd; a pointer to a slot at or above sp reads UNDEF, like a popped bytecode slot
- a dereferenced non-pointer must be a parameter (or a copy of one) holding a pointer into caller memory
- call arguments are in parameter order; every external call leaves its result in REGS[0]
- operators and branch tests have VM semantics (falcom/ed9/vm_values.py)
'''

from typing import Any, Dict, Optional

from falcom.ed9.llil_ext import (
    LowLevelILCallScript, LowLevelILCallScriptNoReturn, LowLevelILGlobalLoad, LowLevelILGlobalStore, LowLevelILPushCallerFrame,
    LowLevelILSyscall,
)
from falcom.ed9.vm_values import Op, Pointer, Raw, UNDEF, VMFault, binary_op, is_zero, normalize, unary_op
from ir.llil import (
    LowLevelILAdd, LowLevelILAnd, LowLevelILBasicBlock, LowLevelILBitwiseNot, LowLevelILCall, LowLevelILConst,
    LowLevelILDebug, LowLevelILDerefLoad, LowLevelILDerefStore, LowLevelILDiv, LowLevelILEq, LowLevelILExpr,
    LowLevelILFrameLoad, LowLevelILFrameStore, LowLevelILFunction, LowLevelILGe, LowLevelILGoto, LowLevelILGt,
    LowLevelILIf, LowLevelILLabelInstr, LowLevelILLe, LowLevelILLogicalAnd, LowLevelILLogicalOr, LowLevelILLt,
    LowLevelILMod, LowLevelILMul, LowLevelILNe, LowLevelILNeg, LowLevelILOr, LowLevelILRegLoad, LowLevelILRegStore,
    LowLevelILRet, LowLevelILSpAdd, LowLevelILStackAddr, LowLevelILStackLoad, LowLevelILStackStore, LowLevelILSub,
    LowLevelILTestZero,
)

from .environment import (
    EVENT_CALL, EVENT_EXTERN, EVENT_EXTERN_NORETURN, EVENT_SYSCALL, MAX_EVENTS, MAX_STEPS, InputPlan, RunStatus, ThreadEnded, Trace,
    WorldState, _RunEnd,
)


STACK_SPACE = 'stack'

BINARY_OPS = {
    LowLevelILAdd:          Op.ADD,
    LowLevelILSub:          Op.SUB,
    LowLevelILMul:          Op.MUL,
    LowLevelILDiv:          Op.DIV,
    LowLevelILMod:          Op.MOD,
    LowLevelILEq:           Op.EQ,
    LowLevelILNe:           Op.NE,
    LowLevelILLt:           Op.LT,
    LowLevelILLe:           Op.LE,
    LowLevelILGt:           Op.GT,
    LowLevelILGe:           Op.GE,
    LowLevelILAnd:          Op.BIT_AND,
    LowLevelILOr:           Op.BIT_OR,
    LowLevelILLogicalAnd:   Op.LOGICAL_AND,
    LowLevelILLogicalOr:    Op.LOGICAL_OR,
}

UNARY_OPS = {
    LowLevelILNeg:          Op.NEG,
    LowLevelILBitwiseNot:   Op.BIT_NOT,
    LowLevelILTestZero:     Op.TEST_ZERO,
}


class _Returned(Exception):
    pass


class LLILInterpreter:
    '''Executes one LLIL function against an InputPlan'''

    def __init__(self, llil_func: LowLevelILFunction, plan: InputPlan, *,
                 max_events: int = MAX_EVENTS, max_steps: int = MAX_STEPS):
        self.func = llil_func
        self.plan = plan
        self.max_steps = max_steps
        self.trace = Trace(max_events = max_events)
        self.world = WorldState(plan, self.trace)
        self.steps = 0
        self.address = llil_func.start_addr
        count = llil_func.num_params
        self.slots: Dict[int, Any] = {}
        # Parameter number whose value a slot holds unchanged, for pointers into caller memory
        self.slot_params: Dict[int, int] = {}
        for slot in range(count):
            self.slots[slot] = plan.param(count - slot)
            self.slot_params[slot] = count - slot

        self.sp = count

    def run(self) -> Trace:
        block = self.func.basic_blocks[0] if self.func.basic_blocks else None
        try:
            while block is not None:
                block = self._run_block(block)

            raise VMFault('fell_off_function')

        except (_Returned, ThreadEnded):
            pass

        except _RunEnd as end:
            self.trace.status = end.status

        except VMFault as fault:
            self.world.fault(fault, self.address)

        return self.trace

    def _run_block(self, block: LowLevelILBasicBlock) -> Optional[LowLevelILBasicBlock]:
        for inst in block.instructions:
            self.steps += 1
            if self.steps > self.max_steps:
                raise _RunEnd(RunStatus.STEP_LIMIT)

            if inst.address:
                self.address = inst.address

            target = self._exec(inst)
            if target is not None:
                return target

        raise VMFault('missing_terminal', block.label)

    # ------------------------------------------------------------- storage --

    def _deref(self, pointer: Pointer) -> Any:
        if pointer.space != STACK_SPACE or pointer.key >= self.sp:
            return UNDEF

        return self.slots.get(pointer.key, UNDEF)

    def _param_of(self, expr: LowLevelILExpr) -> Optional[int]:
        if isinstance(expr, LowLevelILFrameLoad):
            return self.slot_params.get(expr.offset)

        if isinstance(expr, LowLevelILStackLoad):
            return self.slot_params.get(expr.slot_index)

        return None

    def _store_slot(self, slot: int, expr: LowLevelILExpr):
        param = self._param_of(expr)
        self.slots[slot] = self._eval(expr)
        if param is None:
            self.slot_params.pop(slot, None)

        else:
            self.slot_params[slot] = param

    def _pointer_target(self, pointer: LowLevelILExpr):
        '''(Pointer, None) for a stack address, (None, parameter number) for caller memory'''
        value = self._eval(pointer)
        if isinstance(value, Pointer):
            return value, None

        param = self._param_of(pointer)
        if param is None:
            raise VMFault('deref_non_pointer')

        return None, param

    # ---------------------------------------------------------- statements --

    def _exec(self, inst) -> Optional[LowLevelILBasicBlock]:
        if isinstance(inst, LowLevelILStackStore):
            self._store_slot(inst.slot_index, inst.value)

        elif isinstance(inst, LowLevelILFrameStore):
            self._store_slot(inst.offset, inst.value)

        elif isinstance(inst, LowLevelILSpAdd):
            self.sp += inst.delta

        elif isinstance(inst, LowLevelILDerefStore):
            value = self._eval(inst.value)
            pointer, param = self._pointer_target(inst.pointer)
            if pointer is not None:
                if pointer.space == STACK_SPACE and pointer.key < self.sp:
                    self.slots[pointer.key] = value
                    self.slot_params.pop(pointer.key, None)

            else:
                self.world.store_cell(param, value)

        elif isinstance(inst, LowLevelILRegStore):
            self.world.store_reg(inst.reg_index, self._eval(inst.value))

        elif isinstance(inst, LowLevelILGlobalStore):
            self.world.store_global(inst.index, self._eval(inst.value))

        elif isinstance(inst, LowLevelILSyscall):
            self._call((EVENT_SYSCALL, int(inst.subsystem), int(inst.cmd)), inst.args)

        elif isinstance(inst, LowLevelILCallScript):
            self._call((EVENT_EXTERN, str(inst.module), str(inst.func)), inst.args)
            return inst.return_target

        elif isinstance(inst, LowLevelILCallScriptNoReturn):
            # Records the call and ends the run (ThreadEnded)
            self._call((EVENT_EXTERN_NORETURN, str(inst.module), str(inst.func)), inst.args)

        elif isinstance(inst, LowLevelILCall):
            self._call((EVENT_CALL, str(inst.target)), inst.args)
            return inst.return_target

        elif isinstance(inst, LowLevelILIf):
            taken = not is_zero(self._eval(inst.condition))
            target = inst.true_target if taken else inst.false_target
            if target is None:
                raise VMFault('missing_branch_target')

            return target

        elif isinstance(inst, LowLevelILGoto):
            return inst.target

        elif isinstance(inst, LowLevelILRet):
            self.world.finish(self.world.regs.get(0, UNDEF), self._deref, self.address)
            raise _Returned()

        elif isinstance(inst, (LowLevelILDebug, LowLevelILLabelInstr, LowLevelILPushCallerFrame)):
            pass

        else:
            raise VMFault('unsupported_llil', type(inst).__name__)

        return None

    def _call(self, key_prefix, arg_exprs):
        args = [self._eval(arg) for arg in arg_exprs]
        outcome = self.world.external_call(key_prefix, args, self._deref, self.address)
        for pos, value in outcome.out_values.items():
            pointer = args[pos]
            if pointer.space == STACK_SPACE and pointer.key < self.sp:
                self.slots[pointer.key] = value
                self.slot_params.pop(pointer.key, None)

    # --------------------------------------------------------- expressions --

    def _eval(self, expr) -> Any:
        if isinstance(expr, LowLevelILConst):
            return normalize(Raw(int(expr.value)) if expr.is_raw else expr.value)

        if isinstance(expr, LowLevelILStackLoad):
            return self.slots.get(expr.slot_index, UNDEF)

        if isinstance(expr, LowLevelILFrameLoad):
            return self.slots.get(expr.offset, UNDEF)

        if isinstance(expr, LowLevelILStackAddr):
            return Pointer(STACK_SPACE, expr.slot_index)

        if isinstance(expr, LowLevelILDerefLoad):
            pointer, param = self._pointer_target(expr.pointer)
            if pointer is not None:
                return self._deref(pointer)

            return self.world.load_cell(param)

        if isinstance(expr, LowLevelILRegLoad):
            return self.world.load_reg(expr.reg_index)

        if isinstance(expr, LowLevelILGlobalLoad):
            return self.world.load_global(expr.index)

        op = BINARY_OPS.get(type(expr))
        if op is not None:
            return binary_op(op, self._eval(expr.lhs), self._eval(expr.rhs))

        op = UNARY_OPS.get(type(expr))
        if op is not None:
            return unary_op(op, self._eval(expr.operand))

        raise VMFault('unsupported_llil', type(expr).__name__)
