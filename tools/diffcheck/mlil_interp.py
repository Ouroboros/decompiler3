'''MLIL interpreter: executes MLIL, in or out of SSA form, against the same world as the bytecode

- variables are keyed by name, SSA versions by (name, version); version 0 is the entry value
  (the parameter for argN, UNDEF otherwise)
- phi nodes take the source of the predecessor the block was entered from, all at once
- a variable whose address is taken also lives in memory: every assignment updates it, writes
  through pointers and call output parameters change it, and the SSA pseudo-definition
  `var#n = undef` after such a write reads it back
- a dereferenced non-pointer must be a parameter (or a copy of one) holding a pointer into caller memory
- every external call leaves its result in REGS[0]; a return without a value returns REGS[0]
- operators and branch tests have VM semantics (falcom/ed9/vm_values.py)
'''

from typing import Any, Dict, Optional

from falcom.ed9.vm_values import Op, Pointer, UNDEF, VMFault, binary_op, is_zero, normalize, unary_op
from ir.mlil import (
    MediumLevelILBasicBlock, MediumLevelILFunction, MLILAdd, MLILAddressOf, MLILAnd, MLILBitwiseNot, MLILCall,
    MLILConst, MLILDebug, MLILDeref, MLILDiv, MLILEq, MLILGe, MLILGoto, MLILGt, MLILIf, MLILLe,
    MLILLoadGlobal, MLILLoadReg, MLILLogicalAnd, MLILLogicalNot, MLILLogicalOr, MLILLt, MLILMod, MLILMul, MLILNe,
    MLILNeg, MLILNop, MLILOr, MLILRet, MLILSetVar, MLILShl, MLILShr, MLILStoreDeref, MLILStoreGlobal, MLILStoreReg,
    MLILSub, MLILTestZero, MLILUndef, MLILVar, MLILXor,
)
from ir.mlil.mlil import mlil_arg_var_name
from ir.mlil.mlil_ssa import MLILPhi, MLILSetVarSSA, MLILVarSSA

from .environment import MAX_EVENTS, MAX_STEPS, InputPlan, RunStatus, ThreadEnded, Trace, WorldState, _RunEnd, call_event_key


LOCAL_SPACE     = 'local'
ENTRY_VERSION   = 0

BINARY_OPS = {
    MLILAdd:        Op.ADD,
    MLILSub:        Op.SUB,
    MLILMul:        Op.MUL,
    MLILDiv:        Op.DIV,
    MLILMod:        Op.MOD,
    MLILAnd:        Op.BIT_AND,
    MLILOr:         Op.BIT_OR,
    MLILXor:        Op.BIT_XOR,
    MLILShl:        Op.SHL,
    MLILShr:        Op.SHR,
    MLILLogicalAnd: Op.LOGICAL_AND,
    MLILLogicalOr:  Op.LOGICAL_OR,
    MLILEq:         Op.EQ,
    MLILNe:         Op.NE,
    MLILLt:         Op.LT,
    MLILLe:         Op.LE,
    MLILGt:         Op.GT,
    MLILGe:         Op.GE,
}

UNARY_OPS = {
    MLILNeg:        Op.NEG,
    MLILLogicalNot: Op.LOGICAL_NOT,
    MLILBitwiseNot: Op.BIT_NOT,
    MLILTestZero:   Op.TEST_ZERO,
}


class _Returned(Exception):
    pass


class MLILInterpreter:
    '''Executes one MLIL function against an InputPlan'''

    def __init__(self, mlil_func: MediumLevelILFunction, param_count: int, plan: InputPlan, *,
                 max_events: int = MAX_EVENTS, max_steps: int = MAX_STEPS):
        self.func = mlil_func
        self.plan = plan
        self.max_steps = max_steps
        self.trace = Trace(max_events = max_events)
        self.world = WorldState(plan, self.trace)
        self.steps = 0
        self.address = mlil_func.start_addr
        self.values: Dict[Any, Any] = {}
        # Parameter number whose value a variable holds unchanged, for pointers into caller memory
        self.value_params: Dict[Any, int] = {}
        self.memory: Dict[str, Any] = {}
        self.address_taken = set(getattr(mlil_func, 'address_taken', ()))
        self.param_numbers = {mlil_arg_var_name(number): number for number in range(1, param_count + 1)}

    def run(self) -> Trace:
        block = self.func.basic_blocks[0] if self.func.basic_blocks else None
        previous = None
        try:
            while block is not None:
                block, previous = self._run_block(block, previous), block

            raise VMFault('fell_off_function')

        except (_Returned, ThreadEnded):
            pass

        except _RunEnd as end:
            self.trace.status = end.status

        except VMFault as fault:
            self.world.fault(fault, self.address)

        return self.trace

    def _run_block(self, block: MediumLevelILBasicBlock, previous: Optional[MediumLevelILBasicBlock]):
        phis = [inst for inst in block.instructions if isinstance(inst, MLILPhi)]
        if phis:
            # Every phi reads the values live on the incoming edge before any of them is written
            incoming = [(phi, self._phi_source(phi, previous)) for phi in phis]
            for phi, source in incoming:
                self._assign(phi.dest, source[0], source[1])

        for inst in block.instructions:
            if isinstance(inst, MLILPhi):
                continue

            self.steps += 1
            if self.steps > self.max_steps:
                raise _RunEnd(RunStatus.STEP_LIMIT)

            if inst.address:
                self.address = inst.address

            target = self._exec(inst)
            if target is not None:
                return target

        raise VMFault('missing_terminal', block.label)

    def _phi_source(self, phi: MLILPhi, previous: Optional[MediumLevelILBasicBlock]):
        for var, pred in phi.sources:
            if pred is previous:
                key = self._key(var)
                return self._load_key(key, var), self.value_params.get(key)

        return UNDEF, None

    # ------------------------------------------------------------- storage --

    @classmethod
    def _key(cls, var) -> Any:
        if hasattr(var, 'base_var'):
            return (var.base_var.name, var.version)

        return var.name

    @classmethod
    def _base_name(cls, var) -> str:
        return var.base_var.name if hasattr(var, 'base_var') else var.name

    def _load_key(self, key: Any, var) -> Any:
        if key in self.values:
            return self.values[key]

        name = self._base_name(var)
        entry = isinstance(key, str) or key[1] == ENTRY_VERSION
        if entry and name in self.param_numbers:
            return self.plan.param(self.param_numbers[name])

        if name in self.memory and isinstance(key, str):
            return self.memory[name]

        return UNDEF

    def _entry_param(self, key: Any, var) -> Optional[int]:
        if key in self.values:
            return self.value_params.get(key)

        name = self._base_name(var)
        entry = isinstance(key, str) or key[1] == ENTRY_VERSION
        return self.param_numbers.get(name) if entry else None

    def _assign(self, var, value: Any, param: Optional[int]):
        key = self._key(var)
        self.values[key] = value
        if param is None:
            self.value_params.pop(key, None)

        else:
            self.value_params[key] = param

        name = self._base_name(var)
        if name in self.address_taken or isinstance(key, str):
            self.memory[name] = value

    def _deref(self, pointer: Pointer) -> Any:
        if pointer.space != LOCAL_SPACE:
            return UNDEF

        return self.memory.get(pointer.key, UNDEF)

    def _write_memory(self, pointer: Pointer, value: Any):
        if pointer.space != LOCAL_SPACE:
            return

        self.memory[pointer.key] = value
        # Out of SSA the variable itself is the memory cell
        self.values[pointer.key] = value
        self.value_params.pop(pointer.key, None)

    def _param_of(self, expr) -> Optional[int]:
        if isinstance(expr, (MLILVar, MLILVarSSA)):
            return self._entry_param(self._key(expr.var), expr.var)

        return None

    def _pointer_target(self, pointer_expr):
        '''(Pointer, None) for a variable's address, (None, parameter number) for caller memory'''
        value = self._eval(pointer_expr)
        if isinstance(value, Pointer):
            return value, None

        param = self._param_of(pointer_expr)
        if param is None:
            raise VMFault('deref_non_pointer')

        return None, param

    # ---------------------------------------------------------- statements --

    def _exec(self, inst) -> Optional[MediumLevelILBasicBlock]:
        if isinstance(inst, (MLILSetVar, MLILSetVarSSA)):
            name = self._base_name(inst.var)
            if isinstance(inst.value, MLILUndef) and name in self.address_taken:
                # Pseudo-definition after a write through a pointer: the variable holds what was written
                self._assign(inst.var, self.memory.get(name, UNDEF), None)

            else:
                self._assign(inst.var, self._eval(inst.value), self._param_of(inst.value))

        elif isinstance(inst, MLILStoreDeref):
            value = self._eval(inst.value)
            pointer, param = self._pointer_target(inst.pointer)
            if pointer is not None:
                self._write_memory(pointer, value)

            else:
                self.world.store_cell(param, value)

        elif isinstance(inst, MLILStoreReg):
            self.world.store_reg(inst.index, self._eval(inst.value))

        elif isinstance(inst, MLILStoreGlobal):
            self.world.store_global(inst.index, self._eval(inst.value))

        elif isinstance(inst, MLILCall):
            self._call(call_event_key(inst.target), inst.args)

        elif isinstance(inst, MLILIf):
            taken = not is_zero(self._eval(inst.condition))
            target = inst.true_target if taken else inst.false_target
            if target is None:
                raise VMFault('missing_branch_target')

            return target

        elif isinstance(inst, MLILGoto):
            return inst.target

        elif isinstance(inst, MLILRet):
            value = self._eval(inst.value) if inst.value is not None else self.world.regs.get(0, UNDEF)
            self.world.finish(value, self._deref, self.address)
            raise _Returned()

        elif isinstance(inst, (MLILNop, MLILDebug)):
            pass

        else:
            raise VMFault('unsupported_mlil', type(inst).__name__)

        return None

    def _call(self, key_prefix, arg_exprs):
        args = [self._eval(arg) for arg in arg_exprs]
        outcome = self.world.external_call(key_prefix, args, self._deref, self.address)
        for pos, value in outcome.out_values.items():
            self._write_memory(args[pos], value)

    # --------------------------------------------------------- expressions --

    def _eval(self, expr) -> Any:
        if isinstance(expr, MLILConst):
            return normalize(expr.value)

        if isinstance(expr, (MLILVar, MLILVarSSA)):
            return self._load_key(self._key(expr.var), expr.var)

        if isinstance(expr, MLILUndef):
            return UNDEF

        if isinstance(expr, MLILAddressOf):
            if not isinstance(expr.operand, (MLILVar, MLILVarSSA)):
                raise VMFault('unsupported_address_of', type(expr.operand).__name__)

            name = self._base_name(expr.operand.var)
            if name not in self.memory:
                self.memory[name] = self._eval(expr.operand)

            return Pointer(LOCAL_SPACE, name)

        if isinstance(expr, MLILDeref):
            pointer, param = self._pointer_target(expr.operand)
            if pointer is not None:
                return self._deref(pointer)

            return self.world.load_cell(param)

        if isinstance(expr, MLILLoadReg):
            return self.world.load_reg(expr.index)

        if isinstance(expr, MLILLoadGlobal):
            return self.world.load_global(expr.index)

        op = BINARY_OPS.get(type(expr))
        if op is not None:
            return binary_op(op, self._eval(expr.lhs), self._eval(expr.rhs))

        op = UNARY_OPS.get(type(expr))
        if op is not None:
            return unary_op(op, self._eval(expr.operand))

        raise VMFault('unsupported_mlil', type(expr).__name__)
