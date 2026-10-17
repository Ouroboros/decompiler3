'''HLIL interpreter: control flow of the generated TypeScript, values with VM semantics

- operators evaluate as the VM opcodes they were lifted from (values.py)
- conditions of if/while and ! test the 30-bit payload, like POP_JMP_ZERO
- && and || short-circuit, test each operand like LOGICAL_AND / LOGICAL_OR and yield 1/0
- call arguments are evaluated left to right
- switch cases do not fall through; an unlabeled break inside a case leaves the switch
- labeled break/continue target the loop carrying that label
- every external call stores its result in REGS[0], as the VM does
- a return without a value returns REGS[0]
'''

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ir.hlil import (
    BinaryOp, HighLevelILFunction, HLILAddressOf, HLILAssign, HLILBinaryOp, HLILBlock, HLILBreak, HLILCall,
    HLILComment, HLILConst, HLILContinue, HLILDeref, HLILDoWhile, HLILExpression, HLILExprStmt, HLILExternCall, HLILFor,
    HLILIf, HLILReturn, HLILStatement, HLILSwitch, HLILSyscall, HLILUnaryOp, HLILVar, HLILWhile, UnaryOp,
    VariableKind,
)
from ir.mlil.mlil import mlil_arg_var_name

from .environment import (
    EVENT_CALL, EVENT_EXTERN, EVENT_SYSCALL, MAX_EVENTS, MAX_STEPS, InputPlan, RunStatus, Trace, WorldState, _RunEnd,
)
from .values import Op, Pointer, UNDEF, VMFault, binary_op, bool_value, is_zero, logical_truth, normalize, unary_op


SPACE_LOCAL     = 'local'
SPACE_GLOBAL    = 'global'
SPACE_REG       = 'reg'
EXTERN_SEPARATOR = ':'

SIGNAL_BREAK    = 'break'
SIGNAL_CONTINUE = 'continue'
SIGNAL_RETURN   = 'return'

BINARY_OPS = {
    BinaryOp.ADD:       Op.ADD,
    BinaryOp.SUB:       Op.SUB,
    BinaryOp.MUL:       Op.MUL,
    BinaryOp.DIV:       Op.DIV,
    BinaryOp.MOD:       Op.MOD,
    BinaryOp.EQ:        Op.EQ,
    BinaryOp.NE:        Op.NE,
    BinaryOp.LT:        Op.LT,
    BinaryOp.LE:        Op.LE,
    BinaryOp.GT:        Op.GT,
    BinaryOp.GE:        Op.GE,
    BinaryOp.BIT_AND:   Op.BIT_AND,
    BinaryOp.BIT_OR:    Op.BIT_OR,
    BinaryOp.BIT_XOR:   Op.BIT_XOR,
    BinaryOp.SHL:       Op.SHL,
    BinaryOp.SHR:       Op.SHR,
}

UNARY_OPS = {
    UnaryOp.NEG:        Op.NEG,
    UnaryOp.NOT:        Op.LOGICAL_NOT,
    UnaryOp.BIT_NOT:    Op.BIT_NOT,
}


@dataclass
class _Signal:
    kind: str
    label: Optional[str] = None
    value: Any = None


class HLILInterpreter:
    '''Executes one HLIL function against an InputPlan'''

    def __init__(self, hlil_func: HighLevelILFunction, param_count: int, plan: InputPlan, *,
                 max_events: int = MAX_EVENTS, max_steps: int = MAX_STEPS):
        self.func = hlil_func
        self.plan = plan
        self.max_steps = max_steps
        self.trace = Trace(max_events = max_events)
        self.world = WorldState(plan, self.trace)
        self.locals: Dict[str, Any] = {}
        self.steps = 0
        self.address = hlil_func.start_addr
        self.param_numbers: Dict[str, int] = {}
        for number in range(1, param_count + 1):
            self.locals[mlil_arg_var_name(number)] = plan.param(number)
            self.param_numbers[mlil_arg_var_name(number)] = number

    def run(self) -> Trace:
        try:
            signal = self._exec(self.func.body)
            if signal is not None and signal.kind != SIGNAL_RETURN:
                raise VMFault('stray_jump', signal.kind)

            value = signal.value if signal is not None else self.world.regs.get(0, UNDEF)
            self.world.finish(value, self._deref, self.address)

        except _RunEnd as end:
            self.trace.status = end.status

        except VMFault as fault:
            self.world.fault(fault, self.address)

        return self.trace

    # ------------------------------------------------------------- storage --

    def _pointer_to(self, expr: HLILExpression) -> Pointer:
        if not isinstance(expr, HLILVar):
            raise VMFault('unsupported_address_of', type(expr).__name__)

        var = expr.var
        if var.kind == VariableKind.GLOBAL:
            return Pointer(SPACE_GLOBAL, var.index)

        if var.kind == VariableKind.REG:
            return Pointer(SPACE_REG, var.index)

        return Pointer(SPACE_LOCAL, var.name)

    def _deref(self, pointer: Pointer) -> Any:
        if pointer.space == SPACE_GLOBAL:
            return self.world.load_global(pointer.key)

        if pointer.space == SPACE_REG:
            return self.world.regs.get(pointer.key, UNDEF)

        return self.locals.get(pointer.key, UNDEF)

    def _store_pointer(self, pointer: Pointer, value: Any):
        if pointer.space == SPACE_GLOBAL:
            self.world.store_global(pointer.key, value)

        elif pointer.space == SPACE_REG:
            self.world.regs[pointer.key] = value

        else:
            self.locals[pointer.key] = value

    def _load_var(self, expr: HLILVar) -> Any:
        var = expr.var
        if var.kind == VariableKind.GLOBAL:
            return self.world.load_global(var.index)

        if var.kind == VariableKind.REG:
            return self.world.regs.get(var.index, UNDEF)

        return self.locals.get(var.name, UNDEF)

    def _cell_param(self, pointer: HLILExpression) -> Optional[int]:
        '''Parameter number when the pointer is a parameter holding a pointer into caller memory'''
        if isinstance(pointer, HLILVar) and pointer.var.kind in (VariableKind.PARAM, VariableKind.LOCAL):
            return self.param_numbers.get(pointer.var.name)

        return None

    def _load_deref(self, pointer: HLILExpression) -> Any:
        number = self._cell_param(pointer)
        if number is not None:
            return self.world.load_cell(number)

        target = self._eval(pointer)
        if not isinstance(target, Pointer):
            raise VMFault('deref_non_pointer')

        return self._deref(target)

    def _store_deref(self, pointer: HLILExpression, value: Any):
        number = self._cell_param(pointer)
        if number is not None:
            self.world.store_cell(number, value)
            return

        target = self._eval(pointer)
        if not isinstance(target, Pointer):
            raise VMFault('deref_non_pointer')

        self._store_pointer(target, value)

    def _assign(self, dest: HLILExpression, value: Any):
        if isinstance(dest, HLILDeref):
            self._store_deref(dest.operand, value)
            return

        if not isinstance(dest, HLILVar):
            raise VMFault('unsupported_assign_target', type(dest).__name__)

        var = dest.var
        if var.kind == VariableKind.GLOBAL:
            self.world.store_global(var.index, value)

        elif var.kind == VariableKind.REG:
            self.world.regs[var.index] = value

        else:
            self.locals[var.name] = value

    # ---------------------------------------------------------- statements --

    def _tick(self, node):
        self.steps += 1
        if self.steps > self.max_steps:
            raise _RunEnd(RunStatus.STEP_LIMIT)

        if node.address:
            self.address = node.address

    def _exec(self, stmt: HLILStatement) -> Optional[_Signal]:
        self._tick(stmt)

        if isinstance(stmt, HLILBlock):
            for child in stmt.statements:
                signal = self._exec(child)
                if signal is not None:
                    return signal

            return None

        if isinstance(stmt, HLILComment):
            return None

        if isinstance(stmt, HLILAssign):
            self._assign(stmt.dest, self._eval(stmt.src))
            return None

        if isinstance(stmt, HLILExprStmt):
            self._eval(stmt.expr)
            return None

        if isinstance(stmt, HLILIf):
            if not is_zero(self._eval(stmt.condition)):
                return self._exec(stmt.true_block)

            if stmt.false_block is not None:
                return self._exec(stmt.false_block)

            return None

        if isinstance(stmt, HLILWhile):
            return self._exec_loop(stmt.condition, stmt.body, stmt.label, None, check_first = True)

        if isinstance(stmt, HLILDoWhile):
            return self._exec_loop(stmt.condition, stmt.body, None, None, check_first = False)

        if isinstance(stmt, HLILFor):
            if stmt.init is not None:
                self._exec(stmt.init)

            return self._exec_loop(stmt.condition, stmt.body, None, stmt.update, check_first = True)

        if isinstance(stmt, HLILSwitch):
            return self._exec_switch(stmt)

        if isinstance(stmt, HLILBreak):
            return _Signal(SIGNAL_BREAK, stmt.label)

        if isinstance(stmt, HLILContinue):
            return _Signal(SIGNAL_CONTINUE, stmt.label)

        if isinstance(stmt, HLILReturn):
            value = self._eval(stmt.value) if stmt.value is not None else self.world.regs.get(0, UNDEF)
            return _Signal(SIGNAL_RETURN, value = value)

        raise VMFault('unsupported_statement', type(stmt).__name__)

    def _exec_loop(self, condition: Optional[HLILExpression], body: HLILBlock, label: Optional[str],
                   update: Optional[HLILStatement], *, check_first: bool) -> Optional[_Signal]:
        first = True
        while True:
            if (check_first or not first) and condition is not None:
                self._tick(body)
                if is_zero(self._eval(condition)):
                    return None

            first = False
            signal = self._exec(body)
            if signal is not None:
                owned = signal.label is None or signal.label == label
                if signal.kind == SIGNAL_BREAK and owned:
                    return None

                if not (signal.kind == SIGNAL_CONTINUE and owned):
                    return signal

            if update is not None:
                self._exec(update)

    def _exec_switch(self, stmt: HLILSwitch) -> Optional[_Signal]:
        scrutinee = self._eval(stmt.scrutinee)
        chosen = None
        for case in stmt.cases:
            if not case.is_default() and binary_op(Op.EQ, scrutinee, self._eval(case.value)) == bool_value(True):
                chosen = case
                break

        if chosen is None:
            chosen = next((case for case in stmt.cases if case.is_default()), None)

        if chosen is None:
            return None

        signal = self._exec(chosen.body)
        if signal is not None and signal.kind == SIGNAL_BREAK and signal.label is None:
            return None

        return signal

    # --------------------------------------------------------- expressions --

    def _eval(self, expr: HLILExpression) -> Any:
        if isinstance(expr, HLILConst):
            return normalize(expr.value)

        if isinstance(expr, HLILVar):
            return self._load_var(expr)

        if isinstance(expr, HLILBinaryOp):
            if expr.op == BinaryOp.AND:
                return bool_value(logical_truth(self._eval(expr.lhs)) and logical_truth(self._eval(expr.rhs)))

            if expr.op == BinaryOp.OR:
                return bool_value(logical_truth(self._eval(expr.lhs)) or logical_truth(self._eval(expr.rhs)))

            lhs = self._eval(expr.lhs)
            rhs = self._eval(expr.rhs)
            return binary_op(BINARY_OPS[expr.op], lhs, rhs)

        if isinstance(expr, HLILDeref):
            return self._load_deref(expr.operand)

        if isinstance(expr, HLILUnaryOp):
            return unary_op(UNARY_OPS[expr.op], self._eval(expr.operand))

        if isinstance(expr, HLILAddressOf):
            return self._pointer_to(expr.operand)

        if isinstance(expr, HLILCall):
            return self._call((EVENT_CALL, str(expr.func_name)), expr)

        if isinstance(expr, HLILSyscall):
            return self._call((EVENT_SYSCALL, int(expr.subsystem), int(expr.cmd)), expr)

        if isinstance(expr, HLILExternCall):
            module, _, func_name = str(expr.target).partition(EXTERN_SEPARATOR)
            return self._call((EVENT_EXTERN, module, func_name), expr)

        raise VMFault('unsupported_expression', type(expr).__name__)

    def _call(self, key_prefix, expr) -> Any:
        self._tick(expr)
        args: List[Any] = [self._eval(arg) for arg in expr.args]
        outcome = self.world.external_call(key_prefix, args, self._deref, self.address)
        for pos, value in outcome.out_values.items():
            self._store_pointer(args[pos], value)

        return outcome.return_value
