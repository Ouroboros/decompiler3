'''Behavior check of TVM: the bytecode reference interpreter and an HLIL interpreter with TVM semantics

Both run against the same inputs (parameters, initial globals and registers, call results, all drawn
from a seed by meaning) and record the same events: every call with its arguments and the globals
written so far, then the return value, or a fault.
'''

import random
from typing import Any, Dict, List, Optional, Tuple

from ir.core import ValueOp
from ir.hlil import (
    BinaryOp, HLIL_VALUE_OPS, HighLevelILFunction, HLILAssign, HLILBinaryOp, HLILBlock, HLILBreak, HLILCall, HLILComment,
    HLILConst, HLILContinue, HLILDoWhile, HLILExpression, HLILExprStmt, HLILExternalCall, HLILFor, HLILIf, HLILReturn,
    HLILSwitch, HLILUnaryOp, HLILVar, HLILWhile, UnaryOp, VariableKind,
)
from ir.mlil.mlil import mlil_arg_var_name

from .image import Function, Image
from .isa import WORD_SIZE, Op
from .target import RESULT_REGISTER, TVMFault, TVMHost, binary, truth, unary


INPUT_POOL      = (-2, -1, 0, 1, 2, 3, 5, 7, 10, 12)
MAX_STEPS       = 100_000
EVENT_CALL      = 'call'
EVENT_RETURN    = 'return'
EVENT_FAULT     = 'fault'
KIND_FUNCTION   = 'function'
KIND_HOST       = 'host'

BINARY_OPS = {
    Op.ADD: ValueOp.ADD, Op.SUB: ValueOp.SUB, Op.MUL: ValueOp.MUL, Op.DIV: ValueOp.DIV, Op.MOD: ValueOp.MOD,
    Op.EQ: ValueOp.EQ, Op.NE: ValueOp.NE, Op.LT: ValueOp.LT, Op.LE: ValueOp.LE, Op.GT: ValueOp.GT, Op.GE: ValueOp.GE,
    Op.AND: ValueOp.BIT_AND, Op.OR: ValueOp.BIT_OR,
}
UNARY_OPS = {Op.NEG: ValueOp.NEG, Op.NOT: ValueOp.BIT_NOT}
HLIL_UNARY_OPS = {UnaryOp.NEG: ValueOp.NEG, UnaryOp.NOT: ValueOp.LOGICAL_NOT, UnaryOp.BIT_NOT: ValueOp.BIT_NOT}


class _Fault(Exception):
    pass


class _Returned(Exception):
    pass


class World:
    '''Inputs drawn by meaning from a seed, plus globals, registers and the event trace'''

    def __init__(self, seed: Any):
        self.seed = seed
        self.globals: Dict[int, Any] = {}
        self.written: Dict[int, Any] = {}
        self.registers: Dict[int, Any] = {}
        self.calls = 0
        self.events: List[Tuple] = []

    def draw(self, *origin) -> int:
        return random.Random(':'.join(str(part) for part in (self.seed,) + origin)).choice(INPUT_POOL)

    def param(self, number: int) -> int:
        return self.draw('param', number)

    def load_global(self, index: int) -> Any:
        if index not in self.globals:
            self.globals[index] = self.draw('global', index)

        return self.globals[index]

    def store_global(self, index: int, value: Any):
        self.globals[index] = value
        self.written[index] = value

    def load_register(self, index: int) -> Any:
        if index not in self.registers:
            self.registers[index] = self.draw('register', index)

        return self.registers[index]

    def store_register(self, index: int, value: Any):
        self.registers[index] = value

    def call(self, kind: str, name: str, args: List[Any]) -> Any:
        self.events.append((EVENT_CALL, kind, name, tuple(args), tuple(sorted(self.written.items()))))
        result = self.draw('result', self.calls)
        self.calls += 1
        self.registers[RESULT_REGISTER] = result
        return result

    def finish(self, value: Any):
        self.events.append((EVENT_RETURN, value, tuple(sorted(self.written.items()))))

    def fault(self, reason: str):
        self.events.append((EVENT_FAULT,))


class BytecodeMachine:
    '''Reference: executes the TVM bytecode of one function'''

    def __init__(self, image: Image, func: Function, world: World):
        self.image = image
        self.func = func
        self.world = world
        self.code = {inst.offset: inst for inst in func.instructions}
        self.stack: List[Any] = [world.param(number) for number in range(1, len(func.param_flags) + 1)]

    def run(self) -> List[Tuple]:
        try:
            self._run()

        except _Returned:
            pass

        except (TVMFault, _Fault) as fault:
            self.world.fault(str(fault))

        return self.world.events

    def _pop(self) -> Any:
        if not self.stack:
            raise _Fault('stack underflow')

        return self.stack.pop()

    def _slot(self, byte_offset: int) -> int:
        index = len(self.stack) + byte_offset // WORD_SIZE
        if not 0 <= index < len(self.stack):
            raise _Fault('stack offset out of range')

        return index

    def _run(self):
        pc, steps = 0, 0
        while True:
            steps += 1
            if steps > MAX_STEPS:
                raise _Fault('step limit')

            inst = self.code[pc]
            op, operands = inst.op, inst.operands
            pc = inst.offset + inst.size
            if op == Op.PUSH:
                self.stack.append(operands[0])

            elif op == Op.PUSHS:
                self.stack.append(self.image.strings[operands[0]])

            elif op == Op.LDL:
                self.stack.append(self.stack[self._slot(operands[0])])

            elif op == Op.STL:
                value = self._pop()
                self.stack[self._slot(operands[0])] = value

            elif op == Op.LDG:
                self.stack.append(self.world.load_global(operands[0]))

            elif op == Op.STG:
                self.world.store_global(operands[0], self._pop())

            elif op == Op.LDR:
                self.stack.append(self.world.load_register(operands[0]))

            elif op == Op.STR:
                self.world.store_register(operands[0], self._pop())

            elif op == Op.DROP:
                for _ in range(operands[0]):
                    self._pop()

            elif op in BINARY_OPS:
                rhs, lhs = self._pop(), self._pop()
                self.stack.append(binary(BINARY_OPS[op], lhs, rhs))

            elif op in UNARY_OPS:
                self.stack.append(unary(UNARY_OPS[op], self._pop()))

            elif op == Op.JMP:
                pc = operands[0]

            elif op in (Op.JZ, Op.JNZ):
                if truth(self._pop()) == (op == Op.JNZ):
                    pc = operands[0]

            elif op in (Op.CALL, Op.HOST):
                argc = operands[1]
                args = self.stack[len(self.stack) - argc:] if argc else []
                del self.stack[len(self.stack) - argc:]
                if op == Op.CALL:
                    self.world.call(KIND_FUNCTION, self.image.functions[operands[0]].name, args)

                else:
                    self.world.call(KIND_HOST, self.image.strings[operands[0]], args)

            elif op == Op.RET:
                self.world.finish(self.world.load_register(RESULT_REGISTER))
                raise _Returned()

            elif op in (Op.LINE, Op.LOG):
                pass

            else:
                raise _Fault(f'unknown op {op!r}')


class _Signal(Exception):
    def __init__(self, kind: str, label: Optional[str] = None):
        super().__init__(kind)
        self.kind = kind
        self.label = label


SIGNAL_BREAK    = 'break'
SIGNAL_CONTINUE = 'continue'


class HLILMachine:
    '''Executes decompiled HLIL with TVM semantics against the same world'''

    def __init__(self, func: HighLevelILFunction, param_count: int, world: World):
        self.func = func
        self.world = world
        self.locals: Dict[str, Any] = {mlil_arg_var_name(number): world.param(number) for number in range(1, param_count + 1)}
        self.steps = 0

    def run(self) -> List[Tuple]:
        try:
            self._exec(self.func.body)
            self.world.finish(self.world.load_register(RESULT_REGISTER))

        except _Returned:
            pass

        except (TVMFault, _Fault) as fault:
            self.world.fault(str(fault))

        return self.world.events

    def _tick(self):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise _Fault('step limit')

    def _exec(self, stmt):
        self._tick()
        if isinstance(stmt, HLILBlock):
            for child in stmt.statements:
                self._exec(child)

        elif isinstance(stmt, HLILComment):
            pass

        elif isinstance(stmt, HLILAssign):
            self._assign(stmt.dest, self._eval(stmt.src))

        elif isinstance(stmt, HLILExprStmt):
            self._eval(stmt.expr)

        elif isinstance(stmt, HLILIf):
            if truth(self._eval(stmt.condition)):
                self._exec(stmt.true_block)

            elif stmt.false_block is not None:
                self._exec(stmt.false_block)

        elif isinstance(stmt, (HLILWhile, HLILDoWhile, HLILFor)):
            self._loop(stmt)

        elif isinstance(stmt, HLILSwitch):
            self._switch(stmt)

        elif isinstance(stmt, HLILBreak):
            raise _Signal(SIGNAL_BREAK, stmt.label)

        elif isinstance(stmt, HLILContinue):
            raise _Signal(SIGNAL_CONTINUE, stmt.label)

        elif isinstance(stmt, HLILReturn):
            value = self._eval(stmt.value) if stmt.value is not None else self.world.load_register(RESULT_REGISTER)
            self.world.finish(value)
            raise _Returned()

        else:
            raise _Fault(f'unsupported statement {type(stmt).__name__}')

    def _loop(self, stmt):
        label = getattr(stmt, 'label', None)
        if isinstance(stmt, HLILFor) and stmt.init is not None:
            self._exec(stmt.init)

        first = True
        while True:
            check = not (isinstance(stmt, HLILDoWhile) and first)
            first = False
            if check and stmt.condition is not None and not truth(self._eval(stmt.condition)):
                return

            try:
                self._exec(stmt.body)

            except _Signal as signal:
                if signal.label not in (None, label):
                    raise

                if signal.kind == SIGNAL_BREAK:
                    return

            if isinstance(stmt, HLILFor) and stmt.update is not None:
                self._exec(stmt.update)

    def _switch(self, stmt: HLILSwitch):
        scrutinee = self._eval(stmt.scrutinee)
        chosen = next((case for case in stmt.cases if not case.is_default()
                       and any(truth(binary(ValueOp.EQ, scrutinee, self._eval(value))) for value in case.values)), None)
        chosen = chosen or next((case for case in stmt.cases if case.is_default()), None)
        if chosen is None:
            return

        try:
            self._exec(chosen.body)

        except _Signal as signal:
            if not (signal.kind == SIGNAL_BREAK and signal.label is None):
                raise

    def _assign(self, dest, value: Any):
        if not isinstance(dest, HLILVar):
            raise _Fault(f'unsupported assignment to {type(dest).__name__}')

        var = dest.var
        if var.kind == VariableKind.GLOBAL:
            self.world.store_global(var.index, value)

        elif var.kind == VariableKind.REG:
            self.world.store_register(var.index, value)

        else:
            self.locals[var.name] = value

    def _eval(self, expr: HLILExpression) -> Any:
        if isinstance(expr, HLILConst):
            return expr.value

        if isinstance(expr, HLILVar):
            var = expr.var
            if var.kind == VariableKind.GLOBAL:
                return self.world.load_global(var.index)

            if var.kind == VariableKind.REG:
                return self.world.load_register(var.index)

            if var.name not in self.locals:
                raise _Fault(f'read of unset {var.name}')

            return self.locals[var.name]

        if isinstance(expr, HLILBinaryOp):
            if expr.op == BinaryOp.AND:
                return int(truth(self._eval(expr.lhs)) and truth(self._eval(expr.rhs)))

            if expr.op == BinaryOp.OR:
                return int(truth(self._eval(expr.lhs)) or truth(self._eval(expr.rhs)))

            return binary(HLIL_VALUE_OPS[expr.op], self._eval(expr.lhs), self._eval(expr.rhs))

        if isinstance(expr, HLILUnaryOp):
            return unary(HLIL_UNARY_OPS[expr.op], self._eval(expr.operand))

        if isinstance(expr, HLILCall):
            return self.world.call(KIND_FUNCTION, str(expr.func_name), [self._eval(arg) for arg in expr.args])

        if isinstance(expr, HLILExternalCall) and isinstance(expr.target, TVMHost):
            return self.world.call(KIND_HOST, expr.target.name, [self._eval(arg) for arg in expr.args])

        raise _Fault(f'unsupported expression {type(expr).__name__}')
