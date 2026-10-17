'''Symbolic values for branch-directed input search

The reference interpreter attaches to every value derived from run inputs the
expression that produced it. When a branch went one way, the checker looks for
the smallest change of inputs that sends it the other way.
'''

import itertools
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from falcom.ed9.vm_values import Op, UNDEF, VMFault, binary_op, is_zero, normalize, unary_op

from .environment import Origin


# Larger expressions keep only their inputs and are not solved
MAX_SYM_SIZE        = 64
MAX_SOLVE_EVALS     = 4096
UNARY_ARITY         = 1
NEIGHBOR_DELTA      = 1


class Sym:
    '''Expression over run inputs'''
    inputs: FrozenSet[Origin] = frozenset()
    size: int = 1


class SymInput(Sym):
    def __init__(self, origin: Origin):
        self.origin = origin
        self.inputs = frozenset({origin})


class SymConst(Sym):
    def __init__(self, value: Any):
        self.value = value


class SymOp(Sym):
    def __init__(self, op: Op, operands: Tuple[Sym, ...]):
        self.op = op
        self.operands = operands
        self.inputs = frozenset().union(*(operand.inputs for operand in operands))
        self.size = 1 + sum(operand.size for operand in operands)


class SymOpaque(Sym):
    '''Expression too large to evaluate'''

    def __init__(self, inputs: FrozenSet[Origin]):
        self.inputs = inputs


class _Unsolvable(Exception):
    pass


def make_op(op: Op, values: Sequence[Any], syms: Sequence[Optional[Sym]]) -> Optional[Sym]:
    '''Symbolic result of op, or None when no operand depends on an input'''
    if all(sym is None for sym in syms):
        return None

    operands = tuple(sym if sym is not None else SymConst(value) for value, sym in zip(values, syms))
    inputs = frozenset().union(*(operand.inputs for operand in operands))
    if any(isinstance(operand, SymOpaque) for operand in operands):
        return SymOpaque(inputs)

    if sum(operand.size for operand in operands) >= MAX_SYM_SIZE:
        return SymOpaque(inputs)

    return SymOp(op, operands)


def evaluate(sym: Sym, assignment: Dict[Origin, Any], current: Dict[Origin, Any]) -> Any:
    if isinstance(sym, SymConst):
        return sym.value

    if isinstance(sym, SymInput):
        if sym.origin in assignment:
            return assignment[sym.origin]

        return current.get(sym.origin, UNDEF)

    if isinstance(sym, SymOp):
        operands = [evaluate(operand, assignment, current) for operand in sym.operands]
        if len(operands) == UNARY_ARITY:
            return unary_op(sym.op, operands[0])

        return binary_op(sym.op, operands[0], operands[1])

    raise _Unsolvable()


def literals(sym: Sym) -> List[Any]:
    '''Numeric constants of an expression and their neighbors, the natural values to flip it'''
    found: List[Any] = []
    pending = [sym]
    while pending:
        node = pending.pop()
        if isinstance(node, SymConst) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            found.extend((node.value, node.value - NEIGHBOR_DELTA, node.value + NEIGHBOR_DELTA))

        elif isinstance(node, SymOp):
            pending.extend(node.operands)

    return list(dict.fromkeys(normalize(value) for value in found))


def _holds(sym: Sym, observed_zero: bool, assignment: Dict[Origin, Any], current: Dict[Origin, Any]) -> bool:
    try:
        return is_zero(evaluate(sym, assignment, current)) == observed_zero

    except VMFault:
        return False


def flip(condition: Sym, observed_zero: bool, current: Dict[Origin, Any],
         candidates: Dict[Origin, Sequence[Any]],
         path: Sequence[Tuple[Sym, bool]] = ()) -> Optional[Dict[Origin, Any]]:
    '''Fewest input changes that make the branch take the other side while earlier branches keep theirs

    path holds (condition, observed_zero) of the branches executed before this one.
    '''
    if isinstance(condition, SymOpaque):
        return None

    inputs = sorted(condition.inputs, key = str)
    # Only earlier branches reading a changed input can change direction
    path = [(sym, zero) for sym, zero in path if sym.inputs.intersection(condition.inputs) and not isinstance(sym, SymOpaque)]
    extra = literals(condition)
    alternatives = {
        origin: [value for value in dict.fromkeys(list(candidates.get(origin, ())) + extra) if value != current.get(origin)]
        for origin in inputs
    }
    evals = 0
    for count in range(1, len(inputs) + 1):
        for chosen in itertools.combinations(inputs, count):
            evals += 1
            if evals > MAX_SOLVE_EVALS:
                return None

            if any(not alternatives[origin] for origin in chosen):
                continue

            for values in itertools.product(*(alternatives[origin] for origin in chosen)):
                evals += 1
                if evals > MAX_SOLVE_EVALS:
                    return None

                assignment = dict(zip(chosen, values))
                try:
                    if _holds(condition, observed_zero, assignment, current):
                        continue

                except _Unsolvable:
                    return None

                if all(_holds(sym, zero, assignment, current) for sym, zero in path if sym.inputs.intersection(assignment)):
                    return assignment

    return None
