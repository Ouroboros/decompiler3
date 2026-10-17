#!/usr/bin/env python3
'''Generated TypeScript must execute like the HLIL it was emitted from (tools/diffcheck/ts_parse.py).'''

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).parent.parent))

from codegen import generate_typescript
from ir.hlil import (
    BinaryOp, HighLevelILFunction, HLILBinaryOp, HLILBlock, HLILCall, HLILConst, HLILExprStmt, HLILIf, HLILReturn,
    HLILVar, HLILVariable, VariableKind,
)
from tools.diffcheck.environment import ORIGIN_PARAM, InputPlan
from tools.diffcheck.hlil_interp import HLILInterpreter
from tools.diffcheck.ts_parse import TSParseError, parse_typescript


PARAM_COUNT             = 2
# arg1 & arg2 is nonzero, but arg1 & (arg2 != 0) is zero
BIT_LHS                 = 2
BIT_RHS                 = 2
TRUE_RESULT             = 1
FALSE_RESULT            = 0
QUOTED_TEXT             = 'say "hi"\\path\nnext'
SUB_LHS                 = 10
SUB_RHS                 = 3
SCALE                   = 2


def make_function(*statements) -> HighLevelILFunction:
    func = HighLevelILFunction('f')
    func.parameters = [HLILVariable(f'arg{number}', kind = VariableKind.PARAM) for number in range(1, PARAM_COUNT + 1)]
    func.body = HLILBlock(list(statements))
    return func


def arg(number: int) -> HLILVar:
    return HLILVar(HLILVariable(f'arg{number}', kind = VariableKind.PARAM))


def trace_keys(func: HighLevelILFunction, params) -> list:
    forced = {(ORIGIN_PARAM, number): value for number, value in enumerate(params, start = 1)}
    plan = InputPlan('roundtrip', [0], {}, [], PARAM_COUNT, forced)
    return [event.key for event in HLILInterpreter(func, PARAM_COUNT, plan).run().events]


class TestRoundTrip(unittest.TestCase):
    def assert_same_behavior(self, func: HighLevelILFunction, params):
        emitted = parse_typescript(generate_typescript(func))
        self.assertEqual(trace_keys(emitted, params), trace_keys(func, params))

    def test_negated_condition_keeps_operand_parentheses(self):
        # An empty true branch makes the generator negate the condition
        condition = HLILBinaryOp(BinaryOp.EQ, HLILBinaryOp(BinaryOp.BIT_AND, arg(1), arg(2)), HLILConst(0))
        func = make_function(
            HLILIf(condition, HLILBlock(), HLILBlock([HLILReturn(HLILConst(TRUE_RESULT))])),
            HLILReturn(HLILConst(FALSE_RESULT)),
        )
        self.assert_same_behavior(func, (BIT_LHS, BIT_RHS))

    def test_string_literals_are_escaped(self):
        func = make_function(HLILExprStmt(HLILCall('show', [HLILConst(QUOTED_TEXT)])))
        self.assert_same_behavior(func, (0, 0))

    def test_non_associative_operands_keep_parentheses(self):
        nested = HLILBinaryOp(BinaryOp.SUB, arg(1), HLILBinaryOp(BinaryOp.SUB, arg(2), HLILConst(1)))
        scaled = HLILBinaryOp(BinaryOp.MUL, HLILBinaryOp(BinaryOp.ADD, arg(1), HLILConst(1)), HLILConst(SCALE))
        func = make_function(HLILExprStmt(HLILCall('use', [nested, scaled])))
        self.assert_same_behavior(func, (SUB_LHS, SUB_RHS))


class TestParser(unittest.TestCase):
    def test_undeclared_names_are_rejected(self):
        with self.assertRaises(TSParseError):
            parse_typescript('function f(): void {\n    x = 1;\n}')


if __name__ == '__main__':
    unittest.main()
