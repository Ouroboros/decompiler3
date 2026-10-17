#!/usr/bin/env python3
'''VM value semantics of the differential checker, as implemented by VMContext::execute.'''

from pathlib import Path
import math
import sys
import unittest


sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.diffcheck.values import (
    FALSE_VALUE,
    FLOAT_DROPPED_BITS,
    PAYLOAD_SIGN_BIT,
    TRUE_VALUE,
    Op,
    Raw,
    VMFault,
    binary_op,
    is_zero,
    logical_truth,
    observable,
    unary_op,
)


INT30_MAX               = PAYLOAD_SIGN_BIT - 1
INT30_MIN               = -PAYLOAD_SIGN_BIT
DIVIDEND                = -7
DIVISOR                 = 2
TRUNCATED_QUOTIENT      = -3
TRUNCATED_REMAINDER     = -1
FLOAT32_MANTISSA_BITS   = 23
FLOAT32_ULP_AT_ONE      = 2.0 ** -FLOAT32_MANTISSA_BITS
KEPT_FLOAT_STEP         = FLOAT32_ULP_AT_ONE * (1 << FLOAT_DROPPED_BITS)
FLOAT32_EXACT_LIMIT     = 1 << (FLOAT32_MANTISSA_BITS + 1)
LARGE_INT               = FLOAT32_EXACT_LIMIT + 1
LARGE_INT_AS_FLOAT      = float(FLOAT32_EXACT_LIMIT)
FLOAT_LHS               = 1.5
FLOAT_RHS               = 2.5
RAW_PAYLOAD             = 5
STRING_A                = 'mon0004'
STRING_B                = 'mon0000'


def no_deref(pointer):
    raise AssertionError('no pointer expected')


class TestIntegerSemantics(unittest.TestCase):
    def test_results_wrap_to_30_bits(self):
        self.assertEqual(binary_op(Op.ADD, INT30_MAX, 1), INT30_MIN)
        self.assertEqual(binary_op(Op.SUB, INT30_MIN, 1), INT30_MAX)
        self.assertEqual(unary_op(Op.NEG, INT30_MIN), INT30_MIN)

    def test_division_truncates_toward_zero(self):
        self.assertEqual(binary_op(Op.DIV, DIVIDEND, DIVISOR), TRUNCATED_QUOTIENT)
        self.assertEqual(binary_op(Op.MOD, DIVIDEND, DIVISOR), TRUNCATED_REMAINDER)

    def test_division_by_zero_faults(self):
        with self.assertRaises(VMFault):
            binary_op(Op.DIV, DIVIDEND, 0)

        with self.assertRaises(VMFault):
            binary_op(Op.MOD, DIVIDEND, 0)


class TestFloatSemantics(unittest.TestCase):
    def test_results_drop_the_two_lowest_bits(self):
        self.assertEqual(binary_op(Op.ADD, 1.0, FLOAT32_ULP_AT_ONE), 1.0)
        self.assertEqual(binary_op(Op.ADD, 1.0, KEPT_FLOAT_STEP), 1.0 + KEPT_FLOAT_STEP)

    def test_integer_operand_rounds_to_float32(self):
        self.assertEqual(binary_op(Op.ADD, LARGE_INT, 0.0), LARGE_INT_AS_FLOAT)
        self.assertEqual(binary_op(Op.EQ, LARGE_INT, LARGE_INT_AS_FLOAT), TRUE_VALUE)

    def test_division_by_zero_is_infinite(self):
        self.assertEqual(binary_op(Op.DIV, 1.0, 0.0), math.inf)
        self.assertEqual(binary_op(Op.DIV, -1.0, 0.0), -math.inf)


class TestTypeRestrictedOperators(unittest.TestCase):
    def test_ez_not_and_neg_leave_non_integers_unchanged(self):
        self.assertIsInstance(unary_op(Op.TEST_ZERO, 0.0), float)
        self.assertEqual(unary_op(Op.TEST_ZERO, Raw(0)), Raw(0))
        self.assertEqual(unary_op(Op.BIT_NOT, FLOAT_LHS), FLOAT_LHS)
        self.assertEqual(unary_op(Op.NEG, Raw(RAW_PAYLOAD)), Raw(RAW_PAYLOAD))

    def test_ez_and_not_on_integers(self):
        self.assertEqual(unary_op(Op.TEST_ZERO, 0), TRUE_VALUE)
        self.assertEqual(unary_op(Op.BIT_NOT, 0), -1)

    def test_mod_and_bitwise_keep_left_operand_without_integers(self):
        self.assertEqual(binary_op(Op.BIT_AND, FLOAT_LHS, FLOAT_RHS), FLOAT_LHS)
        self.assertEqual(binary_op(Op.MOD, Raw(RAW_PAYLOAD), Raw(DIVISOR)), Raw(RAW_PAYLOAD))


class TestConditions(unittest.TestCase):
    def test_branch_test_uses_the_payload(self):
        self.assertTrue(is_zero(0.0))
        self.assertFalse(is_zero(-0.0))
        self.assertTrue(is_zero(Raw(0)))
        self.assertFalse(is_zero(''))

    def test_logical_operators_compare_floats_with_zero(self):
        self.assertFalse(logical_truth(-0.0))
        self.assertEqual(binary_op(Op.LOGICAL_OR, -0.0, 0), FALSE_VALUE)


class TestEquality(unittest.TestCase):
    def test_strings_compare_by_content(self):
        self.assertEqual(binary_op(Op.EQ, STRING_A, STRING_A), TRUE_VALUE)
        self.assertEqual(binary_op(Op.EQ, STRING_A, STRING_B), FALSE_VALUE)

    def test_raw_and_integer_compare_by_payload(self):
        self.assertEqual(binary_op(Op.EQ, Raw(RAW_PAYLOAD), RAW_PAYLOAD), TRUE_VALUE)
        self.assertEqual(observable(Raw(RAW_PAYLOAD), no_deref), RAW_PAYLOAD)

    def test_floats_are_observed_by_bit_pattern(self):
        self.assertNotEqual(observable(-0.0, no_deref), observable(0.0, no_deref))
        self.assertNotEqual(observable(1.0, no_deref), observable(1, no_deref))


if __name__ == '__main__':
    unittest.main()
