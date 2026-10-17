#!/usr/bin/env python3
'''Constant folding and algebraic identities under the ED9 VM value semantics.'''

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).parent.parent))

from falcom.ed9.constant_folder import ED9ConstantFolder
from falcom.ed9.vm_values import PAYLOAD_SIGN_BIT, TRUE_VALUE
from ir.mlil import MLILAdd, MLILAnd, MLILConst, MLILDiv, MLILEq, MLILLt, MLILMod, MLILMul, MLILOr, MLILSub, MLILTestZero
from ir.mlil.passes import ExpressionSimplificationPass, LatticeValue, NotFoldable


INT30_MAX               = PAYLOAD_SIGN_BIT - 1
INT30_MIN               = -PAYLOAD_SIGN_BIT
DIVIDEND                = -7
DIVISOR                 = 2
TRUNCATED_QUOTIENT      = -3
TRUNCATED_REMAINDER     = -1
FLOAT_VALUE             = 1.5
FLOAT_QUOTIENT          = FLOAT_VALUE / DIVISOR
STRING_VALUE            = 'mon0004'

# EV_03_00_00 in mp2000_ev.dat pushes camera_lookat's second argument as (1.267 + 0.67) - 0.9
EV_ADDEND_A             = 1.2669997215270996
EV_ADDEND_B             = 0.6699998378753662
EV_SUBTRAHEND           = 0.8999998569488525
EV_VM_RESULT            = 1.036999225616455


class TestFolding(unittest.TestCase):
    def test_float_operations_round_every_step_like_the_vm(self):
        total = ED9ConstantFolder.binary(MLILAdd, EV_ADDEND_A, EV_ADDEND_B)
        self.assertEqual(ED9ConstantFolder.binary(MLILSub, total, EV_SUBTRAHEND), EV_VM_RESULT)

    def test_float_division_is_not_floored(self):
        self.assertEqual(ED9ConstantFolder.binary(MLILDiv, FLOAT_VALUE, DIVISOR), FLOAT_QUOTIENT)

    def test_integers_wrap_and_truncate_toward_zero(self):
        self.assertEqual(ED9ConstantFolder.binary(MLILAdd, INT30_MAX, 1), INT30_MIN)
        self.assertEqual(ED9ConstantFolder.binary(MLILDiv, DIVIDEND, DIVISOR), TRUNCATED_QUOTIENT)
        self.assertEqual(ED9ConstantFolder.binary(MLILMod, DIVIDEND, DIVISOR), TRUNCATED_REMAINDER)

    def test_ez_on_integers(self):
        self.assertEqual(ED9ConstantFolder.unary(MLILTestZero, 0), TRUE_VALUE)

    def test_branches_test_the_payload(self):
        self.assertFalse(ED9ConstantFolder.is_true(0.0))
        self.assertTrue(ED9ConstantFolder.is_true(-0.0))
        self.assertTrue(ED9ConstantFolder.is_true(STRING_VALUE))


class TestNotFolded(unittest.TestCase):
    def test_runtime_faults_and_values_without_literals(self):
        with self.assertRaises(NotFoldable):
            ED9ConstantFolder.binary(MLILDiv, DIVIDEND, 0)

        with self.assertRaises(NotFoldable):
            ED9ConstantFolder.binary(MLILDiv, FLOAT_VALUE, 0.0)

    def test_type_errors(self):
        with self.assertRaises(NotFoldable):
            ED9ConstantFolder.unary(MLILTestZero, FLOAT_VALUE)

        with self.assertRaises(NotFoldable):
            ED9ConstantFolder.binary(MLILMod, FLOAT_VALUE, FLOAT_VALUE)

    def test_results_depending_on_string_offsets(self):
        self.assertEqual(ED9ConstantFolder.binary(MLILEq, STRING_VALUE, STRING_VALUE), TRUE_VALUE)
        with self.assertRaises(NotFoldable):
            ED9ConstantFolder.binary(MLILEq, STRING_VALUE, 0)

        with self.assertRaises(NotFoldable):
            ED9ConstantFolder.binary(MLILLt, STRING_VALUE, STRING_VALUE)


class TestAlgebraicIdentities(unittest.TestCase):
    def setUp(self):
        self.simplify = ExpressionSimplificationPass()._apply_algebraic_identity
        self.float_operand = MLILConst(FLOAT_VALUE)
        self.integer_operand = MLILEq(MLILConst(DIVIDEND), MLILConst(DIVISOR))

    def test_float_constants_do_not_act_as_integer_identities(self):
        self.assertIsNone(self.simplify(MLILMul, self.integer_operand, MLILConst(float(1))))
        self.assertIsNone(self.simplify(MLILAdd, self.integer_operand, MLILConst(0.0)))

    def test_rules_that_change_a_float_operand_need_an_integer(self):
        self.assertIsNone(self.simplify(MLILMul, self.float_operand, MLILConst(0)))
        self.assertIsNone(self.simplify(MLILOr, self.float_operand, MLILConst(0)))
        self.assertEqual(self.simplify(MLILMul, self.integer_operand, MLILConst(0)).value, 0)
        self.assertIs(self.simplify(MLILOr, self.integer_operand, MLILConst(0)), self.integer_operand)

    def test_rules_valid_for_any_operand(self):
        self.assertEqual(self.simplify(MLILAnd, self.float_operand, MLILConst(0)).value, 0)


class TestLatticeIdentity(unittest.TestCase):
    def test_constants_of_different_types_or_signs_differ(self):
        self.assertTrue(LatticeValue.constant(0).meet(LatticeValue.constant(0.0)).is_bottom())
        self.assertTrue(LatticeValue.constant(0.0).meet(LatticeValue.constant(-0.0)).is_bottom())
        self.assertEqual(LatticeValue.constant(FLOAT_VALUE), LatticeValue.constant(FLOAT_VALUE))


if __name__ == '__main__':
    unittest.main()
