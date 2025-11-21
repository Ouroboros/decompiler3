'''
Test HLIL basic structures
'''

import unittest
from ir.hlil import *


class TestHLILBasic(unittest.TestCase):
    '''Test HLIL basic instruction creation and formatting'''

    def test_simple_function(self):
        '''Test creating a simple function with if-else'''
        # Create function
        func = HighLevelILFunction('test_func', 0x1000)

        # Create variables
        arg1 = HLILVariable('arg1', 'int')
        arg2 = HLILVariable('arg2', 'int')
        var_x = HLILVariable('var_x', 'int')
        var_y = HLILVariable('var_y', 'int')

        func.parameters = [arg1, arg2]
        func.variables = [var_x, var_y]

        # Build: var_x = arg1 + arg2
        assign_stmt = HLILAssign(
            HLILVar(var_x),
            HLILBinaryOp('+', HLILVar(arg1), HLILVar(arg2))
        )
        func.add_statement(assign_stmt)

        # Build: if (var_x > 0) { return var_x; } else { return 0; }
        true_block = HLILBlock()
        true_block.add_statement(HLILReturn(HLILVar(var_x)))

        false_block = HLILBlock()
        false_block.add_statement(HLILReturn(HLILConst(0)))

        condition = HLILBinaryOp('>', HLILVar(var_x), HLILConst(0))
        if_stmt = HLILIf(condition, true_block, false_block)
        func.add_statement(if_stmt)

        # Format and print
        lines = HLILFormatter.format_function(func)
        output = '\n'.join(lines)

        print('\n=== Simple Function ===')
        print(output)

        # Verify structure
        self.assertEqual(func.name, 'test_func')
        self.assertEqual(len(func.body.statements), 2)
        self.assertIsInstance(func.body.statements[0], HLILAssign)
        self.assertIsInstance(func.body.statements[1], HLILIf)

        # Verify output contains expected keywords
        self.assertIn('function test_func', output)
        self.assertIn('if (var_x > 0)', output)
        self.assertIn('return var_x', output)
        self.assertIn('else', output)

    def test_while_loop(self):
        '''Test creating a while loop'''
        func = HighLevelILFunction('loop_func', 0x2000)

        i = HLILVariable('i', 'int')
        sum_var = HLILVariable('sum', 'int')
        func.variables = [i, sum_var]

        # i = 0
        func.add_statement(HLILAssign(HLILVar(i), HLILConst(0)))

        # sum = 0
        func.add_statement(HLILAssign(HLILVar(sum_var), HLILConst(0)))

        # while (i < 10) { sum = sum + i; i = i + 1; }
        loop_body = HLILBlock()
        loop_body.add_statement(
            HLILAssign(
                HLILVar(sum_var),
                HLILBinaryOp('+', HLILVar(sum_var), HLILVar(i))
            )
        )
        loop_body.add_statement(
            HLILAssign(
                HLILVar(i),
                HLILBinaryOp('+', HLILVar(i), HLILConst(1))
            )
        )

        condition = HLILBinaryOp('<', HLILVar(i), HLILConst(10))
        while_stmt = HLILWhile(condition, loop_body)
        func.add_statement(while_stmt)

        # return sum
        func.add_statement(HLILReturn(HLILVar(sum_var)))

        # Format
        lines = HLILFormatter.format_function(func)
        output = '\n'.join(lines)

        print('\n=== While Loop ===')
        print(output)

        # Verify
        self.assertIn('while (i < 10)', output)
        self.assertIn('sum = sum + i', output)
        self.assertIn('return sum', output)

    def test_nested_if(self):
        '''Test nested if statements'''
        func = HighLevelILFunction('nested_func', 0x3000)

        x = HLILVariable('x', 'int')
        func.parameters = [x]

        # if (x > 0) {
        #   if (x > 10) {
        #     return 1;
        #   } else {
        #     return 2;
        #   }
        # } else {
        #   return 0;
        # }

        # Inner if
        inner_true = HLILBlock()
        inner_true.add_statement(HLILReturn(HLILConst(1)))

        inner_false = HLILBlock()
        inner_false.add_statement(HLILReturn(HLILConst(2)))

        inner_cond = HLILBinaryOp('>', HLILVar(x), HLILConst(10))
        inner_if = HLILIf(inner_cond, inner_true, inner_false)

        # Outer if
        outer_true = HLILBlock()
        outer_true.add_statement(inner_if)

        outer_false = HLILBlock()
        outer_false.add_statement(HLILReturn(HLILConst(0)))

        outer_cond = HLILBinaryOp('>', HLILVar(x), HLILConst(0))
        outer_if = HLILIf(outer_cond, outer_true, outer_false)
        func.add_statement(outer_if)

        # Format
        lines = HLILFormatter.format_function(func)
        output = '\n'.join(lines)

        print('\n=== Nested If ===')
        print(output)

        # Verify
        self.assertIn('if (x > 0)', output)
        self.assertIn('if (x > 10)', output)
        self.assertEqual(output.count('return'), 3)

    def test_break_continue(self):
        '''Test break and continue statements'''
        func = HighLevelILFunction('break_continue_func', 0x4000)

        i = HLILVariable('i', 'int')
        func.variables = [i]

        # while (true) {
        #   if (i < 0) break;
        #   if (i % 2 == 0) continue;
        #   i = i + 1;
        # }

        loop_body = HLILBlock()

        # if (i < 0) break;
        break_if_true = HLILBlock()
        break_if_true.add_statement(HLILBreak())
        break_cond = HLILBinaryOp('<', HLILVar(i), HLILConst(0))
        break_if = HLILIf(break_cond, break_if_true)
        loop_body.add_statement(break_if)

        # if (i % 2 == 0) continue;
        continue_if_true = HLILBlock()
        continue_if_true.add_statement(HLILContinue())
        continue_cond = HLILBinaryOp(
            '==',
            HLILBinaryOp('%', HLILVar(i), HLILConst(2)),
            HLILConst(0)
        )
        continue_if = HLILIf(continue_cond, continue_if_true)
        loop_body.add_statement(continue_if)

        # i = i + 1;
        loop_body.add_statement(
            HLILAssign(
                HLILVar(i),
                HLILBinaryOp('+', HLILVar(i), HLILConst(1))
            )
        )

        while_cond = HLILConst(True)
        while_stmt = HLILWhile(while_cond, loop_body)
        func.add_statement(while_stmt)

        # Format
        lines = HLILFormatter.format_function(func)
        output = '\n'.join(lines)

        print('\n=== Break/Continue ===')
        print(output)

        # Verify
        self.assertIn('break;', output)
        self.assertIn('continue;', output)
        self.assertIn('while (true)', output)


if __name__ == '__main__':
    unittest.main()
