#!/usr/bin/env python3
'''
Unit tests for MLIL (Medium Level IL)
'''

from pathlib import Path
import sys
import unittest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ir.llil import *
from ir.mlil import *
from ir.mlil.llil_to_mlil import LLILToMLILTranslator


class TestMLILBuilder(unittest.TestCase):
    '''Test MLIL builder'''

    def test_create_function(self):
        '''Test function creation'''
        builder = MLILBuilder()
        builder.create_function('test_func', 0x1000)

        self.assertEqual(builder.function.name, 'test_func')
        self.assertEqual(builder.function.start_addr, 0x1000)

    def test_create_block(self):
        '''Test block creation'''
        builder = MLILBuilder()
        builder.create_function('test', 0x1000)

        block = builder.create_block(0x1000, 'entry')
        self.assertEqual(block.start, 0x1000)
        self.assertEqual(block.label, 'entry')

    def test_variable_operations(self):
        '''Test variable creation and operations'''
        builder = MLILBuilder()
        builder.create_function('test', 0x1000)
        block = builder.create_block(0x1000, 'entry')
        builder.set_current_block(block)

        # Create variable
        var_x = builder.get_or_create_var('x')
        self.assertEqual(var_x.name, 'x')

        # Set variable
        builder.set_var(var_x, builder.const_int(42))

        # Check instruction was added
        self.assertEqual(len(block.instructions), 1)
        inst = block.instructions[0]
        self.assertIsInstance(inst, MLILSetVar)
        self.assertEqual(inst.var.name, 'x')
        self.assertIsInstance(inst.value, MLILConst)
        self.assertEqual(inst.value.value, 42)

    def test_arithmetic_operations(self):
        '''Test arithmetic expression building'''
        builder = MLILBuilder()
        builder.create_function('test', 0x1000)
        block = builder.create_block(0x1000, 'entry')
        builder.set_current_block(block)

        var_a = builder.get_or_create_var('a')
        var_b = builder.get_or_create_var('b')

        # a + b
        expr = builder.add(builder.var(var_a), builder.var(var_b))
        self.assertIsInstance(expr, MLILAdd)
        self.assertIsInstance(expr.lhs, MLILVar)
        self.assertIsInstance(expr.rhs, MLILVar)

    def test_control_flow(self):
        '''Test control flow instructions'''
        builder = MLILBuilder()
        builder.create_function('test', 0x1000)

        entry = builder.create_block(0x1000, 'entry')
        exit_block = builder.create_block(0x1010, 'exit')

        builder.set_current_block(entry)

        # Goto
        builder.goto(exit_block)
        self.assertEqual(len(entry.instructions), 1)
        self.assertIsInstance(entry.instructions[0], MLILGoto)

        # Return
        builder.set_current_block(exit_block)
        builder.ret()
        self.assertEqual(len(exit_block.instructions), 1)
        self.assertIsInstance(exit_block.instructions[0], MLILRet)

    def test_conditional_branch(self):
        '''Test conditional branching'''
        builder = MLILBuilder()
        builder.create_function('test', 0x1000)

        entry = builder.create_block(0x1000, 'entry')
        true_block = builder.create_block(0x1010, 'true_branch')
        false_block = builder.create_block(0x1020, 'false_branch')

        builder.set_current_block(entry)

        # if (condition) goto true_block else false_block
        condition = builder.const_int(1)
        builder.branch_if(condition, true_block, false_block)

        self.assertEqual(len(entry.instructions), 1)
        inst = entry.instructions[0]
        self.assertIsInstance(inst, MLILIf)
        self.assertEqual(inst.true_target, true_block)
        self.assertEqual(inst.false_target, false_block)

    def test_finalize_validates_terminals(self):
        '''Test that finalize validates all blocks have terminals'''
        builder = MLILBuilder()
        builder.create_function('test', 0x1000)

        entry = builder.create_block(0x1000, 'entry')
        builder.set_current_block(entry)

        # Block without terminal should fail
        with self.assertRaises(RuntimeError):
            builder.finalize()

        # Add terminal
        builder.ret()

        # Should succeed now
        func = builder.finalize()
        self.assertIsNotNone(func)


class TestLLILToMLILTranslator(unittest.TestCase):
    '''Test LLIL to MLIL translation'''

    def test_translate_stack_operations(self):
        '''Test stack store/load translation to variables'''
        # Create LLIL function with stack operations
        llil_func = LowLevelILFunction('test', 0x1000)
        llil_builder = LowLevelILBuilder(llil_func)

        # Create block and add to function
        entry = LowLevelILBasicBlock(0x1000, 0, 'entry')
        entry.function = llil_func
        llil_func.basic_blocks.append(entry)

        llil_builder.set_current_block(entry)

        # STACK[0] = 42
        llil_builder.push(llil_builder.const_int(42))

        # value = STACK[0]
        value = llil_builder.stack_load(0, 0)

        # STACK[1] = value
        llil_builder.push(value)

        # return
        llil_builder.ret()

        # Translate to MLIL (no need to finalize, llil_func already exists)
        translator = LLILToMLILTranslator()
        mlil_func = translator.translate(llil_func)

        # Check MLIL has variables instead of stack ops
        self.assertEqual(mlil_func.name, 'test')
        self.assertEqual(len(mlil_func.basic_blocks), 1)

        block = mlil_func.basic_blocks[0]

        # Should have: var_s0 = 42, var_s1 = var_s0, return
        instructions = block.instructions

        # Find SetVar instructions (skip SpAdd which are eliminated)
        set_vars = [i for i in instructions if isinstance(i, MLILSetVar)]
        self.assertGreaterEqual(len(set_vars), 2)

        # First: var_s0 = 42
        self.assertEqual(set_vars[0].var.name, 'var_s0')
        self.assertIsInstance(set_vars[0].value, MLILConst)
        self.assertEqual(set_vars[0].value.value, 42)

        # Second: var_s1 = var_s0
        self.assertEqual(set_vars[1].var.name, 'var_s1')
        self.assertIsInstance(set_vars[1].value, MLILVar)

    def test_translate_arithmetic(self):
        '''Test arithmetic operation translation'''
        # Create LLIL: push 10, push 5, add, pop to stack[0]
        llil_func = LowLevelILFunction('test', 0x1000)
        llil_builder = LowLevelILBuilder(llil_func)

        entry = LowLevelILBasicBlock(0x1000, 0, 'entry')
        entry.function = llil_func
        llil_func.basic_blocks.append(entry)

        llil_builder.set_current_block(entry)

        llil_builder.push(llil_builder.const_int(10))
        llil_builder.push(llil_builder.const_int(5))

        lhs = llil_builder.pop()
        rhs = llil_builder.pop()
        result = llil_builder.add(lhs, rhs)
        llil_builder.push(result)

        llil_builder.ret()

        # Translate (no need to finalize)
        translator = LLILToMLILTranslator()
        mlil_func = translator.translate(llil_func)

        # Check result
        block = mlil_func.basic_blocks[0]
        set_vars = [i for i in block.instructions if isinstance(i, MLILSetVar)]

        # Should have var assignments
        self.assertGreater(len(set_vars), 0)

        # Find the add operation
        adds = [i for i in block.instructions if isinstance(i.value if isinstance(i, MLILSetVar) else i, MLILAdd)]
        self.assertGreater(len(adds), 0)

    def test_translate_control_flow(self):
        '''Test control flow translation'''
        # Create LLIL with if statement
        llil_func = LowLevelILFunction('test', 0x1000)
        llil_builder = LowLevelILBuilder(llil_func)

        entry = LowLevelILBasicBlock(0x1000, 0, 'entry')
        true_block = LowLevelILBasicBlock(0x1010, 1, 'true_branch')
        false_block = LowLevelILBasicBlock(0x1020, 2, 'false_branch')

        for block in [entry, true_block, false_block]:
            block.function = llil_func
            llil_func.basic_blocks.append(block)

        llil_builder.set_current_block(entry)
        condition = llil_builder.const_int(1)
        llil_builder.branch_if(condition, true_block, false_block)

        llil_builder.set_current_block(true_block)
        llil_builder.ret()

        llil_builder.set_current_block(false_block)
        llil_builder.ret()

        # Translate (no need to finalize)
        translator = LLILToMLILTranslator()
        mlil_func = translator.translate(llil_func)

        # Check MLIL has same structure
        self.assertEqual(len(mlil_func.basic_blocks), 3)

        entry_mlil = mlil_func.basic_blocks[0]
        # Entry should have If instruction
        if_insts = [i for i in entry_mlil.instructions if isinstance(i, MLILIf)]
        self.assertEqual(len(if_insts), 1)


class TestMLILFormatter(unittest.TestCase):
    '''Test MLIL formatter'''

    def test_format_simple_function(self):
        '''Test formatting a simple MLIL function'''
        builder = MLILBuilder()
        builder.create_function('test_func', 0x1000)

        entry = builder.create_block(0x1000, 'entry')
        builder.set_current_block(entry)

        var_x = builder.get_or_create_var('x')
        builder.set_var(var_x, builder.const_int(42))
        builder.ret()

        func = builder.finalize()

        # Format
        lines = MLILFormatter.format_function(func)

        # Check output
        text = '\n'.join(lines)
        self.assertIn('test_func', text)
        self.assertIn('x = 42', text)
        self.assertIn('return', text)


if __name__ == '__main__':
    unittest.main()
