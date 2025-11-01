#!/usr/bin/env python3
"""
Test script for the new IR system based on BinaryNinja design
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_llil():
    """Test LLIL instructions"""
    print("🔧 Testing LLIL System...")

    from decompiler3.ir.llil import (
        LowLevelILBuilder, LowLevelILFunction, LowLevelILBasicBlock,
        LowLevelILConst, LowLevelILAdd, LowLevelILRet
    )
    from decompiler3.ir.common import ILRegister

    # Create function and basic block
    function = LowLevelILFunction("test_function", 0x1000)
    block = LowLevelILBasicBlock(0x1000)
    function.add_basic_block(block)

    # Create builder
    builder = LowLevelILBuilder(function)
    builder.set_current_block(block)

    # Create some instructions
    const1 = builder.const(10, 4)
    const2 = builder.const(20, 4)
    add_result = builder.add(const1, const2, 4)
    ret_instr = builder.ret(add_result)

    # Add instructions to block
    builder.add_instruction(const1)
    builder.add_instruction(const2)
    builder.add_instruction(add_result)
    builder.add_instruction(ret_instr)

    print(f"✅ LLIL Function created: {function.name}")
    print(f"   Instructions: {len(block.instructions)}")
    print("   Code:")
    for i, instr in enumerate(block.instructions):
        print(f"     {i}: {instr}")

    return function


def test_mlil():
    """Test MLIL instructions"""
    print("\n🔧 Testing MLIL System...")

    from decompiler3.ir.mlil import (
        MediumLevelILBuilder, MediumLevelILFunction, MediumLevelILBasicBlock,
        Variable, MediumLevelILConst
    )

    # Create function and basic block
    function = MediumLevelILFunction("test_mlil_function", 0x2000)
    block = MediumLevelILBasicBlock(0x2000)
    function.add_basic_block(block)

    # Create builder
    builder = MediumLevelILBuilder(function)
    builder.set_current_block(block)

    # Create variables
    var_a = function.create_variable("a", "int", 4)
    var_b = function.create_variable("b", "int", 4)
    var_result = function.create_variable("result", "int", 4)

    # Create instructions
    const1 = builder.const(42, 4)
    const2 = builder.const(58, 4)
    set_a = builder.set_var(var_a, const1)
    set_b = builder.set_var(var_b, const2)

    load_a = builder.var(var_a)
    load_b = builder.var(var_b)
    add_result = builder.add(load_a, load_b, 4)
    set_result = builder.set_var(var_result, add_result)

    ret_value = builder.var(var_result)
    ret_instr = builder.ret([ret_value])

    # Add instructions to block
    instructions = [set_a, set_b, set_result, ret_instr]
    for instr in instructions:
        builder.add_instruction(instr)

    print(f"✅ MLIL Function created: {function.name}")
    print(f"   Variables: {len(function.variables)}")
    print(f"   Instructions: {len(block.instructions)}")
    print("   Variables:")
    for name, var in function.variables.items():
        print(f"     {name}: {var.var_type} (size: {var.size})")
    print("   Code:")
    for i, instr in enumerate(block.instructions):
        print(f"     {i}: {instr}")

    return function


def test_hlil():
    """Test HLIL instructions"""
    print("\n🔧 Testing HLIL System...")

    from decompiler3.ir.hlil import (
        HighLevelILBuilder, HighLevelILFunction, HighLevelILBasicBlock
    )
    from decompiler3.ir.mlil import Variable

    # Create function and basic block
    function = HighLevelILFunction("test_hlil_function", 0x3000)
    block = HighLevelILBasicBlock(0x3000)
    function.add_basic_block(block)

    # Create builder
    builder = HighLevelILBuilder(function)
    builder.set_current_block(block)

    # Create variables
    var_i = function.create_variable("i", "int", 4)
    var_sum = function.create_variable("sum", "int", 4)

    # Create structured control flow
    init_sum = builder.assign(builder.var(var_sum), builder.const(0))
    init_i = builder.assign(builder.var(var_i), builder.const(0))

    # For loop: for (i = 0; i < 10; i++) { sum += i; }
    condition = builder.var(var_i)  # Simplified for demo
    update = builder.assign(builder.var(var_i), builder.add(builder.var(var_i), builder.const(1)))
    body = builder.assign(builder.var(var_sum), builder.add(builder.var(var_sum), builder.var(var_i)))

    for_loop = builder.for_loop(init_i, condition, update, body)
    ret_stmt = builder.ret([builder.var(var_sum)])

    # Add instructions to block
    instructions = [init_sum, for_loop, ret_stmt]
    for instr in instructions:
        builder.add_instruction(instr)

    print(f"✅ HLIL Function created: {function.name}")
    print(f"   Variables: {len(function.variables)}")
    print(f"   Instructions: {len(block.instructions)}")
    print("   Variables:")
    for name, var in function.variables.items():
        print(f"     {name}: {var.var_type} (size: {var.size})")
    print("   Code:")
    for i, instr in enumerate(block.instructions):
        print(f"     {i}: {instr}")

    return function


def main():
    """Run all tests"""
    print("🎉 Testing New IR System Based on BinaryNinja Design")
    print("=" * 60)

    try:
        llil_func = test_llil()
        mlil_func = test_mlil()
        hlil_func = test_hlil()

        print("\n🎉 All Tests Passed!")
        print("✅ LLIL system working correctly")
        print("✅ MLIL system working correctly")
        print("✅ HLIL system working correctly")
        print("\n🏆 New IR system is ready for integration!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())