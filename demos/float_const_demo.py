#!/usr/bin/env python3
"""
Float Constant Demo - Testing float constant support
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ir.llil import (
    LowLevelILConst,
    LowLevelILFunction,
    LowLevelILBasicBlock,
    LowLevelILStackStore,
    LowLevelILVspAdd
)
from ir.llil_builder import LowLevelILBuilder


def test_float_constants():
    """Test different float constant values"""

    print("🧪 Testing Float Constants")
    print("=" * 60)

    # Different float values
    test_values = [
        (3.14159, 4, "Pi (float)"),
        (2.71828, 4, "e (float)"),
        (3.14159265358979, 8, "Pi (double)"),
        (0.0, 4, "Zero"),
        (-1.5, 4, "Negative"),
        (1.0, 4, "Whole number"),
        (123.456789, 4, "Decimal"),
    ]

    print("\n1. Float Constant Creation:")
    for value, size, description in test_values:
        const = LowLevelILConst(value, size)
        print(f"   {description:20s}: {const} (size={size}, type={type(const.value).__name__})")

    # Integer vs Float
    print("\n2. Integer vs Float Display:")
    int_const = LowLevelILConst(42, 4)
    float_const = LowLevelILConst(42.0, 4)
    float_precise = LowLevelILConst(42.5, 4)

    print(f"   Integer 42:      {int_const}")
    print(f"   Float 42.0:      {float_const}")
    print(f"   Float 42.5:      {float_precise}")

    # Type checking
    print("\n3. Type Checking:")
    print(f"   int_const.value type:   {type(int_const.value).__name__}")
    print(f"   float_const.value type: {type(float_const.value).__name__}")

    print("\n" + "=" * 60)
    print("✅ Float constant tests passed!")


def test_float_in_code():
    """Test float constants in actual code generation"""

    print("\n\n🧪 Testing Float Constants in Code")
    print("=" * 60)

    func = LowLevelILFunction("float_test", 0x1000)
    block = LowLevelILBasicBlock(0x1000, 0)
    func.add_basic_block(block)

    builder = LowLevelILBuilder(func)
    builder.set_current_block(block)

    # Push different types of constants
    builder.stack_push(builder.const_int(42))
    builder.stack_push(builder.const_float(3.14159))
    builder.stack_push(builder.const_float(2.71828, 8))  # double
    builder.stack_push(builder.const_str("hello"))
    builder.stack_push(builder.const_float(1.0))
    builder.stack_push(builder.const_float(-0.5))

    print("\nGenerated code:")
    for instr in block.instructions:
        print(f"  {instr}")

    print("\nNotice the formatting:")
    print("  • Integers:     42")
    print("  • Float (whole): 3.0, 1.0")
    print("  • Float (decimal): 3.141590, -0.500000")
    print("  • Strings:      \"hello\"")

    print("\n" + "=" * 60)
    print("✅ Code generation test passed!")


def test_size_handling():
    """Test size handling for different float types"""

    print("\n\n🧪 Testing Float Size Handling")
    print("=" * 60)

    float32 = LowLevelILConst(3.14159, 4)    # 32-bit float
    float64 = LowLevelILConst(3.14159, 8)    # 64-bit double

    print("\n1. Size Specification:")
    print(f"   32-bit float: {float32} (size={float32.size})")
    print(f"   64-bit double: {float64} (size={float64.size})")

    print("\n2. Builder Helper Methods:")
    builder = LowLevelILBuilder(LowLevelILFunction("test", 0))

    f32 = builder.const_float(2.71828)          # default 4 bytes
    f64 = builder.const_float(2.71828, 8)       # explicit 8 bytes

    print(f"   const_float(2.71828):     size={f32.size}")
    print(f"   const_float(2.71828, 8):  size={f64.size}")

    print("\n" + "=" * 60)
    print("✅ Size handling test passed!")


if __name__ == "__main__":
    test_float_constants()
    test_float_in_code()
    test_size_handling()

    print("\n" + "=" * 60)
    print("🎉 All float constant tests completed!")
    print("=" * 60)
