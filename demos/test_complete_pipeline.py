#!/usr/bin/env python3
"""
Complete Pipeline Test for New IR System

Tests the complete flow: LLIL -> MLIL -> HLIL -> TypeScript
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decompiler3.ir.lifter import DecompilerPipeline
from decompiler3.typescript.generator import TypeScriptGenerator


def test_complete_pipeline():
    """Test complete decompilation pipeline"""
    print("🧪 Testing Complete Decompilation Pipeline")
    print("=" * 50)

    # Create pipeline and generator
    pipeline = DecompilerPipeline()
    generator = TypeScriptGenerator()

    # Test 1: Simple arithmetic function
    print("\n📝 Test 1: Simple Arithmetic Function")
    print("-" * 30)

    llil_func = pipeline.create_sample_llil_function()

    print(f"LLIL Function: {llil_func.name}")
    print(f"LLIL Blocks: {len(llil_func.basic_blocks)}")
    print(f"LLIL Instructions: {sum(len(block.instructions) for block in llil_func.basic_blocks)}")

    # Run complete pipeline
    hlil_func = pipeline.decompile_function(llil_func)

    # Generate TypeScript
    ts_code = generator.generate_function(hlil_func)
    print("\n📄 Generated TypeScript:")
    print(ts_code)

    print("\n✅ Complete pipeline test successful!")

    # Test 2: Pretty vs compact styles
    print("\n📝 Test 2: Generator Style Options")
    print("-" * 30)

    generator_compact = TypeScriptGenerator(style="compact")
    ts_code_compact = generator_compact.generate_function(hlil_func)
    print("Compact style:")
    print(ts_code_compact)

    return True


def test_ir_levels():
    """Test each IR level individually"""
    print("\n🔬 Testing Individual IR Levels")
    print("=" * 50)

    pipeline = DecompilerPipeline()
    llil_func = pipeline.create_sample_llil_function()

    # Test LLIL
    print("\n🔧 LLIL Level:")
    for i, block in enumerate(llil_func.basic_blocks):
        print(f"  Block {i}:")
        for j, instr in enumerate(block.instructions):
            print(f"    {j}: {instr}")

    # Test MLIL
    print("\n🔧 MLIL Level:")
    mlil_func = pipeline.llil_to_mlil.lift_function(llil_func)
    for i, block in enumerate(mlil_func.basic_blocks):
        print(f"  Block {i}:")
        for j, instr in enumerate(block.instructions):
            print(f"    {j}: {instr}")

    # Test HLIL
    print("\n🔧 HLIL Level:")
    hlil_func = pipeline.mlil_to_hlil.lift_function(mlil_func)
    for i, block in enumerate(hlil_func.basic_blocks):
        print(f"  Block {i}:")
        for j, instr in enumerate(block.instructions):
            print(f"    {j}: {instr}")

    print("\n✅ IR level tests successful!")
    return True


if __name__ == "__main__":
    try:
        test_complete_pipeline()
        test_ir_levels()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()