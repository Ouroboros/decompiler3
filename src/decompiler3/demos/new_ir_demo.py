#!/usr/bin/env python3
"""
New IR System Demo

Demonstrates the complete BinaryNinja-style IR system:
LLIL -> MLIL -> HLIL -> TypeScript
"""

from ..ir.lifter import DecompilerPipeline
from ..typescript.generator import TypeScriptGenerator
from ..ir.llil import (
    LowLevelILFunction, LowLevelILBasicBlock, LowLevelILBuilder,
    LowLevelILConst, LowLevelILAdd, LowLevelILSetReg, LowLevelILReg, LowLevelILRet,
    LowLevelILIf, LowLevelILGoto, LowLevelILCall
)
from ..ir.common import ILRegister, InstructionIndex


def create_complex_llil_function() -> LowLevelILFunction:
    """Create a more complex LLIL function with control flow"""
    print("🏗️  Creating complex LLIL function with control flow")

    # Create function
    function = LowLevelILFunction("complex_function", 0x1000)

    # Create registers
    reg_eax = ILRegister("eax", 0, 4)
    reg_ebx = ILRegister("ebx", 1, 4)
    reg_ecx = ILRegister("ecx", 2, 4)

    # Block 1: Entry block
    block1 = LowLevelILBasicBlock(0x1000)
    function.add_basic_block(block1)
    builder = LowLevelILBuilder(function)
    builder.set_current_block(block1)

    # eax = 5
    const5 = builder.const(5)
    set_eax = builder.set_reg(reg_eax, const5)
    builder.add_instruction(set_eax)

    # ebx = 10
    const10 = builder.const(10)
    set_ebx = builder.set_reg(reg_ebx, const10)
    builder.add_instruction(set_ebx)

    # if (eax < ebx) goto block2 else block3
    eax_val = builder.reg(reg_eax)
    ebx_val = builder.reg(reg_ebx)
    # For simplicity, just use eax as condition
    if_stmt = builder.if_stmt(eax_val, InstructionIndex(1), InstructionIndex(2))
    builder.add_instruction(if_stmt)

    # Block 2: True branch
    block2 = LowLevelILBasicBlock(0x1020)
    function.add_basic_block(block2)
    builder.set_current_block(block2)

    # ecx = eax + ebx
    add_result = builder.add(eax_val, ebx_val)
    set_ecx = builder.set_reg(reg_ecx, add_result)
    builder.add_instruction(set_ecx)

    # return ecx
    ecx_val = builder.reg(reg_ecx)
    ret_stmt = builder.ret(ecx_val)
    builder.add_instruction(ret_stmt)

    # Block 3: False branch
    block3 = LowLevelILBasicBlock(0x1040)
    function.add_basic_block(block3)
    builder.set_current_block(block3)

    # ecx = eax * 2
    const2 = builder.const(2)
    mul_result = builder.mul(eax_val, const2)
    set_ecx2 = builder.set_reg(reg_ecx, mul_result)
    builder.add_instruction(set_ecx2)

    # return ecx
    ret_stmt2 = builder.ret(ecx_val)
    builder.add_instruction(ret_stmt2)

    print(f"✅ Created LLIL function with {len(function.basic_blocks)} blocks")
    return function


def demo_complete_pipeline():
    """Demonstrate the complete decompilation pipeline"""
    print("🚀 New IR System Complete Pipeline Demo")
    print("=" * 50)

    # Create pipeline components
    pipeline = DecompilerPipeline()
    generator = TypeScriptGenerator()

    # Test 1: Simple function
    print("\n📝 Test 1: Simple Arithmetic Function")
    print("-" * 40)

    simple_func = pipeline.create_sample_llil_function()
    print(f"LLIL: {simple_func.name} ({len(simple_func.basic_blocks)} blocks)")

    # Run pipeline
    hlil_func = pipeline.decompile_function(simple_func)

    # Generate TypeScript
    ts_code = generator.generate_function(hlil_func)
    print("\n📄 Generated TypeScript:")
    print(ts_code)

    # Test 2: Complex function with control flow
    print("\n📝 Test 2: Complex Function with Control Flow")
    print("-" * 40)

    complex_func = create_complex_llil_function()
    print(f"LLIL: {complex_func.name} ({len(complex_func.basic_blocks)} blocks)")

    # Show LLIL structure
    print("\n🔧 LLIL Structure:")
    for i, block in enumerate(complex_func.basic_blocks):
        print(f"  Block {i} (0x{block.start_address:x}):")
        for j, instr in enumerate(block.instructions):
            print(f"    {j}: {instr}")

    # Run pipeline
    hlil_func2 = pipeline.decompile_function(complex_func)

    # Generate TypeScript
    ts_code2 = generator.generate_function(hlil_func2)
    print("\n📄 Generated TypeScript:")
    print(ts_code2)

    print("\n🎉 Complete pipeline demo finished!")


def demo_ir_levels():
    """Demonstrate each IR level transformation"""
    print("\n🔬 IR Level Transformation Demo")
    print("=" * 50)

    pipeline = DecompilerPipeline()
    llil_func = pipeline.create_sample_llil_function()

    print("\n🔧 LLIL (Low Level IL):")
    print("- Register-based operations")
    print("- Direct memory access")
    print("- Low-level control flow")
    for i, block in enumerate(llil_func.basic_blocks):
        print(f"  Block {i}:")
        for j, instr in enumerate(block.instructions):
            print(f"    {j}: {instr}")

    print("\n🔧 MLIL (Medium Level IL):")
    print("- Variable-based operations")
    print("- Abstract memory operations")
    print("- Simplified control flow")
    mlil_func = pipeline.llil_to_mlil.lift_function(llil_func)
    for i, block in enumerate(mlil_func.basic_blocks):
        print(f"  Block {i}:")
        for j, instr in enumerate(block.instructions):
            print(f"    {j}: {instr}")

    print("\n🔧 HLIL (High Level IL):")
    print("- High-level constructs")
    print("- Structured control flow")
    print("- Type-aware operations")
    hlil_func = pipeline.mlil_to_hlil.lift_function(mlil_func)
    for i, block in enumerate(hlil_func.basic_blocks):
        print(f"  Block {i}:")
        for j, instr in enumerate(block.instructions):
            print(f"    {j}: {instr}")

    print("\n📄 TypeScript Output:")
    generator = TypeScriptGenerator()
    ts_code = generator.generate_function(hlil_func)
    print(ts_code)


def demo_instruction_types():
    """Demonstrate different instruction types"""
    print("\n🧩 Instruction Type Demo")
    print("=" * 50)

    from ..ir.llil import (
        LowLevelILConst, LowLevelILAdd, LowLevelILReg, LowLevelILSetReg,
        LowLevelILIf, LowLevelILRet
    )
    from ..ir.common import ILRegister, InstructionIndex

    # Create some example instructions
    reg_eax = ILRegister("eax", 0, 4)
    reg_ebx = ILRegister("ebx", 1, 4)

    print("\n📋 LLIL Instruction Examples:")

    # Arithmetic
    const_val = LowLevelILConst(42, 4)
    add_op = LowLevelILAdd(LowLevelILReg(reg_eax), const_val, 4)
    print(f"  Arithmetic: {add_op}")

    # Register operations
    set_reg = LowLevelILSetReg(reg_ebx, add_op)
    print(f"  Assignment: {set_reg}")

    # Control flow
    if_stmt = LowLevelILIf(LowLevelILReg(reg_eax), InstructionIndex(1), InstructionIndex(2))
    print(f"  Conditional: {if_stmt}")

    # Terminal
    ret_stmt = LowLevelILRet(LowLevelILReg(reg_eax))
    print(f"  Return: {ret_stmt}")

    print("\n🏷️  Instruction Mixins:")
    print(f"  {const_val} -> Constant: {hasattr(const_val, 'constant')}")
    print(f"  {add_op} -> Arithmetic: {hasattr(add_op, 'left')}")
    print(f"  {if_stmt} -> ControlFlow: {hasattr(if_stmt, 'condition')}")
    print(f"  {ret_stmt} -> Terminal: {hasattr(ret_stmt, 'src')}")


if __name__ == "__main__":
    try:
        demo_complete_pipeline()
        demo_ir_levels()
        demo_instruction_types()
        print("\n🎉 All demos completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()