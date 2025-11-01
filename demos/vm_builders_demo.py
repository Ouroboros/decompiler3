#!/usr/bin/env python3
"""
VM Builders Demo - Extensible Builder Architecture

Demonstrates how different VMs can implement their own specialized builders
while inheriting from the common LowLevelILBuilderExtended base.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decompiler3.ir.lifter import DecompilerPipeline
from decompiler3.typescript.generator import TypeScriptGenerator
from decompiler3.ir.llil import (
    LowLevelILFunction, LowLevelILBasicBlock,
    FalcomVMBuilder, JavaVMBuilder, PythonVMBuilder, create_vm_builder
)
from decompiler3.ir.common import ILRegister


def create_falcom_av_function() -> LowLevelILFunction:
    """Create AV_04_0017 using FalcomVMBuilder"""
    function = LowLevelILFunction("AV_04_0017", 0x2000)

    # Use Falcom VM specific builder
    builder = FalcomVMBuilder(function)

    # Create main basic block
    main_block = LowLevelILBasicBlock(0x2000)
    function.add_basic_block(main_block)
    builder.set_current_block(main_block)

    # High-level Falcom patterns
    builder.DEBUG_SET_LINENO(2519)
    builder.falcom_function_call('map_event_box_set_enable', 'loc_243E3', 0, 'AV_04_0017')

    builder.DEBUG_SET_LINENO(2520)
    builder.falcom_function_call('avoice_play', 'loc_243FB', 427)

    builder.DEBUG_SET_LINENO(2521)
    builder.RETURN(0x00000000)

    return function


def create_falcom_game_function() -> LowLevelILFunction:
    """Create a more complex Falcom game function"""
    function = LowLevelILFunction("event_battle_sequence", 0x3000)

    builder = FalcomVMBuilder(function)

    main_block = LowLevelILBasicBlock(0x3000)
    function.add_basic_block(main_block)
    builder.set_current_block(main_block)

    # Game-specific high-level operations
    builder.item_give(101, 5)  # Give 5 of item 101
    builder.battle_start(25, "boss_formation")  # Start boss battle
    builder.item_give(201, 1)  # Give reward item
    builder.RETURN()

    return function


def create_java_function() -> LowLevelILFunction:
    """Create function using JavaVMBuilder"""
    function = LowLevelILFunction("java_method", 0x4000)

    builder = JavaVMBuilder(function)

    main_block = LowLevelILBasicBlock(0x4000)
    function.add_basic_block(main_block)
    builder.set_current_block(main_block)

    # Java VM operations
    builder.add_instruction(builder.aload(0))  # Load 'this'
    builder.add_instruction(builder.push_str("Hello World"))

    # Duplicate string on stack
    dup_instrs = builder.dup()
    for instr in dup_instrs:
        builder.add_instruction(instr)

    builder.add_instruction(builder.invokevirtual("println"))
    builder.add_instruction(builder.ret())

    return function


def create_python_function() -> LowLevelILFunction:
    """Create function using PythonVMBuilder"""
    function = LowLevelILFunction("python_func", 0x5000)

    builder = PythonVMBuilder(function)

    main_block = LowLevelILBasicBlock(0x5000)
    function.add_basic_block(main_block)
    builder.set_current_block(main_block)

    # Python VM operations
    builder.add_instruction(builder.LOAD_GLOBAL("print"))
    builder.add_instruction(builder.push_str("Hello from Python VM"))
    builder.add_instruction(builder.CALL_FUNCTION(1))

    builder.STORE_FAST("result")
    builder.add_instruction(builder.ret())

    return function


def demo_vm_factory():
    """Demonstrate the VM builder factory"""
    print("\n🏭 VM Builder Factory Demo:")
    print("-" * 40)

    base_func = LowLevelILFunction("test_function", 0x6000)

    # Create different VM builders using factory
    falcom_builder = create_vm_builder('falcom', base_func)
    java_builder = create_vm_builder('java', base_func)
    python_builder = create_vm_builder('python', base_func)
    generic_builder = create_vm_builder('generic', base_func)

    print(f"✅ Falcom Builder: {type(falcom_builder).__name__}")
    print(f"✅ Java Builder: {type(java_builder).__name__}")
    print(f"✅ Python Builder: {type(python_builder).__name__}")
    print(f"✅ Generic Builder: {type(generic_builder).__name__}")

    # Show specialized methods
    print(f"\n🔧 Falcom Builder Methods:")
    falcom_methods = [method for method in dir(falcom_builder)
                     if not method.startswith('_') and hasattr(getattr(falcom_builder, method), '__call__')]
    print(f"   Specialized: battle_start, item_give, falcom_function_call")

    print(f"\n🔧 Java Builder Methods:")
    print(f"   Specialized: dup, swap, aload, invokevirtual")

    print(f"\n🔧 Python Builder Methods:")
    print(f"   Specialized: LOAD_GLOBAL, STORE_FAST, CALL_FUNCTION")


def main():
    """VM Builders demo showing extensible architecture"""
    print("🏗️  VM Builders Demo - Extensible Architecture")
    print("=" * 60)

    print("\n📋 This demo shows:")
    print("  🔹 VM-specific builder inheritance")
    print("  🔹 High-level domain-specific operations")
    print("  🔹 Factory pattern for builder creation")
    print("  🔹 Falcom, Java, Python VM examples")

    # Demo Falcom VM Builder
    print("\n🎮 Falcom VM Builder Demo:")
    print("-" * 40)

    falcom_func = create_falcom_av_function()
    print(f"✅ Created Falcom function: {falcom_func.name}")
    print(f"   📊 Instructions: {sum(len(block.instructions) for block in falcom_func.basic_blocks)}")

    print("\n🔧 Falcom LLIL Structure (High-level patterns):")
    for i, block in enumerate(falcom_func.basic_blocks):
        print(f"  Block {i}:")
        for j, instr in enumerate(block.instructions):
            print(f"    {j}: {instr}")
        print()

    # Demo complex Falcom function
    print("\n🎮 Complex Falcom Game Function:")
    print("-" * 40)

    game_func = create_falcom_game_function()
    print(f"✅ Created game function: {game_func.name}")

    print("\n🔧 Game Function LLIL:")
    for i, block in enumerate(game_func.basic_blocks):
        print(f"  Block {i}:")
        for j, instr in enumerate(block.instructions):
            print(f"    {j}: {instr}")
        print()

    # Demo Java VM Builder
    print("\n☕ Java VM Builder Demo:")
    print("-" * 40)

    java_func = create_java_function()
    print(f"✅ Created Java function: {java_func.name}")

    print("\n🔧 Java LLIL Structure:")
    for i, block in enumerate(java_func.basic_blocks):
        print(f"  Block {i}:")
        for j, instr in enumerate(block.instructions):
            print(f"    {j}: {instr}")
        print()

    # Demo Python VM Builder
    print("\n🐍 Python VM Builder Demo:")
    print("-" * 40)

    python_func = create_python_function()
    print(f"✅ Created Python function: {python_func.name}")

    print("\n🔧 Python LLIL Structure:")
    for i, block in enumerate(python_func.basic_blocks):
        print(f"  Block {i}:")
        for j, instr in enumerate(block.instructions):
            print(f"    {j}: {instr}")
        print()

    # Demo factory pattern
    demo_vm_factory()

    print("\n🎯 Builder Architecture Benefits:")
    print("-" * 40)
    print("  ✅ Extensible: Easy to add new VM types")
    print("  ✅ Domain-specific: Each VM has specialized methods")
    print("  ✅ Inheritance: Reuses common functionality")
    print("  ✅ High-level: Complex patterns in simple calls")
    print("  ✅ Factory: Clean instantiation pattern")

    print("\n🔧 Architecture Summary:")
    print("-" * 40)
    print("  1. LowLevelILBuilder (Binary Ninja base)")
    print("  2. LowLevelILBuilderExtended (stack + typed operands)")
    print("  3. FalcomVMBuilder (game engine patterns)")
    print("  4. JavaVMBuilder (JVM bytecode patterns)")
    print("  5. PythonVMBuilder (CPython bytecode patterns)")
    print("  6. create_vm_builder() factory function")

    print("\n🎉 VM Builders demo completed!")
    print("This architecture allows easy extension for any VM type!")


if __name__ == "__main__":
    main()