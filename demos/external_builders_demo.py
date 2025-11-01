#!/usr/bin/env python3
"""
External Builders Demo - Using VM-specific builders from external files

This demonstrates how VM-specific builders can be created and used
outside the core IR system, maintaining clean separation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decompiler3.ir.llil import LowLevelILFunction, LowLevelILBasicBlock
from falcom_vm_builder import FalcomVMBuilder, create_falcom_builder


def demo_external_falcom_builder():
    """Demonstrate external Falcom VM builder"""
    print("\n🎮 External Falcom VM Builder Demo:")
    print("-" * 50)

    # Create function using external builder
    function = LowLevelILFunction("falcom_event_script", 0x1000)
    builder = create_falcom_builder(function)

    # Create main block
    main_block = LowLevelILBasicBlock(0x1000)
    function.add_basic_block(main_block)
    builder.set_current_block(main_block)

    # Use high-level Falcom patterns
    builder.npc_talk(101, "Welcome to our town!")
    builder.item_give(205, 3)  # Give 3 of item 205
    builder.bgm_play(15, 2000)  # Play BGM 15 with 2s fade
    builder.battle_start(42, "dragon_formation")
    builder.cutscene_play("victory_scene")
    builder.RETURN(0)

    print(f"✅ Created function: {function.name}")
    print(f"   📊 Instructions: {sum(len(block.instructions) for block in function.basic_blocks)}")

    print("\n🔧 Generated LLIL (High-level Falcom patterns):")
    for i, block in enumerate(function.basic_blocks):
        print(f"  Block {i}:")
        for j, instr in enumerate(block.instructions):
            print(f"    {j}: {instr}")
        print()

    return function




def demo_builder_modularity():
    """Demonstrate builder modularity and extensibility"""
    print("\n🔧 Builder Modularity Demo:")
    print("-" * 50)

    print("📁 External Builder Files:")
    print("  ✅ demos/falcom_vm_builder.py - Falcom game engine patterns")
    print("  🚀 demos/other_vm_builder.py - Other VMs (could be added as needed)")

    print("\n🏗️ Architecture Benefits:")
    print("  ✅ Clean separation: Core IR vs VM-specific logic")
    print("  ✅ Modularity: Each VM in its own file")
    print("  ✅ Extensibility: Easy to add new VMs")
    print("  ✅ No core pollution: llil.py stays clean")
    print("  ✅ Custom patterns: High-level domain operations")

    print("\n📦 Usage Pattern:")
    print("  1. Import VM-specific builder: from falcom_vm_builder import FalcomVMBuilder")
    print("  2. Create function: func = LowLevelILFunction('name', addr)")
    print("  3. Create builder: builder = FalcomVMBuilder(func)")
    print("  4. Use patterns: builder.battle_start(25, 'boss_formation')")

    # Show how easy it is to create new VMs
    print("\n🚀 Easy Extension Example:")
    print("```python")
    print("# demos/custom_vm_builder.py")
    print("class CustomVMBuilder(LowLevelILBuilderExtended):")
    print("    def custom_pattern(self, arg):")
    print("        self.add_instruction(self.push_int(arg))")
    print("        return self.call_func('custom_op')")
    print("```")


def main():
    """External builders demo showing modular architecture"""
    print("🏗️  External Builders Demo - Modular VM Architecture")
    print("=" * 70)

    print("\n📋 This demo shows:")
    print("  🔹 VM builders outside core IR system")
    print("  🔹 Clean separation of concerns")
    print("  🔹 Easy extensibility for new VMs")
    print("  🔹 High-level domain-specific patterns")

    # Demo external builders
    falcom_func = demo_external_falcom_builder()
    demo_builder_modularity()

    print("\n🎯 Key Advantages of External Builders:")
    print("-" * 50)
    print("  ✅ Core system stays clean and focused")
    print("  ✅ VM-specific logic is modular and isolated")
    print("  ✅ Easy to maintain and extend")
    print("  ✅ No cross-VM contamination")
    print("  ✅ Perfect for plugin-style architecture")

    print("\n📁 File Organization:")
    print("-" * 50)
    print("  📂 decompiler3/ir/llil.py - Core IR (clean!)")
    print("  📂 demos/falcom_vm_builder.py - Falcom VM patterns")
    print("  📂 demos/external_builders_demo.py - Usage examples")

    print("\n🎉 External builders demo completed!")
    print("This shows how to keep core IR clean while supporting any VM!")


if __name__ == "__main__":
    main()