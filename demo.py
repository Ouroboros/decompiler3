#!/usr/bin/env python3
"""
BinaryNinja-style IR System Demo

Main entry point for demonstrating the complete IR system.
Run this script to see the IR system in action.
"""

from decompiler3.ir.lifter import DecompilerPipeline
from decompiler3.typescript.generator import TypeScriptGenerator


def main():
    """Main demo function"""
    print("🎯 BinaryNinja-style IR System Demo")
    print("=" * 50)

    # Create pipeline and generator
    pipeline = DecompilerPipeline()
    generator = TypeScriptGenerator()

    print("\n📋 System Overview:")
    print("  🔹 Three-layer IR system (LLIL → MLIL → HLIL)")
    print("  🔹 BinaryNinja-compatible instruction set")
    print("  🔹 Complete TypeScript code generation")
    print("  🔹 Proper control flow handling")

    print("\n🔄 Running Sample Pipeline:")
    print("-" * 30)

    # Create and process a sample function
    llil_func = pipeline.create_sample_llil_function()
    print(f"✅ Created LLIL function: {llil_func.name}")

    # Run complete decompilation
    hlil_func = pipeline.decompile_function(llil_func)

    # Generate TypeScript
    ts_code = generator.generate_function(hlil_func)

    print("\n📄 Generated TypeScript:")
    print("-" * 30)
    print(ts_code)

    print("\n🎉 Demo completed successfully!")
    print("\nFor more detailed demos, run:")
    print("  📁 python3 demo_ir_system.py")
    print("  📁 python3 -m decompiler3.demos.ir_demo")


if __name__ == "__main__":
    main()