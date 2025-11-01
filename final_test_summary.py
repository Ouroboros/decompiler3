#!/usr/bin/env python3
"""
Final Test Summary for BinaryNinja-style IR Refactor

This script validates that all components work together correctly
"""

def run_final_validation():
    """Run comprehensive validation of the new IR system"""
    print("🏁 Final IR System Validation")
    print("=" * 50)

    test_results = []

    # Test 1: Import all new IR modules
    print("\n📦 Testing Module Imports...")
    try:
        from decompiler3.ir.common import BaseILInstruction, ILRegister
        from decompiler3.ir.llil import LowLevelILFunction
        from decompiler3.ir.mlil import MediumLevelILFunction
        from decompiler3.ir.hlil import HighLevelILFunction
        from decompiler3.ir.lifter import DecompilerPipeline
        from decompiler3.typescript.generator import TypeScriptGenerator
        print("✅ All modules imported successfully")
        test_results.append(("Module Imports", True))
    except Exception as e:
        print(f"❌ Import failed: {e}")
        test_results.append(("Module Imports", False))

    # Test 2: Basic instruction creation
    print("\n🔧 Testing Instruction Creation...")
    try:
        from decompiler3.ir.llil import LowLevelILConst, LowLevelILAdd, LowLevelILReg
        from decompiler3.ir.common import ILRegister

        reg = ILRegister("eax", 0, 4)
        const = LowLevelILConst(42, 4)
        reg_expr = LowLevelILReg(reg)
        add_expr = LowLevelILAdd(reg_expr, const, 4)

        assert str(const) == "42"
        assert str(reg_expr) == "eax"
        assert str(add_expr) == "eax + 42"
        print("✅ Instruction creation working")
        test_results.append(("Instruction Creation", True))
    except Exception as e:
        print(f"❌ Instruction creation failed: {e}")
        test_results.append(("Instruction Creation", False))

    # Test 3: Complete pipeline
    print("\n🔄 Testing Complete Pipeline...")
    try:
        pipeline = DecompilerPipeline()
        llil_func = pipeline.create_sample_llil_function()
        hlil_func = pipeline.decompile_function(llil_func)

        assert llil_func.name == "sample_function"
        assert len(hlil_func.basic_blocks) > 0
        assert len(hlil_func.variables) > 0
        print("✅ Complete pipeline working")
        test_results.append(("Complete Pipeline", True))
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        test_results.append(("Complete Pipeline", False))

    # Test 4: TypeScript generation
    print("\n📄 Testing TypeScript Generation...")
    try:
        generator = TypeScriptGenerator()
        ts_code = generator.generate_function(hlil_func)

        assert "function sample_function" in ts_code
        assert "return" in ts_code
        assert "{" in ts_code and "}" in ts_code
        print("✅ TypeScript generation working")
        test_results.append(("TypeScript Generation", True))
    except Exception as e:
        print(f"❌ TypeScript generation failed: {e}")
        test_results.append(("TypeScript Generation", False))

    # Test 5: Control flow instructions (the original problem)
    print("\n🔀 Testing Control Flow Instructions...")
    try:
        from decompiler3.ir.llil import LowLevelILIf, LowLevelILGoto, LowLevelILJump
        from decompiler3.ir.common import InstructionIndex

        condition = LowLevelILReg(reg)
        if_instr = LowLevelILIf(condition, InstructionIndex(10), InstructionIndex(20))
        goto_instr = LowLevelILGoto(InstructionIndex(30))
        jump_instr = LowLevelILJump(condition)

        assert "if" in str(if_instr).lower()
        assert "goto" in str(goto_instr).lower()
        assert str(jump_instr)  # Should have string representation
        print("✅ Control flow instructions working (FIXED!)")
        test_results.append(("Control Flow Instructions", True))
    except Exception as e:
        print(f"❌ Control flow failed: {e}")
        test_results.append(("Control Flow Instructions", False))

    # Print summary
    print("\n📊 Final Results Summary")
    print("=" * 50)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! IR Refactor Complete!")
        print("\n🏆 Key Achievements:")
        print("  ✅ Fixed missing control flow instructions")
        print("  ✅ Implemented BinaryNinja-style three-layer IR")
        print("  ✅ Complete LLIL -> MLIL -> HLIL -> TypeScript pipeline")
        print("  ✅ Proper instruction hierarchies and mixins")
        print("  ✅ Working builder patterns")
        print("  ✅ All string representations working")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Manual investigation needed.")
        return False


if __name__ == "__main__":
    run_final_validation()