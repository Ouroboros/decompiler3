import unittest
from unittest.mock import MagicMock
from falcom.ed9.lifters.vm_lifter import ED9VMLifter
from falcom.ed9.mlil_translator import translate_falcom_llil_to_mlil
from falcom.ed9.llil_builder import FalcomVMBuilder
from ir.llil.llil import LowLevelILFunction, LowLevelILBasicBlock
from ir.mlil.mlil import MediumLevelILFunction, MLILSetVar, MLILConst
from ir.mlil.mlil_optimizer import optimize_mlil

class TestTranslationFlow(unittest.TestCase):
    def test_optimizer_integration(self):
        # 1. Create a dummy LLIL function
        llil_func = LowLevelILFunction("test_func", 0x1000, 0)
        block = LowLevelILBasicBlock(0x1000, "entry", llil_func)
        llil_func.add_block(block)

        # Add some LLIL instructions that would benefit from optimization
        # e.g., pushing constants and adding them
        # We need to manually construct LLIL instructions or use the builder
        builder = FalcomVMBuilder()
        builder.create_function("test_func", 0x1000, 0)
        llil_block = builder.create_basic_block(0x1000, "entry")
        builder.set_current_block(llil_block)

        # Push 1, Push 2, Add -> Stack has [3]
        builder.push_int(1)
        builder.push_int(2)
        builder.add()

        # Pop to local var (simulated)
        # FalcomVMBuilder doesn't have explicit locals in the same way,
        # but let's assume we pop to a register or global for the sake of MLIL translation
        builder.set_reg(0) # Pop top of stack (3) to reg 0

        llil_func = builder.finalize()

        # 2. Translate to MLIL
        # This calls optimize_mlil(use_ssa=False) internally by default
        mlil_func = translate_falcom_llil_to_mlil(llil_func)

        # 3. Verify MLIL structure
        # Should have: reg0 = 1 + 2 (or folded to 3 if constant folding is enabled in non-SSA)
        # Since I enabled constant folding in BaseMLILOptimizer, it should be 3.

        found_assignment = False
        for block in mlil_func.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, MLILSetVar) and inst.var.index == 0: # reg 0 maps to var 0 usually
                    found_assignment = True
                    # Check if folded
                    if isinstance(inst.value, MLILConst):
                        print(f"Found folded constant: {inst.value.value}")
                        self.assertEqual(inst.value.value, 3)
                    else:
                        print(f"Found expression: {inst.value}")
                        # If not folded, it might be Add(Const(1), Const(2))

        self.assertTrue(found_assignment, "Did not find assignment to register 0")

    def test_ssa_optimization_flow(self):
        # Test explicit SSA optimization call
        llil_func = LowLevelILFunction("test_ssa", 0x2000, 0)
        builder = FalcomVMBuilder()
        builder.create_function("test_ssa", 0x2000, 0)
        llil_block = builder.create_basic_block(0x2000, "entry")
        builder.set_current_block(llil_block)

        # x = 1
        builder.push_int(1)
        builder.set_reg(0)

        # y = x + 2
        builder.get_reg(0)
        builder.push_int(2)
        builder.add()
        builder.set_reg(1)

        llil_func = builder.finalize()

        # Translate (non-SSA first)
        mlil_func = translate_falcom_llil_to_mlil(llil_func)

        # Now optimize with SSA
        mlil_func = optimize_mlil(mlil_func, use_ssa=True)

        # Check results
        # reg1 should be 3
        found_result = False
        for block in mlil_func.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, MLILSetVar) and inst.var.index == 1: # reg 1
                    if isinstance(inst.value, MLILConst) and inst.value.value == 3:
                        found_result = True

        self.assertTrue(found_result, "SSA optimization did not propagate constants to reg1")

if __name__ == '__main__':
    unittest.main()
