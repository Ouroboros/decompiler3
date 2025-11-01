#!/usr/bin/env python3
"""
基础测试 - 最小依赖版本

测试系统的核心组件是否正常工作
"""

def test_basic_imports():
    """测试基础导入"""
    print("测试基础导入...")

    from decompiler3.ir.base import IRFunction, OperationType
    print("✓ IR基础模块导入成功")

    from decompiler3.ir.hlil import HLILConstant
    print("✓ HLIL模块导入成功")

    from decompiler3.target.capability import TargetCapability
    print("✓ 目标能力模块导入成功")

    from decompiler3.builtin.registry import BuiltinRegistry
    print("✓ Built-in注册表模块导入成功")

    return True

def test_ir_creation():
    """测试IR创建"""
    print("\n测试IR创建...")

    from decompiler3.ir.base import IRFunction, IRBasicBlock, OperationType
    from decompiler3.ir.hlil import HLILConstant, HLILReturn

    # 创建函数
    func = IRFunction("test_func")
    block = IRBasicBlock()
    func.basic_blocks.append(block)

    # 创建表达式
    const = HLILConstant(42, 4, "number")
    ret = HLILReturn(const)

    print(f"✓ 创建函数: {func.name}")
    print(f"✓ 创建常量: {const}")
    print(f"✓ 创建返回: {ret}")

    return True

def test_target_capabilities():
    """测试目标能力"""
    print("\n测试目标能力...")

    from decompiler3.target.capability import get_target_capability, X86Capability

    # 测试获取目标能力
    x86 = get_target_capability("x86")
    if x86:
        print(f"✓ X86能力: {x86.name}")

    falcom = get_target_capability("falcom_vm")
    if falcom:
        print(f"✓ Falcom VM能力: {falcom.name}")

    return True

def test_builtin_registry():
    """测试Built-in注册表"""
    print("\n测试Built-in注册表...")

    from decompiler3.builtin.registry import builtin_registry, get_builtin

    # 测试获取built-in
    abs_func = get_builtin("abs")
    if abs_func:
        print(f"✓ 获取abs函数: {abs_func.signature.name}")

    # 测试列出类别
    math_funcs = builtin_registry.list_by_category("math")
    print(f"✓ 数学函数: {len(math_funcs)} 个")

    return True

def test_machine_instruction():
    """测试机器指令"""
    print("\n测试机器指令...")

    from decompiler3.target.instruction_selection import MachineInstruction

    # 创建指令
    instr = MachineInstruction("add", ["eax", "ebx"], cost=1)
    print(f"✓ 创建指令: {instr}")

    return True

def demonstrate_system():
    """演示系统工作"""
    print("\n=== 系统演示 ===")

    print("1. 创建简单的IR函数")
    print("   function test(): number { return 42; }")
    print()

    print("2. IR表示 (HLIL):")
    print("   HLILReturn(HLILConstant(42, type='number'))")
    print()

    print("3. 目标架构支持:")
    print("   - x86: 寄存器机器, 支持复杂寻址")
    print("   - Falcom VM: 栈机器, 简单指令集")
    print("   - ARM: RISC架构, 条件执行")
    print()

    print("4. Built-in函数:")
    print("   - 数学: abs, pow, sqrt, sin, cos")
    print("   - 字符串: strlen, strcmp, strcat")
    print("   - 系统: debug_print, script_call")
    print()

    print("5. 编译流程:")
    print("   TypeScript → HLIL → MLIL → LLIL → 机器码")
    print()

    print("6. 反编译流程:")
    print("   字节码 → LLIL → MLIL → HLIL → TypeScript")

def main():
    """主测试函数"""
    print("BinaryNinja风格IR系统 - 基础测试")
    print("=" * 45)

    tests = [
        test_basic_imports,
        test_ir_creation,
        test_target_capabilities,
        test_builtin_registry,
        test_machine_instruction
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过!")
    elif passed > 0:
        print("⚠️  部分测试通过")
    else:
        print("❌ 测试失败")

    demonstrate_system()

if __name__ == "__main__":
    main()