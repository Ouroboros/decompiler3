#!/usr/bin/env python3
"""
测试新的包结构导入

验证所有模块能正确导入
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """测试基础导入"""
    print("🧪 测试基础导入...")

    # 测试IR模块
    from decompiler3.ir.base import OperationType, IRFunction, IRBasicBlock
    print("✅ IR基础模块导入成功")

    from decompiler3.ir.hlil import HLILConstant, HLILVariable
    print("✅ HLIL模块导入成功")

    # 测试target模块
    from decompiler3.target.capability import TargetCapability
    print("✅ 目标能力模块导入成功")

    # 测试builtin模块
    from decompiler3.builtin.registry import BuiltinRegistry
    print("✅ Built-in注册表模块导入成功")

    # 创建一些对象测试
    func = IRFunction("test", 0x1000)
    const = HLILConstant(42, 4, "number")

    print(f"✅ 测试对象创建: {func.name}, {const}")
    return True

def test_demo_imports():
    """测试演示模块导入"""
    print("\n🎭 测试演示模块导入...")

    # 测试能否导入演示模块（不运行）
    from decompiler3.demos import basic_test
    from decompiler3.examples import add_instruction_example

    print("✅ 演示模块导入成功")
    return True

def main():
    """运行所有测试"""
    print("🚀 测试新包结构")
    print("=" * 30)

    basic_ok = test_basic_imports()
    demo_ok = test_demo_imports()

    print("\n" + "=" * 30)
    if basic_ok and demo_ok:
        print("🎉 所有导入测试通过！")
        print("✅ 包结构重组成功")
        print("✅ 相对导入问题已解决")
    else:
        print("❌ 部分导入测试失败")
        print("需要进一步调试")

if __name__ == "__main__":
    main()