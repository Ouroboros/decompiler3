#!/usr/bin/env python3
"""
运行演示脚本

使用正确的Python包结构运行所有演示
"""

import sys
import os

# 确保能够导入decompiler3包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_basic_test():
    """运行基础测试"""
    print("🔧 运行基础测试")
    print("=" * 30)

    from decompiler3.demos.basic_test import main
    main()
    print("✅ 基础测试完成\n")

def run_real_system_demo():
    """运行真实系统演示"""
    print("🏗️ 运行真实系统演示")
    print("=" * 30)

    from decompiler3.demos.real_system_demo import main
    main()
    print("✅ 真实系统演示完成\n")

def run_generator_design_demo():
    """运行代码生成器设计演示"""
    print("🎯 运行代码生成器设计演示")
    print("=" * 40)

    from decompiler3.demos.correct_generator_design import main
    main()
    print("✅ 代码生成器设计演示完成\n")

def run_add_instruction_example():
    """运行添加指令示例"""
    print("⚙️ 运行添加指令示例")
    print("=" * 30)

    from decompiler3.examples.add_instruction_example import main
    main()
    print("✅ 添加指令示例完成\n")

def run_extend_falcom_vm():
    """运行Falcom VM扩展示例"""
    print("🎮 运行Falcom VM扩展示例")
    print("=" * 35)

    from decompiler3.examples.extend_falcom_vm import main
    main()
    print("✅ Falcom VM扩展示例完成\n")

def run_lifter_demo():
    """运行LLIL Lifter演示"""
    print("🚀 运行LLIL Lifter演示")
    print("=" * 30)

    from decompiler3.demos.lifter_demo import main
    main()
    print("✅ LLIL Lifter演示完成\n")

def main():
    """主函数 - 运行所有演示"""
    print("🚀 BinaryNinja风格IR系统演示")
    print("=" * 50)
    print("使用正确的Python包结构")
    print()

    # 运行所有演示 - 直接执行，不隐藏错误
    demos = [
        ("基础测试", run_basic_test),
        ("真实系统演示", run_real_system_demo),
        ("代码生成器设计", run_generator_design_demo),
        ("添加指令示例", run_add_instruction_example),
        ("Falcom VM扩展", run_extend_falcom_vm),
        ("LLIL Lifter演示", run_lifter_demo),
    ]

    for name, demo_func in demos:
        print(f"\n▶️ 开始执行: {name}")
        demo_func()  # 直接执行，让错误暴露

    print("🎯 演示总结:")
    print("=" * 20)
    print("✅ 所有演示使用正确的包结构")
    print("✅ 不再有相对导入问题")
    print("✅ 使用真实的项目类型")
    print("✅ 所有错误都会直接暴露")

if __name__ == "__main__":
    main()