#!/usr/bin/env python3
"""
decompiler3 命令行接口

提供编译、反编译和演示功能的统一入口点
"""

import argparse
import sys
from typing import Optional

def run_demo(demo_name: str):
    """运行指定的演示"""
    print(f"🚀 运行演示: {demo_name}")

    if demo_name == "basic":
        from decompiler3.demos.basic_test import main
        main()
    elif demo_name == "real_system":
        from decompiler3.demos.real_system_demo import main
        main()
    elif demo_name == "generator":
        from decompiler3.demos.correct_generator_design import main
        main()
    elif demo_name == "add_instruction":
        from decompiler3.examples.add_instruction_example import main
        main()
    elif demo_name == "extend_vm":
        from decompiler3.examples.extend_falcom_vm import main
        main()
    else:
        print(f"❌ 未知演示: {demo_name}")
        print("可用演示: basic, real_system, generator, add_instruction, extend_vm")

def compile_typescript(input_file: str, output_file: str, target: str = "x86"):
    """编译TypeScript到字节码"""
    print(f"🔨 编译 {input_file} -> {output_file} (目标: {target})")

    try:
        from decompiler3.pipeline.compiler import Compiler

        compiler = Compiler(target)
        with open(input_file, 'r') as f:
            typescript_code = f.read()

        bytecode = compiler.compile(typescript_code)

        with open(output_file, 'wb') as f:
            f.write(bytecode)

        print(f"✅ 编译完成: {len(bytecode)} 字节")

    except Exception as e:
        print(f"❌ 编译失败: {e}")
        sys.exit(1)

def decompile_bytecode(input_file: str, output_file: str, target: str = "x86"):
    """反编译字节码到TypeScript"""
    print(f"🔍 反编译 {input_file} -> {output_file} (来源: {target})")

    try:
        from decompiler3.pipeline.decompiler import Decompiler

        decompiler = Decompiler(target)
        with open(input_file, 'rb') as f:
            bytecode = f.read()

        typescript_code = decompiler.decompile(bytecode)

        with open(output_file, 'w') as f:
            f.write(typescript_code)

        print(f"✅ 反编译完成: {len(typescript_code)} 字符")

    except Exception as e:
        print(f"❌ 反编译失败: {e}")
        sys.exit(1)

def list_targets():
    """列出支持的目标架构"""
    try:
        from decompiler3.target.capability import get_available_targets
        targets = get_available_targets()

        print("支持的目标架构:")
        for target in targets:
            print(f"  • {target}")
    except Exception as e:
        print(f"❌ 获取目标列表失败: {e}")

def show_info():
    """显示系统信息"""
    print("decompiler3 - BinaryNinja风格的三层IR系统")
    print("=" * 50)

    try:
        from decompiler3 import __version__
        print(f"版本: {__version__}")
    except:
        print("版本: 开发版")

    print("\n核心组件:")
    print("  • 三层IR架构 (LLIL/MLIL/HLIL + SSA)")
    print("  • Built-in函数系统")
    print("  • 多目标架构后端")
    print("  • 双向TypeScript编译管道")

    print("\n可用命令:")
    print("  • demo - 运行演示")
    print("  • compile - 编译TypeScript")
    print("  • decompile - 反编译字节码")
    print("  • targets - 列出支持的架构")

def main():
    """主入口点"""
    parser = argparse.ArgumentParser(
        prog='decompiler3',
        description='BinaryNinja风格的三层IR系统与双向TypeScript编译管道'
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # demo命令
    demo_parser = subparsers.add_parser('demo', help='运行演示')
    demo_parser.add_argument('name', choices=[
        'basic', 'real_system', 'generator', 'add_instruction', 'extend_vm'
    ], help='演示名称')

    # compile命令
    compile_parser = subparsers.add_parser('compile', help='编译TypeScript')
    compile_parser.add_argument('input', help='输入TypeScript文件')
    compile_parser.add_argument('output', help='输出字节码文件')
    compile_parser.add_argument('--target', default='x86', help='目标架构')

    # decompile命令
    decompile_parser = subparsers.add_parser('decompile', help='反编译字节码')
    decompile_parser.add_argument('input', help='输入字节码文件')
    decompile_parser.add_argument('output', help='输出TypeScript文件')
    decompile_parser.add_argument('--target', default='x86', help='源架构')

    # targets命令
    subparsers.add_parser('targets', help='列出支持的目标架构')

    # info命令
    subparsers.add_parser('info', help='显示系统信息')

    args = parser.parse_args()

    if args.command == 'demo':
        run_demo(args.name)
    elif args.command == 'compile':
        compile_typescript(args.input, args.output, args.target)
    elif args.command == 'decompile':
        decompile_bytecode(args.input, args.output, args.target)
    elif args.command == 'targets':
        list_targets()
    elif args.command == 'info':
        show_info()
    else:
        show_info()

if __name__ == '__main__':
    main()