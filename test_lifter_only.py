#!/usr/bin/env python3
"""
只运行 test_complete_llil_lifter 函数
"""

import sys
import os

# 确保能够导入decompiler3包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """只运行 test_complete_llil_lifter 函数"""
    print("🎯 单独运行 test_complete_llil_lifter 函数")
    print("=" * 60)

    try:
        from decompiler3.demos.lifter_demo import test_complete_llil_lifter
        result = test_complete_llil_lifter()

        print("\n🎉 test_complete_llil_lifter 执行完成!")
        print(f"返回的 MLIL 函数: {result.name}")
        print(f"变量数量: {len(result.variables)}")
        print(f"参数数量: {len(result.parameters)}")
        print(f"基本块数量: {len(result.basic_blocks)}")

        return 0

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())