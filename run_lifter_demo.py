#!/usr/bin/env python3
"""
运行 LLIL Lifter 演示

独立运行脚本，解决 import 问题
"""

import sys
import os

# 确保能够导入decompiler3包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """运行 LLIL Lifter 演示"""
    print("🚀 LLIL Lifter 独立演示")
    print("=" * 40)
    print("包含完整的栈分析、变量恢复、内存访问分析等功能")
    print()

    try:
        from decompiler3.demos.lifter_demo import main as lifter_main
        lifter_main()
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())