#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="decompiler3",
    version="0.1.0",
    description="BinaryNinja风格的三层IR系统与双向TypeScript编译管道",
    author="Claude",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        # 基础依赖在 requirements.txt
    ],
    extras_require={
        "dev": ["pytest", "black", "mypy"],
        "typescript": ["nodejs"],  # TypeScript解析需要Node.js
    },
    entry_points={
        "console_scripts": [
            "decompiler3=decompiler3.cli:main",
        ],
    },
)