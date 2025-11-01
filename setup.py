#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="decompiler3",
    version="1.0.0",
    description="BinaryNinja-style IR decompiler with TypeScript code generation",
    author="Claude AI",
    packages=find_packages(include=['decompiler3', 'decompiler3.*']),
    python_requires=">=3.7",
    install_requires=[
        # No external dependencies - uses only Python standard library
    ],
    extras_require={
        "dev": ["pytest", "black", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "decompiler3=decompiler3.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Compilers",
        "Topic :: Software Development :: Code Generators",
    ],
)