# IR System Demo Guide

This guide shows how to run demonstrations of the BinaryNinja-style IR system.

## Quick Start

Run the main demo:
```bash
python3 demo.py
```

## Available Demos

### 1. Main Demo (`demo.py`)
Simple demonstration showing the complete pipeline in action.

### 2. Comprehensive Demo (`demo_ir_system.py`)
Detailed demonstration with multiple test cases:
```bash
python3 demo_ir_system.py
```

### 3. Module Demo
Run as a Python module:
```bash
python3 -m decompiler3.demos.ir_demo
```

## What the Demos Show

- **Three-layer IR System**: LLIL → MLIL → HLIL transformations
- **Control Flow Instructions**: Proper handling of jumps, branches, and loops
- **TypeScript Generation**: Complete code generation from HLIL
- **BinaryNinja Compatibility**: Following BinaryNinja's proven architecture

## System Features Demonstrated

✅ **Fixed Control Flow**: The original missing branch/jump instructions are now fully implemented
✅ **Complete Pipeline**: End-to-end decompilation from low-level to TypeScript
✅ **Proper Architecture**: BinaryNinja-style instruction hierarchies and mixins
✅ **Builder Patterns**: Clean IR construction with builder classes
✅ **Type System**: Proper variable and type handling across all IR levels

## Expected Output

When you run the demos, you should see:
1. 🔄 Pipeline execution with stage-by-stage progress
2. 📄 Generated TypeScript code with proper formatting
3. ✅ All tests passing successfully
4. 🎉 Confirmation that the system is working correctly

The demos validate that all the core issues from the original IL design have been resolved.