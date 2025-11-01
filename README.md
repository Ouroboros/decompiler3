# Decompiler3: BinaryNinja-Style IR System

A Python implementation of a three-layer IR system (LLIL/MLIL/HLIL + SSA) with bidirectional TypeScript compilation support, inspired by BinaryNinja's architecture.

## Architecture

### Three-Layer IR System
- **LLIL (Low Level IR)**: Close to assembly, handles basic operations and control flow
- **MLIL (Medium Level IR)**: Removes stack operations, introduces variables and structured control flow
- **HLIL (High Level IR)**: High-level constructs, ready for decompilation to TypeScript

### SSA Support
Each IR layer supports Static Single Assignment form for optimization and analysis.

### Built-in System
Unified semantic entry points for script domain extensions, replacing traditional intrinsics.

### Target Backend System
Pluggable backend system with capability models for different target architectures.

## Pipeline

### Decompilation Chain
`bytecode → LLIL → MLIL → HLIL → TypeScript`

### Compilation Chain
`TypeScript → HLIL → MLIL → (legalize) → LLIL → instruction selection → bytecode`

## Usage

```python
from decompiler3 import DecompilerPipeline

# Decompile bytecode to TypeScript
pipeline = DecompilerPipeline()
typescript_code = pipeline.decompile_to_typescript(bytecode)

# Compile TypeScript back to bytecode
bytecode = pipeline.compile_from_typescript(typescript_code, target="falcom_vm")
```