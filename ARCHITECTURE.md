# Decompiler3 Architecture

A BinaryNinja-style three-layer IR system with bidirectional TypeScript compilation support.

## Overview

This system implements a comprehensive intermediate representation (IR) framework inspired by BinaryNinja's architecture, with the following key components:

- **Three-layer IR**: LLIL, MLIL, HLIL with SSA support
- **Built-in System**: Unified semantic entry points replacing intrinsics
- **Target Backend**: Pluggable backend system with capability models
- **Bidirectional Pipeline**: Complete bytecode ↔ TypeScript compilation

## Architecture Components

### 1. IR Layers (`src/ir/`)

#### LLIL (Low Level Intermediate Language)
- Close to assembly level representation
- Stack/register operations
- Memory operations with explicit addressing
- Direct jumps and calls

```python
# Example LLIL
eax = 42
ebx = 10
eax = eax + ebx
return eax
```

#### MLIL (Medium Level Intermediate Language)
- Variables instead of registers/stack locations
- Structured control flow
- Function calls with proper arguments
- Basic type information

```python
# Example MLIL
x = 42
y = 10
result = x + y
return result
```

#### HLIL (High Level Intermediate Language)
- High-level control structures (for, while, switch)
- Rich type information
- Object-oriented constructs
- Ready for TypeScript generation

```python
# Example HLIL
function calculate(x: number, y: number): number {
    return x + y;
}
```

### 2. SSA Support (`src/ir/ssa.py`)

Implements Static Single Assignment form transformation:
- Dominance analysis
- Phi node placement
- Variable renaming
- SSA destruction for final code generation

### 3. Built-in System (`src/builtin/`)

Unified semantic entry points:
- **Registry**: Manages all built-in functions
- **Semantics**: Type checking and validation
- **Definitions**: Standard built-in implementations

Categories:
- Math: `abs`, `pow`, `sqrt`, `sin`, `cos`, `log`
- String: `strlen`, `strcmp`, `strcat`, `substr`
- Memory: `memcpy`, `memset`
- Type: `typeof`, `is_number`, `to_string`
- Script: `script_call`, `get_variable`, `set_variable`

### 4. Target Backend (`src/target/`)

Pluggable backend system:
- **Capability Models**: Define what each target can do
- **Legalization**: Transform IR to target-legal form
- **Instruction Selection**: Map IR to machine instructions

Supported targets:
- **x86**: Traditional register-based architecture
- **Falcom VM**: Stack-based virtual machine
- **ARM**: RISC architecture with conditional execution

### 5. TypeScript Pipeline (`src/typescript/`)

Bidirectional TypeScript support:
- **Generator**: HLIL → TypeScript (pretty/round-trip modes)
- **Parser**: TypeScript → HLIL (with Node.js integration)
- **Pipeline**: Coordinates round-trip compilation

### 6. Compilation Pipelines (`src/pipeline/`)

#### Decompilation Chain
```
bytecode → LLIL → MLIL → HLIL → TypeScript
```

#### Compilation Chain
```
TypeScript → HLIL → MLIL → (legalize) → LLIL → instruction selection → bytecode
```

## Key Features

### 1. Three-Layer Abstraction
Each layer serves a specific purpose:
- **LLIL**: Faithful representation of machine code
- **MLIL**: Intermediate form for analysis and optimization
- **HLIL**: High-level representation for human consumption

### 2. SSA Form Support
All layers support SSA transformation:
- Enables powerful optimizations
- Simplifies data flow analysis
- Standard compiler intermediate form

### 3. Built-in Mechanism
Replaces traditional intrinsics:
- Unified namespace
- Target-specific mappings
- Semantic validation
- Side effect tracking

### 4. Target Capability Model
Describes target limitations:
- Supported operations
- Addressing modes
- Register classes
- Calling conventions

### 5. Bidirectional Compilation
Complete round-trip support:
- Bytecode → TypeScript (decompilation)
- TypeScript → Bytecode (compilation)
- Semantic preservation
- Error handling and diagnostics

## Usage Examples

### Basic Decompilation
```python
from decompiler3 import DecompilerPipeline

pipeline = DecompilerPipeline("falcom_vm", "pretty")
typescript_code = pipeline.decompile_to_typescript(bytecode)
```

### Basic Compilation
```python
from pipeline.compiler import CompilerPipeline

pipeline = CompilerPipeline("falcom_vm")
bytecode = pipeline.compile_from_typescript(typescript_code)
```

### Round-trip Testing
```python
from typescript.pipeline import TypeScriptPipeline

pipeline = TypeScriptPipeline("falcom_vm", "round_trip")
results = pipeline.round_trip_test(bytecode)
```

## Extension Points

### Adding New Targets
1. Create capability model in `target/capability.py`
2. Add legalization rules in `target/legalization.py`
3. Implement instruction selection in `target/instruction_selection.py`

### Adding Built-ins
1. Define in `builtin/definitions.py`
2. Add target mappings
3. Register in global registry

### Extending IR
1. Add new expression types to appropriate layer
2. Update visitors and transformers
3. Add TypeScript generation/parsing support

## Design Principles

1. **Modularity**: Each component is independent and replaceable
2. **Extensibility**: Easy to add new targets and built-ins
3. **Semantic Preservation**: Maintain program meaning across transformations
4. **BinaryNinja Compatibility**: Follow established IR patterns
5. **Round-trip Capability**: Support both directions of compilation

## Dependencies

- Python 3.8+
- Optional: Node.js (for advanced TypeScript parsing)
- typing-extensions
- dataclasses-json
- pydantic

## Future Enhancements

1. **Advanced Optimizations**: Dead code elimination, constant folding
2. **Debug Information**: Source location preservation
3. **Multi-function Support**: Whole-program analysis
4. **Interactive Analysis**: IDE integration and debugging
5. **Additional Targets**: WASM, RISC-V, custom architectures