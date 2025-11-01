"""
TypeScript pipeline coordination

Coordinates between TypeScript generation and parsing for round-trip compatibility.
"""

from typing import Optional, Dict, Any
from ..ir.base import IRFunction
from ..pipeline.decompiler import DecompilerPipeline
from ..pipeline.compiler import CompilerPipeline
from .generator import TypeScriptGenerator, PrettyTypeScriptGenerator, RoundTripTypeScriptGenerator
from .parser import TypeScriptParser


class TypeScriptPipeline:
    """Coordinates TypeScript generation and parsing"""

    def __init__(self, architecture: str = "x86", style: str = "pretty"):
        self.architecture = architecture
        self.style = style

        self.decompiler = DecompilerPipeline(architecture, style)
        self.compiler = CompilerPipeline(architecture)

        # Use appropriate generator based on style
        if style == "round_trip":
            self.generator = RoundTripTypeScriptGenerator()
        else:
            self.generator = PrettyTypeScriptGenerator()

        self.parser = TypeScriptParser()

    def round_trip_test(self, bytecode: bytes) -> Dict[str, Any]:
        """Test round-trip compilation: bytecode → TS → bytecode"""
        results = {
            "original_bytecode": bytecode,
            "typescript_code": None,
            "recompiled_bytecode": None,
            "success": False,
            "errors": []
        }

        try:
            # Decompile to TypeScript
            typescript_code = self.decompiler.decompile_to_typescript(bytecode)
            results["typescript_code"] = typescript_code

            # Recompile to bytecode
            recompiled_bytecode = self.compiler.compile_from_typescript(typescript_code)
            results["recompiled_bytecode"] = recompiled_bytecode

            # Compare results
            results["success"] = self._compare_bytecode(bytecode, recompiled_bytecode)

        except Exception as e:
            results["errors"].append(str(e))

        return results

    def validate_typescript(self, typescript_code: str) -> Dict[str, Any]:
        """Validate TypeScript code can be compiled"""
        validation = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "hlil_function": None
        }

        try:
            hlil_function = self.parser.parse_function(typescript_code)
            validation["hlil_function"] = hlil_function
            validation["valid"] = True
        except Exception as e:
            validation["errors"].append(str(e))

        return validation

    def optimize_for_target(self, typescript_code: str, target: str) -> str:
        """Optimize TypeScript code for specific target"""
        try:
            # Parse to HLIL
            hlil_function = self.parser.parse_function(typescript_code)

            # Create target-specific compiler
            target_compiler = CompilerPipeline(target)

            # Transform through compilation pipeline to get optimized IR
            mlil_function = target_compiler._transform_to_mlil(hlil_function)
            optimized_hlil = target_compiler._transform_to_hlil_optimized(mlil_function)

            # Generate optimized TypeScript
            return self.generator.generate_function(optimized_hlil)

        except Exception:
            return typescript_code  # Return original on error

    def _compare_bytecode(self, original: bytes, recompiled: bytes) -> bool:
        """Compare bytecode for semantic equivalence"""
        # Simple byte comparison for now
        # Real implementation would compare semantic meaning
        return original == recompiled


def create_typescript_pipeline(architecture: str = "x86", style: str = "pretty") -> TypeScriptPipeline:
    """Create a TypeScript pipeline with specified settings"""
    return TypeScriptPipeline(architecture, style)