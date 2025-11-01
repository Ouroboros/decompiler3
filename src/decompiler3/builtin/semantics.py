"""
Semantic analysis and validation for built-in functions

Provides validation, type checking, and semantic analysis for built-in calls.
"""

from typing import List, Dict, Optional, Any, Tuple
from ..ir.base import IRExpression, IRVariable, OperationType
from ..ir.mlil import MLILExpression, MLILBuiltinCall, MLILConstant
from ..ir.hlil import HLILExpression, HLILBuiltinCall, HLILConstant
from .registry import BuiltinFunction, SideEffect, get_builtin


class BuiltinValidator:
    """Validates built-in function calls"""

    def __init__(self):
        self.type_hierarchy = {
            "any": [],
            "number": ["any"],
            "string": ["any"],
            "boolean": ["any"],
            "pointer": ["any"],
            "object": ["any"]
        }

    def validate_call(self, builtin_call: MLILBuiltinCall) -> List[str]:
        """Validate a built-in function call and return error messages"""
        errors = []

        builtin = get_builtin(builtin_call.builtin_name)
        if not builtin:
            errors.append(f"Unknown built-in function: {builtin_call.builtin_name}")
            return errors

        # Validate argument count
        expected_count = len(builtin.signature.parameters)
        actual_count = len(builtin_call.arguments)

        # Handle variadic functions (parameters ending with "...type")
        if (builtin.signature.parameters and
            builtin.signature.parameters[-1].startswith("...")):
            min_count = expected_count - 1
            if actual_count < min_count:
                errors.append(f"Too few arguments for {builtin_call.builtin_name}: "
                             f"expected at least {min_count}, got {actual_count}")
        elif actual_count != expected_count:
            errors.append(f"Wrong number of arguments for {builtin_call.builtin_name}: "
                         f"expected {expected_count}, got {actual_count}")

        # Validate argument types
        for i, (arg, expected_type) in enumerate(zip(builtin_call.arguments, builtin.signature.parameters)):
            if expected_type.startswith("..."):
                break  # Variadic arguments are not type-checked strictly

            actual_type = self._infer_type(arg)
            if not self._is_compatible_type(actual_type, expected_type):
                errors.append(f"Type mismatch for argument {i+1} of {builtin_call.builtin_name}: "
                             f"expected {expected_type}, got {actual_type}")

        # Custom validation
        if not builtin.validate_args(builtin_call.arguments):
            errors.append(f"Custom validation failed for {builtin_call.builtin_name}")

        return errors

    def _infer_type(self, expr: IRExpression) -> str:
        """Infer the type of an expression"""
        if isinstance(expr, (MLILConstant, HLILConstant)):
            if isinstance(expr.value, bool):
                return "boolean"
            elif isinstance(expr.value, (int, float)):
                return "number"
            elif isinstance(expr.value, str):
                return "string"
            else:
                return "any"

        # Check if expression has type annotation
        if hasattr(expr, 'expr_type') and expr.expr_type:
            return expr.expr_type

        # Default to any for unknown types
        return "any"

    def _is_compatible_type(self, actual: str, expected: str) -> bool:
        """Check if actual type is compatible with expected type"""
        if expected == "any" or actual == "any":
            return True

        if actual == expected:
            return True

        # Check type hierarchy
        return expected in self.type_hierarchy.get(actual, [])


class BuiltinAnalyzer:
    """Analyzes built-in function calls for optimization opportunities"""

    def __init__(self):
        self.constant_folder = ConstantFolder()

    def analyze_call(self, builtin_call: MLILBuiltinCall) -> Dict[str, Any]:
        """Analyze a built-in call and return analysis results"""
        analysis = {
            "can_constant_fold": False,
            "constant_value": None,
            "side_effects": [],
            "optimization_hints": []
        }

        builtin = get_builtin(builtin_call.builtin_name)
        if not builtin:
            return analysis

        # Check for constant folding opportunities
        if self.constant_folder.can_fold(builtin_call):
            analysis["can_constant_fold"] = True
            analysis["constant_value"] = self.constant_folder.fold(builtin_call)

        # Analyze side effects
        analysis["side_effects"] = builtin.signature.side_effects

        # Generate optimization hints
        analysis["optimization_hints"] = self._generate_optimization_hints(builtin_call, builtin)

        return analysis

    def _generate_optimization_hints(self, call: MLILBuiltinCall, builtin: BuiltinFunction) -> List[str]:
        """Generate optimization hints for a built-in call"""
        hints = []

        # Check for pure functions that can be memoized
        if builtin.signature.side_effects == [SideEffect.NONE]:
            hints.append("pure_function")

        # Check for expensive operations
        expensive_ops = ["pow", "sqrt", "sin", "cos", "exp", "log"]
        if builtin.signature.name in expensive_ops:
            hints.append("expensive_operation")

        # Check for memory operations that might alias
        if SideEffect.MEMORY_WRITE in builtin.signature.side_effects:
            hints.append("memory_write")

        return hints


class ConstantFolder:
    """Constant folding for built-in functions"""

    def can_fold(self, builtin_call: MLILBuiltinCall) -> bool:
        """Check if a built-in call can be constant folded"""
        builtin = get_builtin(builtin_call.builtin_name)
        if not builtin:
            return False

        # Only fold pure functions
        if builtin.signature.side_effects != [SideEffect.NONE]:
            return False

        # All arguments must be constants
        return all(isinstance(arg, (MLILConstant, HLILConstant))
                  for arg in builtin_call.arguments)

    def fold(self, builtin_call: MLILBuiltinCall) -> Any:
        """Perform constant folding on a built-in call"""
        if not self.can_fold(builtin_call):
            return None

        builtin_name = builtin_call.builtin_name
        args = [arg.value for arg in builtin_call.arguments
                if isinstance(arg, (MLILConstant, HLILConstant))]

        # Implement constant folding for common built-ins
        try:
            if builtin_name == "abs":
                return abs(args[0])
            elif builtin_name == "pow":
                return pow(args[0], args[1])
            elif builtin_name == "sqrt":
                import math
                return math.sqrt(args[0])
            elif builtin_name == "strlen":
                return len(str(args[0]))
            elif builtin_name == "is_number":
                return isinstance(args[0], (int, float))
            # Add more built-ins as needed
        except (ValueError, TypeError, ZeroDivisionError):
            return None

        return None


class BuiltinSemantics:
    """Semantic information and analysis for built-in functions"""

    def __init__(self):
        self.validator = BuiltinValidator()
        self.analyzer = BuiltinAnalyzer()

    def get_semantics(self, builtin_call: MLILBuiltinCall) -> Dict[str, Any]:
        """Get complete semantic information for a built-in call"""
        semantics = {
            "validation_errors": [],
            "analysis": {},
            "signature": None
        }

        # Validate the call
        semantics["validation_errors"] = self.validator.validate_call(builtin_call)

        # Analyze the call
        semantics["analysis"] = self.analyzer.analyze_call(builtin_call)

        # Get signature information
        builtin = get_builtin(builtin_call.builtin_name)
        if builtin:
            semantics["signature"] = {
                "name": builtin.signature.name,
                "parameters": builtin.signature.parameters,
                "return_type": builtin.signature.return_type,
                "side_effects": builtin.signature.side_effects,
                "description": builtin.signature.description,
                "category": builtin.signature.category
            }

        return semantics

    def is_pure(self, builtin_name: str) -> bool:
        """Check if a built-in function is pure (no side effects)"""
        builtin = get_builtin(builtin_name)
        if not builtin:
            return False
        return builtin.signature.side_effects == [SideEffect.NONE]

    def has_side_effect(self, builtin_name: str, effect: SideEffect) -> bool:
        """Check if a built-in function has a specific side effect"""
        builtin = get_builtin(builtin_name)
        if not builtin:
            return False
        return effect in builtin.signature.side_effects

    def get_return_type(self, builtin_name: str) -> Optional[str]:
        """Get the return type of a built-in function"""
        builtin = get_builtin(builtin_name)
        if not builtin:
            return None
        return builtin.signature.return_type


class BuiltinExpander:
    """Expands built-in calls to target-specific implementations"""

    def expand_for_target(self, builtin_call: MLILBuiltinCall, target: str) -> Optional[List[IRExpression]]:
        """Expand a built-in call for a specific target"""
        builtin = get_builtin(builtin_call.builtin_name)
        if not builtin:
            return None

        mapping = builtin.get_mapping(target)
        if not mapping:
            # Try generic mapping
            mapping = builtin.get_mapping("generic")
            if not mapping:
                return None

        if mapping.direct_opcode:
            # Direct mapping to single opcode - return as-is for now
            # This would be handled by instruction selection
            return [builtin_call]

        elif mapping.expansion:
            # Expand to sequence of operations
            # This is a simplified expansion - real implementation would
            # need to properly parse and generate IR from expansion templates
            return self._expand_template(mapping.expansion, builtin_call.arguments)

        elif mapping.library_call:
            # Convert to library function call
            from ..ir.mlil import MLILCall, MLILConstant
            target_expr = MLILConstant(mapping.library_call, 4, "string")
            return [MLILCall(target_expr, builtin_call.arguments)]

        elif mapping.fallback_error:
            # Not supported - would generate error during compilation
            return None

        return None

    def _expand_template(self, template: List[str], args: List[IRExpression]) -> List[IRExpression]:
        """Expand a template with given arguments"""
        # This is a placeholder implementation
        # Real implementation would parse template strings and generate proper IR
        expanded = []

        for instruction in template:
            if instruction.startswith("load_arg_"):
                # Load argument instruction
                arg_index = int(instruction.split("_")[-1])
                if arg_index < len(args):
                    expanded.append(args[arg_index])
            # Add more template instruction handling as needed

        return expanded


# Global semantic analyzer instance
builtin_semantics = BuiltinSemantics()