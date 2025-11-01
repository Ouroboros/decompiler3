"""
Standard built-in function definitions

Provides concrete definitions for commonly used built-in functions
with target-specific mappings and validation rules.
"""

from typing import List
from ..ir.base import IRExpression
from ..ir.mlil import MLILConstant, MLILExpression
from ..ir.hlil import HLILConstant, HLILExpression
from .registry import (
    BuiltinFunction, BuiltinSignature, BuiltinMapping, SideEffect,
    register_builtin
)


def _validate_positive_number(args: List[IRExpression]) -> bool:
    """Validator for functions that require positive numbers"""
    if not args:
        return False

    first_arg = args[0]
    if isinstance(first_arg, (MLILConstant, HLILConstant)):
        if isinstance(first_arg.value, (int, float)):
            return first_arg.value >= 0

    return True  # Can't validate at compile time


def _validate_non_zero_divisor(args: List[IRExpression]) -> bool:
    """Validator for division operations to check for zero divisor"""
    if len(args) < 2:
        return False

    divisor = args[1]
    if isinstance(divisor, (MLILConstant, HLILConstant)):
        if isinstance(divisor.value, (int, float)):
            return divisor.value != 0

    return True  # Can't validate at compile time


def _validate_string_length(args: List[IRExpression]) -> bool:
    """Validator for string operations"""
    if not args:
        return False

    # Basic validation - more sophisticated type checking would be needed
    return True


def register_extended_builtins():
    """Register extended built-in functions with validation"""

    # Mathematical functions with validation
    sin_builtin = BuiltinFunction(BuiltinSignature(
        name="sin",
        parameters=["number"],
        return_type="number",
        side_effects=[SideEffect.NONE],
        description="Sine function",
        category="math"
    ))
    sin_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="sin"
    ))
    sin_builtin.add_target_mapping("x86", BuiltinMapping(
        expansion=["fld", "fsin"]
    ))
    register_builtin(sin_builtin)

    cos_builtin = BuiltinFunction(BuiltinSignature(
        name="cos",
        parameters=["number"],
        return_type="number",
        side_effects=[SideEffect.NONE],
        description="Cosine function",
        category="math"
    ))
    cos_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="cos"
    ))
    cos_builtin.add_target_mapping("x86", BuiltinMapping(
        expansion=["fld", "fcos"]
    ))
    register_builtin(cos_builtin)

    # Logarithm with validation
    log_builtin = BuiltinFunction(BuiltinSignature(
        name="log",
        parameters=["number"],
        return_type="number",
        side_effects=[SideEffect.NONE],
        description="Natural logarithm",
        category="math"
    ))
    log_builtin.add_validator(_validate_positive_number)
    log_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="log"
    ))
    log_builtin.add_target_mapping("x86", BuiltinMapping(
        expansion=["fld", "fyl2x"]
    ))
    register_builtin(log_builtin)

    # Division with zero check
    div_builtin = BuiltinFunction(BuiltinSignature(
        name="div",
        parameters=["number", "number"],
        return_type="number",
        side_effects=[SideEffect.NONE],
        description="Division operation",
        category="math"
    ))
    div_builtin.add_validator(_validate_non_zero_divisor)
    div_builtin.add_target_mapping("generic", BuiltinMapping(
        expansion=["load_arg_0", "load_arg_1", "div"]
    ))
    register_builtin(div_builtin)

    # String manipulation functions
    substr_builtin = BuiltinFunction(BuiltinSignature(
        name="substr",
        parameters=["string", "number", "number"],
        return_type="string",
        side_effects=[SideEffect.MEMORY_READ],
        description="Extract substring",
        category="string"
    ))
    substr_builtin.add_validator(_validate_string_length)
    substr_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="__substr"
    ))
    register_builtin(substr_builtin)

    strfind_builtin = BuiltinFunction(BuiltinSignature(
        name="strfind",
        parameters=["string", "string"],
        return_type="number",
        side_effects=[SideEffect.MEMORY_READ],
        description="Find substring position",
        category="string"
    ))
    strfind_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="strstr"
    ))
    register_builtin(strfind_builtin)

    # Array/collection functions
    array_length_builtin = BuiltinFunction(BuiltinSignature(
        name="array_length",
        parameters=["object"],
        return_type="number",
        side_effects=[SideEffect.MEMORY_READ],
        description="Get array length",
        category="array"
    ))
    array_length_builtin.add_target_mapping("generic", BuiltinMapping(
        expansion=["load_arg_0", "load_field_length"]
    ))
    register_builtin(array_length_builtin)

    array_push_builtin = BuiltinFunction(BuiltinSignature(
        name="array_push",
        parameters=["object", "any"],
        return_type="number",
        side_effects=[SideEffect.MEMORY_WRITE],
        description="Push element to array",
        category="array"
    ))
    array_push_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="__array_push"
    ))
    register_builtin(array_push_builtin)

    # Object manipulation
    object_keys_builtin = BuiltinFunction(BuiltinSignature(
        name="object_keys",
        parameters=["object"],
        return_type="object",
        side_effects=[SideEffect.MEMORY_READ],
        description="Get object property names",
        category="object"
    ))
    object_keys_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="__object_keys"
    ))
    register_builtin(object_keys_builtin)

    # Type conversion functions
    to_number_builtin = BuiltinFunction(BuiltinSignature(
        name="to_number",
        parameters=["any"],
        return_type="number",
        side_effects=[SideEffect.NONE],
        description="Convert value to number",
        category="conversion"
    ))
    to_number_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="__to_number"
    ))
    register_builtin(to_number_builtin)

    to_string_builtin = BuiltinFunction(BuiltinSignature(
        name="to_string",
        parameters=["any"],
        return_type="string",
        side_effects=[SideEffect.MEMORY_WRITE],  # May allocate string
        description="Convert value to string",
        category="conversion"
    ))
    to_string_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="__to_string"
    ))
    register_builtin(to_string_builtin)

    to_boolean_builtin = BuiltinFunction(BuiltinSignature(
        name="to_boolean",
        parameters=["any"],
        return_type="boolean",
        side_effects=[SideEffect.NONE],
        description="Convert value to boolean",
        category="conversion"
    ))
    to_boolean_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="__to_boolean"
    ))
    register_builtin(to_boolean_builtin)

    # Error handling
    throw_builtin = BuiltinFunction(BuiltinSignature(
        name="throw",
        parameters=["string"],
        return_type=None,
        side_effects=[SideEffect.CONTROL, SideEffect.IO],
        description="Throw an exception",
        category="error"
    ))
    throw_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="__throw_exception"
    ))
    register_builtin(throw_builtin)

    # I/O functions
    print_builtin = BuiltinFunction(BuiltinSignature(
        name="print",
        parameters=["...any"],
        return_type=None,
        side_effects=[SideEffect.IO],
        description="Print values to output",
        category="io"
    ))
    print_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="printf"
    ))
    register_builtin(print_builtin)

    read_file_builtin = BuiltinFunction(BuiltinSignature(
        name="read_file",
        parameters=["string"],
        return_type="string",
        side_effects=[SideEffect.IO, SideEffect.MEMORY_WRITE],
        description="Read file contents",
        category="io"
    ))
    read_file_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="__read_file"
    ))
    register_builtin(read_file_builtin)

    # Game/Script specific functions
    play_sound_builtin = BuiltinFunction(BuiltinSignature(
        name="play_sound",
        parameters=["string", "number"],
        return_type=None,
        side_effects=[SideEffect.IO, SideEffect.SYSTEM],
        description="Play sound effect",
        category="game"
    ))
    play_sound_builtin.add_target_mapping("falcom_vm", BuiltinMapping(
        direct_opcode="PLAY_SE"
    ))
    play_sound_builtin.add_target_mapping("generic", BuiltinMapping(
        library_call="__play_sound"
    ))
    register_builtin(play_sound_builtin)

    show_message_builtin = BuiltinFunction(BuiltinSignature(
        name="show_message",
        parameters=["string"],
        return_type=None,
        side_effects=[SideEffect.IO, SideEffect.CONTROL],
        description="Display message dialog",
        category="game"
    ))
    show_message_builtin.add_target_mapping("falcom_vm", BuiltinMapping(
        direct_opcode="SHOW_MSG"
    ))
    register_builtin(show_message_builtin)

    get_variable_builtin = BuiltinFunction(BuiltinSignature(
        name="get_variable",
        parameters=["number"],
        return_type="any",
        side_effects=[SideEffect.MEMORY_READ],
        description="Get game variable by ID",
        category="game"
    ))
    get_variable_builtin.add_target_mapping("falcom_vm", BuiltinMapping(
        direct_opcode="GET_VAR"
    ))
    register_builtin(get_variable_builtin)

    set_variable_builtin = BuiltinFunction(BuiltinSignature(
        name="set_variable",
        parameters=["number", "any"],
        return_type=None,
        side_effects=[SideEffect.MEMORY_WRITE],
        description="Set game variable by ID",
        category="game"
    ))
    set_variable_builtin.add_target_mapping("falcom_vm", BuiltinMapping(
        direct_opcode="SET_VAR"
    ))
    register_builtin(set_variable_builtin)


# Register all extended built-ins when module is imported
register_extended_builtins()