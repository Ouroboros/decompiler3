'''Write a parsed script back to bytes with its original layout

Every table is regenerated from the parser model (header, function entries, parameter flags, defaults,
call debug records, globals) together with the strings they reference; the body after the tables (code
and the string pool) is kept as bytes until the instruction layer can encode it.
'''

import struct

from common.config import default_encoding

from .types_scp import (
    ScpFunctionCallDebugInfo, ScpFunctionCallDebugInfoArg, ScpFunctionEntry, ScpGlobalVar, ScpHeader, ScpParamFlags, ScpValue,
    VALUE_TAG_SHIFT,
)


HEADER_FORMAT       = '<4sIIIII'
ENTRY_FORMAT        = '<IBBBBIIIIII'
DEBUG_FORMAT        = '<IHHI'
WORD_FORMAT         = '<I'
STRING_TERMINATOR   = b'\0'


class ScpWriter:
    '''Serializes an ScpParser model; write() returns the script bytes'''

    def __init__(self, parser):
        self.parser = parser
        self.buffer = bytearray()

    def write(self) -> bytes:
        parser = self.parser
        layout = parser.layout
        self.buffer = bytearray(layout.size)
        self.buffer[layout.body_offset:] = layout.body

        header = parser.header
        struct.pack_into(HEADER_FORMAT, self.buffer, 0, ScpHeader.MAGIC, header.function_entry_offset, len(parser.functions),
                         header.global_var_offset, len(parser.global_vars), header.dword_14)

        for index, func in enumerate(parser.functions):
            self._write_function(header.function_entry_offset + ScpFunctionEntry.SIZE * index, func)

        for index, var in enumerate(parser.global_vars):
            position = header.global_var_offset + ScpGlobalVar.SIZE * index
            self._write_word(position, self._string_word(var.name, var.name_offset))
            self._write_word(position + ScpValue.SIZE, int(var.type))

        return bytes(self.buffer)

    def _write_function(self, position: int, func):
        layout = func.layout
        defaults = [param.default_value for param in func.params if param.default_value is not None]
        struct.pack_into(
            ENTRY_FORMAT, self.buffer, position,
            func.offset, len(func.params), int(func.is_common_func), func.byte06, len(defaults),
            layout.defaults_offset, layout.param_flags_offset, len(func.debug_info), layout.debug_info_offset,
            func.name_hash(), self._string_word(func.name, layout.name_offset),
        )

        for slot, param in enumerate(func.params):
            self._write_word(layout.param_flags_offset + ScpParamFlags.SIZE * slot, param.type.flags)

        for slot, value in enumerate(defaults):
            self._write_value(layout.defaults_offset + ScpValue.SIZE * slot, value)

        for record, info in enumerate(func.debug_info):
            struct.pack_into(DEBUG_FORMAT, self.buffer, layout.debug_info_offset + ScpFunctionCallDebugInfo.SIZE * record,
                             info.func_id, int(info.call_type), len(info.args), info.info_offset)
            for arg_index, arg in enumerate(info.args):
                arg_position = info.info_offset + ScpFunctionCallDebugInfoArg.SIZE * arg_index
                self._write_value(arg_position, arg.value)
                self._write_word(arg_position + ScpValue.SIZE, arg.type)

    def _write_word(self, position: int, word: int):
        struct.pack_into(WORD_FORMAT, self.buffer, position, word)

    def _write_value(self, position: int, value: ScpValue):
        self.buffer[position:position + ScpValue.SIZE] = value.to_bytes()
        if value.type == ScpValue.Type.String:
            self._write_string(value.value, value.offset)

    def _string_word(self, text: str, offset: int) -> int:
        self._write_string(text, offset)
        return offset | (ScpValue.Type.String << VALUE_TAG_SHIFT)

    def _write_string(self, text: str, offset: int):
        data = text.encode(default_encoding()) + STRING_TERMINATOR
        self.buffer[offset:offset + len(data)] = data


def write_scp(parser) -> bytes:
    return ScpWriter(parser).write()
