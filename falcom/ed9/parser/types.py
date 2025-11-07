from ml import *
from common import *

class RawInt(int):
    def __repr__(self) -> str:
        return f'RawInt({super()})'

class ScpValue:
    class Type(IntEnum2):
        Raw             = 0
        Integer         = 1
        Float           = 2
        String          = 3

    ClassMap = {
        RawInt  : Type.Raw,
        int     : Type.Integer,
        float   : Type.Float,
        str     : Type.String,
    }

    def __init__(self, value: int | float | str | RawInt = None, *, fs: fileio.FileStream = None):
        self.value  = value

        if value is not None:
            self.type = self.ClassMap[type(value)]
        else:
            self.type = None

        self.from_stream(fs)

    def from_stream(self, fs: fileio.FileStream):
        if not fs:
            return

        value = fs.ReadULong()
        typ = value >> 30

        match typ:
            case ScpValue.Type.Raw:
                value = RawInt(value)

            case ScpValue.Type.Integer:
                value = (value << 2) & 0xFFFFFFFF
                sign = 0xC0000000 if value & 0x80000000 != 0 else 0
                value = int.from_bytes((sign | (value >> 2)).to_bytes(4, 'little'), 'little', signed = True)

            case ScpValue.Type.Float:
                value = struct.unpack('f', ((value << 2) & 0xFFFFFFFF).to_bytes(4, 'little'))[0]

            case ScpValue.Type.String:
                with fs.PositionSaver:
                    fs.Position = value & 0x3FFFFFFF
                    value = fs.ReadMultiByte()

        self.value = value
        self.type = ScpValue.Type(typ)

    def to_bytes(self) -> bytes:
        match self.type:
            case ScpValue.Type.Raw:
                v = self.value.to_bytes(4, defaultEndian())

            case ScpValue.Type.Integer:
                assert self.value <= 0x3FFFFFFFF if self.value >= 0 else self.value >= -(0x1FFFFFFF + 1)

                v = (self.value & 0x3FFFFFFF) | (ScpValue.Type.Integer << 30)
                v = int(v).to_bytes(4, defaultEndian(), signed = False)

            case ScpValue.Type.Float:
                v = int.from_bytes(struct.pack('f', self.value), defaultEndian())
                v = (v >> 2) | (ScpValue.Type.Float << 30)
                v = v.to_bytes(4, defaultEndian())

            case _:
                raise NotImplementedError(f'unsupported type: {self.value}')

        return v

    def __str__(self) -> str:
        return f'ScenaValue<{self.value}>'

    __repr__ = __str__


class ScpHeader(StrictBase):
    SIZE    = 0x18
    MAGIC   = b'#scp'

    function_entry_offset   : int
    function_count          : int
    global_var_offset       : int
    global_var_count        : int
    dword_14                : int

    def __init__(self, *, fs: fileio.FileStream = None):
        self.from_stream(fs)

    def from_stream(self, fs: fileio.FileStream):
        if not fs:
            return

        assert fs.Read(4) == self.MAGIC

        self.function_entry_offset  = fs.ReadULong()    # 0x04
        self.function_count         = fs.ReadULong()    # 0x08
        self.global_var_offset      = fs.ReadULong()    # 0x0C
        self.global_var_count       = fs.ReadULong()    # 0x10
        self.dword_14               = fs.ReadULong()    # 0x14


class ScenaFunctionEntry(StrictBase):
    SIZE = 0x20

    offset                      : int
    param_count                 : int
    byte05                      : int
    byte06                      : int
    default_params_count        : int
    default_params_offset       : int
    param_flags_offset          : int
    debug_symbol_count          : int
    debug_symbol_offset         : int
    name_crc32                  : int
    name_offset                 : int
    name                        : int

    def __init__(self, *, fs: fileio.FileStream = None):
        self.from_stream(fs)

    def from_stream(self, fs: fileio.FileStream):
        if not fs:
            return

        self.offset                 = fs.ReadULong()
        self.param_count            = fs.ReadByte()
        self.byte05                 = fs.ReadByte()
        self.byte06                 = fs.ReadByte()
        self.default_params_count   = fs.ReadByte()
        self.default_params_offset  = fs.ReadULong()
        self.param_flags_offset     = fs.ReadULong()
        self.debug_symbol_count     = fs.ReadULong()
        self.debug_symbol_offset    = fs.ReadULong()
        self.name_crc32             = fs.ReadULong()
        self.name_offset            = ScpValue(fs = fs).value
