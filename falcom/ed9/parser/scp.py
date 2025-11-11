from .types_scp import *
from .types_parser import *
from pprint import pprint

class ScpParser(StrictBase):
    fs              : fileio.FileStream
    name            : str
    header          : ScpHeader
    functions       : list[Function]
    global_vars     : list[GlobalVar]
    function_map    : dict[str, Function]

    def __init__(self, fs: fileio.FileStream, name: str = ''):
        self.fs = fs

    def get_func_by_name(self, name: str) -> Function:
        return self.function_map[name]

    def get_func_name(self, func_id: int) -> str | None:
        if func_id >= len(self.functions):
            raise ValueError(f'func_id out of range: {func_id} >= {len(self.functions)}')

        return self.functions[func_id].name

    def get_func_argc(self, func_id: int) -> int:
        if func_id >= len(self.functions):
            raise ValueError(f'func_id out of range: {func_id} >= {len(self.functions)}')

        return len(self.functions[func_id].params)

    def parse(self):
        self._read_header()

    def _read_header(self):
        fs = self.fs
        self.header = ScpHeader(fs = fs)

        func_entries        = self._read_function_entries(fs)
        self.functions      = self._read_functions(fs, func_entries)
        self.global_vars    = self._read_global_vars(fs)

        self.function_map = {func.name: func for func in self.functions}

    def _read_function_entries(self, fs: fileio.FileStream):
        return [ScpFunctionEntry(fs = fs) for _ in range(self.header.function_count)]

    def _read_functions(self, fs: fileio.FileStream, func_entries: list[ScpFunctionEntry]) -> list[Function]:
        functions: list[Function] = []

        # load function

        for i, entry in enumerate(func_entries):
            func = Function()

            func.is_common_func = entry.is_common_func == 1
            func.offset = entry.offset

            func_name = ScpValue().from_value(entry.name_offset, fs = fs)
            func.name = func_name.value

            fs.Position = entry.param_flags_offset
            param_flags = [ScpParamFlags(fs = fs) for _ in range(entry.param_count)]

            fs.Position = entry.default_params_offset
            default_params = [ScpValue(fs = fs) for _ in range(entry.default_params_count)]
            if len(default_params) > len(param_flags):
                raise ValueError(f'default_params count is greater than param_flags count: {len(default_params)} > {len(param_flags)}')

            while len(default_params) < len(param_flags):
                default_params.insert(0, None)

            for i in range(entry.param_count):
                param = FunctionParam(type = param_flags[i], default_value = default_params[i])
                func.params.append(param)
            functions.append(func)

        # init debug info

        for i, func in enumerate(functions):
            entry = func_entries[i]
            fs.Position = entry.debug_info_offset
            debug_info_list = self.read_debug_info(fs, entry)

            for dbg_info in debug_info_list:
                if dbg_info.arg_count == 0:
                    continue

                info = FunctionCallDebugInfo()
                info.call_type = dbg_info.call_type

                if dbg_info.func_id != 0xFFFFFFFF:
                    if dbg_info.func_id >= len(functions):
                        raise ValueError(f'debug info func_id out of range: {dbg_info.func_id} >= {len(functions)}')

                    info.func_name = functions[dbg_info.func_id].name

                else:
                    info.func_name = f'{dbg_info.call_type}'

                fs.Position = dbg_info.info_offset
                for _ in range(dbg_info.arg_count):
                    arg_value = ScpValue(fs = fs)
                    arg_type = fs.ReadULong()

                    info.args.append(FunctionCallDebugInfo.ArgInfo(arg_type, arg_value))

                func.debug_info.append(info)

        return functions

    def read_debug_info(self, fs: fileio.FileStream, func_entry: ScpFunctionEntry) -> list[ScpFunctionCallDebugInfo]:
        debug_info_list = []

        if func_entry.debug_info_count == 0:
            return debug_info_list

        fs.Position = func_entry.debug_info_offset
        for _ in range(func_entry.debug_info_count):
            dbg_info = ScpFunctionCallDebugInfo(fs = fs)
            debug_info_list.append(dbg_info)

        return debug_info_list

    def _read_global_vars(self, fs: fileio.FileStream) -> list[ScpGlobalVar]:
        hdr = self.header
        fs.Position = hdr.global_var_offset

        global_vars: list[GlobalVar] = []

        for i in range(hdr.global_var_count):
            gvar_info = ScpGlobalVar(fs = fs)
            var_name = ScpValue(fs = fs).value
            var_type = gvar_info.type

            global_var = GlobalVar(
                index = i,
                name = var_name,
                type = var_type,
            )

            global_vars.append(global_var)

        return global_vars
