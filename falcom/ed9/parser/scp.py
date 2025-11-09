from .types_scp import *
from .types_parser import *
from pprint import pprint

class ScpParser(StrictBase):
    fs      : fileio.FileStream
    name    : str
    header  : ScpHeader

    def __init__(self, fs: fileio.FileStream, name: str = ''):
        self.fs = fs

    def parse(self):
        self.read_header()

    def read_header(self):
        fs = self.fs

        hdr = ScpHeader(fs = fs)
        self.header = hdr

        func_entries    = self.read_function_entries(fs)
        functions       = self.read_functions(fs, func_entries)

    def read_function_entries(self, fs: fileio.FileStream):
        return [ScpFunctionEntry(fs = fs) for _ in range(self.header.function_count)]

    def read_functions(self, fs: fileio.FileStream, func_entries: list[ScpFunctionEntry]) -> list[Function]:
        functions: list[Function] = []

        # load function

        for i, entry in enumerate(func_entries):
            func = Function()

            func.is_common_func = entry.is_common_func == 1

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

            print(func)
            print()

            # debug_info_list = self.read_debug_info(fs, f) if f.debug_info_count != 0 else []
            # for dbg_info in debug_info_list:
            #     info = FunctionCallDebugInfo()

        # for idx, f in enumerate(func_entries):
        #     if f.debug_info_count == 0:
        #         continue

        #     print(f'[{idx}]')
        #     print(f)
        #     for d in f.debug_info:
        #         s = [f'    {l}' for l in str(d).splitlines()]
        #         print(f'{'\n'.join(s)}\n')

        #     print()

    def read_debug_info(self, fs: fileio.FileStream, func_entry: ScpFunctionEntry) -> list[ScpFunctionCallDebugInfo]:
        debug_info_list = []

        if func_entry.debug_info_count == 0:
            return debug_info_list

        fs.Position = func_entry.debug_info_offset
        for _ in range(func_entry.debug_info_count):
            dbg_info = ScpFunctionCallDebugInfo(fs = fs)
            debug_info_list.append(dbg_info)

        return debug_info_list
