'''How the TypeScript backend writes ED9 constructs (ir/core/target.py BackendHints)'''

from typing import Any, List

from ir.core import BackendHints

from .calls import ED9ScriptCall, ED9Syscall


SYSCALL_INTRINSIC           = 'syscall'
EXTERN_INTRINSIC            = 'extern_call'
EXTERN_NORETURN_INTRINSIC   = 'extern_call_noreturn'

DECLARATIONS = f'''// Placeholder function: external script call
function {EXTERN_INTRINSIC}(target: string, ...args: any[]): any {{ return undefined; }}

// Placeholder function: external script call that ends the thread (never returns)
function {EXTERN_NORETURN_INTRINSIC}(target: string, ...args: any[]): never {{ throw undefined; }}

// Placeholder function: system call
function {SYSCALL_INTRINSIC}(subsystem: number, cmd: number, ...args: any[]): any {{ return undefined; }}

'''


class ED9Hints(BackendHints):

    @classmethod
    def typescript_external_call(cls, target, args: List[Any], generator) -> str:
        if isinstance(target, ED9Syscall):
            # Syscalls in common functions stay raw: those functions are the named wrappers themselves
            func = generator.current_function()
            db = generator.signature_db()
            sig = db.get_syscall(target.subsystem, target.cmd) if db and not (func and func.is_common_func) else None
            if sig:
                return f'{sig.name}({generator.format_args_with_signature(args, sig.params)})'

            operands = [str(target.subsystem), str(target.cmd), *[generator.format_expr(arg) for arg in args]]
            return f'{SYSCALL_INTRINSIC}({", ".join(operands)})'

        if isinstance(target, ED9ScriptCall):
            operands = [f"'{target.qualified_name}'", *[generator.format_expr(arg) for arg in args]]
            intrinsic = EXTERN_INTRINSIC if target.returns else EXTERN_NORETURN_INTRINSIC
            return f'{intrinsic}({", ".join(operands)})'

        return super().typescript_external_call(target, args, generator)

    @classmethod
    def typescript_declarations(cls) -> str:
        return DECLARATIONS


def generate_syscall_wrappers(signature_db) -> str:
    '''Generate TypeScript wrapper functions for syscalls'''
    if not signature_db or not signature_db.syscalls:
        return ''

    lines = ['// Syscall wrappers']

    for name, sig in signature_db.syscalls.items():
        # Build parameter list
        params = []
        args = []
        for p in sig.params:
            if p.variadic:
                params.append(f'...{p.name}: ({p.type})[]')
                args.append(f'...{p.name}')

            else:
                params.append(f'{p.name}: {p.type}')
                args.append(p.name)

        params_str = ', '.join(params)
        args_str = ', '.join(args)

        # Determine return type
        if sig.return_hint:
            if sig.return_hint.type == 'void':
                return_type = 'void'
                body = f'syscall({sig.subsystem}, {sig.cmd}, {args_str});' if args_str else f'syscall({sig.subsystem}, {sig.cmd});'

            else:
                return_type = 'number'
                body = f'return syscall({sig.subsystem}, {sig.cmd}, {args_str});' if args_str else f'return syscall({sig.subsystem}, {sig.cmd});'

        else:
            return_type = 'any'
            body = f'return syscall({sig.subsystem}, {sig.cmd}, {args_str});' if args_str else f'return syscall({sig.subsystem}, {sig.cmd});'

        lines.append(f'function {name}({params_str}): {return_type} {{ {body} }}')

    lines.append('')
    return '\n'.join(lines) + '\n'
