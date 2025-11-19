'''
MLIL Formatter - Format MLIL for display

Provides clean, readable output of MLIL functions.
'''

from typing import List

from .mlil import *
from .mlil_ssa import MLILVarSSA, MLILSetVarSSA, MLILPhi


class MLILFormatter:
    '''Format MLIL functions for display'''

    @classmethod
    def format_function(cls, func: MediumLevelILFunction) -> List[str]:
        '''Format entire MLIL function

        Returns:
            List of formatted lines
        '''
        result = [
            f'; ===== MLIL Function {func.name} @ 0x{func.start_addr:X} =====',
            f'; Variables: {len(func.variables)}',
        ]

        # List variables
        if func.variables:
            result.append(';')
            for var_name in sorted(func.variables.keys()):
                var = func.variables[var_name]
                if var.slot_index >= 0:
                    result.append(f';   {var.name} (slot {var.slot_index})')
                else:
                    result.append(f';   {var.name}')

        # List inferred types (if available)
        if func.var_types:
            result.append(';')
            result.append('; Inferred Types:')
            for var_name in sorted(func.var_types.keys()):
                typ = func.var_types[var_name]
                result.append(f';   {var_name}: {typ}')

        result.append('')

        # Format each block
        for block in func.basic_blocks:
            result.extend(cls.format_block(block))
            result.append('')

        return result

    @classmethod
    def format_block(cls, block: MediumLevelILBasicBlock) -> List[str]:
        '''Format a single basic block

        Returns:
            List of formatted lines
        '''
        result = [f'{block.label}:']

        for inst in block.instructions:
            # Skip hidden instructions
            if inst.options.hidden_for_formatter:
                continue

            formatted = cls.format_instruction(inst)
            result.append(f'  {formatted}')

        return result

    @classmethod
    def format_instruction(cls, inst: MediumLevelILInstruction) -> str:
        '''Format a single instruction

        Returns:
            Formatted instruction string
        '''
        # Constants and Variables
        if isinstance(inst, MLILConst):
            return str(inst)

        elif isinstance(inst, MLILVar):
            return str(inst)

        elif isinstance(inst, MLILSetVar):
            return f'{inst.var} = {inst.value}'

        # SSA variable operations
        elif isinstance(inst, MLILVarSSA):
            return str(inst)

        elif isinstance(inst, MLILSetVarSSA):
            return f'{inst.var} = {inst.value}'

        elif isinstance(inst, MLILPhi):
            return str(inst)

        # Binary operations - Arithmetic
        elif isinstance(inst, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod)):
            return str(inst)

        # Binary operations - Bitwise
        elif isinstance(inst, (MLILAnd, MLILOr, MLILXor)):
            return str(inst)

        # Binary operations - Logical
        elif isinstance(inst, (MLILLogicalAnd, MLILLogicalOr)):
            return str(inst)

        # Binary operations - Comparison
        elif isinstance(inst, (MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            return str(inst)

        # Unary operations
        elif isinstance(inst, (MLILNeg, MLILLogicalNot)):
            return str(inst)

        elif isinstance(inst, MLILTestZero):
            return str(inst)

        elif isinstance(inst, MLILAddressOf):
            return str(inst)

        # Control flow
        elif isinstance(inst, MLILGoto):
            return f'goto {inst.target.label}'

        elif isinstance(inst, MLILIf):
            return f'if ({inst.condition}) goto {inst.true_target.label} else {inst.false_target.label}'

        elif isinstance(inst, MLILRet):
            if inst.value is not None:
                return f'return {inst.value}'
            else:
                return 'return'

        # Function calls
        elif isinstance(inst, MLILCall):
            args_str = ', '.join(str(arg) for arg in inst.args)
            return f'{inst.target}({args_str})'

        elif isinstance(inst, MLILSyscall):
            args = [
                f'{inst.subsystem}',
                f'{inst.cmd}',
                *[str(arg) for arg in inst.args],
            ]
            return f'syscall({', '.join(args)})'

        elif isinstance(inst, MLILCallScript):
            args_str = ', '.join(str(arg) for arg in inst.args)
            return f'{inst.module}.{inst.func}({args_str})  ; MLILCallScript'

        # Globals
        elif isinstance(inst, MLILLoadGlobal):
            return str(inst)

        elif isinstance(inst, MLILStoreGlobal):
            return f'GLOBAL[{inst.index}] = {inst.value}'

        # Registers
        elif isinstance(inst, MLILLoadReg):
            return str(inst)

        elif isinstance(inst, MLILStoreReg):
            return f'REG[{inst.index}] = {inst.value}'

        # Debug
        elif isinstance(inst, MLILNop):
            return str(inst)

        elif isinstance(inst, MLILDebug):
            return f'; debug.{inst.debug_type}({inst.value})'

        else:
            raise NotImplementedError(f'Unhandled MLIL instruction type: {type(inst).__name__}')

    @classmethod
    def to_dot(cls, func: MediumLevelILFunction) -> str:
        '''Generate Graphviz DOT format for CFG visualization

        Args:
            func: MLIL function to visualize

        Returns:
            DOT format string

        Example:
            from ir.mlil import MLILFormatter
            dot = MLILFormatter.to_dot(mlil_func)
            with open('cfg.dot', 'w') as f:
                f.write(dot)
            # Then: dot -Tpng cfg.dot -o cfg.png
        '''
        lines = []
        lines.append(f'digraph "{func.name}" {{')
        lines.append('    rankdir=TB;')
        lines.append('    node [shape=box, fontname="Courier New", fontsize=10];')
        lines.append('    edge [fontname="Courier New", fontsize=9];')
        lines.append('')

        # Add nodes (basic blocks)
        for block in func.basic_blocks:
            label_parts = []

            # Block header
            header = f'{block.label} @ 0x{block.start:X}\\l'
            label_parts.append(header)
            label_parts.append('-' * 40 + '\\l')

            # Format instructions
            for inst in block.instructions:
                if inst.options.hidden_for_formatter:
                    continue

                formatted = cls.format_instruction(inst)
                escaped = formatted.replace('\\', '\\\\').replace('"', '\\"')
                label_parts.append(escaped + '\\l')

            label = ''.join(label_parts)

            # Node styling
            if block.index == 0:
                # Entry block
                lines.append(f'    {block.block_name} [label="{label}", style=filled, fillcolor=lightgreen];')
            elif block.has_terminal and isinstance(block.instructions[-1], MLILRet):
                # Exit block
                lines.append(f'    {block.block_name} [label="{label}", style=filled, fillcolor=lightblue];')
            else:
                lines.append(f'    {block.block_name} [label="{label}"];')

        lines.append('')

        # Add edges
        for block in func.basic_blocks:
            if not block.outgoing_edges:
                continue

            last_inst = block.instructions[-1] if block.instructions else None

            for target in block.outgoing_edges:
                # Determine edge label and style
                edge_label = ''
                edge_style = ''

                if isinstance(last_inst, MLILIf):
                    # Conditional branch
                    if target == last_inst.true_target:
                        edge_label = 'true'
                        edge_style = ', color=green'
                    elif target == last_inst.false_target:
                        edge_label = 'false'
                        edge_style = ', color=red'
                elif isinstance(last_inst, MLILGoto):
                    edge_label = 'goto'
                    edge_style = ', color=blue'
                else:
                    # Fall-through
                    edge_label = 'fall-through'
                    edge_style = ', style=dashed'

                if edge_label:
                    lines.append(f'    {block.block_name} -> {target.block_name} [label="{edge_label}"{edge_style}];')
                else:
                    lines.append(f'    {block.block_name} -> {target.block_name}{edge_style};')

        lines.append('}')
        return '\n'.join(lines)


def format_mlil_function(func: MediumLevelILFunction) -> List[str]:
    '''Convenience function to format MLIL function'''
    return MLILFormatter.format_function(func)
