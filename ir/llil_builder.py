'''
LLIL v2 Builder - Layered architecture for convenience
'''

from typing import Union, Optional, List
from .llil import *
from .llil import LowLevelILBinaryOp, LowLevelILOperation


class LowLevelILBuilder:
    '''Mid-level builder with convenience methods'''

    def __init__(self, function: LowLevelILFunction):
        self.function = function
        self.current_block: Optional[LowLevelILBasicBlock] = None
        self.label_map: dict[str, LowLevelILBasicBlock] = {}  # label name -> block
        self.current_sp: int = 0  # Track current stack pointer state (for block sp_in/sp_out)
        self.vstack: List[LowLevelILInstruction] = []  # Virtual stack for expression tracking

    def set_current_block(self, block: LowLevelILBasicBlock, sp: Optional[int] = None):
        '''Set the current basic block for instruction insertion

        Args:
            block: The block to set as current
            sp: Optional stack pointer value. If None, continues from previous block's sp
        '''
        # Save previous block's sp_out if we have a current block
        if self.current_block is not None:
            self.current_block.sp_out = self.current_sp

        self.current_block = block

        # Set new block's sp_in and current sp
        if sp is not None:
            self.current_sp = sp
        block.sp_in = self.current_sp

    def mark_label(self, name: str, block: LowLevelILBasicBlock):
        '''Associate a label name with a block'''
        self.label_map[name] = block

    def get_block_by_label(self, label: str) -> Optional[LowLevelILBasicBlock]:
        '''Get block by label name'''
        return self.label_map.get(label)

    def add_instruction(self, instr: LowLevelILInstruction):
        '''Add instruction to current block and update stack pointer tracking'''
        if self.current_block is None:
            raise RuntimeError('No current basic block set')
        self.current_block.add_instruction(instr)

        # Update stack pointer based on instruction type
        if isinstance(instr, LowLevelILSpAdd):
            self.current_sp += instr.delta

    # === Virtual Stack Management ===

    def _to_expr(self, value: Union[LowLevelILInstruction, int, float, str]) -> LowLevelILInstruction:
        '''Convert value to expression'''
        if isinstance(value, LowLevelILInstruction):
            return value
        elif isinstance(value, int):
            return self.const_int(value)
        elif isinstance(value, float):
            return self.const_float(value)
        elif isinstance(value, str):
            return self.const_str(value)
        else:
            raise TypeError(f'Cannot convert {type(value)} to expression')

    def push(self, value: Union[LowLevelILInstruction, int, float, str], size: int = 4) -> LowLevelILInstruction:
        '''Push value onto stack using StackPush instruction

        Generates LowLevelILStackPush and maintains sp.
        Returns the expression that was pushed.
        '''
        expr = self._to_expr(value)
        # Generate StackPush instruction (semantic: STACK[sp] = value; sp++)
        push_instr = LowLevelILStackPush(expr, size)
        self.add_instruction(push_instr)
        # Maintain sp in builder (sp maintained here, not in add_instruction)
        self.current_sp += 1
        # Track on vstack for expression tracking
        self.vstack.append(expr)
        return expr

    def pop(self, size: int = 4) -> LowLevelILInstruction:
        '''Pop value from stack using StackPop expression

        Returns LowLevelILStackPop expression (value expression with side effect).
        Maintains sp in builder (sp maintained here, not in add_instruction).
        '''
        # Pop from virtual stack for tracking
        if self.vstack:
            self.vstack.pop()

        # Create StackPop expression (semantic: sp--; return STACK[sp])
        pop_expr = LowLevelILStackPop(size)
        # Maintain sp in builder
        self.current_sp -= 1

        return pop_expr

    # === Legacy Stack Operations (kept for compatibility) ===

    def stack_push(self, value: Union[LowLevelILInstruction, int, str], size: int = 4):
        '''STACK[sp++] = value (legacy, use push() instead)'''
        self.push(value, size)

    def stack_pop(self, size: int = 4) -> LowLevelILStackLoad:
        '''STACK[--sp] (legacy, use pop() instead)'''
        return self.pop(size)

    def stack_load(self, offset: int, size: int = 4) -> LowLevelILStackLoad:
        '''STACK[sp + offset] (no sp change)'''
        return LowLevelILStackLoad(offset, size)

    def stack_store(self, value: Union[LowLevelILInstruction, int, str], offset: int, size: int = 4):
        '''STACK[sp + offset] = value (no sp change)'''
        self.add_instruction(LowLevelILStackStore(value, offset, size))

    # === Register Operations ===

    def reg_store(self, reg_index: int, value: Union[LowLevelILInstruction, int], size: int = 4):
        '''R[index] = value'''
        self.add_instruction(LowLevelILRegStore(reg_index, value, size))

    def reg_load(self, reg_index: int, size: int = 4) -> LowLevelILRegLoad:
        '''R[index]'''
        return LowLevelILRegLoad(reg_index, size)

    # === Constants ===

    def const_int(self, value: int, size: int = 4, is_hex: bool = False) -> LowLevelILConst:
        '''Integer constant

        Args:
            value: Integer value
            size: Size in bytes (default 4)
            is_hex: If True, display as hex; if False, use auto detection (default)
        '''
        return LowLevelILConst(value, size, is_hex)

    def const_float(self, value: float, size: int = 4) -> LowLevelILConst:
        '''Float constant (size: 4 for float, 8 for double)'''
        return LowLevelILConst(value, size, False)

    def const_str(self, value: str) -> LowLevelILConst:
        '''String constant'''
        return LowLevelILConst(value, 0, False)

    # === Binary Operations ===

    def _binary_op(self, op_class, lhs = None, rhs = None, *, push: bool = True, size: int = 4) -> LowLevelILInstruction:
        '''Generic binary operation handler

        Args:
            op_class: The operation class (e.g., LowLevelILAdd)
            lhs: Left operand (None = pop from vstack)
            rhs: Right operand (None = pop from vstack)
            push: Whether to push result back to vstack
            size: Operation size

        Returns:
            The operation expression
        '''
        # Get operands - both must be None or both must be provided
        if lhs is None and rhs is None:
            # Implicit mode: pop both from vstack
            rhs = self.pop(size)
            lhs = self.pop(size)
        elif lhs is not None and rhs is not None:
            # Explicit mode: both provided
            lhs = self._to_expr(lhs)
            rhs = self._to_expr(rhs)
        else:
            raise ValueError('Binary operation requires both operands or neither (lhs and rhs must both be None or both be provided)')

        # Create operation with operands
        op = op_class(lhs, rhs, size)
        self.add_instruction(op)

        # Optionally push result (binary operations implicitly push their result)
        if push:
            self.vstack.append(op)

        return op

    def add(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''ADD operation'''
        return self._binary_op(LowLevelILAdd, lhs, rhs, push = push, size = size)

    def sub(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''SUB operation'''
        from ir.llil import LowLevelILSub
        return self._binary_op(LowLevelILSub, lhs, rhs, push = push, size = size)

    def mul(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''MUL operation'''
        return self._binary_op(LowLevelILMul, lhs, rhs, push = push, size = size)

    def div(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''DIV operation'''
        from ir.llil import LowLevelILDiv
        return self._binary_op(LowLevelILDiv, lhs, rhs, push = push, size = size)

    # === Comparison Operations ===

    def eq(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''EQ operation (==)'''
        from ir.llil import LowLevelILEq
        return self._binary_op(LowLevelILEq, lhs, rhs, push = push, size = size)

    def ne(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''NE operation (!=)'''
        from ir.llil import LowLevelILNe
        return self._binary_op(LowLevelILNe, lhs, rhs, push = push, size = size)

    def lt(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''LT operation (<)'''
        from ir.llil import LowLevelILLt
        return self._binary_op(LowLevelILLt, lhs, rhs, push = push, size = size)

    def le(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''LE operation (<=)'''
        from ir.llil import LowLevelILLe
        return self._binary_op(LowLevelILLe, lhs, rhs, push = push, size = size)

    def gt(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''GT operation (>)'''
        from ir.llil import LowLevelILGt
        return self._binary_op(LowLevelILGt, lhs, rhs, push = push, size = size)

    def ge(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''GE operation (>=)'''
        from ir.llil import LowLevelILGe
        return self._binary_op(LowLevelILGe, lhs, rhs, push = push, size = size)

    # === Control Flow ===

    def jmp(self, target: Union[str, LowLevelILBasicBlock]):
        '''Unconditional jump - target can be label or block'''
        if isinstance(target, str):
            target_block = self.get_block_by_label(target)
            if target_block is None:
                raise ValueError(f'Undefined label: {target}')
            target = target_block
        self.add_instruction(LowLevelILJmp(target))

    def branch_if(self, condition: LowLevelILInstruction, target: Union[str, LowLevelILBasicBlock]):
        '''Branch if condition is true - target can be label or block'''
        if isinstance(target, str):
            target_block = self.get_block_by_label(target)
            if target_block is None:
                raise ValueError(f'Undefined label: {target}')
            target = target_block
        self.add_instruction(LowLevelILBranch(condition, target))

    def call(self, target: Union[str, LowLevelILInstruction], stack_cleanup: Optional[int] = None):
        '''Function call

        Args:
            target: Function name or address
            stack_cleanup: Number of stack slots to pop after call (for callee cleanup)
                          If None, no automatic cleanup is performed
        '''
        self.add_instruction(LowLevelILCall(target))
        # In Falcom VM, callee cleans up the stack (including func_id, ret_addr, and arguments)
        if stack_cleanup is not None:
            self.current_sp -= stack_cleanup

    def ret(self):
        '''Return'''
        self.add_instruction(LowLevelILRet())

    # === Special ===

    def label(self, name: str):
        '''Label - marks current block with this label name'''
        if self.current_block is None:
            raise RuntimeError('No current block to label')
        self.mark_label(name, self.current_block)
        self.add_instruction(LowLevelILLabelInstr(name))

    def debug_line(self, line_no: int):
        '''Debug line number'''
        self.add_instruction(LowLevelILDebug('line', line_no))

    def syscall(self, catalog: int, cmd: int, arg_count: int):
        '''System call'''
        self.add_instruction(LowLevelILSyscall(catalog, cmd, arg_count))


class LLILFormatter:
    '''Formatting layer for beautiful output'''

    @staticmethod
    def indent_lines(lines: List[str], indent: str) -> List[str]:
        '''Add indentation to multiple lines

        Args:
            lines: List of lines to indent
            indent: Indentation string to prepend

        Returns:
            List of indented lines
        '''
        return [indent + line for line in lines]

    @staticmethod
    def format_instruction(instr: LowLevelILInstruction) -> str:
        '''Format a single instruction - can be customized per instruction type

        Returns a single line for simple instructions.
        For multi-line instructions, use format_instruction_expanded().
        '''
        # For now, use the instruction's __str__ method
        # This can be extended with custom formatting logic for specific instruction types
        return str(instr)

    @staticmethod
    def format_instruction_expanded(instr: LowLevelILInstruction) -> List[str]:
        '''Format instruction with expanded stack operations (multi-line)

        Returns a list of lines showing explicit stack behavior.
        For binary operations like EQ, MUL, ADD, this shows:
        - Pop operations to get operands
        - The actual operation
        - Push operation for result
        '''
        # Binary operations: pop 2, compute, push 1
        if isinstance(instr, LowLevelILBinaryOp):
            # Get operation name from enum (e.g., 'LLIL_ADD' -> 'ADD')
            op_name = instr.operation.name.replace('LLIL_', '')

            # Map operation types to their expression strings
            expr_map = {
                LowLevelILOperation.LLIL_ADD: 'lhs + rhs',
                LowLevelILOperation.LLIL_SUB: 'lhs - rhs',
                LowLevelILOperation.LLIL_MUL: 'lhs * rhs',
                LowLevelILOperation.LLIL_DIV: 'lhs / rhs',
                LowLevelILOperation.LLIL_EQ: '(lhs == rhs) ? 1 : 0',
                LowLevelILOperation.LLIL_NE: '(lhs != rhs) ? 1 : 0',
                LowLevelILOperation.LLIL_LT: '(lhs < rhs) ? 1 : 0',
                LowLevelILOperation.LLIL_LE: '(lhs <= rhs) ? 1 : 0',
                LowLevelILOperation.LLIL_GT: '(lhs > rhs) ? 1 : 0',
                LowLevelILOperation.LLIL_GE: '(lhs >= rhs) ? 1 : 0',
            }

            expr = expr_map[instr.operation]

            return [
                f'; {op_name}()',
                'rhs = STACK[--sp]',
                'lhs = STACK[--sp]',
                f'STACK[sp++] = {expr}'
            ]

        # For non-binary operations, return single line
        return [str(instr)]

    @staticmethod
    def format_instruction_sequence(instructions: List[LowLevelILInstruction], indent: str = '  ') -> list[str]:
        '''Format sequence of instructions - returns list of lines

        Args:
            instructions: List of instructions to format
            indent: Indentation string to prepend to each line (default: '  ')

        Returns:
            List of formatted lines with indentation
        '''
        result = []

        for instr in instructions:
            # Use expanded format for multi-line instructions
            expanded = LLILFormatter.format_instruction_expanded(instr)
            if len(expanded) > 1:
                # Multi-line instruction
                result.extend(LLILFormatter.indent_lines(expanded, indent))
            else:
                # Single line instruction
                result.append(f'{indent}{expanded[0]}')

        return result

    @staticmethod
    def format_llil_function(func: LowLevelILFunction) -> list[str]:
        assert isinstance(func, LowLevelILFunction)

        '''Format entire LLIL function with beautiful output - returns list of lines'''
        result = [
            f'; ---------- {func.name} ----------',
        ]

        for block in func.basic_blocks:
            # Block header: {block_N}  label_name: [sp = 0]

            block_info = [
                f'block_{block.index}(0x{block.start:04X})',
            ]

            if block.instructions and isinstance(block.instructions[0], LowLevelILLabelInstr):
                label_name = block.instructions[0].name
                block_info.append(label_name)
                instructions_to_format = block.instructions[1:]
            else:
                instructions_to_format = block.instructions

            block_info.append(f'[sp = {block.sp_in}]')

            result.append(', '.join(block_info))

            # Format instructions - now returns list
            indent = '  '
            result.extend(LLILFormatter.format_instruction_sequence(instructions_to_format, indent))
            result.append('')

        return result
