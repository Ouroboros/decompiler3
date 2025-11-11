"""
Formatter for ED9 VM disassembly output
"""

from common import *
from typing import TYPE_CHECKING, Callable
from dataclasses import dataclass

if TYPE_CHECKING:
    from .basic_block import BasicBlock

__all__ = (
    'Formatter',
    'FormatterContext',
)


@dataclass
class FormatterContext:
    """Context for formatter with callbacks"""
    get_func_name: Callable[[int], str | None] = None  # func_id -> func_name


class Formatter:
    """Format disassembled code for output"""

    def __init__(self, context: FormatterContext, indent: str = '    '):
        self.context = context
        self.indent = indent
        self.formatted_offsets: set[int] = set()
        self.formatted_labels: set[str] = set()

    def format_function(self, entry_block: 'BasicBlock') -> list[str]:
        """
        Format a function starting from entry block.

        Args:
            entry_block: Entry basic block

        Returns:
            List of formatted lines
        """
        # Reset formatted tracking
        self.formatted_offsets.clear()
        self.formatted_labels.clear()

        # Collect all blocks
        blocks = self._collect_blocks(entry_block)

        # Sort by offset
        blocks.sort(key = lambda b: b.offset)

        # Format all blocks
        lines = []
        func_name = entry_block.name or f'func_{entry_block.offset:X}'
        lines.append(f'def {func_name}():')
        lines.append('')

        for i, block in enumerate(blocks):
            block_lines = self._format_block(block, gen_label = i != 0)
            lines.extend(block_lines)
            if lines and lines[-1] != '':
                lines.append('')

        # Remove trailing empty line
        if lines and lines[-1] == '':
            lines.pop()

        return lines

    def format_block(self, block: 'BasicBlock') -> list[str]:
        """
        Format a single basic block.

        Args:
            block: Basic block to format

        Returns:
            List of formatted lines
        """
        return self._format_block(block, gen_label = True)

    def _collect_blocks(self, entry: 'BasicBlock') -> list['BasicBlock']:
        """Collect all blocks reachable from entry"""
        blocks = []
        visited = set()
        todo = [entry]

        while todo:
            block = todo.pop()
            if block.offset in visited:
                continue

            visited.add(block.offset)
            blocks.append(block)
            todo.extend(block.branches)

        return blocks

    def _format_block(self, block: 'BasicBlock', gen_label: bool = True) -> list[str]:
        """Format block instructions"""
        lines = []

        if not block.instructions:
            return lines

        # Generate label if needed
        if gen_label and block.name and block.name not in self.formatted_labels:
            self.formatted_labels.add(block.name)
            lines.append(f"label('{block.name}')")
            lines.append('')

        # Format instructions
        for inst in block.instructions:
            if inst.offset in self.formatted_offsets:
                continue

            self.formatted_offsets.add(inst.offset)

            # Format instruction with context
            formatted = self._format_instruction(inst)
            lines.append(formatted)

        return lines

    def _format_instruction(self, inst) -> str:
        """Format instruction with context support"""
        return inst.descriptor.format_instruction(inst, self.context)

    def format_label(self, name: str) -> str:
        """Format a label"""
        return f"label('{name}')"
