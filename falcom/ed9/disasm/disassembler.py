"""
Generic recursive descent disassembler
"""

from common import *
from ml import fileio
from typing import Callable

from .instruction_table import *
from .instruction import *
from .basic_block import *


@dataclass
class DisassemblerContext:
    """Context for disassembler with callbacks"""
    instruction_table      : 'InstructionTable' = None
    get_func_argc          : Callable[[int], int | None] = None  # func_id -> argc
    create_fallthrough_jump: Callable[[int, int, 'InstructionTable'], 'Instruction'] = None  # offset, target, inst_table -> synthetic_jmp
    on_disasm_function      : Callable[['DisassemblerContext', int, str], None] = None  # context, offset, name
    on_block_start          : Callable[['DisassemblerContext', int], None] = None  # context, offset
    on_instruction_decoded  : Callable[['DisassemblerContext', 'Instruction', 'BasicBlock'], list[BranchTarget]] = None  # context, current_inst, block -> branch_targets
    on_pre_add_branch       : Callable[['DisassemblerContext', BranchTarget], None] = None  # context, target


class Disassembler:
    """
    Generic recursive descent disassembler.

    Works with any instruction set via pluggable InstructionTable.
    """

    def __init__(self, instruction_table: InstructionTable, context: DisassemblerContext):
        self.instruction_table = instruction_table
        self.context = context
        self.context.instruction_table = instruction_table

        # Tracking dictionaries
        self.disassembled_blocks: dict[int, BasicBlock] = {}   # offset -> completed block
        self.disassembled_offset: dict[int, Instruction] = {}  # offset -> instruction
        self.allocated_blocks: dict[int, BasicBlock] = {}      # offset -> allocated block
        self.offset_to_block: dict[int, BasicBlock] = {}       # instruction offset -> block

        # Current state
        self.current_block: BasicBlock | None = None

    def disasm_function(
        self,
        stream: bytes | fileio.FileStream,
        offset: int = 0,
        name: str = ''
    ) -> BasicBlock:
        """
        Disassemble a function starting at offset.

        Args:
            bytecode: Function bytecode or stream
            offset: Starting offset in bytecode
            name: Optional function name

        Returns:
            Entry basic block
        """
        # Create FileStream from bytecode

        if isinstance(stream, fileio.FileStream):
            fs = stream

        elif isinstance(stream, bytes):
            fs = fileio.FileStream(encoding = default_encoding())
            fs.OpenMemory(stream)

        else:
            raise ValueError(f'Invalid bytecode type: {type(stream)}')

        fs.Position = offset

        # Call on_disasm_function callback
        if self.context.on_disasm_function:
            self.context.on_disasm_function(self.context, offset, name)

        entry_block = self.disasm_block(fs)

        if name:
            entry_block.name = name

        return entry_block

    def disasm_block(self, fs) -> BasicBlock:
        """
        Recursively disassemble a basic block.

        Args:
            fs: File stream positioned at block start

        Returns:
            Disassembled basic block
        """
        offset = fs.Position

        # Check if already disassembled
        if offset in self.disassembled_blocks:
            return self.disassembled_blocks[offset]

        # Create new block
        block = self.create_block(offset)

        # Mark as disassembled (prevents infinite recursion)
        self.disassembled_blocks[offset] = block

        # Call on_block_start callback to restore saved state
        if self.context.on_block_start:
            self.context.on_block_start(self.context, offset)

        # Save previous block context
        previous_block = self.current_block
        self.current_block = block

        # Disassemble instructions in this block
        while True:
            pos = fs.Position

            # Check if we've reached another block's instructions
            if pos in self.disassembled_offset:
                # Reached an instruction that's part of another block
                break

            if pos == 0xEEC92:
                pass

            # Decode instruction
            inst = self.instruction_table.decode_instruction(fs, pos)

            # Record instruction
            self.disassembled_offset[pos] = inst
            self.offset_to_block[pos] = block
            block.instructions.append(inst)

            # Get descriptor
            desc = inst.descriptor

            # Call on_instruction_decoded callback for every instruction
            pending_targets = []
            if self.context.on_instruction_decoded:
                pending_targets = self.context.on_instruction_decoded(self.context, inst, block)

            # If block terminates or declares future targets, collect them
            if desc.is_end_block():
                pending_targets.extend(desc.get_branch_targets(inst, fs.Position))

            if desc.is_start_block():
                pending_targets.extend(desc.get_branch_targets(inst, fs.Position))

            # Add any pending targets
            for target in pending_targets:
                if self.context.on_pre_add_branch:
                    self.context.on_pre_add_branch(self.context, target)

                if target.offset == 0xEEC8C:
                    pass

                target_block = self.ensure_block_at(target.offset)
                block.add_branch(target_block, target.kind)

            if desc.is_end_block():
                break

            # Check if next position is an allocated block
            if fs.Position in self.allocated_blocks:
                # Next instruction belongs to a pre-allocated block
                self._add_fallthrough_jump(block, pos, fs.Position)
                break

        # Recursively disassemble all successors
        # Make a copy of succs list since split_block may modify it during iteration
        succs_to_process = list(block.succs)
        for succ in succs_to_process:
            saved_pos = fs.Position
            fs.Position = succ.offset
            # Recursively disassemble (modifies succ in place, no need to reassign)
            self.disasm_block(fs)
            fs.Position = saved_pos

        # Restore previous block context
        self.current_block = previous_block

        return block

    def _add_fallthrough_jump(self, block: BasicBlock, src_offset: int, target_offset: int):
        """
        Emit a synthetic fallthrough jump from block to the block at target_offset.

        Args:
            block: Current basic block
            src_offset: Offset of the last decoded instruction (jump origin)
            target_offset: Offset of the next block (jump destination)
        """
        next_block = self.ensure_block_at(target_offset)
        synthetic_jmp = self.context.create_fallthrough_jump(
            src_offset,
            target_offset,
            self.instruction_table
        )
        block.instructions.append(synthetic_jmp)
        block.add_branch(next_block, BranchKind.UNCONDITIONAL)

    def create_block(self, offset: int) -> BasicBlock:
        """
        Create or get a basic block at offset.

        Args:
            offset: Block starting offset

        Returns:
            BasicBlock instance
        """
        if offset in self.allocated_blocks:
            return self.allocated_blocks[offset]

        block = BasicBlock(start_offset=offset)
        self.allocated_blocks[offset] = block

        return block

    def ensure_block_at(self, target_offset: int) -> BasicBlock:
        """
        Ensure target_offset is the start of a BasicBlock.

        If target_offset is in the middle of an existing block, split that block.

        Args:
            target_offset: Target offset

        Returns:
            BasicBlock starting at target_offset
        """
        # Already a block start
        if target_offset in self.allocated_blocks:
            return self.allocated_blocks[target_offset]

        # Check if target is in middle of existing block using offset_to_block
        # This works even for blocks still being disassembled
        if target_offset in self.offset_to_block:
            owner = self.offset_to_block[target_offset]
            # Target is in middle of owner block, need to split
            if owner.start_offset != target_offset:
                return self.split_block(owner, target_offset)
            # Target is already at start of owner (shouldn't happen, but handle it)
            return owner

        # No existing block, create new empty block
        block = BasicBlock(start_offset = target_offset)
        self.allocated_blocks[target_offset] = block
        return block

    def split_block(self, owner: BasicBlock, split_offset: int) -> BasicBlock:
        """
        Split a block at split_offset.

        Args:
            owner: Block to split
            split_offset: Offset to split at

        Returns:
            Tail block (starts at split_offset)
        """
        # Find split index
        split_idx = None
        for i, inst in enumerate(owner.instructions):
            if inst.offset == split_offset:
                split_idx = i
                break

        if split_idx is None:
            inst_offsets = [f'0x{inst.offset:X}' for inst in owner.instructions]
            raise ValueError(
                f'No instruction at split_offset 0x{split_offset:X} in block {owner.name}\n'
                f'Block range: 0x{owner.start_offset:X}-0x{owner.computed_end_offset:X}\n'
                f'Instructions at: {", ".join(inst_offsets)}'
            )

        # Create tail block
        tail = BasicBlock(
            start_offset = split_offset,
            end_offset   = owner.end_offset,
        )
        tail.instructions = owner.instructions[split_idx:]
        tail.succs = owner.succs.copy()
        tail.true_succs = owner.true_succs.copy()
        tail.false_succs = owner.false_succs.copy()

        # Update owner (head)
        owner.end_offset = split_offset
        owner.instructions = owner.instructions[:split_idx]

        # Add synthetic fallthrough jump instruction to owner to make it terminate properly
        # This is needed for LLIL conversion to ensure every block ends with a terminal instruction

        synthetic_jmp = self.context.create_fallthrough_jump(
            split_offset,
            tail.start_offset,
            self.instruction_table
        )
        owner.instructions.append(synthetic_jmp)

        owner.succs = [tail]
        owner.true_succs = []
        owner.false_succs = []

        # Register tail in all tracking dictionaries
        self.allocated_blocks[tail.start_offset] = tail
        # Mark tail as disassembled since it already has all its instructions
        self.disassembled_blocks[tail.start_offset] = tail
        # Update offset_to_block mapping for all instructions in tail
        for inst in tail.instructions:
            self.offset_to_block[inst.offset] = tail

        return tail

    def add_branch(self, offset: int, kind: BranchKind | None = None) -> BasicBlock:
        """
        Add a branch from current block to target offset.

        Args:
            offset: Target offset
            kind: Branch kind (for conditional branches)

        Returns:
            Target basic block
        """
        target = self.create_block(offset)

        if self.current_block:
            self.current_block.add_branch(target, kind)

        return target

    def get_instruction(self, offset: int) -> Instruction | None:
        """Get instruction at offset, if disassembled"""
        return self.disassembled_offset.get(offset)

    def get_block(self, offset: int) -> BasicBlock | None:
        """Get basic block at offset, if disassembled"""
        return self.disassembled_blocks.get(offset)
