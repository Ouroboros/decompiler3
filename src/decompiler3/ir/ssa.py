"""
SSA (Static Single Assignment) transformation and support

Implements SSA construction and destruction for all IR levels (LLIL/MLIL/HLIL).
Following standard algorithms like those used in BinaryNinja.
"""

from typing import List, Dict, Set, Optional, Union, Tuple, Any
from collections import defaultdict, deque
from .base import (
    IRExpression, IRVariable, IRBasicBlock, IRFunction, IRVisitor, IRTransformer,
    OperationType
)
from .llil import LLILExpression
from .mlil import MLILExpression, MLILVariable, MLILAssignment
from .hlil import HLILExpression, HLILVariable, HLILAssignment


class PhiNode(IRExpression):
    """Phi node for SSA form"""

    def __init__(self, variable: IRVariable, incoming_values: List[Tuple[IRBasicBlock, IRExpression]]):
        super().__init__(OperationType.PHI, variable.size)
        self.variable = variable
        self.incoming_values = incoming_values  # (block, value) pairs
        self.operands = [value for _, value in incoming_values]

    def __str__(self) -> str:
        values_str = ", ".join(f"{block.id[:8]}:{value}" for block, value in self.incoming_values)
        return f"{self.variable} = φ({values_str})"

    def accept(self, visitor) -> Any:
        if hasattr(visitor, 'visit_phi'):
            return visitor.visit_phi(self)
        return visitor.visit_expression(self)

    def add_incoming(self, block: IRBasicBlock, value: IRExpression):
        """Add an incoming value from a block"""
        self.incoming_values.append((block, value))
        self.operands.append(value)

    def remove_incoming(self, block: IRBasicBlock):
        """Remove incoming value from a block"""
        self.incoming_values = [(b, v) for b, v in self.incoming_values if b != block]
        self.operands = [v for _, v in self.incoming_values]


class DominanceAnalysis:
    """Dominance analysis for SSA construction"""

    def __init__(self, function: IRFunction):
        self.function = function
        self.dominators: Dict[IRBasicBlock, Set[IRBasicBlock]] = {}
        self.immediate_dominators: Dict[IRBasicBlock, Optional[IRBasicBlock]] = {}
        self.dominance_frontiers: Dict[IRBasicBlock, Set[IRBasicBlock]] = {}
        self._compute_dominance()

    def _compute_dominance(self):
        """Compute dominance information using iterative algorithm"""
        blocks = self.function.basic_blocks
        if not blocks:
            return

        # Initialize
        entry_block = blocks[0]
        self.dominators[entry_block] = {entry_block}

        for block in blocks[1:]:
            self.dominators[block] = set(blocks)

        # Iteratively compute dominators
        changed = True
        while changed:
            changed = False
            for block in blocks[1:]:
                # Intersect dominators of all predecessors
                new_dominators = set(blocks)
                for pred in block.predecessors:
                    if pred in self.dominators:
                        new_dominators &= self.dominators[pred]
                new_dominators.add(block)

                if new_dominators != self.dominators[block]:
                    self.dominators[block] = new_dominators
                    changed = True

        # Compute immediate dominators
        self._compute_immediate_dominators()

        # Compute dominance frontiers
        self._compute_dominance_frontiers()

    def _compute_immediate_dominators(self):
        """Compute immediate dominators"""
        for block in self.function.basic_blocks:
            dominators = self.dominators[block] - {block}
            if not dominators:
                self.immediate_dominators[block] = None
                continue

            # Find the immediate dominator (closest dominator)
            idom = None
            for candidate in dominators:
                if all(candidate in self.dominators[other] or candidate == other
                       for other in dominators):
                    idom = candidate
                    break
            self.immediate_dominators[block] = idom

    def _compute_dominance_frontiers(self):
        """Compute dominance frontiers"""
        for block in self.function.basic_blocks:
            self.dominance_frontiers[block] = set()

        for block in self.function.basic_blocks:
            if len(block.predecessors) >= 2:  # Join node
                for pred in block.predecessors:
                    runner = pred
                    while runner and runner != self.immediate_dominators.get(block):
                        self.dominance_frontiers[runner].add(block)
                        runner = self.immediate_dominators.get(runner)

    def dominates(self, a: IRBasicBlock, b: IRBasicBlock) -> bool:
        """Check if block a dominates block b"""
        return a in self.dominators.get(b, set())


class SSATransformer(IRTransformer):
    """Transforms IR to SSA form"""

    def __init__(self):
        self.variable_versions: Dict[str, int] = defaultdict(int)
        self.variable_stacks: Dict[str, List[int]] = defaultdict(list)
        self.dominance_analysis: Optional[DominanceAnalysis] = None

    def transform_function(self, function: IRFunction) -> IRFunction:
        """Transform function to SSA form"""
        if function.ssa_form:
            return function

        # Compute dominance information
        self.dominance_analysis = DominanceAnalysis(function)

        # Collect all variables that need SSA treatment
        variables = self._collect_variables(function)

        # Place phi nodes
        self._place_phi_nodes(function, variables)

        # Rename variables
        self._rename_variables(function)

        function.ssa_form = True
        return function

    def transform_expression(self, expr: IRExpression) -> IRExpression:
        """Transform individual expression"""
        # This is handled by the function-level transformation
        return expr

    def _collect_variables(self, function: IRFunction) -> Set[str]:
        """Collect all variables that are assigned in the function"""
        variables = set()

        for block in function.basic_blocks:
            for instruction in block.instructions:
                variables.update(self._get_assigned_variables(instruction))

        return variables

    def _get_assigned_variables(self, expr: IRExpression) -> Set[str]:
        """Get variables that are assigned in an expression"""
        variables = set()

        if isinstance(expr, MLILAssignment):
            variables.add(expr.dest.variable.name)
        elif isinstance(expr, HLILAssignment):
            variables.add(expr.dest.variable.name)

        # Recursively check operands
        for operand in expr.operands:
            variables.update(self._get_assigned_variables(operand))

        return variables

    def _place_phi_nodes(self, function: IRFunction, variables: Set[str]):
        """Place phi nodes at appropriate locations"""
        if not self.dominance_analysis:
            return

        # For each variable, place phi nodes at dominance frontiers
        for var_name in variables:
            # Find all blocks that assign to this variable
            defining_blocks = set()
            for block in function.basic_blocks:
                for instruction in block.instructions:
                    if self._assigns_variable(instruction, var_name):
                        defining_blocks.add(block)

            # Place phi nodes at dominance frontiers
            phi_blocks = set()
            worklist = list(defining_blocks)

            while worklist:
                block = worklist.pop(0)
                for frontier_block in self.dominance_analysis.dominance_frontiers[block]:
                    if frontier_block not in phi_blocks:
                        # Create phi node
                        variable = function.variables.get(var_name)
                        if variable:
                            phi = PhiNode(variable, [])
                            frontier_block.instructions.insert(0, phi)
                            phi_blocks.add(frontier_block)
                            worklist.append(frontier_block)

    def _assigns_variable(self, expr: IRExpression, var_name: str) -> bool:
        """Check if expression assigns to a variable"""
        if isinstance(expr, (MLILAssignment, HLILAssignment)):
            return expr.dest.variable.name == var_name

        for operand in expr.operands:
            if self._assigns_variable(operand, var_name):
                return True

        return False

    def _rename_variables(self, function: IRFunction):
        """Rename variables to SSA form using recursive algorithm"""
        if not function.basic_blocks:
            return

        # Initialize variable stacks
        for var_name in function.variables:
            self.variable_stacks[var_name] = []

        # Start renaming from entry block
        self._rename_block(function.basic_blocks[0])

    def _rename_block(self, block: IRBasicBlock):
        """Rename variables in a block and its dominated blocks"""
        # Remember how many variables we push for this block
        pushed_counts: Dict[str, int] = defaultdict(int)

        # Process phi nodes first
        for instruction in block.instructions:
            if isinstance(instruction, PhiNode):
                new_version = self._get_new_version(instruction.variable.name)
                instruction.variable.ssa_version = new_version
                self.variable_stacks[instruction.variable.name].append(new_version)
                pushed_counts[instruction.variable.name] += 1

        # Process other instructions
        for instruction in block.instructions:
            if not isinstance(instruction, PhiNode):
                # Rename uses
                self._rename_uses_in_expression(instruction)

                # Rename definitions
                assigned_vars = self._rename_defs_in_expression(instruction)
                for var_name in assigned_vars:
                    pushed_counts[var_name] += 1

        # Update phi nodes in successor blocks
        for successor in block.successors:
            for instruction in successor.instructions:
                if isinstance(instruction, PhiNode):
                    var_name = instruction.variable.name
                    if var_name in self.variable_stacks and self.variable_stacks[var_name]:
                        current_version = self.variable_stacks[var_name][-1]
                        # Create SSA variable reference
                        ssa_var = IRVariable(var_name, instruction.variable.size)
                        ssa_var.ssa_version = current_version

                        # Create appropriate variable expression based on IR level
                        if any(isinstance(inst, MLILExpression) for inst in block.instructions):
                            var_expr = MLILVariable(ssa_var)
                        elif any(isinstance(inst, HLILExpression) for inst in block.instructions):
                            var_expr = HLILVariable(ssa_var)
                        else:
                            var_expr = MLILVariable(ssa_var)  # Default to MLIL

                        instruction.add_incoming(block, var_expr)

        # Recursively process dominated blocks
        if self.dominance_analysis:
            for dominated_block in self.function.basic_blocks:
                if (self.dominance_analysis.immediate_dominators.get(dominated_block) == block and
                    dominated_block != block):
                    self._rename_block(dominated_block)

        # Pop variables we pushed
        for var_name, count in pushed_counts.items():
            for _ in range(count):
                if self.variable_stacks[var_name]:
                    self.variable_stacks[var_name].pop()

    def _rename_uses_in_expression(self, expr: IRExpression):
        """Rename variable uses in an expression"""
        if isinstance(expr, (MLILVariable, HLILVariable)):
            var_name = expr.variable.name
            if var_name in self.variable_stacks and self.variable_stacks[var_name]:
                expr.variable.ssa_version = self.variable_stacks[var_name][-1]

        for operand in expr.operands:
            self._rename_uses_in_expression(operand)

    def _rename_defs_in_expression(self, expr: IRExpression) -> List[str]:
        """Rename variable definitions in an expression"""
        assigned_vars = []

        if isinstance(expr, (MLILAssignment, HLILAssignment)):
            var_name = expr.dest.variable.name
            new_version = self._get_new_version(var_name)
            expr.dest.variable.ssa_version = new_version
            self.variable_stacks[var_name].append(new_version)
            assigned_vars.append(var_name)

        for operand in expr.operands:
            assigned_vars.extend(self._rename_defs_in_expression(operand))

        return assigned_vars

    def _get_new_version(self, var_name: str) -> int:
        """Get a new SSA version number for a variable"""
        self.variable_versions[var_name] += 1
        return self.variable_versions[var_name]


class SSADestroyer(IRTransformer):
    """Transforms SSA form back to normal form"""

    def transform_function(self, function: IRFunction) -> IRFunction:
        """Transform function from SSA form"""
        if not function.ssa_form:
            return function

        # Remove phi nodes and insert copies at appropriate locations
        self._remove_phi_nodes(function)

        # Remove SSA version numbers from variables
        self._remove_ssa_versions(function)

        function.ssa_form = False
        return function

    def transform_expression(self, expr: IRExpression) -> IRExpression:
        """Transform individual expression"""
        return expr

    def _remove_phi_nodes(self, function: IRFunction):
        """Remove phi nodes and insert appropriate copies"""
        for block in function.basic_blocks:
            phi_nodes = [inst for inst in block.instructions if isinstance(inst, PhiNode)]

            for phi in phi_nodes:
                # Remove phi node
                block.instructions.remove(phi)

                # Insert copies at the end of predecessor blocks
                for pred_block, value in phi.incoming_values:
                    # Create assignment: phi_var = value
                    if isinstance(value, (MLILVariable, MLILExpression)):
                        from .mlil import MLILAssignment, MLILVariable
                        dest_var = MLILVariable(phi.variable)
                        assignment = MLILAssignment(dest_var, value)
                    elif isinstance(value, (HLILVariable, HLILExpression)):
                        from .hlil import HLILAssignment, HLILVariable
                        dest_var = HLILVariable(phi.variable)
                        assignment = HLILAssignment(dest_var, value)
                    else:
                        continue

                    # Insert at end of predecessor block (before terminator)
                    if pred_block.instructions:
                        pred_block.instructions.insert(-1, assignment)
                    else:
                        pred_block.instructions.append(assignment)

    def _remove_ssa_versions(self, function: IRFunction):
        """Remove SSA version numbers from variables"""
        for block in function.basic_blocks:
            for instruction in block.instructions:
                self._remove_ssa_from_expression(instruction)

        # Remove versions from function variables
        for var in function.variables.values():
            var.ssa_version = None

    def _remove_ssa_from_expression(self, expr: IRExpression):
        """Remove SSA versions from an expression recursively"""
        if isinstance(expr, (MLILVariable, HLILVariable)):
            expr.variable.ssa_version = None

        for operand in expr.operands:
            self._remove_ssa_from_expression(operand)


class SSAAnalysis:
    """Analysis utilities for SSA form IR"""

    @staticmethod
    def is_ssa_form(function: IRFunction) -> bool:
        """Check if function is in SSA form"""
        return function.ssa_form

    @staticmethod
    def get_def_use_chains(function: IRFunction) -> Dict[IRVariable, List[IRExpression]]:
        """Get definition-use chains for SSA variables"""
        def_use_chains = defaultdict(list)

        for block in function.basic_blocks:
            for instruction in block.instructions:
                SSAAnalysis._collect_uses(instruction, def_use_chains)

        return dict(def_use_chains)

    @staticmethod
    def _collect_uses(expr: IRExpression, def_use_chains: Dict[IRVariable, List[IRExpression]]):
        """Collect variable uses in an expression"""
        if isinstance(expr, (MLILVariable, HLILVariable)):
            def_use_chains[expr.variable].append(expr)

        for operand in expr.operands:
            SSAAnalysis._collect_uses(operand, def_use_chains)

    @staticmethod
    def get_reaching_definitions(function: IRFunction) -> Dict[IRBasicBlock, Dict[str, IRExpression]]:
        """Get reaching definitions for each basic block"""
        reaching_defs: Dict[IRBasicBlock, Dict[str, IRExpression]] = {}

        # Initialize
        for block in function.basic_blocks:
            reaching_defs[block] = {}

        # Iterative analysis
        changed = True
        while changed:
            changed = False
            for block in function.basic_blocks:
                # Merge definitions from predecessors
                new_defs = {}
                for pred in block.predecessors:
                    for var_name, definition in reaching_defs[pred].items():
                        new_defs[var_name] = definition

                # Apply definitions in this block
                for instruction in block.instructions:
                    if isinstance(instruction, (MLILAssignment, HLILAssignment)):
                        new_defs[instruction.dest.variable.name] = instruction

                if new_defs != reaching_defs[block]:
                    reaching_defs[block] = new_defs
                    changed = True

        return reaching_defs


# Convenience functions
def to_ssa(function: IRFunction) -> IRFunction:
    """Convert function to SSA form"""
    transformer = SSATransformer()
    return transformer.transform_function(function)

def from_ssa(function: IRFunction) -> IRFunction:
    """Convert function from SSA form"""
    destroyer = SSADestroyer()
    return destroyer.transform_function(function)