'''Base IL traits for LLIL, MLIL, HLIL'''

from abc import ABC, abstractmethod


class ILInstruction(ABC):
    '''Base class for all IL instructions'''
    pass


class ControlFlow(ILInstruction):
    '''Control flow instruction trait'''
    pass


class Terminal(ControlFlow):
    '''Terminal control flow instruction trait'''
    pass


class Constant(ILInstruction):
    '''Constant value expression trait'''
    pass


class BinaryOperation(ILInstruction):
    '''Binary operation trait'''
    pass


class UnaryOperation(ILInstruction):
    '''Unary operation trait'''
    pass
