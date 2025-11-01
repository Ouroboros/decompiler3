#!/usr/bin/env python3
"""
Java VM Builder - External JVM-specific builder

This demonstrates how to create custom builders for JVM bytecode
without modifying the core IR system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional
from decompiler3.ir.llil import LowLevelILFunction, LowLevelILCall
from falcom_vm_builder import LowLevelILBuilderExtended


class JavaVMBuilder(LowLevelILBuilderExtended):
    """Java Virtual Machine specific builder"""

    def __init__(self, function: LowLevelILFunction):
        super().__init__(function)

    # JVM specific stack operations
    def dup(self) -> List:
        """JVM: Duplicate top stack value"""
        return [self.push(self.pop()), self.push(self.pop())]

    def dup2(self) -> List:
        """JVM: Duplicate top two stack values"""
        val1 = self.pop()
        val2 = self.pop()
        return [
            self.push(val2), self.push(val1),
            self.push(val2), self.push(val1)
        ]

    def swap(self) -> List:
        """JVM: Swap top two stack values"""
        val1 = self.pop()
        val2 = self.pop()
        return [self.push(val1), self.push(val2)]

    def aload(self, index: int):
        """JVM: Load reference from local variable"""
        return self.push_int(index)  # Simplified

    def iload(self, index: int):
        """JVM: Load int from local variable"""
        return self.push_int(index)  # Simplified

    def astore(self, index: int) -> None:
        """JVM: Store reference in local variable"""
        value = self.pop()
        from decompiler3.ir.common import ILRegister
        reg = ILRegister(f"local_{index}", index, 8)
        self.add_instruction(self.set_reg(reg, value))

    def istore(self, index: int) -> None:
        """JVM: Store int in local variable"""
        value = self.pop()
        from decompiler3.ir.common import ILRegister
        reg = ILRegister(f"local_{index}", index, 4)
        self.add_instruction(self.set_reg(reg, value))

    # JVM method invocations
    def invokevirtual(self, method_name: str) -> LowLevelILCall:
        """JVM: Invoke virtual method"""
        return self.call_func(f"virtual_{method_name}")

    def invokestatic(self, method_name: str) -> LowLevelILCall:
        """JVM: Invoke static method"""
        return self.call_func(f"static_{method_name}")

    def invokespecial(self, method_name: str) -> LowLevelILCall:
        """JVM: Invoke special method (constructor, private, super)"""
        return self.call_func(f"special_{method_name}")

    def invokeinterface(self, method_name: str) -> LowLevelILCall:
        """JVM: Invoke interface method"""
        return self.call_func(f"interface_{method_name}")

    # JVM object operations
    def new(self, class_name: str):
        """JVM: Create new object"""
        return self.push_str(f"new_{class_name}")

    def checkcast(self, class_name: str):
        """JVM: Check cast"""
        return self.push_str(f"cast_{class_name}")

    def instanceof(self, class_name: str):
        """JVM: Instance of check"""
        obj = self.pop()
        # In real implementation, would do proper type checking
        return self.push_int(1)  # Simplified: always true

    # JVM array operations
    def newarray(self, array_type: str):
        """JVM: Create new array"""
        size = self.pop()
        return self.push_str(f"array_{array_type}")

    def arraylength(self):
        """JVM: Get array length"""
        array_ref = self.pop()
        return self.push_int(0)  # Simplified

    def aaload(self):
        """JVM: Load reference from array"""
        index = self.pop()
        array_ref = self.pop()
        return self.push_str("array_element")  # Simplified

    def aastore(self) -> None:
        """JVM: Store reference in array"""
        value = self.pop()
        index = self.pop()
        array_ref = self.pop()
        # Simplified: just consume the values

    # High-level JVM patterns
    def java_method_call(self, class_name: str, method_name: str, *args) -> None:
        """High-level Java method call pattern"""
        # Push arguments
        for arg in args:
            if isinstance(arg, int):
                self.add_instruction(self.push_int(arg))
            elif isinstance(arg, str):
                self.add_instruction(self.push_str(arg))

        # Invoke method
        self.add_instruction(self.invokevirtual(f"{class_name}.{method_name}"))

    def system_println(self, message: str) -> None:
        """Java: System.out.println() pattern"""
        self.add_instruction(self.push_str("System.out"))
        self.add_instruction(self.push_str(message))
        self.add_instruction(self.invokevirtual("PrintStream.println"))


# Factory function for creating Java VM builders
def create_java_builder(function: LowLevelILFunction) -> JavaVMBuilder:
    """Factory to create Java VM builder"""
    return JavaVMBuilder(function)


if __name__ == "__main__":
    # Simple test
    func = LowLevelILFunction("test", 0x1000)
    builder = create_java_builder(func)

    print("✅ JavaVMBuilder created successfully!")
    print(f"   Builder type: {type(builder).__name__}")
    print(f"   Available methods: dup, swap, invokevirtual, new, newarray, system_println")