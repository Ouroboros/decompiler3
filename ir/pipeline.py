'''
IR Pipeline - Pass Manager

Provides generic pass execution framework with chainable composition.
'''

from typing import Any, List


class Pass:
    '''Pass base class

    All IR transformation/optimization passes should inherit from this class
    and implement the run method.
    '''

    def run(self, input_data: Any) -> Any:
        '''Execute the pass

        Args:
            input_data: Input data (can be any type of IR)

        Returns:
            Transformed data
        '''
        raise NotImplementedError(f'{self.__class__.__name__}.run() not implemented')


class Pipeline:
    '''Pass Pipeline

    Manages and executes a sequence of passes with chainable calls.
    '''

    def __init__(self, passes: List[Pass] = None):
        '''Initialize pipeline

        Args:
            passes: Initial pass list (optional)
        '''
        self.passes = passes or []

    def add_pass(self, pass_obj: Pass) -> 'Pipeline':
        '''Add a pass to the pipeline

        Args:
            pass_obj: Pass object

        Returns:
            self, for method chaining
        '''
        self.passes.append(pass_obj)
        return self

    def run(self, input_data: Any) -> Any:
        '''Run all passes in the pipeline

        Args:
            input_data: Input data

        Returns:
            Data after processing through all passes
        '''
        result = input_data
        for pass_obj in self.passes:
            result = pass_obj.run(result)
        return result
