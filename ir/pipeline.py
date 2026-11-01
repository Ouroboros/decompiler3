'''IR Pipeline - Pass Manager'''

from typing import Any, Callable, List, Optional


# Called after every pass with the pass name and its result (the IR as that pass left it)
PassObserver = Callable[[str, Any], None]


class Pass:
    '''Pass base class'''

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def run(self, input_data: Any) -> Any:
        '''Execute the pass'''
        raise NotImplementedError(f'{self.__class__.__name__}.run() not implemented')

    def run_observed(self, input_data: Any, observer: PassObserver) -> Any:
        '''Execute the pass and report its result; passes made of sub-passes report each of them'''
        result = self.run(input_data)
        observer(self.name, result)
        return result


class Pipeline:
    '''Pass Pipeline'''

    def __init__(self, passes: List[Pass] = None):
        '''Initialize pipeline'''
        self.passes = passes or []

    def add_pass(self, pass_obj: Pass) -> 'Pipeline':
        '''Add a pass to the pipeline'''
        self.passes.append(pass_obj)
        return self

    def run(self, input_data: Any, debug: bool = False, observer: Optional[PassObserver] = None) -> Any:
        '''Run all passes in the pipeline'''
        result = input_data
        for pass_obj in self.passes:
            if debug:
                print(f'  Running {pass_obj.__class__.__name__}...')

            if observer is not None:
                result = pass_obj.run_observed(result, observer)

            else:
                result = pass_obj.run(result)

            if debug:
                print(f'  {pass_obj.__class__.__name__} done')

        return result
