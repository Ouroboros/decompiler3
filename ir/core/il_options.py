'''IL Options - Generic options for all IL levels'''

from common import *


class ILOptions(StrictBase):
    '''Attributes:'''

    hidden_for_formatter: bool

    def __init__(self):
        self.hidden_for_formatter = False
