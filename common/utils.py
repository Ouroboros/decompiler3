from .config import *

FLOAT_ROUND_REL_TOL = 1e-6
FLOAT_ROUND_ABS_TOL = 1e-9

def format_float(value: float) -> str:
    precision = default_float_precision_decimals()
    round_value = round(value, precision)

    rel_tol = FLOAT_ROUND_REL_TOL
    abs_tol = FLOAT_ROUND_ABS_TOL

    if abs(round_value - value) <= max(abs(value) * rel_tol, abs_tol):
        # if value != round_value:
        #     print(f'float: {precision} {value} -> {round_value}')
        value = round_value

    return f'{value}'
