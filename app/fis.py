from telnetlib import XASCII
from fuzzylogic.classes import Domain, Rule
from fuzzylogic.functions import gauss, bounded_sigmoid, number
from pydantic import BaseModel


def zmf(h: number, l: number):
    """
    Z-shaped membership function
    """
    return bounded_sigmoid(l, h, inverse=True)


def smf(h: number, l: number):
    """
    S-shaped membership function
    """
    return bounded_sigmoid(h, l, inverse=False)


x_avg = Domain("x_avg", -1_000, 1_000, res=0.1)
x_avg.L = gauss(-1000, 200)
x_avg.O = gauss(-220, 180)
x_avg.R = gauss(500, 230)

y_avg = Domain("y_avg", -10, 2_000, res=0.1)
y_avg.O = gauss(0, 750)
y_avg.N = gauss(2000, 250)

speed_mean = Domain("speed_mean", 0, 12, res=0.1)
speed_mean.O = zmf(6, 12)
speed_mean.H = gauss(15, 2)

speed_std = Domain("speed_std", 0, 8, res=0.1)
speed_std.O = zmf(4, 8)
speed_std.H = gauss(8, 1.2)

output = Domain("output", -1, 1, res=0.1)
output.F = gauss(-1, 0.6)
output.T = gauss(1, 0.6)

_rules = [
    Rule({(x_avg.O, y_avg.O, speed_mean.O, speed_std.O): output.T}),
    Rule({(x_avg.L): output.F}),
    Rule({(x_avg.R): output.F}),
    Rule({(y_avg.N): output.F}),
    Rule({(speed_mean.H): output.F}),
    Rule({(speed_std.H): output.F}),
    Rule({(x_avg.O, y_avg.N): output.F}),
    Rule({(x_avg.L, speed_mean.O): output.F}),
    Rule({(x_avg.R, speed_mean.O): output.F}),
    Rule({(y_avg.O, speed_mean.H): output.F}),
    Rule({(y_avg.O, speed_std.H): output.F}),
]
rules = sum(_rules)


__all__ = ["FisInput", "evaluate_fis"]


class FisInput(BaseModel, frozen=True):
    x_avg: float
    y_avg: float
    speed_mean: float
    speed_std: float


def evaluate_fis(fis_input: FisInput) -> float:
    assert isinstance(rules, Rule)
    r = rules(fis_input.model_dump())  # type: ignore
    assert isinstance(r, float)
    return r
