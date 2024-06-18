import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pydantic import BaseModel
from typing import Callable, Union

# https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/docs/examples/plot_tipping_problem_newapi.py
# Define the universe of discourse for each variable
xAvg = ctrl.Antecedent(np.arange(-1000, 1001, 1), "xAvg")
yAvg = ctrl.Antecedent(np.arange(-10, 2001, 1), "yAvg")
speedMean = ctrl.Antecedent(np.arange(0, 13, 1), "speedMean")
speedStd = ctrl.Antecedent(np.arange(0, 9, 1), "speedStd")
output = ctrl.Consequent(np.arange(-1, 2, 1), "output")
output.defuzzify_method = "centroid"

# Define membership functions for xAvg
xAvg["XL"] = fuzz.gaussmf(xAvg.universe, -1000.0, 200.0)
xAvg["XO"] = fuzz.gaussmf(xAvg.universe, -220.0, 180.0)
xAvg["XR"] = fuzz.gaussmf(xAvg.universe, 500.0, 230.0)

# Define membership functions for yAvg
yAvg["YO"] = fuzz.gaussmf(yAvg.universe, 0.0, 750.0)
yAvg["YN"] = fuzz.gaussmf(yAvg.universe, 2000.0, 250.0)

# Define membership functions for speedMean
speedMean["SMO"] = fuzz.zmf(speedMean.universe, 6, 12)
speedMean["SMH"] = fuzz.gaussmf(speedMean.universe, 15.0, 2.0)

# Define membership functions for speedStd
speedStd["SSO"] = fuzz.zmf(speedStd.universe, 4, 8)
speedStd["SSH"] = fuzz.gaussmf(speedStd.universe, 8.0, 1.2)

# Define membership functions for output
output["OF"] = fuzz.gaussmf(output.universe, -1.0, 0.6)
output["OT"] = fuzz.gaussmf(output.universe, 1.0, 0.6)

# Define fuzzy rules
rule1 = ctrl.Rule(
    xAvg["XO"] & yAvg["YO"] & speedMean["SMO"] & speedStd["SSO"], output["OT"]
)
rule2 = ctrl.Rule(xAvg["XL"], output["OF"])
rule3 = ctrl.Rule(xAvg["XR"], output["OF"])
rule4 = ctrl.Rule(yAvg["YN"], output["OF"])
rule5 = ctrl.Rule(speedMean["SMH"], output["OF"])
rule6 = ctrl.Rule(speedStd["SSH"], output["OF"])
rule7 = ctrl.Rule(xAvg["XO"] & yAvg["YN"], output["OF"])
rule8 = ctrl.Rule(xAvg["XL"] & speedMean["SMO"], output["OF"])
rule9 = ctrl.Rule(xAvg["XR"] & speedMean["SMO"], output["OF"])
rule10 = ctrl.Rule(yAvg["YO"] & speedMean["SMH"], output["OF"])
rule11 = ctrl.Rule(yAvg["YO"] & speedStd["SSH"], output["OF"])

# Create control system and simulation
_fis_ctrl = ctrl.ControlSystem(
    [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11]
)
_fis = ctrl.ControlSystemSimulation(_fis_ctrl)


class FisInput(BaseModel, frozen=True):
    xAvg: float
    yAvg: float
    speedMean: float
    speedStd: float


Num = Union[int, float]


def centroid(
    domain: range, mf: Callable[[Num], float], segmentation: int = 100
) -> float:
    x = [
        domain.start + i * (domain.stop - domain.start) / (segmentation - 1)
        for i in range(segmentation)
    ]
    dx = (domain.stop - domain.start) / (segmentation - 1)
    y = [mf(xi) for xi in x]

    def trapz(dx: Num, y: list[Num]) -> Num:
        return (2 * sum(y) - y[0] - y[-1]) * dx / 2

    integral_f = trapz(dx, y)
    integral_xf = trapz(dx, [xi * yi for xi, yi in zip(x, y)])
    centroid_x = integral_xf / integral_f
    return centroid_x


def mapRange(x: Num, inMin: Num, inMax: Num, outMin: Num, outMax: Num) -> float:
    return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin


def infer_raw(input: FisInput) -> float:
    _fis.input["xAvg"] = input.xAvg
    _fis.input["yAvg"] = input.yAvg
    _fis.input["speedMean"] = input.speedMean
    _fis.input["speedStd"] = input.speedStd
    _fis.compute()
    return _fis.output["output"]


def gauss_fn(x: float, mean: float, sigma: float):
    return np.exp(-((x - mean) ** 2.0) / (2 * sigma**2.0))


MAX_VAL = centroid(
    range(-1, 1),
    lambda x: gauss_fn(x, 1, 0.6),
)
MIN_VAL = centroid(
    range(-1, 1),
    lambda x: gauss_fn(x, -1, 0.6),
)


def infer(input: FisInput) -> float:
    raw = infer_raw(input)
    return mapRange(raw, MIN_VAL, MAX_VAL, -1, 1)
