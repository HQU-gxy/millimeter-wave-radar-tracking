import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pydantic import BaseModel
from .utils import centroid, mapRange

# https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/docs/examples/plot_tipping_problem_newapi.py
# Define the universe of discourse for each variable
xAvg = ctrl.Antecedent(np.arange(-1000, 1000, 1), "xAvg")
yAvg = ctrl.Antecedent(np.arange(-10, 2000, 1), "yAvg")
speedMean = ctrl.Antecedent(np.arange(0, 13, 1), "speedMean")
speedStd = ctrl.Antecedent(np.arange(0, 9, 1), "speedStd")
output = ctrl.Consequent(np.arange(-1, 2, 1), "output")
output.defuzzify_method = "centroid"

# Define membership functions for xAvg
xAvg["XL"] = fuzz.gaussmf(xAvg.universe, -1000.0, 200.0)
xAvg["XO"] = fuzz.gaussmf(xAvg.universe, -220.0, 180.0)
xAvg["XR"] = fuzz.gaussmf(xAvg.universe, 500.0, 230.0)

# Define membership functions for yAvg
yAvg["YO"] = fuzz.gaussmf(yAvg.universe, 0.0, 1000.0)
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
rules = [
    ctrl.Rule(xAvg["XO"] & yAvg["YO"] & speedMean["SMO"] & speedStd["SSO"],
              output["OT"]),
    ctrl.Rule(xAvg["XL"], output["OF"]),
    ctrl.Rule(xAvg["XR"], output["OF"]),
    ctrl.Rule(yAvg["YN"], output["OF"]),
    ctrl.Rule(speedMean["SMH"], output["OF"]),
    ctrl.Rule(speedStd["SSH"], output["OF"]),
    ctrl.Rule(xAvg["XO"] & yAvg["YN"], output["OF"]),
    ctrl.Rule(xAvg["XL"] & speedMean["SMO"], output["OF"]),
    ctrl.Rule(xAvg["XR"] & speedMean["SMO"], output["OF"]),
    ctrl.Rule(yAvg["YO"] & speedMean["SMH"], output["OF"]),
    ctrl.Rule(yAvg["YO"] & speedStd["SSH"], output["OF"]),
]

_fis_ctrl = ctrl.ControlSystem(rules)
_fis = ctrl.ControlSystemSimulation(_fis_ctrl)


class FisInput(BaseModel, frozen=True):
    xAvg: float
    yAvg: float
    speedMean: float
    speedStd: float


def infer_raw(fis_in: FisInput) -> float:
    _fis.input["xAvg"] = fis_in.xAvg
    _fis.input["yAvg"] = fis_in.yAvg
    _fis.input["speedMean"] = fis_in.speedMean
    _fis.input["speedStd"] = fis_in.speedStd
    _fis.compute()
    return _fis.output["output"]


def gauss_fn(x: float, mean: float, sigma: float) -> float:
    """
    Gaussian function, nothing special.
    """
    return float(np.exp(-((x - mean)**2.0) / (2 * sigma**2.0)))


# MAX_VAL = centroid(
#     range(-1, 1),
#     lambda x: gauss_fn(x, 1, 0.6),
# )
MAX_VAL = 0.45
# MIN_VAL = centroid(
#     range(-1, 1),
#     lambda x: gauss_fn(x, -1, 0.6),
# )
MIN_VAL = -0.45


def infer(fis_in: FisInput) -> float:
    raw = infer_raw(fis_in)
    return mapRange(raw, MIN_VAL, MAX_VAL, -1, 1)
