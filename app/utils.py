from typing import Callable, Union

Num = Union[int, float]


def centroid(domain: range,
             mf: Callable[[Num], float],
             segmentation: int = 100) -> float:
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
    assert inMin < inMax, "inMin must be less than inMax"
    assert outMin < outMax, "outMin must be less than outMax"
    if x < inMin:
        return outMin
    if x > inMax:
        return outMax
    return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin
