from __future__ import annotations

from collections.abc import Sequence


def harrell_c_index(time: Sequence[float], event: Sequence[int], risk: Sequence[float]) -> float:
    concordant = comparable = 0.0
    for i in range(len(time)):
        for j in range(len(time)):
            if i == j or int(event[i]) != 1 or float(time[i]) >= float(time[j]):
                continue
            comparable += 1.0
            concordant += 1.0 if float(risk[i]) > float(risk[j]) else 0.5 if float(risk[i]) == float(risk[j]) else 0.0
    if comparable == 0:
        raise ValueError("No comparable survival pairs")
    return concordant / comparable

