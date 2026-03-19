import numpy as np


class ShapeMismatchError(Exception):
    pass


def adaptive_filter(
    Vs: np.ndarray,
    Vj: np.ndarray,
    diag_A: np.ndarray,
) -> np.ndarray:
    if Vs.shape[0] != Vj.shape[0]:
        raise ShapeMismatchError

    if Vj.shape[1] != diag_A.size:
        raise ShapeMismatchError

    E = np.eye(Vj.shape[1])
    Vjh = np.conj(Vj).T
    A = np.diag(diag_A)

    M = np.linalg.inv(E + Vjh @ Vj @ A)
    y = Vs - Vj @ M @ (Vjh @ Vs)

    return y
