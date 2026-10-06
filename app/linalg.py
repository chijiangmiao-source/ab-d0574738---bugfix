"""极简稠密线性代数（纯 Python，无第三方依赖）。

仅覆盖固定滞后平滑器所需的定长矩阵运算与正定/求逆检查。
矩阵统一表示为 ``list[list[float]]``，向量为 ``list[float]``。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Matrix = List[List[float]]
Vector = List[float]


def is_finite_vector(v: Sequence[float]) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in v)


def is_finite_matrix(m: Sequence[Sequence[float]]) -> bool:
    return all(is_finite_vector(row) for row in m)


def shape(m: Sequence[Sequence[float]]) -> Tuple[int, int]:
    return len(m), (len(m[0]) if m else 0)


def zeros(n: int, m: int) -> Matrix:
    return [[0.0 for _ in range(m)] for _ in range(n)]


def identity(n: int) -> Matrix:
    m = zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def copy_matrix(m: Matrix) -> Matrix:
    return [row[:] for row in m]


def transpose(m: Sequence[Sequence[float]]) -> Matrix:
    r, c = shape(m)
    return [[float(m[i][j]) for i in range(r)] for j in range(c)]


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    ar, ac = shape(a)
    br, bc = shape(b)
    if ac != br:
        raise ValueError(f"矩阵维度不匹配: ({ar},{ac}) x ({br},{bc})")
    out = zeros(ar, bc)
    for i in range(ar):
        for k in range(ac):
            aik = float(a[i][k])
            if aik == 0.0:
                continue
            brow = b[k]
            orow = out[i]
            for j in range(bc):
                orow[j] += aik * float(brow[j])
    return out


def matvec(a: Sequence[Sequence[float]], v: Sequence[float]) -> Vector:
    ar, ac = shape(a)
    if ac != len(v):
        raise ValueError("矩阵/向量维度不匹配")
    return [sum(float(a[i][j]) * float(v[j]) for j in range(ac)) for i in range(ar)]


def add(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    return [[float(a[i][j]) + float(b[i][j]) for j in range(len(a[0]))] for i in range(len(a))]


def sub(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    return [[float(a[i][j]) - float(b[i][j]) for j in range(len(a[0]))] for i in range(len(a))]


def symmetric_part(m: Matrix) -> Matrix:
    r, c = shape(m)
    return [[0.5 * (m[i][j] + m[j][i]) for j in range(c)] for i in range(r)]


def is_symmetric(m: Sequence[Sequence[float]], atol: float = 1e-8, rtol: float = 1e-8) -> bool:
    r, c = shape(m)
    if r != c:
        return False
    for i in range(r):
        for j in range(i + 1, r):
            a, b = float(m[i][j]), float(m[j][i])
            tol = atol + rtol * max(abs(a), abs(b))
            if abs(a - b) > tol:
                return False
    return True


def cholesky(m: Sequence[Sequence[float]]) -> Matrix:
    """对称正定矩阵的 Cholesky 分解 L（M = L L^T）；非正定抛 LinAlgError。"""
    r, c = shape(m)
    if r != c:
        raise np_linalg_error()
    n = r
    L = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = float(m[i][j]) - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0.0 or not math.isfinite(s):
                    raise np_linalg_error()
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def np_linalg_error() -> "LinAlgError":
    return LinAlgError("矩阵分解失败")


class LinAlgError(Exception):
    """与 numpy.linalg.LinAlgError 对应的本地异常。"""


def inv2(m: Sequence[Sequence[float]]) -> Matrix:
    """2x2 矩阵求逆（带行列式检查）。"""
    a, b = float(m[0][0]), float(m[0][1])
    c, d = float(m[1][0]), float(m[1][1])
    det = a * d - b * c
    if not math.isfinite(det) or abs(det) < 1e-15:
        raise np_linalg_error()
    return [[d / det, -b / det], [-c / det, a / det]]


def eigvalsh2(m: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """2x2 对称矩阵的两个（实）特征值，升序返回。"""
    a, b = float(m[0][0]), float(m[0][1])
    d = float(m[1][1])
    disc = math.sqrt(max(0.0, (a - d) ** 2) + 4.0 * b * b)
    return (a + d - disc) / 2.0, (a + d + disc) / 2.0


def diag(m: Sequence[Sequence[float]]) -> Vector:
    return [float(m[i][i]) for i in range(len(m))]
