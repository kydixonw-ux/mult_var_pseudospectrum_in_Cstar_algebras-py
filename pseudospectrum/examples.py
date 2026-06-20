"""Ports of the examples from the paper and original MATLAB scripts."""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def universal_order_two_pair(z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the two order-two unitaries U and V_z from Equation (6.3)."""

    z = _finite_float(z, "z")
    if not -1.0 <= z <= 1.0:
        raise ValueError("z must lie in [-1, 1]")
    root = np.sqrt(max(1.0 - z * z, 0.0))
    u = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    v = np.array([[z, root], [root, -z]], dtype=complex)
    return u, v


def universal_order_two_quadratic_fixed_z(x: float, y: float, z: float) -> float:
    """Closed form from Lemma 6.1 for fixed z."""

    x = _finite_float(x, "x")
    y = _finite_float(y, "y")
    z = _finite_float(z, "z")
    inner = x * x + 2.0 * z * x * y + y * y
    value = x * x + y * y + 2.0 - 2.0 * np.sqrt(max(inner, 0.0))
    return float(np.sqrt(max(value, 0.0)))


def universal_order_two_clifford_fixed_z(x: float, y: float, z: float) -> float:
    """Closed form from Lemma 6.1 for fixed z."""

    x = _finite_float(x, "x")
    y = _finite_float(y, "y")
    z = _finite_float(z, "z")
    inner = x * x + 2.0 * x * y * z + y * y + 1.0 - z * z
    value = x * x + y * y + 2.0 - 2.0 * np.sqrt(max(inner, 0.0))
    return float(np.sqrt(max(value, 0.0)))


def universal_order_two_quadratic(x: float, y: float) -> float:
    """Universal quadratic pseudospectrum: distance to the four corners."""

    x = _finite_float(x, "x")
    y = _finite_float(y, "y")
    corners = ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
    return float(min(np.hypot(x - cx, y - cy) for cx, cy in corners))


def universal_order_two_clifford(x: float, y: float) -> float:
    """Universal Clifford pseudospectrum from Theorem 6.2."""

    x = _finite_float(x, "x")
    y = _finite_float(y, "y")
    xy = x * y
    if -1.0 <= xy <= 1.0:
        inner = x * x * y * y + x * x + y * y + 1.0
        value = x * x + y * y + 2.0 - 2.0 * np.sqrt(max(inner, 0.0))
        return float(np.sqrt(max(value, 0.0)))
    return universal_order_two_quadratic(x, y)


def projection_pair(z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the pair of projections used in ``Projection_calc.m``."""

    z = _finite_float(z, "z")
    if not 0.0 <= z <= 1.0:
        raise ValueError("z must lie in [0, 1]")
    root = np.sqrt(max(z - z * z, 0.0))
    p = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    q = np.array([[1.0 - z, root], [root, z]], dtype=complex)
    return p, q


def projection_pair_quadratic_fixed_z(x: float, y: float, z: float) -> float:
    """Closed form ported from ``Projection_calc.m``."""

    x = _finite_float(x, "x")
    y = _finite_float(y, "y")
    z = _finite_float(z, "z")
    inner = (x + y) ** 2 - 2.0 * y - 2.0 * x + 1.0
    inner -= (2.0 * x - 1.0) * (2.0 * y - 1.0) * z
    value = x * x + y * y - x - y + 1.0 - np.sqrt(max(inner, 0.0))
    return float(np.sqrt(max(value, 0.0)))


def projection_pair_clifford_fixed_z(x: float, y: float, z: float) -> float:
    """Closed form ported from ``Projection_calc.m``."""

    x = _finite_float(x, "x")
    y = _finite_float(y, "y")
    z = _finite_float(z, "z")
    inner = (x + y) ** 2 - 2.0 * y - 2.0 * x + 1.0
    inner -= (2.0 * x - 1.0) * (2.0 * y - 1.0) * z
    inner += z - z * z
    value = x * x + y * y - x - y + 1.0 - np.sqrt(max(inner, 0.0))
    return float(np.sqrt(max(value, 0.0)))


def shift_triple_dense(size: int, b: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite truncation of the unilateral-shift triple from Equation (7.1)."""

    if size < 1:
        raise ValueError("size must be positive")
    b = _finite_float(b, "b")
    shift = np.zeros((size, size), dtype=complex)
    for index in range(size - 1):
        shift[index + 1, index] = 1.0
    a1 = 0.5 * (shift + shift.conj().T)
    a2 = 0.5j * (shift - shift.conj().T)
    a3 = np.zeros((size, size), dtype=complex)
    a3[0, 0] = float(b)
    return a1, a2, a3


def shift_triple_sparse(size: int, b: float = 1.0) -> tuple[Any, Any, Any]:
    """SciPy sparse version of :func:`shift_triple_dense`."""

    try:
        import scipy.sparse as sparse
    except Exception as exc:  # pragma: no cover - depends on user environment
        raise ImportError("shift_triple_sparse requires scipy") from exc

    if size < 1:
        raise ValueError("size must be positive")
    b = _finite_float(b, "b")
    lower = sparse.diags([np.ones(size - 1)], offsets=[-1], shape=(size, size), dtype=complex)
    upper = lower.conj().T
    a1 = 0.5 * (lower + upper)
    a2 = 0.5j * (lower - upper)
    a3 = sparse.diags([float(b)] + [0.0] * (size - 1), offsets=0, format="csr", dtype=complex)
    return a1.tocsr(), a2.tocsr(), a3.tocsr()


def hemisphere_polynomial(x: float, y: float, z: float, b: float = 1.0) -> float:
    """Polynomial surface from Example 7.2.

    The radial variable in the paper is ``r2 = x**2 + y**2``.
    """

    x = _finite_float(x, "x")
    y = _finite_float(y, "y")
    z = _finite_float(z, "z")
    b = _finite_float(b, "b")
    r2 = x * x + y * y
    return float(
        b**4 * z
        + b**3 * r2
        - 3.0 * b**3 * z * z
        - b**3
        - b * b * r2 * z
        + 3.0 * b * b * z**3
        + 2.0 * b * b * z
        + b * r2 * r2
        - b * r2
        - b * z**4
        - b * z * z
        + r2 * z
    )


def shift_region_function(x: float, z: float, b: float = 1.0) -> float:
    """Inequality function f(x, z) from the shift-triple calculation."""

    x = _finite_float(x, "x")
    z = _finite_float(z, "z")
    b = _finite_float(b, "b")
    return float(x * (b - z) - (b - z) + z * x * x + z * (b - z) ** 2)
