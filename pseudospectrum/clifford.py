"""Clifford matrix generators used by the spectral localizer."""

from __future__ import annotations

from functools import reduce
from typing import Iterable, Sequence

import numpy as np


def pauli_matrices(dtype: np.dtype | type = complex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the Pauli X, Y, and Z matrices."""

    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.complexfloating):
        raise ValueError("Pauli matrices require a complex dtype")
    x = np.array([[0, 1], [1, 0]], dtype=dtype)
    y = np.array([[0, -1j], [1j, 0]], dtype=dtype)
    z = np.array([[1, 0], [0, -1]], dtype=dtype)
    return x, y, z


def _kron_all(matrices: Iterable[np.ndarray]) -> np.ndarray:
    matrices = list(matrices)
    if not matrices:
        return np.ones((1, 1), dtype=complex)
    return reduce(np.kron, matrices)


def clifford_generators(dimension: int, dtype: np.dtype | type = complex) -> list[np.ndarray]:
    """Build an irreducible complex representation of the Clifford relations.

    The returned Hermitian matrices Gamma_j satisfy Gamma_j^2 = I and
    Gamma_j Gamma_k = - Gamma_k Gamma_j for j != k. The size is
    2**floor(dimension / 2), matching the convention in the paper.
    """

    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.complexfloating):
        raise ValueError("Clifford generators require a complex dtype")

    if dimension == 1:
        return [np.ones((1, 1), dtype=dtype)]

    x, y, z = pauli_matrices(dtype)
    identity = np.eye(2, dtype=dtype)
    pairs = dimension // 2
    generators: list[np.ndarray] = []

    for index in range(pairs):
        prefix = [z] * index
        suffix = [identity] * (pairs - index - 1)
        generators.append(_kron_all(prefix + [x] + suffix).astype(dtype, copy=False))
        generators.append(_kron_all(prefix + [y] + suffix).astype(dtype, copy=False))

    if dimension % 2:
        generators.append(_kron_all([z] * pairs).astype(dtype, copy=False))

    return generators[:dimension]


def validate_clifford(
    generators: Sequence[np.ndarray],
    *,
    atol: float = 1e-12,
    raise_on_error: bool = True,
) -> bool:
    """Validate the Clifford relations for a sequence of generators."""

    if not generators:
        if raise_on_error:
            raise ValueError("at least one generator is required")
        return False

    shape = generators[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        if raise_on_error:
            raise ValueError("generators must be square matrices")
        return False

    identity = np.eye(shape[0], dtype=complex)
    for index, gamma in enumerate(generators):
        if gamma.shape != shape:
            if raise_on_error:
                raise ValueError("all generators must have the same shape")
            return False
        if not np.allclose(gamma, gamma.conj().T, atol=atol):
            if raise_on_error:
                raise ValueError(f"generator {index} is not Hermitian")
            return False
        if not np.allclose(gamma @ gamma, identity, atol=atol):
            if raise_on_error:
                raise ValueError(f"generator {index} does not square to identity")
            return False

    for left in range(len(generators)):
        for right in range(left + 1, len(generators)):
            anticommutator = generators[left] @ generators[right]
            anticommutator += generators[right] @ generators[left]
            if not np.allclose(anticommutator, 0.0, atol=atol):
                if raise_on_error:
                    raise ValueError(f"generators {left} and {right} do not anticommute")
                return False

    return True
