"""MPO/tensor-network backend for spectral localizers.

The MPO convention is: each tensor has shape
``(left_bond, right_bond, physical_row, physical_col)``. The helpers here avoid
dense materialization until an exact small-system solve is requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import sys

import numpy as np

from .backends import smallest_singular_value_dense
from .clifford import clifford_generators, validate_clifford

MPO = list[np.ndarray]


@dataclass
class TensorNetworkSolveInfo:
    """Metadata returned by tensor-network pseudospectrum calls."""

    backend: str
    dimension: int
    localizer_shape: tuple[int, int]
    details: dict[str, Any] = field(default_factory=dict)


def _copy_mpo(mpo: Sequence[np.ndarray]) -> MPO:
    tensors = [np.asarray(tensor, dtype=complex).copy() for tensor in mpo]
    _validate_mpo(tensors)
    return tensors


def _validate_mpo(mpo: Sequence[np.ndarray]) -> None:
    if not mpo:
        raise ValueError("MPO must contain at least one tensor")
    for index, tensor in enumerate(mpo):
        if tensor.ndim != 4:
            raise ValueError(f"MPO tensor {index} must have rank 4")
        if any(size <= 0 for size in tensor.shape):
            raise ValueError(f"MPO tensor {index} has an empty dimension")
        if tensor.shape[2] != tensor.shape[3]:
            raise ValueError(f"MPO tensor {index} must have square physical legs")


def _local_dims(mpo: Sequence[np.ndarray]) -> list[int]:
    _validate_mpo(mpo)
    return [int(tensor.shape[2]) for tensor in mpo]


def _total_dim_from_dims(dims: Sequence[int]) -> int:
    total = 1
    for dim in dims:
        total *= int(dim)
    return int(total)


def _require_same_site_dimensions(mpos: Sequence[Sequence[np.ndarray]]) -> list[int]:
    dims = _local_dims(mpos[0])
    site_count = len(mpos[0])
    for mpo in mpos:
        if len(mpo) != site_count or _local_dims(mpo) != dims:
            raise ValueError("all MPOs must have the same site dimensions")
    return dims


def dense_matrix_to_mpo(matrix: Any) -> MPO:
    """Represent a dense matrix as a one-site MPO."""

    array = np.asarray(matrix, dtype=complex)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    return [array.reshape(1, 1, array.shape[0], array.shape[1])]


def as_mpo(operator: Any) -> MPO:
    """Coerce an object with ``.mpo()``, a raw MPO, or a dense matrix to an MPO."""

    if hasattr(operator, "mpo") and callable(operator.mpo):
        return _copy_mpo(operator.mpo())
    if hasattr(operator, "to_mpo") and callable(operator.to_mpo):
        materialized = operator.to_mpo()
        if hasattr(materialized, "to_mpo") and materialized is not operator:
            materialized = materialized.to_mpo()
        if isinstance(materialized, (list, tuple)):
            return _copy_mpo(materialized)
        raise TypeError("to_mpo() did not return a materialized MPO")
    if isinstance(operator, (list, tuple)) and operator and np.asarray(operator[0]).ndim == 4:
        return _copy_mpo(operator)
    return dense_matrix_to_mpo(operator)


def identity_mpo_like(mpo: Sequence[np.ndarray]) -> MPO:
    """Build an identity MPO with the same site dimensions as ``mpo``."""

    return [
        np.eye(tensor.shape[2], dtype=complex).reshape(1, 1, tensor.shape[2], tensor.shape[2])
        for tensor in mpo
    ]


def scale_mpo(mpo: Sequence[np.ndarray], scalar: complex) -> MPO:
    """Scale an MPO without changing its bond dimensions."""

    tensors = _copy_mpo(mpo)
    tensors[0] *= scalar
    return tensors


def add_mpos(mpos: Sequence[Sequence[np.ndarray]]) -> MPO:
    """Direct-sum MPO representation of a sum of operators."""

    if not mpos:
        raise ValueError("at least one MPO is required")
    copied = [_copy_mpo(mpo) for mpo in mpos]
    _require_same_site_dimensions(copied)

    if all(_is_open_boundary_mpo(mpo) for mpo in copied):
        return _add_open_boundary_mpos(copied)

    if not all(_is_periodic_square_mpo(mpo) for mpo in copied):
        raise ValueError(
            "MPO sums require either all open-boundary MPOs or all square "
            "periodic/block-diagonal MPOs"
        )

    result: MPO = []
    site_count = len(copied[0])
    for site in range(site_count):
        tensors = [mpo[site] for mpo in copied]
        left = sum(tensor.shape[0] for tensor in tensors)
        right = sum(tensor.shape[1] for tensor in tensors)
        dim = tensors[0].shape[2]
        block = np.zeros((left, right, dim, dim), dtype=complex)
        left_offset = 0
        right_offset = 0
        for tensor in tensors:
            left_size, right_size = tensor.shape[:2]
            block[
                left_offset : left_offset + left_size,
                right_offset : right_offset + right_size,
                :,
                :,
            ] = tensor
            left_offset += left_size
            right_offset += right_size
        result.append(block)
    return result


def _add_open_boundary_mpos(mpos: Sequence[MPO]) -> MPO:
    site_count = len(mpos[0])
    dims = _local_dims(mpos[0])
    if site_count == 1:
        matrix = sum((mpo[0][0, 0] for mpo in mpos), np.zeros((dims[0], dims[0]), dtype=complex))
        return [matrix.reshape(1, 1, dims[0], dims[0])]

    result: MPO = []
    for site in range(site_count):
        tensors = [mpo[site] for mpo in mpos]
        dim = dims[site]

        if site == 0:
            right = sum(tensor.shape[1] for tensor in tensors)
            block = np.zeros((1, right, dim, dim), dtype=complex)
            offset = 0
            for tensor in tensors:
                width = tensor.shape[1]
                block[0, offset : offset + width] = tensor[0]
                offset += width
        elif site == site_count - 1:
            left = sum(tensor.shape[0] for tensor in tensors)
            block = np.zeros((left, 1, dim, dim), dtype=complex)
            offset = 0
            for tensor in tensors:
                height = tensor.shape[0]
                block[offset : offset + height, 0] = tensor[:, 0]
                offset += height
        else:
            left = sum(tensor.shape[0] for tensor in tensors)
            right = sum(tensor.shape[1] for tensor in tensors)
            block = np.zeros((left, right, dim, dim), dtype=complex)
            left_offset = 0
            right_offset = 0
            for tensor in tensors:
                left_size, right_size = tensor.shape[:2]
                block[
                    left_offset : left_offset + left_size,
                    right_offset : right_offset + right_size,
                    :,
                    :,
                ] = tensor
                left_offset += left_size
                right_offset += right_size
        result.append(block)

    return result


def product_mpos(mpos: Sequence[Sequence[np.ndarray]]) -> MPO:
    """Ordered product of MPOs using local physical-leg contraction."""

    if not mpos:
        raise ValueError("at least one MPO is required")
    result = _copy_mpo(mpos[0])
    for mpo in mpos[1:]:
        other = _copy_mpo(mpo)
        _require_same_site_dimensions([result, other])
        next_result: MPO = []
        for left_tensor, right_tensor in zip(result, other):
            if left_tensor.shape[3] != right_tensor.shape[2]:
                raise ValueError("adjacent MPO physical dimensions do not align")
            la, ra, d_row, d_mid = left_tensor.shape
            lb, rb, _, d_col = right_tensor.shape
            combined = np.einsum(
                "abij,cdjk->acbdik",
                left_tensor,
                right_tensor,
                optimize=True,
            )
            next_result.append(combined.reshape(la * lb, ra * rb, d_row, d_col))
        result = next_result
    return result


def adjoint_mpo(mpo: Sequence[np.ndarray]) -> MPO:
    """Hermitian adjoint of an MPO, preserving the virtual convention."""

    return [np.conjugate(tensor).swapaxes(2, 3).copy() for tensor in _copy_mpo(mpo)]


def shift_mpo(mpo: Sequence[np.ndarray], coordinate: float) -> MPO:
    """Build ``mpo - coordinate * I``."""

    if coordinate == 0:
        return _copy_mpo(mpo)
    return add_mpos([mpo, scale_mpo(identity_mpo_like(mpo), -complex(coordinate))])


def append_site(mpo: Sequence[np.ndarray], matrix: np.ndarray) -> MPO:
    """Append a one-site local matrix to an MPO."""

    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("appended site matrix must be square")
    tensors = _copy_mpo(mpo)
    if _is_open_boundary_mpo(tensors):
        tensors.append(matrix.reshape(1, 1, matrix.shape[0], matrix.shape[1]))
        return tensors

    if all(tensor.shape[0] == tensor.shape[1] for tensor in tensors):
        virtual_dims = {tensor.shape[0] for tensor in tensors}
        if len(virtual_dims) == 1:
            virtual_dim = virtual_dims.pop()
            appended = np.zeros(
                (virtual_dim, virtual_dim, matrix.shape[0], matrix.shape[1]),
                dtype=complex,
            )
            for index in range(virtual_dim):
                appended[index, index] = matrix
            tensors.append(appended)
            return tensors

    raise ValueError("cannot append a site to an MPO with incompatible virtual bonds")


def localizer_mpo(
    operators: Sequence[Any],
    point: Sequence[float],
    *,
    gammas: Sequence[np.ndarray] | None = None,
) -> MPO:
    """Build an MPO for ``sum_j (A_j - lambda_j I) tensor Gamma_j``.

    The Clifford spinor is appended as one additional site. This lets ordinary
    MPO product/sum utilities represent the spectral localizer without
    increasing every physical site dimension.
    """

    mpos = [as_mpo(operator) for operator in operators]
    if not mpos:
        raise ValueError("at least one operator is required")
    point_array = np.asarray(point, dtype=float)
    if point_array.shape != (len(mpos),):
        raise ValueError(f"point must have length {len(mpos)}")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("point coordinates must be finite real numbers")
    _require_same_site_dimensions(mpos)

    if gammas is None:
        gammas = clifford_generators(len(mpos))
    gammas = [np.asarray(gamma, dtype=complex) for gamma in gammas]
    validate_clifford(gammas)

    terms = [
        append_site(shift_mpo(mpo, coordinate), gamma)
        for mpo, coordinate, gamma in zip(mpos, point_array, gammas)
    ]
    return add_mpos(terms)


def quadratic_mpo(operators: Sequence[Any], point: Sequence[float]) -> MPO:
    """Build an MPO for ``Q_lambda(A) = sum_j (A_j - lambda_j I)^2``."""

    mpos = [as_mpo(operator) for operator in operators]
    if not mpos:
        raise ValueError("at least one operator is required")
    point_array = np.asarray(point, dtype=float)
    if point_array.shape != (len(mpos),):
        raise ValueError(f"point must have length {len(mpos)}")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("point coordinates must be finite real numbers")
    _require_same_site_dimensions(mpos)
    shifted = [shift_mpo(mpo, coordinate) for mpo, coordinate in zip(mpos, point_array)]
    return add_mpos([product_mpos([mpo, mpo]) for mpo in shifted])


def _is_open_boundary_mpo(mpo: Sequence[np.ndarray]) -> bool:
    if mpo[0].shape[0] != 1 or mpo[-1].shape[1] != 1:
        return False
    return all(mpo[index].shape[1] == mpo[index + 1].shape[0] for index in range(len(mpo) - 1))


def _is_periodic_square_mpo(mpo: Sequence[np.ndarray]) -> bool:
    return (
        all(tensor.shape[0] == tensor.shape[1] for tensor in mpo)
        and len({tensor.shape[0] for tensor in mpo}) == 1
    )


def _mpo_to_dense_open(mpo: Sequence[np.ndarray]) -> np.ndarray:
    acc = mpo[0][0, :, :, :]
    for tensor in mpo[1:]:
        acc = np.tensordot(acc, tensor, axes=([0], [0]))
        right_axis = len(acc.shape) - 3
        acc = np.moveaxis(acc, right_axis, 0)
    if acc.shape[0] != 1:
        raise ValueError("open-boundary MPO did not end with right bond dimension 1")
    body = acc[0]
    site_count = len(mpo)
    row_axes = list(range(0, 2 * site_count, 2))
    col_axes = list(range(1, 2 * site_count, 2))
    dims = _local_dims(mpo)
    return body.transpose(row_axes + col_axes).reshape(
        _total_dim_from_dims(dims),
        _total_dim_from_dims(dims),
    )


def _mpo_to_dense_periodic(mpo: Sequence[np.ndarray]) -> np.ndarray:
    dims = _local_dims(mpo)
    if not _is_periodic_square_mpo(mpo):
        raise ValueError("periodic MPO contraction needs a common virtual bond dimension")

    total = _total_dim_from_dims(dims)
    dense = np.zeros((total, total), dtype=complex)
    for row_flat, row_multi in enumerate(np.ndindex(*dims)):
        for col_flat, col_multi in enumerate(np.ndindex(*dims)):
            virtual = mpo[0][:, :, row_multi[0], col_multi[0]]
            for site in range(1, len(mpo)):
                virtual = virtual @ mpo[site][:, :, row_multi[site], col_multi[site]]
            dense[row_flat, col_flat] = np.trace(virtual)
    return dense


def mpo_to_dense(mpo: Sequence[np.ndarray], *, boundary: str = "auto") -> np.ndarray:
    """Exactly contract a small MPO to a dense matrix."""

    tensors = _copy_mpo(mpo)
    if boundary not in {"auto", "open", "periodic"}:
        raise ValueError("boundary must be 'auto', 'open', or 'periodic'")
    if boundary == "open" or (boundary == "auto" and _is_open_boundary_mpo(tensors)):
        return _mpo_to_dense_open(tensors)
    return _mpo_to_dense_periodic(tensors)


def _run_dmrg_on_positive_mpo(
    positive_mpo: Sequence[np.ndarray],
    *,
    dmrg_module_path: str | Path | None = None,
    dmrg_kwargs: dict[str, Any] | None = None,
) -> tuple[float, Any]:
    """Minimise ``<psi|L* L|psi>`` with an optional external DMRG driver.

    The driver is any importable module exposing a ``DMRG`` class with a
    ``run(mpo) -> (energy, state)`` method. Point ``dmrg_module_path`` at the
    directory containing that module (it is prepended to ``sys.path``), or make
    it importable some other way (installed package, existing ``sys.path``
    entry). This keeps large-MPO solving optional so the core package stays
    dependency-free.
    """

    if dmrg_module_path is not None:
        path_text = str(Path(dmrg_module_path))
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    try:
        from dmrg import DMRG  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional driver
        raise RuntimeError(
            "DMRG tensor-network solving requires an external DMRG driver -- a "
            "module exposing a `DMRG` class -- supplied via `dmrg_module_path` "
            "or importable on sys.path. Use solver='dense' for small MPOs."
        ) from exc

    kwargs = dict(dmrg_kwargs or {})
    solver = DMRG(**kwargs)
    return solver.run(list(positive_mpo))


def clifford_pseudovalue_tensor(
    operators: Sequence[Any],
    point: Sequence[float],
    *,
    gammas: Sequence[np.ndarray] | None = None,
    solver: str = "auto",
    max_exact_dim: int = 4096,
    dmrg_module_path: str | Path | None = None,
    dmrg_kwargs: dict[str, Any] | None = None,
) -> tuple[float, TensorNetworkSolveInfo]:
    """Return ``mu_C(point)`` from MPO/tensor-network operators.

    ``solver='auto'`` contracts the MPO exactly when the total Hilbert-space
    dimension is at most ``max_exact_dim``. Larger problems fall back to an
    optional external DMRG driver on ``L* L`` if one is available (see
    ``dmrg_module_path`` and :func:`_run_dmrg_on_positive_mpo`).
    """

    localizer = localizer_mpo(operators, point, gammas=gammas)
    dims = _local_dims(localizer)
    total_dim = _total_dim_from_dims(dims)
    localizer_shape = (total_dim, total_dim)

    if solver not in {"auto", "dense", "dmrg"}:
        raise ValueError("solver must be 'auto', 'dense', or 'dmrg'")
    if not isinstance(max_exact_dim, int) or max_exact_dim < 1:
        raise ValueError("max_exact_dim must be a positive integer")

    if solver == "dense" or (solver == "auto" and total_dim <= max_exact_dim):
        dense = mpo_to_dense(localizer)
        value = smallest_singular_value_dense(dense, hermitian=True)
        return value, TensorNetworkSolveInfo(
            backend="dense-mpo",
            dimension=total_dim,
            localizer_shape=localizer_shape,
        )

    positive = product_mpos([adjoint_mpo(localizer), localizer])
    energy, state = _run_dmrg_on_positive_mpo(
        positive,
        dmrg_module_path=dmrg_module_path,
        dmrg_kwargs=dmrg_kwargs,
    )
    value = float(np.sqrt(max(float(np.real(energy)), 0.0)))
    return value, TensorNetworkSolveInfo(
        backend="dmrg",
        dimension=total_dim,
        localizer_shape=localizer_shape,
        details={"ground_energy_LstarL": float(np.real(energy)), "state": state},
    )
