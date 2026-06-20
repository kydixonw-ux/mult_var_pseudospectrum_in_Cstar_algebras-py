"""Dense and sparse matrix backends for multivariable pseudospectra."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal

import numpy as np

from .clifford import clifford_generators, validate_clifford

BackendName = Literal["dense", "sparse", "tensor", "tn", "mpo"]


def _try_import_scipy() -> tuple[Any, Any]:
    try:
        import scipy.sparse as sparse
        import scipy.sparse.linalg as sparse_linalg
    except Exception as exc:  # pragma: no cover - depends on user environment
        raise ImportError(
            "Sparse pseudospectrum evaluation requires SciPy. Install scipy or "
            "use backend='dense' for small matrices."
        ) from exc
    return sparse, sparse_linalg


def _as_dense_operator(operator: Any) -> np.ndarray:
    if hasattr(operator, "dense") and callable(operator.dense):
        operator = operator.dense()
    elif hasattr(operator, "toarray") and callable(operator.toarray):
        operator = operator.toarray()
    array = np.asarray(operator, dtype=complex)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("operators must be square matrices")
    return array


def _as_point(point: Sequence[float], expected: int) -> np.ndarray:
    values = np.asarray(point, dtype=float)
    if values.shape != (expected,):
        raise ValueError(f"point must have length {expected}")
    if not np.all(np.isfinite(values)):
        raise ValueError("point coordinates must be finite real numbers")
    return values


def _dense_operators(
    operators: Sequence[Any],
    *,
    check_hermitian: bool = False,
    hermitian_atol: float = 1e-10,
) -> list[np.ndarray]:
    dense = [_as_dense_operator(operator) for operator in operators]
    if not dense:
        raise ValueError("at least one operator is required")
    dimension = dense[0].shape[0]
    if any(operator.shape != (dimension, dimension) for operator in dense):
        raise ValueError("all operators must have the same square shape")
    if check_hermitian:
        for index, operator in enumerate(dense):
            if not _is_hermitian(operator, atol=hermitian_atol):
                raise ValueError(f"operator {index} is not Hermitian")
    return dense


def _default_gammas(count: int, gammas: Sequence[np.ndarray] | None) -> list[np.ndarray]:
    if gammas is None:
        gammas = clifford_generators(count)
    gammas = [np.asarray(gamma, dtype=complex) for gamma in gammas]
    if len(gammas) != count:
        raise ValueError(f"expected {count} Clifford generators")
    validate_clifford(gammas)
    return gammas


def _is_hermitian(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    return np.allclose(matrix, matrix.conj().T, atol=atol)


def smallest_singular_value_dense(matrix: np.ndarray, *, hermitian: bool | None = None) -> float:
    """Compute the smallest singular value of a dense matrix."""

    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if hermitian is None:
        hermitian = matrix.shape[0] == matrix.shape[1] and _is_hermitian(matrix)
    if hermitian:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return float(np.min(np.abs(eigenvalues)))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return float(np.min(singular_values))


def dense_localizer(
    operators: Sequence[Any],
    point: Sequence[float],
    *,
    gammas: Sequence[np.ndarray] | None = None,
    check_hermitian: bool = True,
    hermitian_atol: float = 1e-10,
) -> np.ndarray:
    """Build the dense spectral localizer at a point."""

    dense = _dense_operators(
        operators,
        check_hermitian=check_hermitian,
        hermitian_atol=hermitian_atol,
    )
    point_array = _as_point(point, len(dense))
    gammas = _default_gammas(len(dense), gammas)

    dimension = dense[0].shape[0]
    gamma_size = gammas[0].shape[0]
    identity = np.eye(dimension, dtype=complex)
    localizer = np.zeros((dimension * gamma_size, dimension * gamma_size), dtype=complex)

    for operator, coordinate, gamma in zip(dense, point_array, gammas):
        localizer += np.kron(operator - coordinate * identity, gamma)

    return localizer


def clifford_pseudovalue_dense(
    operators: Sequence[Any],
    point: Sequence[float],
    *,
    gammas: Sequence[np.ndarray] | None = None,
    assume_hermitian: bool = True,
    check_hermitian: bool = True,
    hermitian_atol: float = 1e-10,
) -> float:
    """Return mu_C(point) for dense matrices."""

    localizer = dense_localizer(
        operators,
        point,
        gammas=gammas,
        check_hermitian=assume_hermitian and check_hermitian,
        hermitian_atol=hermitian_atol,
    )
    return smallest_singular_value_dense(localizer, hermitian=assume_hermitian)


def quadratic_matrix_dense(
    operators: Sequence[Any],
    point: Sequence[float],
    *,
    check_hermitian: bool = True,
    hermitian_atol: float = 1e-10,
) -> np.ndarray:
    """Build Q_lambda(A) = sum_j (A_j - lambda_j I)^2 densely."""

    dense = _dense_operators(
        operators,
        check_hermitian=check_hermitian,
        hermitian_atol=hermitian_atol,
    )
    point_array = _as_point(point, len(dense))
    dimension = dense[0].shape[0]
    identity = np.eye(dimension, dtype=complex)
    quadratic = np.zeros((dimension, dimension), dtype=complex)
    for operator, coordinate in zip(dense, point_array):
        shifted = operator - coordinate * identity
        quadratic += shifted @ shifted
    return quadratic


def quadratic_pseudovalue_dense(
    operators: Sequence[Any],
    point: Sequence[float],
    *,
    check_hermitian: bool = True,
    hermitian_atol: float = 1e-10,
) -> float:
    """Return mu_Q(point) for dense Hermitian matrices."""

    quadratic = quadratic_matrix_dense(
        operators,
        point,
        check_hermitian=check_hermitian,
        hermitian_atol=hermitian_atol,
    )
    eigenvalues = np.linalg.eigvalsh((quadratic + quadratic.conj().T) / 2.0)
    return float(np.sqrt(max(float(np.min(eigenvalues)), 0.0)))


def _as_sparse_operator(operator: Any, sparse: Any) -> Any:
    if hasattr(operator, "sparse") and callable(operator.sparse):
        operator = operator.sparse()
    elif hasattr(operator, "dense") and callable(operator.dense):
        operator = operator.dense()
    if sparse.issparse(operator):
        matrix = operator.tocsr()
    else:
        matrix = sparse.csr_matrix(np.asarray(operator, dtype=complex))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operators must be square matrices")
    return matrix


def _validate_sparse_hermitian(
    operators: Sequence[Any],
    *,
    hermitian_atol: float,
) -> None:
    for index, operator in enumerate(operators):
        difference = (operator - operator.getH()).tocoo()
        if difference.nnz and float(np.max(np.abs(difference.data))) > hermitian_atol:
            raise ValueError(f"operator {index} is not Hermitian")


def sparse_localizer(
    operators: Sequence[Any],
    point: Sequence[float],
    *,
    gammas: Sequence[np.ndarray] | None = None,
    check_hermitian: bool = True,
    hermitian_atol: float = 1e-10,
) -> Any:
    """Build a SciPy sparse spectral localizer."""

    sparse, _ = _try_import_scipy()
    sparse_ops = [_as_sparse_operator(operator, sparse) for operator in operators]
    if not sparse_ops:
        raise ValueError("at least one operator is required")
    dimension = sparse_ops[0].shape[0]
    if any(operator.shape != (dimension, dimension) for operator in sparse_ops):
        raise ValueError("all operators must have the same square shape")
    if check_hermitian:
        _validate_sparse_hermitian(sparse_ops, hermitian_atol=hermitian_atol)

    point_array = _as_point(point, len(sparse_ops))
    gammas = _default_gammas(len(sparse_ops), gammas)

    identity = sparse.eye(dimension, dtype=complex, format="csr")
    localizer = None
    for operator, coordinate, gamma in zip(sparse_ops, point_array, gammas):
        shifted = operator - coordinate * identity
        gamma_sparse = sparse.csr_matrix(gamma)
        term = sparse.kron(shifted, gamma_sparse, format="csr")
        localizer = term if localizer is None else localizer + term
    return localizer.tocsr()


def _smallest_singular_value_sparse(
    matrix: Any,
    *,
    hermitian: bool = True,
    dense_fallback_dimension: int = 512,
) -> float:
    sparse, sparse_linalg = _try_import_scipy()
    matrix = matrix.tocsr()
    if matrix.shape[0] <= dense_fallback_dimension:
        return smallest_singular_value_dense(matrix.toarray(), hermitian=hermitian)

    if hermitian:
        try:
            eigenvalues = sparse_linalg.eigsh(
                matrix,
                k=1,
                sigma=0.0,
                which="LM",
                return_eigenvectors=False,
            )
        except Exception:
            eigenvalues = sparse_linalg.eigsh(
                matrix,
                k=1,
                which="SM",
                return_eigenvectors=False,
            )
        return float(np.min(np.abs(eigenvalues)))

    singular_values = sparse_linalg.svds(matrix, k=1, which="SM", return_singular_vectors=False)
    return float(np.min(np.abs(singular_values)))


def clifford_pseudovalue_sparse(
    operators: Sequence[Any],
    point: Sequence[float],
    *,
    gammas: Sequence[np.ndarray] | None = None,
    dense_fallback_dimension: int = 512,
    assume_hermitian: bool = True,
    check_hermitian: bool = True,
    hermitian_atol: float = 1e-10,
) -> float:
    """Return mu_C(point) using SciPy sparse matrices."""

    localizer = sparse_localizer(
        operators,
        point,
        gammas=gammas,
        check_hermitian=assume_hermitian and check_hermitian,
        hermitian_atol=hermitian_atol,
    )
    return _smallest_singular_value_sparse(
        localizer,
        hermitian=assume_hermitian,
        dense_fallback_dimension=dense_fallback_dimension,
    )


def quadratic_pseudovalue_sparse(
    operators: Sequence[Any],
    point: Sequence[float],
    *,
    dense_fallback_dimension: int = 512,
    check_hermitian: bool = True,
    hermitian_atol: float = 1e-10,
) -> float:
    """Return mu_Q(point) using SciPy sparse matrices."""

    sparse, sparse_linalg = _try_import_scipy()
    sparse_ops = [_as_sparse_operator(operator, sparse) for operator in operators]
    if not sparse_ops:
        raise ValueError("at least one operator is required")
    dimension = sparse_ops[0].shape[0]
    if any(operator.shape != (dimension, dimension) for operator in sparse_ops):
        raise ValueError("all operators must have the same square shape")
    if check_hermitian:
        _validate_sparse_hermitian(sparse_ops, hermitian_atol=hermitian_atol)

    point_array = _as_point(point, len(sparse_ops))
    identity = sparse.eye(dimension, dtype=complex, format="csr")
    quadratic = sparse.csr_matrix((dimension, dimension), dtype=complex)
    for operator, coordinate in zip(sparse_ops, point_array):
        shifted = operator - coordinate * identity
        quadratic += shifted @ shifted

    if dimension <= dense_fallback_dimension:
        quadratic_dense = quadratic.toarray()
        eigenvalues = np.linalg.eigvalsh((quadratic_dense + quadratic_dense.conj().T) / 2.0)
    else:
        eigenvalues = sparse_linalg.eigsh(
            quadratic,
            k=1,
            which="SA",
            return_eigenvectors=False,
        )
    return float(np.sqrt(max(float(np.min(np.real(eigenvalues))), 0.0)))


def clifford_pseudovalue(
    operators: Sequence[Any],
    point: Sequence[float],
    *,
    backend: BackendName = "dense",
    gammas: Sequence[np.ndarray] | None = None,
    **kwargs: Any,
) -> float:
    """Dispatch mu_C(point) to the requested backend."""

    backend = backend.lower()
    if backend == "dense":
        return clifford_pseudovalue_dense(operators, point, gammas=gammas, **kwargs)
    if backend == "sparse":
        return clifford_pseudovalue_sparse(operators, point, gammas=gammas, **kwargs)
    if backend in {"tensor", "tn", "mpo"}:
        from .tensor_network import clifford_pseudovalue_tensor

        value, _ = clifford_pseudovalue_tensor(operators, point, gammas=gammas, **kwargs)
        return value
    raise ValueError("backend must be 'dense', 'sparse', or 'tensor'")


def quadratic_pseudovalue(
    operators: Sequence[Any],
    point: Sequence[float],
    *,
    backend: BackendName = "dense",
    **kwargs: Any,
) -> float:
    """Dispatch mu_Q(point) to the requested backend."""

    backend = backend.lower()
    if backend == "dense":
        return quadratic_pseudovalue_dense(operators, point, **kwargs)
    if backend == "sparse":
        return quadratic_pseudovalue_sparse(operators, point, **kwargs)
    if backend in {"tensor", "tn", "mpo"}:
        from .tensor_network import quadratic_mpo, mpo_to_dense

        mpo = quadratic_mpo(operators, point)
        dense = mpo_to_dense(mpo)
        eigenvalues = np.linalg.eigvalsh((dense + dense.conj().T) / 2.0)
        return float(np.sqrt(max(float(np.min(eigenvalues)), 0.0)))
    raise ValueError("backend must be 'dense', 'sparse', or 'tensor'")


def epsilon_mask(values: np.ndarray | Iterable[float], epsilon: float) -> np.ndarray:
    """Return a Boolean mask for the closed epsilon-pseudospectrum."""

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be a finite nonnegative number")
    return np.asarray(values, dtype=float) <= epsilon


def grid_values(
    evaluator: Callable[[tuple[float, ...]], float],
    axes: Sequence[Sequence[float] | np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Evaluate a scalar pseudospectrum function over a tensor-product grid."""

    axis_arrays = [np.asarray(axis, dtype=float) for axis in axes]
    if not axis_arrays:
        raise ValueError("at least one grid axis is required")
    for index, axis in enumerate(axis_arrays):
        if axis.ndim != 1 or axis.size == 0:
            raise ValueError(f"axis {index} must be a nonempty one-dimensional array")
        if not np.all(np.isfinite(axis)):
            raise ValueError(f"axis {index} contains nonfinite values")
    meshes = np.meshgrid(*axis_arrays, indexing="ij")
    values = np.empty(meshes[0].shape, dtype=float)
    for index in np.ndindex(values.shape):
        point = tuple(float(mesh[index]) for mesh in meshes)
        values[index] = evaluator(point)
    return list(meshes), values
