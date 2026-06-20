"""Multivariable pseudospectrum tools.

The public functions implement the Clifford and quadratic pseudospectra used
in the C*-algebra examples:

    L_lambda(A) = sum_j (A_j - lambda_j I) kron Gamma_j
    mu_C(lambda) = s_min(L_lambda(A))

and

    Q_lambda(A) = sum_j (A_j - lambda_j I)^2
    mu_Q(lambda) = sqrt(lambda_min(Q_lambda(A))).
"""

from .backends import (
    clifford_pseudovalue,
    clifford_pseudovalue_dense,
    clifford_pseudovalue_sparse,
    dense_localizer,
    epsilon_mask,
    grid_values,
    quadratic_matrix_dense,
    quadratic_pseudovalue,
    quadratic_pseudovalue_dense,
    quadratic_pseudovalue_sparse,
    sparse_localizer,
)
from .clifford import clifford_generators, pauli_matrices, validate_clifford
from .tensor_network import (
    TensorNetworkSolveInfo,
    add_mpos,
    adjoint_mpo,
    as_mpo,
    clifford_pseudovalue_tensor,
    dense_matrix_to_mpo,
    identity_mpo_like,
    localizer_mpo,
    mpo_to_dense,
    product_mpos,
    quadratic_mpo,
    scale_mpo,
    shift_mpo,
)

__all__ = [
    "TensorNetworkSolveInfo",
    "add_mpos",
    "adjoint_mpo",
    "as_mpo",
    "clifford_generators",
    "clifford_pseudovalue",
    "clifford_pseudovalue_dense",
    "clifford_pseudovalue_sparse",
    "clifford_pseudovalue_tensor",
    "dense_localizer",
    "dense_matrix_to_mpo",
    "epsilon_mask",
    "grid_values",
    "identity_mpo_like",
    "localizer_mpo",
    "mpo_to_dense",
    "pauli_matrices",
    "product_mpos",
    "quadratic_matrix_dense",
    "quadratic_mpo",
    "quadratic_pseudovalue",
    "quadratic_pseudovalue_dense",
    "quadratic_pseudovalue_sparse",
    "scale_mpo",
    "shift_mpo",
    "sparse_localizer",
    "validate_clifford",
]
