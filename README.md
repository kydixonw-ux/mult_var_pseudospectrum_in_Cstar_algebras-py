# Multivariable Pseudospectrum in C*-Algebras

This folder keeps the original MATLAB derivation/plotting scripts and adds a
production-oriented Python package for computing the multivariable
pseudospectra from `2402.15934v2.pdf`.

The main quantity is the Clifford spectral localizer

```text
L_lambda(A) = sum_j (A_j - lambda_j I) tensor Gamma_j
mu_C(lambda) = smallest singular value of L_lambda(A).
```

The package also includes the quadratic pseudospectrum

```text
Q_lambda(A) = sum_j (A_j - lambda_j I)^2
mu_Q(lambda) = sqrt(lambda_min(Q_lambda(A))).
```

## Install

One-shot bootstrap from this folder (creates `./.venv` and installs the
package in editable mode):

```bash
./setup.sh            # Linux / macOS / Git-Bash
```

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1     # Windows PowerShell
```

Add the optional extras with `--sparse`, `--tensor`, or `--all`
(`-Sparse` / `-Tensor` / `-All` for the PowerShell script).

Manual install (Python >= 3.10):

```bash
python -m venv .venv && . .venv/bin/activate     # .venv\Scripts\Activate.ps1 on Windows
python -m pip install -e .
```

The core package only requires NumPy. Optional extras:

```bash
python -m pip install -e .[sparse]    # SciPy sparse-matrix backends
python -m pip install -e .[tensor]    # SciPy + quimb tensor-network stack
```

`sparse` enables the SciPy sparse-matrix backends; `tensor` installs the
SciPy/quimb stack used by the optional large-MPO DMRG path.

## Test

```bash
python -m unittest discover -s tests
```

## Quick Start

Example dense calculation:

```python
from pseudospectrum import clifford_pseudovalue, quadratic_pseudovalue
from pseudospectrum.examples import universal_order_two_pair

ops = universal_order_two_pair(z=0.35)
point = (0.25, -0.7)

mu_c = clifford_pseudovalue(ops, point, backend="dense")
mu_q = quadratic_pseudovalue(ops, point, backend="dense")
```

Sparse matrices are supported with `backend="sparse"` when SciPy is installed.
Inputs are validated as Hermitian by default because the C*-algebra definitions
in the paper require Hermitian tuples. For a pure singular-value experiment on
non-Hermitian matrices, pass `assume_hermitian=False` to the Clifford dense or
sparse evaluator.

Tensor-network/MPO inputs use the convention that each MPO tensor has shape
`(left_bond, right_bond, physical_row, physical_col)`.
The Clifford spinor is appended as one extra MPO site, so the localizer can be
kept as an MPO:

```python
from pseudospectrum import clifford_pseudovalue_tensor, dense_matrix_to_mpo
from pseudospectrum.examples import universal_order_two_pair

u, v = universal_order_two_pair(z=0.2)
value, info = clifford_pseudovalue_tensor(
    [dense_matrix_to_mpo(u), dense_matrix_to_mpo(v)],
    (0.1, -0.3),
)
```

For small MPOs this contracts exactly with NumPy. For large MPOs,
`solver="dmrg"` bridges to an optional external DMRG driver -- any importable
module exposing a `DMRG` class, supplied via `dmrg_module_path` -- so the core
package itself needs only NumPy.

## API Map

- `dense_localizer`, `sparse_localizer`: build `L_lambda(A)`.
- `clifford_pseudovalue`: dispatch `mu_C(lambda)` to dense, sparse, or tensor.
- `quadratic_pseudovalue`: dispatch `mu_Q(lambda)` to dense, sparse, or tensor.
- `localizer_mpo`, `quadratic_mpo`: build MPO representations without
  densifying.
- `mpo_to_dense`: exact small-MPO contraction for validation and tests.
- `grid_values`, `epsilon_mask`: evaluate grid data and threshold an
  epsilon-pseudospectrum.

## Ported Examples

`pseudospectrum.examples` includes:

- the universal order-two unitary pair from Section 6,
- the projection-pair formulas from `Projection_calc.m`,
- finite dense and sparse truncations of the unilateral-shift triple from
  Section 7 / `one_shift.m` / `variable_shift.m`,
- the hemisphere polynomial and inequality function used in the shift-triple
  symbolic calculation.
