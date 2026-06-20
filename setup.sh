#!/usr/bin/env bash
# =============================================================================
# setup.sh -- environment bootstrap for the multivariable-pseudospectrum package.
#
# Usage (from this folder):
#   ./setup.sh            # create ./.venv and install the core package (NumPy)
#   ./setup.sh --sparse   # also install the SciPy sparse extra
#   ./setup.sh --tensor   # also install the SciPy + quimb tensor-network extra
#   ./setup.sh --all      # install both optional extras
#
# The script is idempotent: re-running it updates the existing environment.
# Works on Linux, macOS, and Windows (Git Bash / MSYS / WSL).
# =============================================================================

set -euo pipefail   # abort on first error, undefined variable, or pipe failure

# --- locate the project root (directory containing this script) --------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- parse the optional extras flag ------------------------------------------
EXTRAS=""                             # comma-separated pip extras, e.g. "sparse"
for arg in "$@"; do
    case "$arg" in
        --sparse) EXTRAS="sparse" ;;
        --tensor) EXTRAS="tensor" ;;
        --all)    EXTRAS="sparse,tensor" ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

# --- find a Python 3.10+ interpreter ------------------------------------------
# Try the common launcher names in order and keep the first that satisfies the
# version requirement of pyproject.toml (>=3.10).
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3 python py; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: no Python >= 3.10 interpreter found on PATH." >&2
    exit 1
fi
echo "Using interpreter: $($PYTHON --version) [$PYTHON]"

# --- create (or reuse) the virtual environment --------------------------------
VENV="$ROOT/.venv"
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV"
    "$PYTHON" -m venv "$VENV"
fi

# venv layout differs between POSIX (bin/) and Windows (Scripts/).
if [ -x "$VENV/bin/python" ]; then
    VPY="$VENV/bin/python"
else
    VPY="$VENV/Scripts/python.exe"
fi

# --- install ------------------------------------------------------------------
echo "Upgrading pip..."
"$VPY" -m pip install --quiet --upgrade pip

if [ -n "$EXTRAS" ]; then
    echo "Installing package with extras: [$EXTRAS] ..."
    "$VPY" -m pip install --quiet -e ".[$EXTRAS]"
else
    echo "Installing core package (NumPy only)..."
    "$VPY" -m pip install --quiet -e .
fi

# --- smoke test ----------------------------------------------------------------
echo "Running import smoke test..."
"$VPY" - <<'EOF'
import numpy
from pseudospectrum import clifford_pseudovalue
from pseudospectrum.examples import universal_order_two_pair
ops = universal_order_two_pair(z=0.35)
mu_c = clifford_pseudovalue(ops, (0.25, -0.7), backend="dense")
print(f"multivariable-pseudospectrum ready (numpy {numpy.__version__}); "
      f"sample mu_C = {mu_c:.6f}")
EOF

echo
echo "Done. Activate the environment with:"
echo "  source .venv/bin/activate          (Linux / macOS)"
echo "  .venv\\Scripts\\Activate.ps1         (Windows PowerShell)"
echo "Run the tests with:  python -m unittest discover -s tests"
