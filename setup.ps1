# =============================================================================
# setup.ps1 -- environment bootstrap for the multivariable-pseudospectrum
#              package (Windows PowerShell twin of setup.sh).
#
# Usage (from this folder):
#   powershell -ExecutionPolicy Bypass -File setup.ps1            # core (NumPy)
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Sparse    # + SciPy sparse
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Tensor    # + SciPy/quimb
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -All       # both extras
# =============================================================================
param(
    [switch]$Sparse,   # install the SciPy sparse-matrix extra
    [switch]$Tensor,   # install the SciPy + quimb tensor-network extra
    [switch]$All       # install both optional extras
)

$ErrorActionPreference = "Stop"            # abort on the first error
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# --- resolve the optional extras ---------------------------------------------
$extras = @()
if ($All) {
    $extras = @("sparse", "tensor")
} else {
    if ($Sparse) { $extras += "sparse" }
    if ($Tensor) { $extras += "tensor" }
}

# --- find a Python >= 3.10 interpreter ---------------------------------------
$python = $null
foreach ($candidate in @("python3.12", "python3.11", "python3.10", "python", "py")) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        & $candidate -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $python = $candidate; break }
    }
}
if ($null -eq $python) { throw "No Python >= 3.10 interpreter found on PATH." }
Write-Host "Using interpreter: $(& $python --version) [$python]"

# --- create (or reuse) the virtual environment --------------------------------
$venv = Join-Path $Root ".venv"
if (-not (Test-Path $venv)) {
    Write-Host "Creating virtual environment at $venv"
    & $python -m venv $venv
}
$vpy = Join-Path $venv "Scripts\python.exe"

# --- install ------------------------------------------------------------------
Write-Host "Upgrading pip..."
& $vpy -m pip install --quiet --upgrade pip
if ($extras.Count -gt 0) {
    $spec = ".[{0}]" -f ($extras -join ",")
    Write-Host "Installing package with extras: [$($extras -join ',')] ..."
} else {
    $spec = "."
    Write-Host "Installing core package (NumPy only)..."
}
& $vpy -m pip install --quiet -e $spec

# --- smoke test ----------------------------------------------------------------
Write-Host "Running import smoke test..."
& $vpy -c @"
import numpy
from pseudospectrum import clifford_pseudovalue
from pseudospectrum.examples import universal_order_two_pair
ops = universal_order_two_pair(z=0.35)
mu_c = clifford_pseudovalue(ops, (0.25, -0.7), backend='dense')
print(f'multivariable-pseudospectrum ready (numpy {numpy.__version__}); sample mu_C = {mu_c:.6f}')
"@

Write-Host ""
Write-Host "Done. Activate the environment with: .venv\Scripts\Activate.ps1"
Write-Host "Run the tests with:  python -m unittest discover -s tests"
