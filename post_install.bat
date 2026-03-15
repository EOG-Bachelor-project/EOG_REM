@echo off
REM Run this after creating the conda environment to install GSSc from GitHub 
REM without letting pip override the pinned numpy version.

echo Installing GSSC from GitHub without dependencies...
pip install git+https://github.com/jshanna100/gssc.git --no-deps

echo Re-pinning numpy to stay below 2.0 (required for numba)...
pip install "numpy>=1.24,<2.0" --force-install

echo Done. Verifying versions:
python -c "import numpy; print(f'numpy:     {numpy.__version__}')"
python -c "import numba; print(f'numba:     {numba.__version__}')"
python -c "import gssc; print(f'gssc:     installed OK')"

pause