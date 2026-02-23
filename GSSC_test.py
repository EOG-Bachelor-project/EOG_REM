# GSSC_test.py

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Imports
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
from __future__ import annotations

import torch
import gssc.networks

torch.serialization.add_safe_globals([gssc.networks.ResSleep])

from gssc.infer import EEGInfer
from pathlib import Path
import mne



# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
# Predefined variables
# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠

file_path = Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_1_a")
edf_file = next(file_path.glob("*.edf"))
channels = ['EOGH-A1', 'EOGV-A2']

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Function
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

# GSSC test function
def test_GSSC(folder_path: str):
    """
    Load EDF file from the specified folder path, run the GSSC inference, and return the results as a DataFrame.

    Parameters
    ----------
    folder_path : str | Path
        The path to the folder containing the EDF file to be loaded.
    """

    # Load EDF signal via MNE
    raw = mne.io.read_raw_edf(folder_path, preload=False)

    # Make inferencer
    infer = EEGInfer()

    # Run inferencer
    hypnogram = infer.mne_infer(inst=raw)
    print(hypnogram)

    return hypnogram

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Test
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

test_GSSC(Path("L:\Auditdata\RBD PD\PD-RBD Glostrup Database_ok\DCSM_1_a\contiguous.edf"))

# OUTPUT
# (BPML) PS C:\Users\AKLO0022\EOG_REM> & C:/Users/AKLO0022/AppData/Local/anaconda3/envs/BPML/python.exe c:/Users/AKLO0022/EOG_REM/GSSC_test.py
#OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
#OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://openmp.llvm.org/