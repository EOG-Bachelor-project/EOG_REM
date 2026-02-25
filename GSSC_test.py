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
(BPML) PS C:\Users\AKLO0022\EOG_REM> python -c "import numpy; print('numpy ok')"
(BPML) PS C:\Users\AKLO0022\EOG_REM> python -X faulthandler -c "import numpy; print('numpy ok')" 2>&1 | % { $_ }
python : Windows fatal exception: code 0xc06d007e
At line:1 char:1
+ python -X faulthandler -c "import numpy; print('numpy ok')" 2>&1 | %  ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (Windows fatal exception: code 0xc06d007e:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
    + CategoryInfo          : NotSpecified: (Windows fatal exception: code 0xc06d007e:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError

    + CategoryInfo          : NotSpecified: (Windows fatal exception: code 0xc06d007e:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
    + CategoryInfo          : NotSpecified: (Windows fatal exception: code 0xc06d007e:String) [], RemoteException
    + CategoryInfo          : NotSpecified: (Windows fatal exception: code 0xc06d007e:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError

Current thread 0x00002314 (most recent call first):
  File "C:\Users\AKLO0022\AppData\Local\anaconda3\envs\BPML\Lib\site-packages\numpy\__init__.py", line 881 in blas_fpe_check
  File "C:\Users\AKLO0022\AppData\Local\anaconda3\envs\BPML\Lib\site-packages\numpy\__init__.py", line 890 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 940 in exec_module
  File "<frozen importlib._bootstrap>", line 690 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1147 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1176 in _find_and_load
  File "<string>", line 1 in <module>