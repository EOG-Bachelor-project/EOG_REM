# TEST_LOAD_pyedflib.py

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Imports
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
from __future__ import annotations
from pathlib import Path
import pyedflib
import numpy as np

# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
# Predefined variables
# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
file_path = Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_1_a")
edf_file = next(file_path.glob("*.edf"))

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Function
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

def load_edf_w_pyedflib(edf_path: Path):
    """
    Load an EDF file using pyedflib and return the signal data and channel names.

    Parameters
    ----------
    edf_path : Path
        The path to the EDF file to be loaded.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        A tuple containing the signal data as a NumPy array and the list of channel names.
    """

    print(f"Loading EDF file: {edf_path}")
    with pyedflib.EdfReader(str(edf_path)) as f:
        n_channels = f.signals_in_file                                        # Get number of channels in the EDF file     
        channel_names = f.getSignalLabels()                                   # Get channel names/labels
        sample_rates = f.getSampleFrequencies()                               # Get sample rates 
        n_samples = [f.getNSamples()[i] for i in range(n_channels)]           # Get number of samples in the EDF file

        # ---------------------------------------------------------------------
        # Summarize EDF file information
        # ---------------------------------------------------------------------
        print("\nEDF SUMMARY")
        print("=" * 60)
        print(f"{'Number of Channels':<30} : {n_channels}")
        print(f"{'Unique sampling rates':<30} : {sorted(set(sample_rates))}")
        print("=" * 60)

        print("\nCHANNEL LIST")
        print("-" * 60)
        for i, (lab, fs, n) in enumerate(zip(channel_names, sample_rates, n_samples), start=1):
            print(f"{i:>3}. {lab:<20} {fs:>8} Hz {n:>10} samples")
        print("=" * 60)

        # ---------------------------------------------------------------------
        # Load signals
        # ---------------------------------------------------------------------
        signal_data = {channel_names[i]: f.readSignal(i) for i in range(n_channels)}
    
    return signal_data, channel_names, sample_rates

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Test
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
signals, channels, fs = load_edf_w_pyedflib(edf_file)
print(signals)