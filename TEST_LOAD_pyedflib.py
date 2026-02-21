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
        sample_rates = [f.getSampleFrequencies(i) for i in range(n_channels)] # Get sample rates for each channel
        n_samples = f.getNSamples()                                           # Get number of samples in the EDF file

        # ---------------------------------------------------------------------
        # Summarize EDF file information
        # ---------------------------------------------------------------------
        print("\nEDF SUMMARY (pyEDFlib)")
        print("=" * 50)
        print(f"{'Number of Channels':<28} : {n_channels}")
        print(f"{'Samples per channel':<28} : {n_samples[0]}")
        print(f"{'Unique sampling rates':<28} : {sorted(set(sample_rates))}")
        print("=" * 50)

        print("\nCHANNEL LIST")
        print("-" * 50)
        for i, (lab, fs) in enumerate(zip(channel_names, sample_rates), start=1):
            print(f"{i:>3}. {lab:<25} ({fs} Hz)")
        print("=" * 50)

        # ---------------------------------------------------------------------
        # Load signals into a NumPy array
        # ---------------------------------------------------------------------
        signal_data = np.vstack([f.readSignal(i) for i in range(n_channels)]).T  # Shape: (n_samples, n_channels)
    
    return signal_data, channel_names, sample_rates

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Test
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
signals, channels, fs = load_edf_w_pyedflib(edf_file)
print("\nArray shape (samples, channels):", signals.shape)