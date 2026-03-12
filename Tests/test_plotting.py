import matplotlib.pyplot as plt
import os 
import numpy as np 
import mne 


def test_plot_mne(edf_file):

    raw = mne.io.read_raw_edf(edf_file, preload = True) 
    raw.plot()

test_plot_mne(r"C:\Users\rasmu\Desktop\6. Semester\Bachelor Projekt\Test edf filer\cfs-visit5-800331.edf")
