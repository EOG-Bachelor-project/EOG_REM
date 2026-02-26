import glob
import mne
import numpy as np
from os import path
from ..infer import EEGInfer
from ..utils import inst_load, output_stages, pick_chan_names, graph_summary

def main():

    # -------- ONLY EDIT THIS --------
    data_folder = r"C:\full\path\to\your\folder"  # <<< CHANGE
    # --------------------------------

    # Find EDF files automatically
    edf_files = glob.glob(path.join(data_folder, "*.edf"))

    if len(edf_files) == 0:
        raise FileNotFoundError("No EDF file found in folder")

    # Use first EDF found
    file_path = edf_files[0]

    opt = {
        "file": file_path,
        "out_form": "csv",
        "cut_from": "back",
        "use_cuda": True,
        "chunk_size": 0,
        "eeg": ["eeg", "none"],  # disable EEG
        "eog": ["eog", "EOGH-A1", "EOGHV-A2"],  # only these channels
        "drop_eeg": True,
        "drop_eog": True,
        "filter": True,
        "graph": False
    }

    inst = inst_load(opt["file"])

    eeginfer = EEGInfer(cut=opt["cut_from"], use_cuda=opt["use_cuda"],
                        chunk_n=opt["chunk_size"])

    if len(opt["eeg"]) == 1:
        eeg = "eeg"
    elif opt["eeg"][1] == "none":
        eeg = []
    else:
        eeg = opt["eeg"][1:]

    if len(opt["eog"]) == 1:
        eog = "eog"
    elif opt["eog"][1] == "none":
        eog = []
    else:
        eog = opt["eog"][1:]

    stages, times, probs = eeginfer.mne_infer(
        inst,
        eeg=eeg,
        eog=eog,
        eeg_drop=opt["drop_eeg"],
        eog_drop=opt["drop_eog"],
        filter=opt["filter"]
    )

    filepath, filename = path.split(opt["file"])
    fileroot, fileext = path.splitext(filename)

    output_stages(stages, times, probs, opt["out_form"], filepath, fileroot)

    if opt["graph"]:
        eegs = pick_chan_names(inst, "eeg") if isinstance(eeg, str) else eeg
        graph_summary(stages, times, inst, eegs, filepath, fileroot)


if __name__ == "__main__":
    main()