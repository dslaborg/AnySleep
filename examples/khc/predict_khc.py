import glob
from pathlib import Path

import mne
import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.preprocessing import RobustScaler

from anysleep_no_hydra import AnySleep


def load_edf(input_file: str, channels_to_include, resample_rate):
    """
    Load and preprocess an EDF file for sleep staging.

    Args:
        input_file (str): Path to the EDF file.
        channels_to_include (list or None): List of channel names to load.
            If None, all channels are loaded.
        resample_rate (int): Target sampling rate in Hz.

    Returns:
        dict: Channel name → preprocessed signal array mapping.
    """
    input_file = Path(input_file)
    if channels_to_include is None:
        raw = mne.io.read_raw_edf(input_fname=input_file)
    else:
        raw = mne.io.read_raw_edf(input_fname=input_file, include=channels_to_include)
    print(f"Available channels: {raw.ch_names}")
    eeg = dict(zip(raw.ch_names, raw.get_data()))

    # eeg = {k: v * 1e6 for k, v in eeg.items()}  # Convert to microvolts
    data = preprocess(eeg, int(raw.info["sfreq"]), resample_rate)
    return data


def clip_noisy_values(psg, min_max_times_global_iqr=20):
    """
    Clip extreme values to reduce noise influence.

    Values beyond (IQR × threshold) are clipped to the threshold.
    This follows the preprocessing from U-Sleep (Perslev et al., 2021).

    Args:
        psg (np.ndarray): PSG data of shape (channels, time).
        min_max_times_global_iqr (int): Clipping threshold as multiple of IQR.

    Returns:
        np.ndarray: Clipped PSG data.
    """
    for chan in range(psg.shape[0]):
        chan_psg = psg[chan]

        # Compute global IQR
        iqr = np.subtract(*np.percentile(chan_psg, [75, 25]))
        threshold = iqr * min_max_times_global_iqr

        # Zero out noisy epochs in the particular channel
        psg[chan] = np.clip(chan_psg, -threshold, threshold)
    return psg


def preprocess(data, sample_rate, resample_rate):
    """
    Full preprocessing pipeline for EEG/EOG data.

    Steps:
        1. Resample to target rate (if different from source)
        2. Clip noisy values (20× IQR threshold)
        3. Robust scaling (subtract median, divide by IQR)

    Args:
        data (dict): Channel name → raw signal array mapping.
        sample_rate (int): Original sampling rate in Hz.
        resample_rate (int): Target sampling rate in Hz.

    Returns:
        dict: Channel name → preprocessed signal array mapping.
    """
    channels = list(data.keys())
    data = np.array(list(data.values()))  # (channels, time)

    # Set different sample rate of PSG?
    if sample_rate != resample_rate:
        data = resample_poly(data, int(resample_rate), int(sample_rate), axis=1)

    # Run over epochs and assess if epoch-specific changes should be
    # made to limit the influence of very high noise level epochs etc.
    data = clip_noisy_values(data, min_max_times_global_iqr=20)

    robust_scaler = RobustScaler()
    data = robust_scaler.fit_transform(data.T).T

    return {chan: data[i] for i, chan in enumerate(channels)}


def main():
    # =========================================================================
    # USER CONFIGURATION - Modify these variables before running
    # =========================================================================

    # Path to the input EDF file containing PSG data
    input_eegs = glob.glob("/mnt/data/KHC/*.edf")

    # Channel names to extract from the EDF file (must match exactly)
    # Set to None to use all available channels
    # Example channel names vary by dataset:
    #   - Sleep-EDF: "EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"
    input_channels = [
                         "C3-A2",
                         "C4-A1",
                         "EOG-L",
                         "EOG-R",
                         "O1-A2",
                         "O2-A1",
                         "A1",
                         "A2",
                         "C3",
                         "C4",
                         "EOG Left",
                         "EOG Right",
                         "O1",
                         "O2",
                     ] or None  # None reads all channels

    pool_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 640, 960, 1920, 3840]
    model_paths = [
        "../models/anysleep-run1.pth",
        "../models/anysleep-run2.pth",
        "../models/anysleep-run3.pth",
    ]

    # Device for inference: "cuda" for GPU, "cpu" for CPU
    device = "cuda"

    # =========================================================================
    # END USER CONFIGURATION
    # =========================================================================

    resample_rate = 128  # Model expects 128 Hz input

    results_to_save = {}

    for ps in pool_sizes:
        print(f"Processing pool size {ps}")

        for mp in model_paths:
            m_id = mp.split("/")[-1].split(".")[0]
            print(f"Processing {m_id}")

            model = AnySleep(
                path=mp,
                sleep_stage_frequency=ps,
            )
            model.eval()
            model.to(device)

            for input_eeg in input_eegs:
                f_id = input_eeg.split("/")[-1].split(".")[0]
                print(f"Processing {f_id}")

                with torch.no_grad():
                    data = load_edf(input_eeg, input_channels, resample_rate)
                    data = np.array(list(data.values())).T  # (time, channels)
                    data = torch.from_numpy(data).float().unsqueeze(0).to(device)
                    predict = model(data)
                    np_predict = predict.detach().cpu().numpy()
                    results_to_save[f"{f_id}#{m_id}#{ps}"] = np_predict[0].argmax(1)

    np.savez_compressed(f"output/KHC/predictions_c.npz", **results_to_save)


if __name__ == "__main__":
    main()
