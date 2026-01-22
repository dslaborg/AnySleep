"""
Predict sleep stages from an EDF file and visualize the hypnogram.

This script demonstrates how to use the standalone AnySleep model to:
1. Load and preprocess EEG/EOG data from an EDF file
2. Run inference to get sleep stage predictions
3. Save raw logits and visualize the predicted hypnogram

The script handles the full preprocessing pipeline including:
- Channel selection
- Resampling to 128 Hz
- Noise clipping (20× IQR threshold)
- Robust scaling (median/IQR normalization)

Output:
    - prediction_{file_id}.npy: Raw logits array of shape (1, epochs, 5)
    - Matplotlib figure showing predicted hypnogram with probability overlay

(Edit the USER CONFIGURATION section in main() before running)
"""

from pathlib import Path

import mne
import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
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
    input_eeg = "path/to/file.edf"

    # Channel names to extract from the EDF file (must match exactly)
    # Set to None to use all available channels
    # Example channel names vary by dataset:
    #   - Sleep-EDF: "EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"
    input_channels = [
        "EEG Fpz-Cz",
        "EEG Pz-Oz",
        "EOG horizontal",
    ] or None  # None reads all channels

    # Device for inference: "cuda" for GPU, "cpu" for CPU
    device = "cuda"

    # Number of sleep stage predictions per 30-second epoch
    # 1 = standard 30s resolution, higher values for finer temporal resolution
    sleep_stage_frequency = 1

    # =========================================================================
    # END USER CONFIGURATION
    # =========================================================================

    resample_rate = 128  # Model expects 128 Hz input

    model = AnySleep(
        path="../models/anysleep-run1.pth", sleep_stage_frequency=sleep_stage_frequency
    )
    model.eval()
    model.to(device)

    f_id = input_eeg.split("/")[-1].split(".")[0]
    print(f"Processing {f_id}")
    with torch.no_grad():
        data = load_edf(input_eeg, input_channels, resample_rate)
        data = np.array(list(data.values())).T  # (time, channels)
        data = torch.from_numpy(data).float().unsqueeze(0).to(device)
        predict = model(data)
        np_predict = predict.detach().cpu().numpy()
        np.save(f"prediction_{f_id}.npy", np_predict)

    # plot hypnogram
    np_predict_prob = scipy.special.softmax(np_predict[0], axis=1)
    predicted_stages = np.argmax(np_predict[0], axis=1)

    stage_names = ["N3", "N2", "N1", "REM", "Wake"]

    # bring predicted_stages into correct order for plotting
    reorder_stage_map = {0: 4, 1: 2, 2: 1, 3: 0, 4: 3}
    predicted_stages = np.array([reorder_stage_map[s] for s in predicted_stages])

    plt.figure(figsize=(10, 5))
    plt.plot(predicted_stages, c="black")
    plt.yticks([0, 1, 2, 3, 4], stage_names)

    # add logits
    bottom = np.zeros(len(predicted_stages))
    for i in range(5):
        plt.bar(
            range(len(predicted_stages)),
            np_predict_prob[:, i] * 4,
            bottom=bottom,
            alpha=0.7,
            width=1,
            label=stage_names[reorder_stage_map[i]],
        )
        bottom += np_predict_prob[:, i] * 4

    plt.ylim(0, 4.1)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
