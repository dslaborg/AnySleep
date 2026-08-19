import glob
from pathlib import Path

import mne
import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.preprocessing import RobustScaler

from usleep_no_hydra import USleep


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


def calc_mode(data, axis=0):
    """
    Compute the mode (most frequent value) along an axis with random tie-breaking.

    This function is used for majority voting across channel predictions.
    When multiple values have the same count (a tie), one is chosen randomly.
    This random tie-breaking ensures unbiased consensus in case of disagreement.

    Args:
        data (np.ndarray): Input array of integer predictions.
            Values should be in range [0, n_stages-1] where n_stages is
            configured in cfg.data.stages.
        axis (int): Axis along which to compute the mode. Defaults to 0.

    Returns:
        np.ndarray: Array of mode values with the specified axis removed.

    Example:
        >>> # Three channels predicting sleep stages for 4 epochs
        >>> predictions = np.array([
        ...     [0, 2, 2, 3],  # Channel 1
        ...     [0, 2, 1, 3],  # Channel 2
        ...     [1, 2, 2, 3],  # Channel 3
        ... ])
        >>> consensus = calc_mode(predictions, axis=0)
        >>> # Result: [0, 2, 2, 3] - majority vote per epoch
    """
    n_stages = 5
    bins = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=n_stages), axis=axis, arr=data
    )
    # resolve ties randomly by adding a small random number to the counts
    rand_mat = np.random.random_sample(bins.shape)
    bins = bins + rand_mat
    return np.argmax(bins, axis=axis)


def main():
    # =========================================================================
    # USER CONFIGURATION - Modify these variables before running
    # =========================================================================

    # Path to the input EDF file containing PSG data
    input_eegs = glob.glob("/mnt/data/anphy/*/*.edf")

    # Channel names to extract from the EDF file (must match exactly)
    # Set to None to use all available channels
    # Example channel names vary by dataset:
    #   - Sleep-EDF: "EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"
    input_channels_eeg = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "F4-",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "FZ",
        "CZ",
        "PZ",
        # "SO1",
        # "SO2",
        "F9",
        "F10",
        # "ZY1",
        # "ZY2",
        "T9",
        "T10",
        "P9",
        "P10",
        "AF7",
        "AF3",
        "F11",
        "F5",
        "F1",
        "FT11",
        "FT9",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCZ",
        "C5",
        "C1",
        "C1-",
        "TP11",
        "TP9",
        "TP7",
        "CP3",
        "CP1",
        "P11",
        "P5",
        "P1",
        "PO7",
        "PO3",
        "POZ",
        "OZ",
        "FPZ",
        "AFZ",
        "AF4",
        "AF8",
        "F2",
        "F6",
        "F12",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "FT10",
        "FT12",
        "C6",
        "C2",
        "CPZ",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "TP10",
        "TP12",
        "P2",
        "P6",
        "P12",
        "PO4",
        "PO8",
        "CP5",
        "Fp1-Ref",
        "Fp2-Ref",
        "F3-Ref",
        "F4-Ref",
        "C3-Ref",
        "C4-Ref",
        "P3-Ref",
        "P4-Ref",
        "O1-Ref",
        "O2-Ref",
        "F7-Ref",
        "F8-Ref",
        "T3-Ref",
        "T4-Ref",
        "T5-Ref",
        "T6-Ref",
        "FZ-Ref",
        "CZ-Ref",
        "PZ-Ref",
        # "SO1-Ref",
        # "SO2-Ref",
        "F9-Ref",
        "F10-Ref",
        # "ZY1",
        # "ZY2",
        "T9-Ref",
        "T10-Ref",
        "P9-Ref",
        "P10-Ref",
        "AF7-Ref",
        "AF3-Ref",
        "F11",
        "F5-Ref",
        "F1-Ref",
        "FT11",
        "FT9-Ref",
        "FT7-Ref",
        "FC5-Ref",
        "FC3-Ref",
        "FC1-Ref",
        "FCZ-Ref",
        "C5-Ref",
        "C1-Ref",
        "TP11",
        "TP9-Ref",
        "TP7-Ref",
        "CP3-Ref",
        "CP1-Ref",
        "P11",
        "P5-Ref",
        "P1-Ref",
        "PO7-Ref",
        "PO3-Ref",
        "POZ-Ref",
        "OZ-Ref",
        "FPZ-Ref",
        "AFZ-Ref",
        "AF4-Ref",
        "AF8-Ref",
        "F2-Ref",
        "F6-Ref",
        "F12",
        "FC2-Ref",
        "FC4-Ref",
        "FC6-Ref",
        "FT8-Ref",
        "FT10-Ref",
        "FT12",
        "C6-Ref",
        "C2-Ref",
        "CPZ-Ref",
        "CP2-Ref",
        "CP4-Ref",
        "CP6-Ref",
        "TP8-Ref",
        "TP10-Ref",
        "TP12",
        "P2-Ref",
        "P6-Ref",
        "P12",
        "PO4-Ref",
        "PO8-Ref",
    ]
    input_channels_eog = ["EOG2", "EOG1"]

    pool_sizes = [1]
    model_paths = [
        "../models/usleep-run1.pth",
        "../models/usleep-run2.pth",
        "../models/usleep-run3.pth",
    ]

    # Device for inference: "cuda" for GPU, "cpu" for CPU
    device = "cuda"

    # =========================================================================
    # END USER CONFIGURATION
    # =========================================================================

    resample_rate = 128  # Model expects 128 Hz input

    results_to_save = {}

    model_cache = {}

    for input_eeg in input_eegs:
        f_id = input_eeg.split("/")[-1].split(".")[0]
        print(f"Processing {f_id}")
        data = load_edf(
            input_eeg, input_channels_eeg + input_channels_eog, resample_rate
        )
        # data_eeg = np.array([v for k,v in data.items() if k in input_channels_eeg]).T  # (time, channels)
        # data_eog = np.array([v for k,v in data.items() if k in input_channels_eog]).T
        # data_eeg = torch.from_numpy(data).float().unsqueeze(0).to(device)
        available_eeg_ch = [k for k in data.keys() if k in input_channels_eeg]
        available_eog_ch = [k for k in data.keys() if k in input_channels_eog]
        eeg_eog_pairs = [
            (eeg, eog) for eeg in available_eeg_ch for eog in available_eog_ch
        ]
        print(len(eeg_eog_pairs))

        batch_size = 8

        for ps in pool_sizes:
            print(f"Processing pool size {ps}")
            if ps not in model_cache:
                model_cache[ps] = {}

            for mp in model_paths:
                m_id = mp.split("/")[-1].split(".")[0]
                print(f"Processing {m_id}")
                if m_id in model_cache[ps]:
                    model = model_cache[ps][m_id]
                else:
                    model = USleep(
                        path=mp,
                        sleep_stage_frequency=ps,
                    )
                    model_cache[ps][m_id] = model
                model.eval()
                model.to(device)

                pred_to_vote = []
                for ch_idx in range(0, len(eeg_eog_pairs), batch_size):
                    data_batch = np.array(
                        [
                            [data[eeg_ch], data[eog_ch]]
                            for eeg_ch, eog_ch in eeg_eog_pairs[
                            ch_idx: ch_idx + batch_size
                        ]
                        ]
                    )
                    data_batch = data_batch.transpose(0, 2, 1)
                    data_batch = torch.from_numpy(data_batch).float().to(device)

                    with torch.no_grad():
                        predict = model(data_batch)
                        np_predict = predict.detach().cpu().numpy()
                        pred_to_vote.append(np_predict.argmax(-1))

                model.to("cpu")

                majority_vote = calc_mode(np.concat(pred_to_vote, axis=0), axis=0)
                results_to_save[f"{f_id}#{m_id}#{ps}"] = majority_vote

    np.savez_compressed(f"output/anphy/predictions_usleep_c.npz", **results_to_save)


if __name__ == "__main__":
    main()
