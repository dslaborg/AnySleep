"""
Multi-panel figure for arousal detection analysis on MASS dataset.

This script creates a publication figure combining:
1. Arousal detection performance metrics at different temporal resolutions
2. Sleep stage distribution during detected arousals
3. Example recording visualization with EEG, spectrogram, and predictions

Figure Layout:
--------------
Top row (3 panels, example MASS recording):
    - Top: EEG spectrogram (0.5-25 Hz)
    - Middle: Raw EEG signal
    - Bottom: Hypnogram with:
        - Stacked bar chart of class probabilities
        - Ground truth (black) and predicted (red) sleep stages
        - Ground truth (green) and predicted (orange) arousal markers

Bottom row (2 panels):
    - Left: Sleep stage composition during arousals (Wake vs Non-Wake %)
        Shows what sleep stages the model predicts during arousal events
    - Right: Arousal detection metrics (IoU Precision, Recall, F1)
        Performance at 20% IoU threshold across temporal resolutions

Data Dependencies:
------------------
From scripts/arousals/:
    - arousals_aasm_merged.json: Predicted arousal events
    - arousals_iou_scores_aasm_merged.json: IoU-based evaluation metrics
    - mass_offsets.py: Timing offsets for annotation alignment

From hf_scores.py:
    - arousals_per_ss: Sleep stage distribution during arousals

External (user must configure paths in create_bottom_plot):
    - MASS dataset EDF files (PSG recordings and annotations)
    - High-frequency prediction sweep folder with logits

Output:
-------
    mass_arousal_logits_panels.svg

Note:
    Requires MASS dataset access. Update mass_base_path and sleep_stage_folder
    variables in create_bottom_plot() before running.
"""

import datetime
import glob
import json
from os.path import dirname
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from lspopt import spectrogram_lspopt
from matplotlib import patches
from matplotlib.colors import Normalize
from mne.filter import filter_data
from scipy.signal import resample_poly
from scipy.special import softmax
from sklearn.preprocessing import RobustScaler

from hf_scores import arousals_per_ss
from scripts.arousals.mass_offsets import offsets


def load_edf(input_file: str, channels_to_include, resample_rate, s_id):
    """
    Load and preprocess EDF file with timing offset correction.

    Applies the MASS-specific timing offset from mass_offsets.py to align
    the EEG signal with arousal annotations.

    Args:
        input_file: Path to EDF file.
        channels_to_include: List of channel names to load.
        resample_rate: Target sampling rate in Hz.
        s_id: Subject ID (e.g., "mass-c1_01-01-0001") for offset lookup.

    Returns:
        Tuple of (preprocessed_data_dict, recording_start_time).
    """
    input_file = Path(input_file)
    raw = mne.io.read_raw_edf(input_fname=input_file)
    print(f"Available channels: {raw.ch_names}")
    raw = mne.io.read_raw_edf(input_fname=input_file, include=channels_to_include)
    eeg = dict(zip(raw.ch_names, raw.get_data()))
    eeg = {
        k: v[int(offsets[s_id.split("_")[0]][s_id.split("_")[1]] * raw.info["sfreq"]) :]
        for k, v in eeg.items()
    }
    # eeg = {k: v * 1e6 for k, v in eeg.items()}  # Convert to microvolts
    data = preprocess(eeg, int(raw.info["sfreq"]), resample_rate)
    return data, raw.info["meas_date"]


def preprocess(data, sample_rate, resample_rate):
    channels = list(data.keys())
    data = np.array(list(data.values()))  # (channels, time)

    # Apply mne.mne.filter.filter_data function with specified settings
    filter_kwargs = {
        "l_freq": 0.3,
        "h_freq": 35,
        "method": "iir",
        "iir_params": {"order": 4, "ftype": "butter", "output": "sos"},
    }
    data = filter_data(data, sample_rate, **filter_kwargs)

    # Set different sample rate of PSG?
    if sample_rate != resample_rate:
        data = resample_poly(data, int(resample_rate), int(sample_rate), axis=1)

    # Run over epochs and assess if epoch-specific changes should be
    # made to limit the influence of very high noise level epochs etc.
    data = clip_noisy_values(data, min_max_times_global_iqr=20)

    robust_scaler = RobustScaler()
    data = robust_scaler.fit_transform(data.T).T

    return {chan: data[i] for i, chan in enumerate(channels)}


def clip_noisy_values(psg, min_max_times_global_iqr=20):
    for chan in range(psg.shape[0]):
        chan_psg = psg[chan]

        # Compute global IQR
        iqr = np.subtract(*np.percentile(chan_psg, [75, 25]))
        threshold = iqr * min_max_times_global_iqr

        # Zero out noisy epochs in the particular channel
        psg[chan] = np.clip(chan_psg, -threshold, threshold)
    return psg


def load_hf_sleep_stages(sweep_folder, ss_sr):
    labels_glob = f"{sweep_folder}/*/labels.npz"
    preds_glob = f"{sweep_folder}/*/predictions.npz"
    labels_files = sorted(glob.glob(labels_glob), key=lambda x: int(x.split("/")[-2]))
    preds_files = sorted(glob.glob(preds_glob), key=lambda x: int(x.split("/")[-2]))

    files_per_sr = {}
    for label_file, pred_file in zip(labels_files, preds_files):
        with open(dirname(label_file) + "/predict-high-freq_full_logits.log") as f:
            lines = f.readlines()
            sr_line = [line for line in lines if "sleep_stage_frequency" in line][0]
            sleep_stage_sr = int(sr_line.split("=")[1])
            model_line = [line for line in lines if "model.path" in line][0]
            model = model_line.split("=")[1].strip()
            if model != "train-usleep-2025-07-21_12-03-51-final.pth":
                continue
        files_per_sr[sleep_stage_sr] = (label_file, pred_file)

    labels = np.load(files_per_sr[ss_sr][0])
    preds = np.load(files_per_sr[ss_sr][1])
    return labels, preds


def load_arousals(ann_file, channel, s_id):
    # <Event channel="EEG C4-LER" groupName="MicroArousal" name="CARSM expert" scoringType="expert"/>
    data = mne.io.read_raw_edf(ann_file)
    arousals = [
        (ann["onset"], ann["onset"] + ann["duration"])
        for ann in data.annotations
        if f"{channel}" in ann["description"] and "MicroArousal" in ann["description"]
    ]
    arousals = np.array(arousals)
    arousals -= offsets[s_id.split("_")[0]][s_id.split("_")[1]]
    return arousals


def load_pred_arousals(s_id):
    with open("../arousals/arousals_aasm_merged.json") as f:
        data = json.load(f)
        models = [m for m in data.keys()]
        data = data[models[0]]["8"][s_id]
    return np.array(data)[:, :2].astype(float)


def calc_mode(data, axis=0):
    bins = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=5), axis=axis, arr=data
    )
    # resolve ties randomly by adding a small random number to the counts
    rand_mat = np.random.random_sample(bins.shape)
    bins = bins + rand_mat
    return np.argmax(bins, axis=axis)


def calc_bincount(data, axis=0, minlength=5):
    return np.apply_along_axis(
        lambda x: np.bincount(x, minlength=minlength), axis=axis, arr=data
    )


def plot_spectrogram(
    data,
    sf,
    win_sec=30,
    fmin=0.5,
    fmax=25,
    trimperc=2.5,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    fig=None,
    axis=None,
    t_offset=0,
):
    # Calculate multi-taper spectrogram
    nperseg = int(win_sec * sf)
    assert data.size > 2 * nperseg, "`data` length must be at least 2 * `win_sec`."
    f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=0)
    Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

    # Select only relevant frequencies (up to 30 Hz)
    good_freqs = np.logical_and(f >= fmin, f <= fmax)
    Sxx = Sxx[good_freqs, :]
    f = f[good_freqs]
    # t /= 3600  # Convert t to hours
    t += t_offset

    # Normalization
    if vmin is None:
        vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
        if axis is None:
            return vmin, vmax
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Open figure
    if axis is not None:
        ax1 = axis
    else:
        fig, ax1 = plt.subplots(nrows=1, figsize=(12, 4))

    # Draw Spectrogram
    im = ax1.pcolormesh(
        t, f, Sxx, norm=norm, cmap=cmap, antialiased=True, shading="auto"
    )
    ax1.set_xlim(t_offset, t.max())
    ax1.set_ylabel("Frequency [Hz]")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax1, shrink=0.95, fraction=0.1, aspect=25)
    cbar.ax.set_ylabel("Log power (dB / Hz)", rotation=270, labelpad=20)

    return vmin, vmax


def create_bottom_plot(topfig):
    """
    Create the metrics panels (arousal detection performance and stage distribution).

    Left panel: Sleep stage distribution during detected arousals
        - Wake vs Non-Wake percentage at each temporal resolution
        - Shows how arousal classification changes with finer predictions

    Right panel: Arousal detection IoU metrics
        - Precision, Recall, F1 at 20% IoU threshold
        - X-axis: temporal resolution (seconds per prediction)
    """
    top_axs = topfig.subplots(1, 2)
    top_axs = top_axs[::-1]

    # panel arousal prediction scores
    with open("../arousals/arousals_iou_scores_aasm_merged.json", "r") as f:
        scores = json.load(f, parse_float=float)
    ovt_idx = 2  # 20%
    f1s, precs, recs = [], [], []
    for model in scores.keys():
        f1s_m, precs_m, recs_m = [], [], []
        for ss_sr in scores[model].keys():
            f1s_m.append(
                [
                    s_id_dict["f1"][ovt_idx]
                    for _, s_id_dict in scores[model][ss_sr].items()
                ]
            )
            precs_m.append(
                [
                    s_id_dict["prec"][ovt_idx]
                    for _, s_id_dict in scores[model][ss_sr].items()
                ]
            )
            recs_m.append(
                [
                    s_id_dict["rec"][ovt_idx]
                    for _, s_id_dict in scores[model][ss_sr].items()
                ]
            )
        f1s.append(f1s_m)
        precs.append(precs_m)
        recs.append(recs_m)
    print(f1s)
    f1_array = np.array(f1s).mean(axis=0)  # shape (n_models, n_srs, n_subjects)
    prec_array = np.array(precs).mean(axis=0)
    rec_array = np.array(recs).mean(axis=0)

    srs = sorted(arousals_per_ss[0].keys())
    srs_axis = [30 / s for s in sorted(arousals_per_ss[0].keys())]
    n_ssrs = 7
    print(f"F1: {f1_array.mean(axis=1)[:n_ssrs]}")
    top_axs[0].plot(
        srs_axis[:n_ssrs],
        f1_array.mean(axis=1)[:n_ssrs],
        label="IoU F1",
        marker="^",
        c="#13274f",
    )
    top_axs[0].fill_between(
        srs_axis[:n_ssrs],
        f1_array.mean(axis=1)[:n_ssrs] - f1_array.std(axis=1)[:n_ssrs],
        f1_array.mean(axis=1)[:n_ssrs] + f1_array.std(axis=1)[:n_ssrs],
        alpha=0.2,
        color="#13274f",
    )
    print(f"Precision: {prec_array.mean(axis=1)[:n_ssrs]}")
    top_axs[0].plot(
        srs_axis[:n_ssrs],
        prec_array.mean(axis=1)[:n_ssrs],
        label="IoU Precision",
        marker="s",
        c="#ce1141",
    )
    print(f"Recall: {rec_array.mean(axis=1)[:n_ssrs]}")
    top_axs[0].plot(
        srs_axis[:n_ssrs],
        rec_array.mean(axis=1)[:n_ssrs],
        label="IoU Recall",
        marker="o",
        c="darkorange",
    )

    top_axs[0].set_xlabel("Time scale (s)")
    top_axs[0].set_ylabel("Metric")
    top_axs[0].set_xscale("log")
    top_axs[0].legend(loc="lower right")
    top_axs[0].grid()
    top_axs[0].set_xticks([1, 10, 30], [1, 10, 30])

    colors = ["#13274f", "darkorange", "#2ca02c", "#d62728", "#9467bd"]
    means = np.array([np.mean([hfdm[sr][0] for hfdm in arousals_per_ss]) for sr in srs])
    print(f"Wake: {means}")
    stds = np.array([np.std([hfdm[sr][0] for hfdm in arousals_per_ss]) for sr in srs])
    top_axs[1].plot(
        srs_axis,
        means,
        label=f"Wake",
        color=colors[0],
        marker="^",
    )
    top_axs[1].fill_between(
        srs_axis, means - stds, means + stds, alpha=0.2, color=colors[0]
    )
    means = np.array(
        [
            np.mean(
                [sum([hfdm[sr][i] for i in range(1, 5)]) for hfdm in arousals_per_ss]
            )
            for sr in srs
        ]
    )
    print(f"Non-Wake: {means}")
    stds = np.array(
        [
            np.std(
                [sum([hfdm[sr][i] for i in range(1, 5)]) for hfdm in arousals_per_ss]
            )
            for sr in srs
        ]
    )
    top_axs[1].plot(
        srs_axis,
        means,
        label=f"Non-Wake",
        color=colors[1],
        # ls="--",
        marker="o",
    )
    top_axs[1].fill_between(
        srs_axis, means - stds, means + stds, alpha=0.2, color=colors[1]
    )
    top_axs[1].legend()
    top_axs[1].grid()
    top_axs[1].set_xlabel("Time scale (s)")
    top_axs[1].set_ylabel("Arousal duration (% total arousal time)")
    top_axs[1].set_xscale("log")
    top_axs[1].set_xticks([1e-2, 1e-1, 1, 10, 30], [1e-2, 1e-1, 1, 10, 30])

    for ax in top_axs:
        # flip x axis
        ax.invert_xaxis()
        ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            top=False,
            left=True,
        )


def create_top_plot(bottomfig):
    """
    Create the example recording visualization panels.

    Shows a 3-minute window (WINDOW_SIZE=180s) from a MASS recording with:
        - Top: Multi-taper spectrogram (0.5-25 Hz)
        - Middle: Raw EEG signal (C4-CLE channel)
        - Bottom: Hypnogram with class probabilities, ground truth and
                  predicted sleep stages, and arousal markers

    Note:
        Requires MASS dataset. Update mass_base_path and sleep_stage_folder
        paths before running.
    """
    bottom_axs = bottomfig.subplots(3, 1, sharex=True)

    s_id = "mass-c1_01-01-0001"
    ss_sr = 8
    current_position = 4050
    # TODO: you need to specify these, as we are not allowed to share the MASS data
    mass_base_path = "path/to/MASS"
    eeg_file = f"{mass_base_path}/{s_id[s_id.find('_')+1:]} PSG.edf"
    sleep_stage_folder = "path/to/mass/logits"
    arousal_file = f"{mass_base_path}/{s_id[s_id.find('_')+1:]} Annotations.edf"
    eeg_data, start_time = load_edf(eeg_file, ["EEG C4-CLE"], 128, s_id)

    pred_sleep_stages = load_hf_sleep_stages(sleep_stage_folder, ss_sr)[1]
    pred_subj = s_id.replace("_", "#")
    pred_sleep_stages = [
        v for k, v in pred_sleep_stages.items() if k.startswith(pred_subj)
    ][0]
    true_sleep_stages = load_hf_sleep_stages(sleep_stage_folder, ss_sr)[0][s_id]

    arousals = load_arousals(arousal_file, "EEG C4-LER", s_id)
    pred_arousals = load_pred_arousals(s_id)

    sfreq = 128
    stage_map = {0: 4, 1: 2, 2: 1, 3: 0, 4: 3, 8: 5, 9: 5}

    start_sample = int(current_position * sfreq)
    end_sample = int(start_sample + WINDOW_SIZE * sfreq)
    eeg_channel_data = next(iter(eeg_data.values()))
    eeg_window = eeg_channel_data[start_sample:end_sample]
    time_axis_eeg = np.arange(len(eeg_window)) / sfreq + current_position

    # update spectrogram
    min_eeg = np.min(eeg_channel_data)
    max_eeg = np.max(eeg_channel_data)
    spec_vmin, spec_vmax = plot_spectrogram(eeg_channel_data, sfreq)
    plot_spectrogram(
        eeg_window,
        sfreq,
        win_sec=1,
        vmin=spec_vmin,
        vmax=spec_vmax,
        fig=bottomfig,
        axis=bottom_axs[0],
        t_offset=current_position,
    )

    bottom_axs[1].plot(time_axis_eeg, eeg_window, linewidth=0.5)
    bottom_axs[1].set_xlim([current_position, current_position + WINDOW_SIZE])
    bottom_axs[1].set_ylabel(f"EEG data")
    bottom_axs[1].set_ylim([min_eeg, max_eeg])
    bottom_axs[1].grid()

    # update sleep stage plot
    start_ss = int(current_position / 30)
    end_ss = int(start_ss + WINDOW_SIZE / 30)
    true_ss_window = true_sleep_stages[start_ss:end_ss]
    start_ss = int(current_position / 30 * ss_sr)
    end_ss = int(start_ss + WINDOW_SIZE / 30 * ss_sr)
    ss_window = pred_sleep_stages[start_ss:end_ss]

    time_axis_ss = np.arange(len(ss_window) + 1) * 30 / ss_sr + current_position
    bottom = np.zeros(len(time_axis_ss) - 1)
    ss_window_prob = softmax(ss_window, axis=-1) * 4
    logit_bars = []
    logit_bar_labels = []
    for i in [3, 2, 1, 4, 0]:
        label = {
            0: "Wake",
            1: "N1",
            2: "N2",
            3: "N3",
            4: "REM",
        }[i]
        logit_bar_labels.append(label)
        width = 30 / ss_sr
        bar = bottom_axs[2].bar(
            time_axis_ss[:-1],
            ss_window_prob[:, i],  # Probabilities for class i
            width=width,
            align="edge",
            bottom=bottom,
            alpha=0.4,
        )
        logit_bars.append(bar)
        bottom += ss_window_prob[:, i]

    ss_window = ss_window.argmax(axis=-1)
    ss_window = np.insert(ss_window, -1, ss_window[-1])
    ss_window = np.array([stage_map[s] for s in ss_window.tolist()])
    (pred_hyp_line,) = bottom_axs[2].plot(
        time_axis_ss,
        ss_window + 0.07,
        drawstyle="steps-post",
        color="red",
    )
    time_axis_ss = np.arange(len(true_ss_window) + 1) * 30 + current_position
    true_ss_window = np.insert(true_ss_window, -1, true_ss_window[-1])
    true_ss_window = np.array([stage_map[s] for s in true_ss_window.tolist()])
    (gt_hyp_line,) = bottom_axs[2].plot(
        time_axis_ss,
        true_ss_window + 0.07,
        drawstyle="steps-post",
        color="black",
    )

    # add highlighted areas for arousals
    arousals = arousals[
        (arousals[:, 0] >= current_position)
        & (arousals[:, 1] <= current_position + WINDOW_SIZE)
    ]
    for arousal in arousals:
        gt_ar_patch = bottom_axs[2].add_patch(
            patches.Rectangle(
                (arousal[0], 4.5),
                arousal[1] - arousal[0],
                0.2,
                linewidth=0,
                edgecolor="green",
                facecolor="green",
            )
        )

    # add highlighted areas for predicted arousals
    arousals = pred_arousals[
        (pred_arousals[:, 0] >= current_position)
        & (pred_arousals[:, 1] <= current_position + WINDOW_SIZE)
    ]
    for arousal in arousals:
        pred_ar_patch = bottom_axs[2].add_patch(
            patches.Rectangle(
                (arousal[0], 4.8),
                arousal[1] - arousal[0],
                0.2,
                linewidth=0,
                edgecolor="darkorange",
                facecolor="darkorange",
            )
        )

    bottom_axs[2].set_xlabel("Time (HH:MM:SS)")
    bottom_axs[2].set_ylabel(f"Annotations")
    bottom_axs[2].set_xlim([current_position, current_position + WINDOW_SIZE])
    bottom_axs[2].set_ylim([0, 5.5])
    bottom_axs[2].set_xticks(
        np.arange(current_position, current_position + WINDOW_SIZE + 1, 30)
    )
    bottom_axs[2].set_xticklabels(
        [
            (start_time + datetime.timedelta(seconds=int(t))).strftime("%H:%M:%S")
            for t in np.arange(current_position, current_position + WINDOW_SIZE + 1, 30)
        ]
    )
    bottom_axs[2].set_yticks([0, 1, 2, 3, 4])
    bottom_axs[2].set_yticklabels(["N3", "N2", "N1", "REM", "WAKE"])
    bottom_axs[2].grid(axis="x")

    legend1 = plt.legend(
        logit_bars[::-1],
        logit_bar_labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Class\nprobabilities",
        alignment="left",
    )
    legend2 = plt.legend(
        [gt_ar_patch, pred_ar_patch, gt_hyp_line, pred_hyp_line],
        [
            "Ground truth arousal",
            "Predicted arousal",
            "Ground truth hypnogram",
            "Predicted hypnogram",
        ],
        loc="center left",
        bbox_to_anchor=(1.1, 0.5),
    )
    bottom_axs[2].add_artist(legend1)
    bottom_axs[2].add_artist(legend2)


def main():
    fig = plt.figure(figsize=(15, 10))
    (topfig, bottomfig) = fig.subfigures(2, 1)

    create_bottom_plot(bottomfig)
    create_top_plot(topfig)

    # increase wspace between upper subplots
    topfig.subplots_adjust(bottom=0.15, top=0.9)
    bottomfig.subplots_adjust(wspace=0.3, bottom=0.15, top=0.7)

    plt.savefig("mass_arousal_logits_panels.svg")

    plt.show()


if __name__ == "__main__":
    WINDOW_SIZE = 90 * 2

    main()
