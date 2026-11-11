import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, hilbert

# ----------------------------------------------------------------------
# Global parameters
# ----------------------------------------------------------------------
HIGHPASS_HZ = 100.0          # high-pass cutoff used for ALL analyses
THRESHOLD_FRAC = 0.1       # decay threshold (fraction of envelope max)
BANDS = ((100, 500), (500, 2000), (2000, 8000))  # Hz, for band-energy fractions
FIGSIZE = (9, 6)
plt.rcParams["figure.figsize"] = FIGSIZE
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'lines.linewidth': 4,
    'axes.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
})

HEIGHTS = [35, 45, 55]     # cm
VOLUMES = ["no_layer", "5microl", "20microl", "50microl", "100microl"]
VOLUMES_NUMBER = [0, 5, 20, 50, 100]
N_REPEATS = 4
DATA_DIR = "data_acc/10nov"


def wav_path(height, volume, i):
    return f"{DATA_DIR}/{height}cm/{volume}/{height}cm_{i}.wav"


# ----------------------------------------------------------------------
# Single-file spot check with plot (decay time only)
# ----------------------------------------------------------------------
def decay_time_from_wav(path, plot=False, highpass_hz=HIGHPASS_HZ,
                        threshold_frac=THRESHOLD_FRAC):
    """Return the time (in seconds) from the envelope maximum to the first
    point where the envelope drops below `threshold_frac` of that maximum.

    Returns np.nan if the envelope never drops below the threshold
    after the peak.
    """
    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    raw = data.astype(float)
    time = np.arange(len(raw)) / fs

    b, a = butter(4, highpass_hz / (0.5 * fs), btype="high")
    filtered = filtfilt(b, a, raw)

    envelope = np.abs(hilbert(filtered))
    peak_idx = np.argmax(envelope)
    peak_val = envelope[peak_idx]
    threshold = threshold_frac * peak_val

    below = np.where(envelope[peak_idx:] < threshold)[0]
    if len(below) == 0:
        decay_idx = None
        decay_time = np.nan
    else:
        decay_idx = peak_idx + below[0]
        decay_time = time[decay_idx] - time[peak_idx]

    if plot:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time, filtered, color="darkorange", alpha=0.8, linewidth=0.8,
                label="Filtered signal")
        ax.plot(time, envelope, color="black", linestyle="--", alpha=0.7,
                label="Envelope")
        ax.axvline(time[peak_idx], color="red", linewidth=1.2,
                   label=f"Max ({time[peak_idx] * 1000:.1f} ms)")
        ax.plot(time[peak_idx], peak_val, "ro")
        ax.axhline(threshold, color="blue", linestyle=":", linewidth=1,
                   alpha=0.7, label=f"{threshold_frac:.0%} of max envelope")
        if decay_idx is not None:
            ax.axvline(time[decay_idx], color="blue", linewidth=1.2,
                       label=f"{threshold_frac:.0%} cutoff ({time[decay_idx] * 1000:.1f} ms)")
            ax.plot(time[decay_idx], envelope[decay_idx], "bo")
        ax.set_title(f"{os.path.basename(path)}  |  decay time = {decay_time * 1000:.2f} ms")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return decay_time


# ----------------------------------------------------------------------
# All features of one file
# ----------------------------------------------------------------------
def impact_features(path, highpass_hz=HIGHPASS_HZ, threshold_frac=THRESHOLD_FRAC,
                    bands=BANDS):
    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    x = data.astype(float)

    b, a = butter(4, highpass_hz / (fs / 2), btype="high")
    y = filtfilt(b, a, x)
    t = np.arange(len(y)) / fs

    env = np.abs(hilbert(y))
    env_s = uniform_filter1d(env, size=max(1, int(0.001 * fs)))  # ~1 ms smoothing
    pk = np.argmax(env_s)
    pk_val = env_s[pk]

    # Decay time + log-envelope decay rate
    below = np.where(env_s[pk:] < threshold_frac * pk_val)[0]
    if len(below):
        dk = pk + below[0]
        decay_time = t[dk] - t[pk]
        seg = slice(pk, max(dk, pk + 2))
        slope = np.polyfit(t[seg], np.log(env_s[seg] + 1e-12), 1)[0]
        decay_rate_db_per_ms = 20 * np.log10(np.e) * slope / 1000
    else:
        decay_time, decay_rate_db_per_ms = np.nan, np.nan

    # Rise time (last point below threshold before the peak -> peak)
    pre = np.where(env_s[:pk] < threshold_frac * pk_val)[0]
    rise_time = t[pk] - t[pre[-1]] if len(pre) else np.nan

    # Spectrum-based features
    power = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / fs)
    total = np.sum(power)
    centroid = np.sum(freqs * power) / total
    band_frac = {
        f"band_{lo}_{hi}": power[(freqs >= lo) & (freqs < min(hi, fs / 2))].sum() / total
        for lo, hi in bands
    }

    return {
        "decay_time": decay_time,
        "decay_rate_dB_per_ms": decay_rate_db_per_ms,
        "rise_time": rise_time,
        "peak_accel": np.max(np.abs(y)),
        "energy": np.sum(y ** 2) / fs,
        "spectral_centroid": centroid,
        "peak_freq": freqs[np.argmax(power)],
        **band_frac,
    }


# ----------------------------------------------------------------------
# Data processing: compute all features once per file
# ----------------------------------------------------------------------
# results[height][feature] = {"mean": [per volume], "std": [per volume]}
results = {}

for h in HEIGHTS:
    per_volume = []  # list (over volumes) of list (over repeats) of feature dicts
    for volume in VOLUMES:
        per_volume.append([
            impact_features(wav_path(h, volume, i))
            for i in range(1, N_REPEATS + 1)
        ])

    feature_names = per_volume[0][0].keys()
    results[h] = {}
    for feat in feature_names:
        vals = np.array([[d[feat] for d in reps] for reps in per_volume], dtype=float)
        # NaNs (e.g. envelope never decays below threshold) are ignored in the stats
        results[h][feat] = {
            "mean": np.nanmean(vals, axis=1),
            "std": np.nanstd(vals, axis=1),
        }

# ----------------------------------------------------------------------
# Plotting: two figures (line + errorbar) per feature, like before
# ----------------------------------------------------------------------
LABELS = {
    "decay_time": "Decay time (s)",
    "decay_rate_dB_per_ms": "Decay rate (dB/ms)",
    "rise_time": "Rise time (s)",
    "peak_accel": "Peak acceleration (a.u.)",
    "energy": "Total energy (a.u.)",
    "spectral_centroid": "Spectral centroid (Hz)",
    "peak_freq": "Dominant frequency (Hz)",
}


def pretty_label(feat):
    if feat in LABELS:
        return LABELS[feat]
    if feat.startswith("band_"):
        _, lo, hi = feat.split("_")
        return f"Energy fraction {lo}-{hi} Hz"
    return feat


def plot_feature(feat):
    label = pretty_label(feat)
    title = f"{label.split(' (')[0]} as function of water layer volume for different drop heights"
    title += f"\n(high-pass {HIGHPASS_HZ:g} Hz)"

    plt.figure()
    for h in HEIGHTS:
        plt.plot(VOLUMES_NUMBER, results[h][feat]["mean"], label=f"{h}cm")
    plt.title(title)
    plt.xlabel("Water layer volume (μL)")
    plt.ylabel(label)
    plt.legend()
    plt.tight_layout()

    plt.figure()
    for h in HEIGHTS:
        plt.errorbar(VOLUMES_NUMBER, results[h][feat]["mean"],
                     yerr=results[h][feat]["std"], fmt="o", capsize=3,
                     label=f"{h}cm")
    plt.title(title)
    plt.xlabel("Water layer volume (μL)")
    plt.ylabel(label)
    plt.legend()
    plt.tight_layout()


for feat in results[HEIGHTS[0]]:
    plot_feature(feat)

plt.show()  # one blocking call at the end shows all figures