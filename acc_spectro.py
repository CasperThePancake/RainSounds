import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, welch


def inspect_spectrum(path, cutoff_hz=None, nperseg=512, fmin=10, fmax=None):
    """Waveform, spectrogram and PSD of the UNFILTERED file.

    cutoff_hz: draws a line at your high-pass cutoff for reference.
    nperseg:   spectrogram window length in samples. Smaller = better time
               resolution (use for short, fast decays), larger = better
               frequency resolution.
    """
    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    x = data.astype(float)
    x -= x.mean()  # remove DC offset
    t = np.arange(len(x)) / fs
    fmax = fmax or fs / 2

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Waveform
    ax1.plot(t, x, color="gray", linewidth=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"{os.path.basename(path)} (unfiltered, DC removed)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Spectrogram (log frequency axis)
    f, tt, S = spectrogram(x, fs, nperseg=nperseg, noverlap=int(nperseg * 0.9))
    S_db = 10 * np.log10(S + 1e-20)
    mesh = ax2.pcolormesh(tt, f, S_db, shading="auto", cmap="magma",
                          vmin=S_db.max() - 80, vmax=S_db.max())
    ax2.set_yscale("log")
    ax2.set_ylim(fmin, fmax)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_title("Spectrogram (dB)")
    fig.colorbar(mesh, ax=ax2, label="dB")
    if cutoff_hz:
        ax2.axhline(cutoff_hz, color="cyan", linestyle="--", linewidth=1)

    # PSD (log-log) - a peak shows as a bump, drift shows as a steady rise
    fw, P = welch(x, fs, nperseg=min(len(x), 4096))
    ax3.loglog(fw[1:], P[1:], color="darkorange")
    ax3.set_xlim(fmin, fmax)
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("PSD")
    ax3.set_title("Power spectral density")
    ax3.grid(True, which="both", linestyle="--", alpha=0.4)
    if cutoff_hz:
        ax3.axvline(cutoff_hz, color="blue", linestyle="--", linewidth=1,
                    label=f"cutoff {cutoff_hz:g} Hz")
        ax3.legend()

    plt.tight_layout()
    plt.show()


def compare_psd(paths, labels=None, nperseg=16384, fmin=10, fmax=None,
                window_s=None, normalize=False, markers=None):
    """Overlay the PSDs of two (or more) files to compare resonances.

    paths:     list of wav file paths
    labels:    legend labels (defaults to file names)
    nperseg:   Welch window length. Larger = finer frequency resolution
               (needed to tell resonance shifts apart); it is capped at the
               signal length.
    window_s:  if set, only analyse this many seconds starting 2 ms before
               the maximum of |signal| (i.e. the impact), instead of the
               whole file. Same window rule is applied to every file.
    normalize: scale each PSD to its own maximum, to compare peak positions
               and shapes regardless of absolute amplitude.
    markers:   optional list of frequencies (Hz) to mark with vertical lines.
    """
    labels = labels or [os.path.basename(p) for p in paths]

    fig, ax = plt.subplots(figsize=(14, 6))
    for path, label in zip(paths, labels):
        fs, data = wavfile.read(path)
        if data.ndim > 1:
            data = data[:, 0]
        x = data.astype(float)
        x -= x.mean()

        if window_s is not None:
            pk = np.argmax(np.abs(x))
            i0 = max(0, pk - int(0.002 * fs))
            x = x[i0:i0 + int(window_s * fs)]

        f, P = welch(x, fs, nperseg=min(len(x), nperseg))
        if normalize:
            P = P / P.max()
        ax.loglog(f[1:], P[1:], linewidth=1, alpha=0.8, label=label)

    if markers:
        for m in markers:
            ax.axvline(m, color="gray", linestyle=":", linewidth=1)

    ax.set_xlim(fmin, fmax or fs / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD" if normalize else "PSD")
    ax.set_title("PSD comparison (35cm)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(prop={'size': 12})
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    a = "data_acc/10nov/35cm/no_layer/35cm_1.wav"
    b = "data_acc/10nov/35cm/100microl/35cm_1.wav"
    c = "data_acc/10nov/35cm/50microl/35cm_1.wav"
    d = "data_acc/10nov/35cm/20microl/35cm_1.wav"

    # Overlay: no layer vs thickest layer, same height
    compare_psd([a, b, c, d], labels=["no layer", "100 µL", "50 µL", "20 µL"])

    # Single-file inspection (waveform + spectrogram + PSD)
    # inspect_spectrum(a, cutoff_hz=100)