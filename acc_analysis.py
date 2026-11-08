import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.io import wavfile

# 1. Load the WAV file and convert to float (prevents integer overflow)
sample_rate, data = wavfile.read("data_acc/s567217_20cm_dry.wav")
if len(data.shape) > 1:
  data = data[:, 0]
data = data.astype(float)

# 2. Define the time segment
start_time = 0.360
end_time = 0.566
start_sample = max(0, int(start_time * sample_rate))
end_sample = min(len(data), int(end_time * sample_rate))

# Slice the raw (unfiltered) segment
unfiltered_segment = data[start_sample:end_sample]
time = np.arange(len(unfiltered_segment)) / sample_rate


# 3. Define and apply High-Pass Filter (e.g., cutoff at 100 Hz)
def highpass_filter(signal_data, cutoff_freq, fs, order=4):
  nyquist = 0.5 * fs
  normal_cutoff = cutoff_freq / nyquist
  b, a = butter(order, normal_cutoff, btype="high", analog=False)
  return filtfilt(b, a, signal_data)


filtered_segment = highpass_filter(
    unfiltered_segment, cutoff_freq=100.0, fs=sample_rate
)

# 4. Calculate Metrics on Filtered Data
peak_accel = np.max(np.abs(filtered_segment))
total_energy = np.sum(filtered_segment**2) / sample_rate

# Envelope and Decay Time
analytic_signal = hilbert(filtered_segment)
envelope = np.abs(analytic_signal)
peak_idx = np.argmax(envelope)
peak_val = envelope[peak_idx]

decay_threshold = 0.1 * peak_val
post_peak_envelope = envelope[peak_idx:]
post_peak_time = time[peak_idx:] - time[peak_idx]

below_threshold = np.where(post_peak_envelope < decay_threshold)[0]
decay_idx_rel = (
    below_threshold[0] if len(below_threshold) > 0 else len(post_peak_time) - 1
)
decay_time = post_peak_time[decay_idx_rel]

# Absolute times (on the plot's time axis) for the markers
peak_time_abs = time[peak_idx]
decay_time_abs = time[peak_idx + decay_idx_rel]

# Spectral Centroid
fft_vals = np.fft.rfft(filtered_segment)
fft_freqs = np.fft.rfftfreq(len(filtered_segment), 1 / sample_rate)
power_spectrum = np.abs(fft_vals) ** 2
spectral_centroid = np.sum(fft_freqs * power_spectrum) / np.sum(power_spectrum)

print(f"--- Impact Metrics ---")
print(f"Peak Acceleration: {peak_accel:.4f}")
print(f"Total Energy: {total_energy:.4f}")
print(f"Time to decay (90% drop): {decay_time * 1000:.2f} ms")
print(f"Spectral Centroid: {spectral_centroid:.2f} Hz")

# 5. Plotting: Unfiltered vs. Filtered (Stacked Subplots)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Top Subplot: Unfiltered
ax1.plot(time, unfiltered_segment, color="gray", alpha=0.7, linewidth=0.8)
ax1.axvline(peak_time_abs, color="red", linestyle="-", linewidth=1.2,
            label=f"Peak ({peak_time_abs * 1000:.1f} ms)")
ax1.axvline(decay_time_abs, color="blue", linestyle="-", linewidth=1.2,
            label=f"10% cutoff ({decay_time_abs * 1000:.1f} ms)")
ax1.set_title("Unfiltered Signal (Contains Drift / Low-Freq Noise)")
ax1.set_ylabel("Amplitude")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.5)

# Bottom Subplot: Filtered
ax2.plot(
    time, filtered_segment, color="darkorange", alpha=0.8, linewidth=0.8
)
ax2.plot(time, envelope, color="black", linestyle="--", alpha=0.6, label="Envelope")
ax2.axvline(peak_time_abs, color="red", linestyle="-", linewidth=1.2,
            label=f"Peak ({peak_time_abs * 1000:.1f} ms)")
ax2.axvline(decay_time_abs, color="blue", linestyle="-", linewidth=1.2,
            label=f"10% cutoff ({decay_time_abs * 1000:.1f} ms)")
ax2.axhline(decay_threshold, color="blue", linestyle=":", linewidth=1,
            alpha=0.7, label="10% of peak envelope")
ax2.set_title("Filtered Signal (100 Hz High-Pass) with Envelope")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Amplitude")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()