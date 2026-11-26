import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from models import solve_raindrop_membrane, OutputType

def wav_amplitude_spectrum(wav_path):
    """Loads a .wav file and returns single-sided amplitude spectrum."""
    sr, data = wavfile.read(wav_path)

    # Convert to mono float in [-1, 1]
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)

    n = len(data)
    win = np.hanning(n)
    spec = np.fft.rfft(data * win)
    # Amplitude scaling (coherent gain compensation)
    amps = 2.0 * np.abs(spec) / max(np.sum(win), 1e-12)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    valid_idx = freqs > 0
    return freqs[valid_idx], amps[valid_idx]


def compute_third_octave_spectrum(freqs, amplitudes, fmin=20, fmax=20000):
    preferred_centers = np.array([
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
        1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
    ])
    fc_list = preferred_centers[(preferred_centers >= fmin) & (preferred_centers <= fmax)]
    factor = 2**(1 / 6)

    third_octave_db = []
    valid_centers = []

    for fc in fc_list:
        f_lower = fc / factor
        f_upper = fc * factor
        idx = np.where((freqs >= f_lower) & (freqs < f_upper))[0]
        if len(idx) > 0:
            rms_band = np.sqrt(np.sum(amplitudes[idx]**2))
            db_val = 20 * np.log10(max(rms_band, 1e-12))
            third_octave_db.append(db_val)
            valid_centers.append(fc)

    return np.array(valid_centers), np.array(third_octave_db)


# --- INPUTS ---
wav_path = "majidData/Ch 5_4_5_0.wav"  # Path to your .wav file

# --- COMPUTE ---
sim_freqs, sim_amps = solve_raindrop_membrane(1.46,1.28,212.5,0.25E-3,1760,2.1E9,0.45,25.0,0.0,None,None,OutputType.PRESSURE,0.84, 0.78, 0.09,radius_raindrop=2.5E-3,speed_raindrop=5.7,max_mode=80)
sim_amps = np.abs(sim_amps)
sim_fc, sim_db = compute_third_octave_spectrum(sim_freqs, sim_amps)

wav_freqs, wav_amps = wav_amplitude_spectrum(wav_path)
wav_fc, wav_db = compute_third_octave_spectrum(wav_freqs, wav_amps)

# --- PLOT ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sim_fc, sim_db, marker='o', linewidth=1.8, label=f"Simulation")
ax.plot(wav_fc, wav_db, marker='s', linewidth=1.8, label=f"WAV: {wav_path}")

ax.set_xscale('log')
ax.set_xticks([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_title("1/3-Octave Band Spectrum Comparison", fontsize=13)
ax.set_xlabel("1/3-Octave Center Frequency (Hz)", fontsize=11)
ax.set_ylabel("Band Level (dB, relative)", fontsize=11)
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.show()