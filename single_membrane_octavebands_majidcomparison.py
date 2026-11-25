import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm


# Possible further improvements:
# drop force not a delta spatially but some gaussian
# frequency-dependent bending stiffness or structural loss factor
# quantitatively taking water layers into account

def solve_raindrop_membrane(mem_tension):
    # MATERIAL PROPERTIES
    membrane_width = 1.46
    membrane_height = 1.28

    mem_thickness = 0.25E-3
    mem_density = 1760
    mem_youngmod = 2.1E9
    mem_poisratio = 0.45
    mem_viscousdamping = 25.0
    mem_structloss = 0.0

    c_air = 343
    rho_air = 1.225
    imped = [c_air * rho_air, c_air * rho_air]

    mem_bendstiffness = (
        mem_youngmod * mem_thickness**3 / (12 * (1 - mem_poisratio**2))
        * (1 + 1j * mem_structloss)
    )

    # RAIN PROPERTIES
    density_rainwater = 1000
    radius_raindrop = 2.5E-3
    speed_raindrop = 5.7

    # NUMERICAL PARAMETERS
    max_mode = 80
    fs = 44100
    N_fft = 32768

    freqs_hz = np.fft.rfftfreq(N_fft, d=1 / fs)
    omegas = 2 * np.pi * freqs_hz
    Nomega = len(omegas)

    # IMPACT FUNCTION DECOMPOSITION
    q_mn_freq = np.zeros((Nomega, max_mode, max_mode), dtype=complex)
    x0 = membrane_width / 2
    y0 = membrane_height / 2 # Majid data taken at middle of his membrane (4_5 data, best mic = 5)

    c = (10*np.sqrt(np.pi)/9)*density_rainwater*(2*radius_raindrop)**2*speed_raindrop**2
    tau = (3/10)*(2*radius_raindrop)/speed_raindrop

    m_arr = np.arange(1, max_mode + 1)
    n_arr = np.arange(1, max_mode + 1)
    M, N = np.meshgrid(m_arr, n_arr, indexing='ij')

    spatial_factor = (
        (4.0 / (membrane_width * membrane_height))
        * np.sin(M * np.pi * x0 / membrane_width)
        * np.sin(N * np.pi * y0 / membrane_height)
    )

    for k, omega in enumerate(omegas):
        if omega == 0:
            Q_k = 0.0
        else:
            Q_k = c*tau*np.sqrt(np.pi) / (2*(1-1j*omega*tau)**(3/2)) # Mitchell Model: smooth sawtooth
        q_mn_freq[k] = Q_k * spatial_factor

    # LINEAR SYSTEM SOLVER
    def solve_system(omega, omega_k):
        prop = 1j * omega / c_air
        k_prime_sq = (M * np.pi / membrane_width)**2 + (N * np.pi / membrane_height)**2

        k = np.sqrt(-prop**2 - k_prime_sq + 0j)
        k = np.where(k.imag < 0, -k, k)
        C = 1j * k / (prop * imped[0])

        S = (
            mem_tension * k_prime_sq
            + mem_bendstiffness * k_prime_sq**2
            - mem_density * mem_thickness * (omega**2)
            + 1j * omega * mem_viscousdamping
        )

        b = np.zeros((max_mode, max_mode, 3, 1), dtype=complex)
        b[:, :, 0, 0] = q_mn_freq[omega_k]

        A = np.zeros((max_mode, max_mode, 3, 3), dtype=complex)
        A[:, :, 0, 0] = S
        A[:, :, 0, 1] = 1.0
        A[:, :, 0, 2] = -1.0
        A[:, :, 1, 0] = -1j * omega
        A[:, :, 1, 1] = -C
        A[:, :, 2, 0] = -1j * omega
        A[:, :, 2, 2] = C

        return np.linalg.solve(A, b).squeeze(-1)

    # QUANTITY SOLVER
    x_listen, y_listen, z_listen = 0.84, 0.78, 0.09
    p5_freq = np.zeros(Nomega, dtype=complex)
    k_prime_sq = (M * np.pi / membrane_width)**2 + (N * np.pi / membrane_height)**2
    phi_listener = (
        np.sin(M * np.pi * x_listen / membrane_width)
        * np.sin(N * np.pi * y_listen / membrane_height)
    )

    for k, omega in enumerate(tqdm(omegas, desc=f"Solving T={mem_tension} N/m")):
        if k == 0:
            continue
        x_sol = solve_system(omega, k)
        prop = 1j * omega / c_air
        k_mn = np.sqrt(-prop**2 - k_prime_sq + 0j)
        k_mn = np.where(k_mn.imag < 0, -k_mn, k_mn)
        p5_freq[k] = np.sum(x_sol[:, :, 2] * np.exp(1j * z_listen * k_mn) * phi_listener)

    valid_idx = freqs_hz > 0
    return freqs_hz[valid_idx], np.abs(p5_freq[valid_idx])


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
tension = 212.5          # Single membrane tension (N/m)
wav_path = "majidData/Ch 5_4_5_0.wav"  # Path to your .wav file

# --- COMPUTE ---
sim_freqs, sim_amps = solve_raindrop_membrane(tension)
sim_fc, sim_db = compute_third_octave_spectrum(sim_freqs, sim_amps)

wav_freqs, wav_amps = wav_amplitude_spectrum(wav_path)
wav_fc, wav_db = compute_third_octave_spectrum(wav_freqs, wav_amps)

# --- PLOT ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sim_fc, sim_db, marker='o', linewidth=1.8, label=f"Simulation (T = {tension:.1f} N/m)")
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