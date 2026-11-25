import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def solve_raindrop_membrane(mem_tension):
    # MATERIAL PROPERTIES
    membrane_width = 1 # Width of the membrane (a)
    membrane_height = 1 # Height of the membrane (b)

    mem_thickness = 0.25E-3 # Thicknesses of membranes (h)
    mem_density = 1750 # Densities of membranes (rho)
    mem_youngmod = 0.52E9 # Young's moduli of membranes (E)
    mem_poisratio = 0.45 # Poisson's ratios of membranes (nu)
    mem_viscousdamping = 0.0 # Viscous damping coefficient (c_d)
    mem_structloss = 0.03 # Structural loss factor, for Im(B) (eta)

    c_air = 343 # Speed of sound in air used to calculate propagation constants based on frequency (c0)
    rho_air = 1.225 # Density of air used to calculate characteristic impedance

    imped = [c_air*rho_air, c_air*rho_air] # Characteristic impedances of volumes (Z)

    mem_bendstiffness = mem_youngmod*mem_thickness**3 / (12*(1-mem_poisratio**2)) * (1 + 1j*mem_structloss) # Bending stiffnesses of membranes (B)

    # RAIN PROPERTIES
    density_rainwater = 1000 # Density of water in the raindrop (rho_r)
    radius_raindrop = 2.5E-3 # Radius of the raindrop (r)
    speed_raindrop = 7 # Rainfall velocity (v_d)

    # NUMERICAL PARAMETERS
    Nx = 50 # Number of grid cells to compute in the x direction
    Ny = 50 # Number of grid cells to compute in the y direction

    x = np.linspace(0, membrane_width, Nx)
    y = np.linspace(0, membrane_height, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij') # 2D grids for (x,y)

    max_mode = 30 # Highest mode number (all combinations to (m,n)=(max_mode,max_mode)) to compute in the sum

    fs = 44100 # Sampling rate
    N_fft = 32768 # FFT size

    # FFT frequencies to solve
    freqs_hz = np.fft.rfftfreq(N_fft, d=1/fs)

    # Omegas to solve
    omegas = 2 * np.pi * freqs_hz

    Nomega = len(omegas) # Number of omegas to compute for

    # IMPACT FUNCTION DECOMPOSITION
    # The force function q(x,y,t) must be decomposed into q_mn(omega) (for values in our grid) to use in the solver
    q_mn_freq = np.zeros((Nomega, max_mode, max_mode), dtype=complex)

    # We assume q is a delta function in space and some model in time (sharp or smooth sawtooth)
    x0 = membrane_width / 2 # Must lie in [0,membrane_width]
    y0 = membrane_height / 2 # Must lie in [0,membrane_height]

    alpha = density_rainwater * np.pi * radius_raindrop**2 * speed_raindrop**2 # Eq. from Toyoda paper
    beta = (3/8) * density_rainwater * np.pi * radius_raindrop * speed_raindrop**3 # Eq. from Toyoda paper
    T = alpha / beta

    # Decompose
    spatial_factor = np.zeros((max_mode, max_mode)) # Pre-calculated to prevent doing this Nomega times
    for m in range(1, max_mode + 1):
        for n in range(1, max_mode + 1):
            spatial_factor[m-1, n-1] = (4.0 / (membrane_width * membrane_height)) * np.sin(m * np.pi * x0 / membrane_width) * np.sin(n * np.pi * y0 / membrane_height)

    for k, omega in enumerate(omegas):
        if omega == 0:
            Q_k = 0
        else: # Pre-derived Fourier transforms
            Q_k = alpha / (omega * 1j) * (np.exp(1j * omega * T) - 1) - beta / (omega**2) * (1 - np.exp(1j * omega * T) * (1 - 1j * omega * T))

        q_mn_freq[k, :, :] = Q_k * spatial_factor

    # LINEAR SYSTEM SOLVER
    # Output form (3 solved unknowns): [W_1mn, P_0mn-, P_5mn+]
    def solve_system(omega, max_mode, omega_k):
        # We solve all systems (for all (m,n)) at once for this given omega
        # Per system to solve (3x3): Ax=b

        # Grid of (m,n)
        m_arr = np.arange(1, max_mode + 1)
        n_arr = np.arange(1, max_mode + 1)
        M, N = np.meshgrid(m_arr, n_arr, indexing='ij')

        # Calculate combined constants
        prop = 1j * omega / c_air # Propagation constant gamma=i*omega/c0
        C = np.zeros((2, max_mode, max_mode), dtype=complex)
        k = np.zeros((2, max_mode, max_mode), dtype=complex)
        S = np.zeros((1, max_mode, max_mode), dtype=complex)

        k_prime_sq = (M * np.pi / membrane_width)**2 + (N * np.pi / membrane_height)**2

        for i in range(2):
            k[i] = np.sqrt(-prop**2 - k_prime_sq + 0j) # The 0j makes sure negative roots work properly

        k = np.where(k.imag < 0, -k, k) # Force k_z = i|k_z|, making sure no negative roots were numerically selected

        for i in range(2):
            C[i] = 1j * k[i] / (prop * imped[i])

        for i in range(1):
            S[i] = mem_tension * k_prime_sq + mem_bendstiffness * k_prime_sq**2 - mem_density * mem_thickness * (omega**2) + 1j * omega * mem_viscousdamping

        # Construct RHS vector (b)
        b = np.zeros((max_mode, max_mode, 3, 1), dtype=complex)
        b[:, :, 0, 0] = q_mn_freq[omega_k, :, :]

        # Construct the matrix (A)
        A = np.zeros((max_mode, max_mode, 3, 3), dtype=complex)

        # Equation (1)
        A[:, :, 0, 0] = S[0]
        A[:, :, 0, 1] = 1.0
        A[:, :, 0, 2] = -1.0

        # Equation (2)
        A[:, :, 1, 0] = -1j * omega
        A[:, :, 1, 1] = -C[0]

        # Equation (3)
        A[:, :, 2, 0] = -1j * omega
        A[:, :, 2, 2] = C[1]

        x_mn = np.linalg.solve(A, b)

        return x_mn.squeeze(-1) # Shape: (max_mode, max_mode, 3)

    # QUANTITY SOLVER
    x_listen, y_listen, z_listen = membrane_width/2, membrane_height/2, 0.3 # Coordinates of microphone for pressure frequency spectrum plot
    p5_freq = np.zeros(Nomega, dtype=complex) # Complex amplitudes per frequency of p5 at specified mic (x,y,z)

    # Precompute static mode arrays outside the loop
    m_arr = np.arange(1, max_mode + 1)
    n_arr = np.arange(1, max_mode + 1)
    M, N = np.meshgrid(m_arr, n_arr, indexing='ij')
    k_prime_sq = (M * np.pi / membrane_width)**2 + (N * np.pi / membrane_height)**2

    # Precompute listener spatial factors
    phi_listener = np.sin(M * np.pi * x_listen / membrane_width) * np.sin(N * np.pi * y_listen / membrane_height)

    for k, omega in enumerate(tqdm(omegas, desc=f"Solving for tension {mem_tension}...")):
        if k == 0:
            continue

        # Solve system
        x_sol_all_modes = solve_system(omega, max_mode, k)

        # Keep track of P1
        # Vectorized vertical wavenumbers for all (m,n) at this omega
        prop = 1j * omega / c_air

        k_mn = np.sqrt(-prop**2 - k_prime_sq + 0j)
        k_mn = np.where(k_mn.imag < 0, -k_mn, k_mn)

        # Vectorized pressure calculation
        p3_modes = x_sol_all_modes[:, :, 2] # P_1 (index 2)
        p5_freq[k] = np.sum(p3_modes * np.exp(1j * z_listen * k_mn) * phi_listener)

    # Return valid frequencies and absolute amplitudes
    valid_idx = freqs_hz > 0
    return [freqs_hz[valid_idx], np.abs(p5_freq[valid_idx])]

def compute_third_octave_spectrum(freqs, amplitudes, fmin=20, fmax=20000):
    """
    Converts narrowband frequency spectrum to 1/3-octave band spectrum.
    """
    # Nominal 1/3-octave center frequencies (ISO standard preferred numbers)
    preferred_centers = np.array([
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
        1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
    ])

    # Filter center frequencies to range of interest
    fc_list = preferred_centers[(preferred_centers >= fmin) & (preferred_centers <= fmax)]

    third_octave_db = []
    valid_centers = []

    # Factor for 1/3 octave band limits: 2^(1/6)
    factor = 2**(1/6)

    for fc in fc_list:
        f_lower = fc / factor
        f_upper = fc * factor

        # Find indices of frequencies in this band
        idx = np.where((freqs >= f_lower) & (freqs < f_upper))[0]

        if len(idx) > 0:
            # Integrate sound power (energy) across the band
            power_in_band = np.sum(amplitudes[idx]**2)
            rms_band = np.sqrt(power_in_band)

            # Convert to dB (ref 1e-12 or 20uPa)
            db_val = 20 * np.log10(max(rms_band, 1e-12))

            third_octave_db.append(db_val)
            valid_centers.append(fc)

    return np.array(valid_centers), np.array(third_octave_db)

# Define your tension range
tensions = np.linspace(10, 1000, 20)

# Set up explicit figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(tensions)))

for i, T in enumerate(tensions):
    freqs, amplitudes = solve_raindrop_membrane(T)

    # Compute 1/3 Octave spectrum
    fc_centers, octave_db = compute_third_octave_spectrum(freqs, amplitudes)

    # Plot as line with markers or step plot
    ax.plot(fc_centers, octave_db, marker='o', markersize=3, color=colors[i], linewidth=1.5, label=f"T = {T:.1f} N/m")

ax.set_xscale('log')
# Set standard 1/3 octave ticks for clarity
ax.set_xticks([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_title("1/3-Octave Band Pressure Spectrum Across Membrane Tensions", fontsize=13, pad=10)
ax.set_xlabel("1/3-Octave Center Frequency (Hz)", fontsize=11)
ax.set_ylabel("Band Pressure Level (dB)", fontsize=11)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=tensions[0], vmax=tensions[-1]))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Membrane Tension (N/m)', fontsize=11)

plt.tight_layout()
plt.show()