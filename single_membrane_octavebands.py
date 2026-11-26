import numpy as np
import matplotlib.pyplot as plt
from models import solve_raindrop_membrane, OutputType

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
    freqs, amps = solve_raindrop_membrane(1.46,1.28,T,0.25E-3,1760,2.1E9,0.45,25.0,0.0,None,None,OutputType.PRESSURE,0.84, 0.78, 0.09,radius_raindrop=2.5E-3,speed_raindrop=5.7,max_mode=80)

    # Compute 1/3 Octave spectrum
    fc_centers, octave_db = compute_third_octave_spectrum(freqs, abs(amps))

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