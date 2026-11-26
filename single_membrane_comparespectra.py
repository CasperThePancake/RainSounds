import numpy as np
import matplotlib.pyplot as plt
from models import solve_raindrop_membrane, OutputType

# Define your tension range
tensions = np.linspace(10, 1000, 5)

# Set up explicit figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Use a colormap so each tension line has a distinct color
colors = plt.cm.viridis(np.linspace(0, 1, len(tensions)))

for i, T in enumerate(tensions):
    # Call your function
    freqs, amplitudes = solve_raindrop_membrane(1,1,T,0.25E-3,1750,0.52E9,0.45,0,0,None,None,OutputType.PRESSURE,max_mode=30)

    # Convert amplitude to Decibels (dB)
    amp_db = 20 * np.log10(np.maximum(np.abs(amplitudes), 1e-12))

    # Plot each curve on 'ax'
    ax.plot(freqs, amp_db, color=colors[i], linewidth=1.2, label=f"T = {T:.1f} N/m")

# Formatting the plot using explicit ax methods
ax.set_xscale('log')
ax.set_title("Pressure Frequency Spectrum (dB) Across Different Membrane Tensions", fontsize=13, pad=10)
ax.set_xlabel("Frequency (Hz)", fontsize=11)
ax.set_ylabel("Pressure Magnitude (dB)", fontsize=11)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

# Attach colorbar explicitly to 'ax' using fig.colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=tensions[0], vmax=tensions[-1]))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Membrane Tension (N/m)', fontsize=11)

plt.tight_layout()
plt.show()