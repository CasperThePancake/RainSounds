import numpy as np
import matplotlib.pyplot as plt

# --- 0. SLIDE-FRIENDLY STYLE (large fonts, thick lines, high contrast) ---
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

# --- 1. MATERIAL PARAMETERS (Membrane 1) ---
tension = 250.0       # Tension force (T) in N/m
thickness = 0.25E-3   # Thickness (h) in m
density = 1750.0      # Density (rho) in kg/m^3
young_mod = 0.52E9    # Young's modulus (E) in Pa
pois_ratio = 0.45     # Poisson's ratio (nu)

eta = 0.005           # Structural loss factor for hysteretic damping
cd = 25.0             # Viscous damping coefficient (Ns/m^3 - tuned for illustration)

# Calculate bending stiffness (B) (real)
bending_stiffness = (young_mod * thickness**3) / (12.0 * (1.0 - pois_ratio**2))

# --- 2. FREQUENCY RANGE ---
frequencies_hz = np.linspace(10.0, 20000.0, 1000)
omegas = 2.0 * np.pi * frequencies_hz
rho_h = density * thickness

# Storage arrays for complex wavenumbers
k_baseline = np.zeros(len(omegas), dtype=complex)
k_hysteretic = np.zeros(len(omegas), dtype=complex)
k_viscous = np.zeros(len(omegas), dtype=complex)
k_combined = np.zeros(len(omegas), dtype=complex)

for i, omega in enumerate(omegas):
    # 1. Baseline Case (Purely Elastic)
    a_base = bending_stiffness
    b_base = tension
    c_base = -rho_h * (omega**2)
    X_base = (-b_base + np.sqrt(b_base**2 - 4 * a_base * c_base)) / (2.0 * a_base)
    k_baseline[i] = np.sqrt(X_base)

    # 2. Hysteretic Damping Case (Complex Bending: B * (1 + j*eta))
    a_hys = bending_stiffness * (1.0 + 1j * eta)
    b_hys = tension
    c_hys = -rho_h * (omega**2)
    X_hys = (-b_hys + np.sqrt(b_hys**2 - 4 * a_hys * c_hys)) / (2.0 * a_hys)
    k_hysteretic[i] = np.sqrt(X_hys)

    # 3. Viscous Damping Case (Real B + c_d * dw/dt)
    a_vis = bending_stiffness
    b_vis = tension
    c_vis = -rho_h * (omega**2) + 1j * omega * cd
    X_vis = (-b_vis + np.sqrt(b_vis**2 - 4 * a_vis * c_vis)) / (2.0 * a_vis)
    k_viscous[i] = np.sqrt(X_vis)

    # 4. Combined Hysteretic + Viscous Damping Case
    a_comb = bending_stiffness * (1.0 + 1j * eta)
    b_comb = tension
    c_comb = -rho_h * (omega**2) + 1j * omega * cd
    X_comb = (-b_comb + np.sqrt(b_comb**2 - 4 * a_comb * c_comb)) / (2.0 * a_comb)
    k_combined[i] = np.sqrt(X_comb)

# --- 3. EXTRACT PROPAGATION AND ATTENUATION ---
c_base = omegas / np.real(k_baseline)
c_hys = omegas / np.real(k_hysteretic)
c_vis = omegas / np.real(k_viscous)
c_comb = omegas / np.real(k_combined)

attn_base = np.imag(k_baseline)
attn_hys = np.imag(k_hysteretic)
attn_vis = np.imag(k_viscous)
attn_comb = np.imag(k_combined)

# --- 4. PLOTTING RESULTS ---
# 16:9 landscape to match a slide
fig, ax1 = plt.subplots(1, 1, figsize=(13.33, 7.5))

# Plot 1: Phase Velocity Comparison
ax1.plot(frequencies_hz, c_base, color='#1f77b4', label='Baseline (Purely Elastic)')
# ax1.plot(frequencies_hz, c_hys, color='#ff7f0e', linewidth=2, linestyle='--', label=f'Hysteretic ($\eta$ = {eta})')
# ax1.plot(frequencies_hz, c_vis, color='#2ca02c', linewidth=2, linestyle='-.', label=f'Viscous ($c_d$ = {cd})')
# ax1.plot(frequencies_hz, c_comb, color='#d62728', linewidth=2, linestyle=':', label='Combined (Hysteretic + Viscous)')
ax1.set_title('ETFE Membrane Phase Velocity Dispersion Curve', pad=16, fontweight='bold')
ax1.set_ylabel('Phase Velocity $c$ (m/s)', labelpad=10)
ax1.set_xlabel('Frequency (Hz)', labelpad=10)
ax1.set_xlim(0, 20000)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:g}k' if x >= 1000 else f'{x:g}'))
ax1.grid(True, which='both', linestyle='--', alpha=0.6)
ax1.legend(loc='best', frameon=True, framealpha=0.95)

# Plot 2: Spatial Attenuation Constant Comparison
# ax2.plot(frequencies_hz, attn_base, color='#1f77b4', linewidth=2, label='Baseline (Zero Attenuation)')
# ax2.plot(frequencies_hz, attn_hys, color='#ff7f0e', linewidth=2, linestyle='--', label='Hysteretic $\\text{Im}(k)$')
# ax2.plot(frequencies_hz, attn_vis, color='#2ca02c', linewidth=2, linestyle='-.', label='Viscous $\\text{Im}(k)$')
# ax2.plot(frequencies_hz, attn_comb, color='#d62728', linewidth=2, linestyle=':', label='Combined $\\text{Im}(k)$')
# ax2.set_title('Spatial Attenuation Constant per Frequency', fontsize=13, pad=10)
# ax2.set_xlabel('Frequency (Hz)', fontsize=11)
# ax2.set_ylabel('Attenuation $\\text{Im}(k)$ (rad/m)', fontsize=11)
# ax2.grid(True, which='both', linestyle='--', alpha=0.6)
# ax2.legend(fontsize=10)

plt.tight_layout()
# Optional: save a high-res image to drop straight into the slide
# plt.savefig('phase_velocity_slide.png', dpi=300, transparent=False)
plt.show()