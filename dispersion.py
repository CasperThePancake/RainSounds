import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline

# --- 0. SLIDE-FRIENDLY STYLE (large fonts, thick lines, high contrast) ---
# plt.rcParams.update({
#     'font.size': 20,
#     'axes.titlesize': 28,
#     'axes.labelsize': 24,
#     'xtick.labelsize': 20,
#     'ytick.labelsize': 20,
#     'legend.fontsize': 20,
#     'lines.linewidth': 4,
#     'axes.linewidth': 2,
#     'xtick.major.width': 2,
#     'ytick.major.width': 2,
#     'xtick.major.size': 8,
#     'ytick.major.size': 8,
# })

# --- 1. MATERIAL PARAMETERS (Membrane 1) ---
tension = 250.0       # Tension force (T) in N/m
thickness = 0.25E-3   # Thickness (h) in m
density = 1750.0      # Density (rho) in kg/m^3
young_mod = 0.52E9    # Young's modulus (E) in Pa
pois_ratio = 0.45     # Poisson's ratio (nu)

eta = 0.03           # Structural loss factor for hysteretic damping
cd = 25.0             # Viscous damping coefficient (Ns/m^3 - tuned for illustration)

# Calculate bending stiffness (B) (real)
bending_stiffness = (young_mod * thickness**3) / (12.0 * (1.0 - pois_ratio**2))

# --- 2. FREQUENCY RANGE ---
frequencies_hz = np.linspace(10.0, 5000.0, 1000)
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

# --- 3. EXTRACT PROPAGATION AND ATTENUATION ---
c_base = omegas / np.real(k_baseline)
c_hys = omegas / np.real(k_hysteretic)
c_vis = omegas / np.real(k_viscous)

attn_base = np.imag(k_baseline)
attn_hys = np.imag(k_hysteretic)
attn_vis = np.imag(k_viscous)

# Majid RT(f) attenuation data
freqs_majid = [63,
80,
100,
125,
160,
200,
250,
315,
400,
500,
630,
800,
1000,
1250,
1600,
2000,
2500,
3150,
4000,
5000]

p50_RT_majid = [5.217,
3.716,
3.289,
2.453,
2.379,
1.644,
1.258,
0.928,
0.694,
0.552,
0.484,
0.348,
0.205,
0.16,
0.132,
0.103,
0.083,
0.066,
0.052,
0.042]

p250_RT_majid = [4.687,
3.673,
3.126,
3.049,
1.759,
1.831,
1.494,
1.519,
1.14,
0.764,
0.894,
0.392,
0.329,
0.271,
0.198,
0.294,
0.13,
0.1,
0.064,
0.067]

p450_RT_majid = [5.068,
3.292,
3.137,
3.198,
2.225,
1.809,
1.287,
1.468,
1,
0.869,
0.538,
0.419,
0.216,
0.189,
0.158,
0.116,
0.083,
0.066,
0.052,
0.041]

p700_RT_majid = [3.575,
3.355,
2.375,
2.356,
1.535,
1.745,
1.437,
0.989,
0.979,
0.849,
0.773,
0.532,
0.41,
0.239,
0.146,
0.131,
0.087,
0.07,
0.052,
0.042]

p1100_RT_majid = [3.319,
4.429,
2.966,
2.467,
1.579,
1.735,
1.477,
1.257,
1.072,
0.686,
0.653,
0.482,
0.295,
0.337,
0.188,
0.134,
0.104,
0.07,
0.053,
0.042]

c_interp = interp1d(frequencies_hz, c_base, kind='nearest')

# Original data points
attn_majid_data_p50 = -3 * np.log(10) / (
    c_interp(freqs_majid) * np.array(p50_RT_majid)
)
attn_majid_data_p250 = -3 * np.log(10) / (
    c_interp(freqs_majid) * np.array(p250_RT_majid)
)
attn_majid_data_p450 = -3 * np.log(10) / (
    c_interp(freqs_majid) * np.array(p450_RT_majid)
)
attn_majid_data_p700 = -3 * np.log(10) / (
    c_interp(freqs_majid) * np.array(p700_RT_majid)
)
attn_majid_data_p1100 = -3 * np.log(10) / (
    c_interp(freqs_majid) * np.array(p1100_RT_majid)
)

# Fit
fit_majid_p50 = make_interp_spline(
    np.log10(freqs_majid),
    p50_RT_majid,
    k=3
)
fit_majid_p250 = make_interp_spline(
    np.log10(freqs_majid),
    p250_RT_majid,
    k=3
)
fit_majid_p450 = make_interp_spline(
    np.log10(freqs_majid),
    p450_RT_majid,
    k=3
)
fit_majid_p700 = make_interp_spline(
    np.log10(freqs_majid),
    p700_RT_majid,
    k=3
)
fit_majid_p1100 = make_interp_spline(
    np.log10(freqs_majid),
    p1100_RT_majid,
    k=3
)

p50_fit = fit_majid_p50(np.log10(frequencies_hz))
p250_fit = fit_majid_p250(np.log10(frequencies_hz))
p450_fit = fit_majid_p450(np.log10(frequencies_hz))
p700_fit = fit_majid_p700(np.log10(frequencies_hz))
p1100_fit = fit_majid_p1100(np.log10(frequencies_hz))

attn_majid_p50 = -3 * np.log(10) / (c_base * p50_fit)
attn_majid_p250 = -3 * np.log(10) / (c_base * p250_fit)
attn_majid_p450 = -3 * np.log(10) / (c_base * p450_fit)
attn_majid_p700 = -3 * np.log(10) / (c_base * p700_fit)
attn_majid_p1100 = -3 * np.log(10) / (c_base * p1100_fit)

plt.figure()
plt.scatter(freqs_majid, attn_majid_data_p50, label='Original data (50 Pa)')
plt.plot(frequencies_hz, attn_majid_p50, label='Spline fit (50 Pa)')
plt.scatter(freqs_majid, attn_majid_data_p250, label='Original data (250 Pa)')
plt.plot(frequencies_hz, attn_majid_p250, label='Spline fit (250 Pa)')
plt.scatter(freqs_majid, attn_majid_data_p450, label='Original data (450 Pa)')
plt.plot(frequencies_hz, attn_majid_p450, label='Spline fit (450 Pa)')
plt.scatter(freqs_majid, attn_majid_data_p700, label='Original data (700 Pa)')
plt.plot(frequencies_hz, attn_majid_p700, label='Spline fit (700 Pa)')
plt.scatter(freqs_majid, attn_majid_data_p1100, label='Original data (1100 Pa)')
plt.plot(frequencies_hz, attn_majid_p1100, label='Spline fit (1100 Pa)')
plt.legend()
plt.grid()
plt.show()

# --- 4. PLOTTING RESULTS ---
# 16:9 landscape to match a slide
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.33, 7.5))

# Plot 1: Phase Velocity Comparison
ax1.plot(frequencies_hz, c_base, color='#1f77b4', label='Baseline (Purely Elastic)')
ax1.plot(frequencies_hz, c_hys, color='#ff7f0e', linewidth=2, linestyle='--', label=f'Hysteretic ($\eta$ = {eta})')
ax1.plot(frequencies_hz, c_vis, color='#2ca02c', linewidth=2, linestyle='-.', label=f'Viscous ($c_d$ = {cd})')
ax1.set_title('ETFE Membrane Phase Velocity Dispersion Curve', pad=16, fontweight='bold')
ax1.set_ylabel('Phase Velocity $c$ (m/s)', labelpad=10)
ax1.set_xlabel('Frequency (Hz)', labelpad=10)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:g}k' if x >= 1000 else f'{x:g}'))
ax1.grid(True, which='both', linestyle='--', alpha=0.6)
ax1.legend(loc='best', frameon=True, framealpha=0.95)

# Plot 2: Spatial Attenuation Constant Comparison
ax2.plot(frequencies_hz, attn_base, color='#1f77b4', linewidth=2, label='Baseline (Zero Attenuation)')
ax2.plot(frequencies_hz, attn_hys, color='#ff7f0e', linewidth=2, linestyle='--', label='Hysteretic $\\text{Im}(k)$')
ax2.plot(frequencies_hz, attn_vis, color='#2ca02c', linewidth=2, linestyle='-.', label='Viscous $\\text{Im}(k)$')
ax2.plot(frequencies_hz, attn_majid_p50, color='black', linewidth=2, linestyle=':', label='Majid RT(f) data (50 Pa) $\\text{Im}(k)$')
ax2.plot(frequencies_hz, attn_majid_p250, color='black', linewidth=2, linestyle=':', label='Majid RT(f) data (250 Pa) $\\text{Im}(k)$')
ax2.plot(frequencies_hz, attn_majid_p450, color='black', linewidth=2, linestyle=':', label='Majid RT(f) data (450 Pa) $\\text{Im}(k)$')
ax2.plot(frequencies_hz, attn_majid_p700, color='black', linewidth=2, linestyle=':', label='Majid RT(f) data (700 Pa) $\\text{Im}(k)$')
ax2.plot(frequencies_hz, attn_majid_p1100, color='black', linewidth=2, linestyle=':', label='Majid RT(f) data (1100 Pa) $\\text{Im}(k)$')
ax2.set_title('Spatial Attenuation Constant per Frequency', fontsize=13, pad=10)
ax2.set_xlabel('Frequency (Hz)', fontsize=11)
ax2.set_ylabel('Attenuation $\\text{Im}(k)$ (rad/m)', fontsize=11)
ax2.grid(True, which='both', linestyle='--', alpha=0.6)
ax2.legend(fontsize=10)

plt.tight_layout()
# Optional: save a high-res image to drop straight into the slide
# plt.savefig('phase_velocity_slide.png', dpi=300, transparent=False)
plt.show()