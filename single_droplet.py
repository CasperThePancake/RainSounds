# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# UNIVERSAL CONSTANTS

# None...

# MATERIAL PROPERTIES

membrane_width = 1 # Width of the membrane (a)
membrane_height = 1 # Height of the membrane (b)

mem_tension = [250,250,250] # Tension forces for membranes (T)
mem_thickness = [0.2E-3,0.2E-3,0.2E-3] # Thicknesses of membranes (h)
mem_density = [1200,1200,1200] # Densities of membranes (rho)

imped = [413.2,488.5-114.8j,488.5-114.8j,488.5-114.8j,488.5-114.8j,413.2] # Characteristic impedances of volumes (Z)
prop = [-18.31j,3.37-22.84j,3.37-22.84j,3.37-22.84j,3.37-22.84j,-18.31j] # Propagation constants of volumes (gamma)

d1 = 0.02 # Distance from top membrane to first air boundary (d_1)
d2 = 0.04 # Distance from top membrane to middle membrane (d_2)
d3 = 0.06 # Distance from top membrane to second air boundary (d_3)
d4 = 0.08 # Distance from top membrane to bottom membrane (d_4)

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

max_mode = 15 # Highest mode number (all combinations to (m,n)=(max_mode,max_mode)) to compute in the sum

fs = 44100 # Sampling rate
N_fft = 32768 # FFT size

# FFT frequencies to solve

freqs_hz = np.fft.rfftfreq(N_fft, d=1/fs)

# Omegas to solve
omegas = 2 * np.pi * freqs_hz
Nomega = len(omegas) # Number of omegas to compute for

# IMPACT FUNCTION DECOMPOSITION
# The force function q(x,y,t) must be decomposed into q_mn(omega) (for values in our grid) to use in the solver

q_mn_freq = np.zeros((Nomega,max_mode,max_mode),dtype=complex)

# We assume q is a delta function in space and a sawtooth in time (q(t) = alpha-beta*t over [0,T] with T = alpha/beta)

x0 = membrane_width/2 # Must lie in [0,membrane_width]
y0 = membrane_height/2 # Must lie in [0,membrane_height]

alpha = density_rainwater * np.pi * radius_raindrop**2 * speed_raindrop**2 # Eq. from Toyoda paper
beta = 3/8 * density_rainwater * np.pi * radius_raindrop * speed_raindrop**3 # Eq. from Toyoda paper
T = alpha/beta

# Decompose

spatial_factor = np.zeros((max_mode, max_mode)) # Pre-calculated to prevent doing this Nomega times
for m in range(1, max_mode + 1):
    for n in range(1, max_mode + 1):
        spatial_factor[m-1, n-1] = (4.0 / (membrane_width * membrane_height)) * np.sin(m * np.pi * x0 / membrane_width) * np.sin(n * np.pi * y0 / membrane_height)

for k, omega in enumerate(omegas):
    if omega == 0:
        Q_k = 0
    else:
        Q_k = alpha / (omega*1j) * (np.exp(1j*omega*T) - 1) - beta/(omega**2) * (1 - np.exp(1j*omega*T) * (1-1j*omega*T) ) # Pre-derived Fourier transform of sawtooth q(t)

    q_mn_freq[k, :, :] = Q_k * spatial_factor

# LINEAR SYSTEM SOLVER
# Output form (13 solved unknowns): [W_1mn, W_2mn, W_3mn, P_0mn-, P_1mn+, P_1mn-, P_2mn+, P_2mn-, P_3mn+, P_3mn-, P_4mn+, P_4mn-, P_5mn+]

def solve_system(omega,max_mode,omega_k):
    # We solve all systems (for all (m,n)) at once for this given omega
    # Per system to solve (13x13): Ax=b

    # Grid of (m,n)
    m_arr = np.arange(1, max_mode + 1)
    n_arr = np.arange(1, max_mode + 1)
    M, N = np.meshgrid(m_arr, n_arr, indexing='ij')

    # Calculate combined constants
    C = np.zeros((6, max_mode,max_mode), dtype=complex)
    k = np.zeros((6, max_mode,max_mode),dtype=complex)
    S = np.zeros((3, max_mode,max_mode),dtype=complex)

    k_prime_sq = (M * np.pi / membrane_width)**2 + (N * np.pi / membrane_height)**2

    for i in range(6):
        k[i] = np.sqrt(-prop[i]**2 - k_prime_sq + 0j) # The 0j makes sure negative roots work properly
        C[i] = 1j * k[i] / (prop[i] * imped[i])

    for i in range(3):
        S[i] = mem_tension[i]*k_prime_sq - mem_density[i]*mem_thickness[i]*(omega**2)

    # Construct RHS vector (b)
    b = np.zeros((max_mode,max_mode,13,1), dtype=complex)
    b[:, :, 0, 0] = q_mn_freq[omega_k, :, :]

    # Construct the matrix (A)
    A = np.zeros((max_mode, max_mode, 13,13),dtype=complex)

    # Equation (1)
    A[:,:,0,0] = S[0]
    A[:,:,0,3] = -1.0
    A[:,:,0,4] = 1.0
    A[:,:,0,5] = 1.0

    # Equation (2)
    A[:,:,1,0] = 1j*omega
    A[:,:,1,3] = C[0]

    # Equation (3)
    A[:,:,2,0] = 1j*omega
    A[:,:,2,4] = -C[1]
    A[:,:,2,5] = C[1]

    # Equation (4)
    A[:,:,3,4] = np.exp(1j * k[1] * d1)
    A[:,:,3,5] = np.exp(-1j * k[1] * d1)
    A[:,:,3,6] = -np.exp(1j * k[2] * d1)
    A[:,:,3,7] = -np.exp(-1j * k[2] * d1)

    # Equation (5)
    A[:,:,4,4] = -C[1]*np.exp(1j*k[1]*d1)
    A[:,:,4,5] = C[1]*np.exp(-1j*k[1]*d1)
    A[:,:,4,6] = C[2]*np.exp(1j*k[2]*d1)
    A[:,:,4,7] = -C[2]*np.exp(-1j*k[2]*d1)

    # Equation (6)
    A[:,:,5,1] = S[1]
    A[:,:,5,6] = -np.exp(1j*k[2]*d2)
    A[:,:,5,7] = -np.exp(-1j*k[2]*d2)
    A[:,:,5,8] = np.exp(1j*k[3]*d2)
    A[:,:,5,9] = np.exp(-1j*k[3]*d2)

    # Equation (7)
    A[:,:,6,1] = 1j*omega
    A[:,:,6,6] = -C[2]*np.exp(1j*k[2]*d2)
    A[:,:,6,7] = C[2]*np.exp(-1j*k[2]*d2)

    # Equation (8)
    A[:,:,7,1] = 1j*omega
    A[:,:,7,8] = -C[3]*np.exp(1j*k[3]*d2)
    A[:,:,7,9] = C[3]*np.exp(-1j*k[3]*d2)

    # Equation (9)
    A[:,:,8,8] = np.exp(1j*k[3]*d3)
    A[:,:,8,9] = np.exp(-1j*k[3]*d3)
    A[:,:,8,10] = -np.exp(1j*k[4]*d3)
    A[:,:,8,11] = -np.exp(-1j*k[4]*d3)

    # Equation (10)
    A[:,:,9,8] = -C[3]*np.exp(1j*k[3]*d3)
    A[:,:,9,9] = C[3]*np.exp(-1j*k[3]*d3)
    A[:,:,9,10] = C[4]*np.exp(1j*k[4]*d3)
    A[:,:,9,11] = -C[4]*np.exp(-1j*k[4]*d3)

    # Equation (11)
    A[:,:,10,2] = S[2]
    A[:,:,10,10] = -np.exp(1j*k[4]*d4)
    A[:,:,10,11] = -np.exp(-1j*k[4]*d4)
    A[:,:,10,12] = np.exp(1j*k[5]*d4)

    # Equation (12)
    A[:,:,11,2] = 1j*omega
    A[:,:,11,10] = -C[4]*np.exp(1j*k[4]*d4)
    A[:,:,11,11] = C[4]*np.exp(-1j*k[4]*d4)

    # Equation (13)
    A[:,:,12,2] = 1j*omega
    A[:,:,12,12] = -C[5]*np.exp(1j*k[5]*d4)

    x_mn = np.linalg.solve(A,b)

    return x_mn.squeeze(-1) # Shape: (max_mode,max_mode,13) (Squeezing to remove the extra b dimension we had)

# QUANTITY SOLVER

# Precompute modal functions for each (x,y,m,n) to save time
phi = np.zeros((max_mode,max_mode,Nx,Ny)) # 4th-order tensor
for m in range(1, max_mode+1):
    for n in range(1, max_mode+1):
        phi[m-1,n-1,:,:] = np.sin(m*np.pi*X / membrane_width) * np.sin(n*np.pi*Y / membrane_height)

# Solve for actual quantities
# All quantities we wish to compute must be initialized here

w1_space_freq = np.zeros((Nomega, Nx, Ny), dtype=complex) # Displacement of top membrane (w_1)

for k, omega in enumerate(tqdm(omegas, desc="Solving system...")): # tqdm displays a progress bar
    if k == 0:
        continue

    x_sol_all_modes = solve_system(omega,max_mode,k)

    w1mn_modes = x_sol_all_modes[:,:,0]

    w1_space_freq[k, :, :] = np.einsum("mn,mnij->ij", w1mn_modes, phi) # Fill up a layer of the w1 solution (layer for this omega)

# Convert each quantity to the time domain

w1_space_time = np.fft.irfft(w1_space_freq, axis=0)

# PLOTTING AND INTERPRETING (3D WAVE PROPAGATION) (GEMINI)
nt, nx, ny = w1_space_time.shape

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# 1. Prevent clipping: Set z limits to true absolute max + 15% head room
absolute_peak = np.abs(w1_space_time).max()
z_max = absolute_peak * 1.15 if absolute_peak > 0 else 1e-6

# 2. Time-scaling / Frame striding:
# At fs = 44100, 1 frame = 0.0226 ms. Skipping frames speeds up playback to a realistic rate.
frame_step = 8  # Step through 8 frames per render step (~350 us per animation frame)
frame_indices = np.arange(0, min(nt, 2000), frame_step)

# Initialize surface plot
color_range = absolute_peak * 0.20  # Boosts color contrast for small waves
surf = [ax.plot_surface(X, Y, w1_space_time[0], cmap="RdBu_r",
                        vmin=-color_range, vmax=color_range,
                        rstride=2, cstride=2, antialiased=True)]

# Axis bounds and labels
ax.set_xlim(0, membrane_width)
ax.set_ylim(0, membrane_height)
ax.set_zlim(-z_max, z_max)

ax.set_xlabel("x (m)", labelpad=10)
ax.set_ylabel("y (m)", labelpad=10)
ax.set_zlabel("Displacement (m)", labelpad=10)

# Set initial camera view for optimal 3D perspective
ax.view_init(elev=28, azim=-125)

def update(frame_idx):
    t_frame = frame_indices[frame_idx]

    # Remove previous frame surface
    surf[0].remove()

    # Re-draw updated displacement surface
    surf[0] = ax.plot_surface(
        X, Y, w1_space_time[t_frame],
        cmap="RdBu_r",
        vmin=-color_range,
        vmax=color_range,
        rstride=2,
        cstride=2,
        antialiased=True
    )

    # Calculate physical time in milliseconds
    t_ms = t_frame * (1000.0 / fs)
    ax.set_title(f"Top Membrane Transient Response | t = {t_ms:.2f} ms", fontsize=12)

# interval=15ms for high FPS playback (~60 fps)
anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=15, blit=False)
plt.show()