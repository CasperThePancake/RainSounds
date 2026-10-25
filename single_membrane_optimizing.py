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

mem_tension = 250 # Tension forces for membranes (T)
mem_thickness = 0.25E-3 # Thicknesses of membranes (h)
mem_density = 1750 # Densities of membranes (rho)
mem_youngmod = 0.52E9 # Young's moduli of membranes (E)
mem_poisratio = 0.45 # Poisson's ratios of membranes (nu)

c_air = 343 # Speed of sound in air used to calculate propagation constants based on frequency (c0)
rho_air = 1.225 # Density of air used to calculate characteristic impedance

imped = [c_air*rho_air, c_air*rho_air] # Characteristic impedances of volumes (Z)

mem_bendstiffness = mem_youngmod*mem_thickness**3 / (12*(1-mem_poisratio**2)) # Bending stiffnesses of membranes (B)

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

q_mn_freq = np.zeros((Nomega,max_mode,max_mode),dtype=complex)

# We assume q is a delta function in space and some model in time (sharp or smooth sawtooth)

x0 = membrane_width/2 # Must lie in [0,membrane_width]
y0 = membrane_height/2 # Must lie in [0,membrane_height]

# alpha = density_rainwater * np.pi * radius_raindrop**2 * speed_raindrop**2 # Eq. from Toyoda paper
# beta = 3/8 * density_rainwater * np.pi * radius_raindrop * speed_raindrop**3 # Eq. from Toyoda paper
# T = alpha/beta

c = (10*np.sqrt(np.pi)/9)*density_rainwater*(2*radius_raindrop)**2*speed_raindrop**2
tau = (3/10)*(2*radius_raindrop)/speed_raindrop

# Decompose

spatial_factor = np.zeros((max_mode, max_mode)) # Pre-calculated to prevent doing this Nomega times
for m in range(1, max_mode + 1):
    for n in range(1, max_mode + 1):
        spatial_factor[m-1, n-1] = (4.0 / (membrane_width * membrane_height)) * np.sin(m * np.pi * x0 / membrane_width) * np.sin(n * np.pi * y0 / membrane_height)

for k, omega in enumerate(omegas):
    if omega == 0:
        Q_k = 0
    else: # Pre-derived Fourier transforms
        # Q_k = alpha / (omega*1j) * (np.exp(1j*omega*T) - 1) - beta/(omega**2) * (1 - np.exp(1j*omega*T) * (1-1j*omega*T) ) # Model 1 (Toyoda): sharp sawtooth
        Q_k = c*tau*np.sqrt(np.pi) / (2*(1-1j*omega*tau)**(3/2)) # Model 2 (Mitchell): smooth sawtooth

    q_mn_freq[k, :, :] = Q_k * spatial_factor

# LINEAR SYSTEM SOLVER
# Output form (13 solved unknowns): [W_1mn, P_0mn-, P_5mn+]

def solve_system(omega,max_mode,omega_k):
    # We solve all systems (for all (m,n)) at once for this given omega
    # Per system to solve (3x3): Ax=b

    # Grid of (m,n)
    m_arr = np.arange(1, max_mode + 1)
    n_arr = np.arange(1, max_mode + 1)
    M, N = np.meshgrid(m_arr, n_arr, indexing='ij')

    # Calculate combined constants
    prop = 1j*omega/c_air # Propagation constant gamma=i*omega/c0
    C = np.zeros((2, max_mode,max_mode), dtype=complex)
    k = np.zeros((2, max_mode,max_mode),dtype=complex)
    S = np.zeros((1, max_mode,max_mode),dtype=complex)

    k_prime_sq = (M * np.pi / membrane_width)**2 + (N * np.pi / membrane_height)**2

    for i in range(2):
        k[i] = np.sqrt(-prop**2 - k_prime_sq + 0j) # The 0j makes sure negative roots work properly

    k = np.where(k.imag < 0, -k, k) # Force k_z = i|k_z|, making sure no negative roots were numerically selected

    for i in range(2):
        C[i] = 1j * k[i] / (prop * imped[i])

    for i in range(1):
        S[i] = mem_tension*k_prime_sq + mem_bendstiffness*k_prime_sq**2 - mem_density*mem_thickness*(omega**2)

    # Time bench comparison:
    # Matrix solving: 14 minutes
    # Exact known solutions: ?? minutes

    # Construct RHS vector (b)
    b = np.zeros((max_mode,max_mode,3,1), dtype=complex)
    b[:, :, 0, 0] = q_mn_freq[omega_k, :, :]

    # Construct the matrix (A)
    A = np.zeros((max_mode, max_mode, 3,3),dtype=complex)

    # Equation (1)
    A[:,:,0,0] = S[0]
    A[:,:,0,1] = 1.0
    A[:,:,0,2] = -1.0

    # Equation (2)
    A[:,:,1,0] = -1j*omega
    A[:,:,1,1] = -C[0]

    # Equation (3)
    A[:,:,2,0] = -1j*omega
    A[:,:,2,2] = C[1]

    x_mn = np.linalg.solve(A,b)

    return x_mn.squeeze(-1) # Shape: (max_mode,max_mode,13) (Squeezing to remove the extra b dimension we had)

    # q_mn = q_mn_freq[omega_k, :, :]
    # S_mn = S[0]
    #
    # # Exact closed-form solutions derived from the 3x3 system
    # denom = S_mn - (1j * omega / C[0]) - (1j * omega / C[1])
    # W_1 = q_mn / denom
    # P_0 = (-1j * omega / C[0]) * W_1
    # P_1 = (1j * omega / C[1]) * W_1
    #
    # # Stack into the expected output shape: (max_mode, max_mode, 3)
    # return np.stack([W_1, P_0, P_1], axis=-1)

# QUANTITY SOLVER

# Precompute modal functions for each (x,y,m,n) to save time
phi = np.zeros((max_mode,max_mode,Nx,Ny)) # 4th-order tensor
for m in range(1, max_mode+1):
    for n in range(1, max_mode+1):
        phi[m-1,n-1,:,:] = np.sin(m*np.pi*X / membrane_width) * np.sin(n*np.pi*Y / membrane_height)

# Solve for actual quantities
# All quantities we wish to compute must be initialized here

w1_modal_freq = np.zeros((Nomega, max_mode, max_mode), dtype=complex) # Keep track of W_1mn for each m,n per omega

x_listen, y_listen, z_listen = membrane_width/2, membrane_height/2, 0.3 # Coordinates of microphone for pressure frequency spectrum plot
p5_freq = np.zeros(Nomega,dtype=complex) # Complex amplitudes per frequency of p5 at specified mic (x,y,z)

# Precompute static mode arrays outside the loop
m_arr = np.arange(1, max_mode + 1)
n_arr = np.arange(1, max_mode + 1)
M, N = np.meshgrid(m_arr, n_arr, indexing='ij')
k_prime_sq = (M * np.pi / membrane_width)**2 + (N * np.pi / membrane_height)**2

# Precompute listener spatial factors
phi_listener = np.sin(M * np.pi * x_listen / membrane_width) * np.sin(N * np.pi * y_listen / membrane_height)

for k, omega in enumerate(tqdm(omegas, desc="Solving system...")): # tqdm displays a progress bar
    if k == 0:
        continue

    # Solve system
    x_sol_all_modes = solve_system(omega,max_mode,k)

    # Keep track of W1
    w1_modal_freq[k, :, :] = x_sol_all_modes[:,:,0]

    # Keep track of P1
    # Vectorized vertical wavenumbers for all (m,n) at this omega
    prop = 1j * omega / c_air
    k_mn = np.sqrt(-prop**2 - k_prime_sq + 0j)
    k_mn = np.where(k_mn.imag < 0, -k_mn, k_mn)

    # Vectorized pressure calculation (replaces the nested m, n loops)
    p3_modes = x_sol_all_modes[:, :, 2] # P_1 (index 2)
    p5_freq[k] = np.sum(p3_modes * np.exp(1j * z_listen * k_mn) * phi_listener)

# PLOTTING AND INTERPRETING (GEMINI)

# 1. Convert modal coefficients to time domain first (shape: Nt, max_mode, max_mode)
w1_modal_time = np.fft.irfft(w1_modal_freq, axis=0)
nt, _, _ = w1_modal_time.shape

# 2. Setup frame indices for animation
frame_step = 8  # Step through 8 frames per render step (~350 us per animation frame)
frame_indices = np.arange(0, min(nt, 2000), frame_step)

print("Reconstructing spatial grids for animation frames...")
# 3. Only compute spatial grids for the exact frames we will display!
w1_space_time_frames = np.zeros((len(frame_indices), Nx, Ny))
for idx, t_frame in enumerate(frame_indices):
    w1_space_time_frames[idx, :, :] = np.einsum("mn,mnij->ij", w1_modal_time[t_frame], phi)

# W1 DISPLACEMENT (GEMINI)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# 4. Prevent clipping: Set z limits to true absolute max + 15% head room
absolute_peak = np.abs(w1_space_time_frames).max()
z_max = absolute_peak * 1.15 if absolute_peak > 0 else 1e-6

# Initialize surface plot
color_range = absolute_peak * 0.20  # Boosts color contrast for small waves
surf = [ax.plot_surface(X, Y, w1_space_time_frames[0], cmap="RdBu_r",
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
    # t_frame gives us the true physical time index for the title
    t_frame = frame_indices[frame_idx]

    # Remove previous frame surface
    surf[0].remove()

    # Re-draw updated displacement surface using sequential animation index (frame_idx)
    surf[0] = ax.plot_surface(
        X, Y, w1_space_time_frames[frame_idx],
        cmap="RdBu_r",
        vmin=-color_range,
        vmax=color_range,
        rstride=2,
        cstride=2,
        antialiased=True
    )

    # Calculate physical time in milliseconds using t_frame
    t_ms = t_frame * (1000.0 / fs)
    ax.set_title(f"Top Membrane Transient Response | t = {t_ms:.2f} ms", fontsize=12)

# interval=15ms for high FPS playback (~60 fps)
anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=15, blit=False)
plt.show()

# P5 frequency spectrum at mic position (x,y,z)

plt.figure()
plt.plot(freqs_hz,abs(p5_freq))
plt.title("Pressure frequency spectrum (magnitude) at mic position")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Pressure magnitude")
plt.show()

# P5 audio file creation (at mic position (x,y,z)) (CLAUDE)

from scipy.io import wavfile

p5_time = np.fft.irfft(np.conj(p5_freq), n=N_fft) # np.conj does time reversal (the audio was reversed so this fixes it)

# The raindrop impact sits at t=0, so the click is at the very start of the buffer.
# Optional: remove DC/subsonic junk
p5_time -= np.mean(p5_time)

# Normalize to avoid clipping (leave a little headroom)
p5_time /= (np.max(np.abs(p5_time)) + 1e-30)

# Write 16-bit PCM WAV (N_fft/fs ≈ 0.74 s long)
wavfile.write("raindrop_p5.wav", fs, (0.9 * p5_time * 32767).astype(np.int16))

# Quick look at the waveform
t_audio = np.arange(N_fft) / fs
plt.figure()
plt.plot(t_audio * 1000, p5_time)
plt.title("Pressure at mic position (time domain)")
plt.xlabel("Time (ms)")
plt.ylabel("Normalized pressure")
plt.show()