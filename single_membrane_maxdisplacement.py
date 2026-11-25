# LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# UNIVERSAL CONSTANTS

# None...

# MATERIAL PROPERTIES

def max_displacement(tension=250):
    membrane_width = 1 # Width of the membrane (a)
    membrane_height = 1 # Height of the membrane (b)

    mem_tension = tension # Tension forces for membranes (T)
    mem_thickness = 0.25E-3 # Thicknesses of membranes (h)
    mem_density = 1750 # Densities of membranes (rho)
    mem_youngmod = 0.52E9 # Young's moduli of membranes (E)
    mem_poisratio = 0.45 # Poisson's ratios of membranes (nu)
    mem_viscousdamping = 25.0 # Viscous damping coefficient (c_d)
    mem_structloss = 0.0 # Structural loss factor, for Im(B) (eta)

    c_air = 343 # Speed of sound in air used to calculate propagation constants based on frequency (c0)
    rho_air = 1.225 # Density of air used to calculate characteristic impedance

    imped = [c_air*rho_air, c_air*rho_air] # Characteristic impedances of volumes (Z)

    mem_bendstiffness = mem_youngmod*mem_thickness**3 / (12*(1-mem_poisratio**2)) * (1 + 1j*mem_structloss) # Bending stiffnesses of membranes (B)

    # RAIN PROPERTIES

    density_rainwater = 1000 # Density of water in the raindrop (rho_r)
    radius_raindrop = 2.5E-3 # Radius of the raindrop (r)
    speed_raindrop = 7 # Rainfall velocity (v_d)

    # NUMERICAL PARAMETERS

    Nx = 51 # Number of grid cells to compute in the x direction
    Ny = 51 # Number of grid cells to compute in the y direction

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

    alpha = density_rainwater * np.pi * radius_raindrop**2 * speed_raindrop**2 # Eq. from Toyoda paper
    beta = 3/8 * density_rainwater * np.pi * radius_raindrop * speed_raindrop**3 # Eq. from Toyoda paper
    T = alpha/beta

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
            Q_k = alpha / (omega*1j) * (np.exp(1j*omega*T) - 1) - beta/(omega**2) * (1 - np.exp(1j*omega*T) * (1-1j*omega*T) ) # Model 1 (Toyoda): sharp sawtooth
            # Q_k = c*tau*np.sqrt(np.pi) / (2*(1-1j*omega*tau)**(3/2)) # Model 2 (Mitchell): smooth sawtooth

        q_mn_freq[k, :, :] = Q_k * spatial_factor

    # LINEAR SYSTEM SOLVER
    # Output form (3 solved unknowns): [W_1mn, P_0mn-, P_5mn+]

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
            S[i] = mem_tension*k_prime_sq + mem_bendstiffness*k_prime_sq**2 - mem_density*mem_thickness*(omega**2) + 1j * omega * mem_viscousdamping

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

    for k, omega in enumerate(tqdm(omegas, desc="Solving system...")): # tqdm displays a progress bar
        if k == 0:
            continue

        # Solve system
        x_sol_all_modes = solve_system(omega,max_mode,k)

        # Keep track of W1
        w1_modal_freq[k, :, :] = x_sol_all_modes[:,:,0]

    # DISPLACEMENT RECONSTRUCTION

    w1_modal_time = np.fft.irfft(w1_modal_freq, axis=0)
    nt, _, _ = w1_modal_time.shape

    # Evaluate every time sample, in chunks to keep memory low
    phi_flat = phi.reshape(max_mode*max_mode, Nx*Ny)
    w_modal_flat = w1_modal_time.reshape(nt, -1)

    # MAXIMUM DISPLACEMENT

    absolute_peak = 0.0
    chunk = 2048
    for start in range(0, nt, chunk):
        w_chunk = w_modal_flat[start:start+chunk] @ phi_flat
        absolute_peak = max(absolute_peak, np.abs(w_chunk).max())

    return absolute_peak

tensions = np.linspace(10,1000,20)
maxdisplacements = []

for tension in tensions:
    maxdisplacements.append(max_displacement(tension))

plt.figure()
plt.plot(tensions,maxdisplacements)
plt.xlabel("Tension (N/m)")
plt.ylabel("Maximum displacement (m)")
plt.show()