# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from models import solve_raindrop_membrane, OutputType

# MATERIAL PROPERTIES

membrane_width = 1 # Width of the membrane (a)
membrane_height = 1 # Height of the membrane (b)

mem_tension = 250 # Tension forces for membranes (T)
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

def save_rain(radius,speed,drop_x,drop_y):
    freqs, pressure = solve_raindrop_membrane(1,1,250,0.25E-3,1750,0.52E9,0.45,25,0,None,None,OutputType.PRESSURE,radius_raindrop=radius,speed_raindrop=speed,x0=drop_x,y0=drop_y)

    # P5 audio file creation (at mic position (x,y,z))
    from scipy.io import wavfile

    p5_time = np.fft.irfft(np.conj(pressure), n=32768) # np.conj does time reversal (the audio was reversed so this fixes it)

    # The raindrop impact sits at t=0, so click is at the very start of the buffer.
    # Optional: remove DC/subsonic junk
    p5_time -= np.mean(p5_time)

    # Normalize to avoid clipping (leave a little headroom)
    p5_time /= (np.max(np.abs(p5_time)) + 1e-30)

    # Write 16-bit PCM WAV (N_fft/fs ≈ 0.74 s long)
    wavfile.write(f"cached_rain/{drop_x}_{drop_y}_{radius}.wav", 44100, (0.9 * p5_time * 32767).astype(np.int16))

# MAIN

# Params
N_spatial = 5
N_radius = 5

min_radius, max_radius = 0.5E-3, 2E-3

# Code

x_lin = np.linspace(0.1, membrane_width, N_spatial)
y_lin = np.linspace(0.1, membrane_height, N_spatial)

# Flattened coordinate arrays
xS, yS = np.meshgrid(x_lin, y_lin, indexing='ij')
xS_flat = xS.ravel()
yS_flat = yS.ravel()

rS = np.linspace(min_radius, max_radius, N_radius)
vS = 9.65 - 10.3 * np.exp(-0.6 * rS)

# Loop cleanly in a single pass over the pairs:
from itertools import product

for (x_coord, y_coord), (r_val, v_val) in tqdm(product(zip(xS_flat, yS_flat), zip(rS, vS)),
                                                total=len(xS_flat) * len(rS),
                                                desc="Generating Cache"):
    save_rain(r_val, v_val, x_coord, y_coord)