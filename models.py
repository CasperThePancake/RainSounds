#!/usr/bin/env python

"""Models.py: Description of what it does."""

__author__ = "Casper Vermeeren"
__copyright__ = "Copyright 2026"
__credits__ = ["Loïck Sansen", "Majid Lavasani", "Christ Glorieux", "Masahiro Toyoda", "Benjamin Mitchell"]
__license__ = "GNU"
__email__ = "vermeerencasper@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm

from enum import Enum

class OutputType(Enum):
    DISPLACEMENT = "displacement"
    PRESSURE = "pressure"


def solve_raindrop_membrane(mem_width: float, mem_height: float, mem_tension: float, mem_thickness: float, mem_density: float, mem_youngmod: float, mem_poisratio: float, mem_viscousdamping: float, mem_structloss: float, x0: float | None, y0: float | None, outputType, x_listen: float | None = None, y_listen: float | None = None, z_listen: float = 0.3, Nx: int = 50, Ny: int = 50, c_air: float = 343, rho_air: float = 1.225, density_rainwater: float = 1000, radius_raindrop: float = 2.5E-3, speed_raindrop: float = 6, mem_tension_x: float = 0, mem_tension_y: float = 0, max_mode: int = 50, fs: int = 44100, N_fft: int = 32768, progressBar = True):
    """
    Solve the impact of a single raindrop on a single-layered membrane

    NOTE: If both mem_tension_x and mem_tension_y are zero, the isotropic mem_tension will be used instead

    :param mem_width: Width of the membrane in meters (x direction)
    :type mem_width: float
    :param mem_height: Height of the membrane in meters (y direction)
    :type mem_height: float
    :param mem_tension: Tension applied to the membrane in N/m
    :type mem_tension: float
    :param mem_thickness: Thickness of the membrane in meters (z direction)
    :type mem_thickness: float
    :param mem_density: Density of the membrane material in kg/m³
    :type mem_density: float
    :param mem_youngmod: Young's modulus of the membrane material in Pa
    :type mem_youngmod: float
    :param mem_poisratio: Poisson ratio of the membrane material (dimensionless)
    :type mem_poisratio: float
    :param mem_viscousdamping: Viscous damping coefficient (0 = no viscous damping)
    :type mem_viscousdamping: float
    :param mem_structloss: Structural loss damping factor (0 = no hysteresis damping)
    :type mem_structloss: float
    :param x0: Impact coordinate of the raindrop in meters (x direction), if None defaults to mem_width/2
    :type x0: float | None
    :param y0: Impact coordinate of the raindrop in meters (y direction), if None defaults to mem_height/2
    :type y0: float | None
    :param outputType: Enum specifying desired output of the model (see return clause for more details)
    :type outputType: OutputType
    :param x_listen: Coordinate of the listener for pressure output in meters (x direction), if None defaults to mem_width/2, default is None
    :type x_listen: float | None
    :param y_listen: Coordinate of the listener for pressure output in meters (y direction), if None defaults to mem_height/2, default is None
    :type y_listen: float | None
    :param z_listen: Coordinate of the listener for pressure output in meters (z direction), defaults to 0.3
    :type z_listen: float
    :param Nx: Number of spatial grid points to compute for displacement output (x direction), default is 50
    :type Nx: int
    :param Ny: Number of spatial grid points to compute for displacement output (y direction), default is 50
    :type Ny: int
    :param c_air: Speed of sound in air in m/s, default is 343
    :type c_air: float
    :param rho_air: Density of air in kg/m³, default is 1.225
    :type rho_air: float
    :param density_rainwater: Density of rainwater in kg/m³, default is 1000
    :type density_rainwater: float
    :param radius_raindrop: Radius of the raindrop in meters, default is 2.5E-3
    :type radius_raindrop: float
    :param speed_raindrop: Speed of the raindrop in m/s, default is 6
    :type speed_raindrop: float
    :param mem_tension_x: Anisotropic tension applied to the membrane (x direction), default is 0
    :type mem_tension_x: float
    :param mem_tension_y: Anisotropic tension applied to the membrane (y direction), default is 0
    :type mem_tension_y: float
    :param max_mode: Maximum number of spatial modes to compute in both directions, defaults to 50
    :type max_mode: int
    :param fs: Sampling frequency for the signal in Hertz, default is 44100
    :type fs: int
    :param N_fft: Number of frequencies to compute, default is 32768
    :type N_fft: int
    :param progressBar: Whether or not to display a progress bar in the console while computing output, default is True
    :type progressBar: bool
    :return: If outputType is OutputType.PRESSURE: list of frequencies and list of their complex magnitudes in the pressure signal; if outputType is OutputType.DISPLACEMENT: list of time values (in seconds) and list of X grid points and list of Y grid points and 3D NumPy array (time,x,y) for displacement (in meters)
    :rtype: list
    """
    # Checks
    if not isinstance(outputType, OutputType):
        raise TypeError(f"Output type must be valid enum of OutputType, got {type(outputType).__name__}")

    # Parameter-based defaulting
    if x0 is None:
        x0 = mem_width / 2
    if y0 is None:
        y0 = mem_height / 2
    if x_listen is None:
        x_listen = mem_width / 2
    if y_listen is None:
        y_listen = mem_height / 2
    if z_listen is None:
        z_listen = 0.3

    # DERIVED PROPERTIES
    imped = [c_air * rho_air, c_air * rho_air]

    mem_bendstiffness = (
        mem_youngmod * mem_thickness**3 / (12 * (1 - mem_poisratio**2))
        * (1 + 1j * mem_structloss)
    )

    freqs_hz = np.fft.rfftfreq(N_fft, d=1 / fs)
    omegas = 2 * np.pi * freqs_hz
    Nomega = len(omegas)

    # IMPACT FUNCTION DECOMPOSITION
    q_mn_freq = np.zeros((Nomega, max_mode, max_mode), dtype=complex)

    c = (10*np.sqrt(np.pi)/9)*density_rainwater*(2*radius_raindrop)**2*speed_raindrop**2
    tau = (3/10)*(2*radius_raindrop)/speed_raindrop

    m_arr = np.arange(1, max_mode + 1)
    n_arr = np.arange(1, max_mode + 1)
    M, N = np.meshgrid(m_arr, n_arr, indexing='ij')

    spatial_factor = (
        (4.0 / (mem_width * mem_height))
        * np.sin(M * np.pi * x0 / mem_width)
        * np.sin(N * np.pi * y0 / mem_height)
    )

    for k, omega in enumerate(omegas):
        if omega == 0:
            Q_k = 0.0
        else:
            Q_k = c*tau*np.sqrt(np.pi) / (2*(1-1j*omega*tau)**(3/2)) # Mitchell Model: smooth sawtooth
        q_mn_freq[k] = Q_k * spatial_factor

    # LINEAR SYSTEM SOLVER (solves for a given omega, for all (m,n) up to max at once)
    def solve_system(omega, omega_k):
        prop = 1j * omega / c_air
        k_prime_sq = (M * np.pi / mem_width)**2 + (N * np.pi / mem_height)**2

        k = np.sqrt(-prop**2 - k_prime_sq + 0j)
        k = np.where(k.imag < 0, -k, k)
        C = 1j * k / (prop * imped[0])

        if mem_tension_x != 0 or mem_tension_y != 0: # Anisotropic version
            S = (
                mem_tension_x * (M * np.pi / mem_width)**2 + mem_tension_y * (N * np.pi / mem_height)**2
                + mem_bendstiffness * k_prime_sq**2
                - mem_density * mem_thickness * (omega**2)
                + 1j * omega * mem_viscousdamping
            )
        else: # Normal version
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
    # Precompute modal functions for each (x,y,m,n) to save time
    if outputType == OutputType.DISPLACEMENT:
        x = np.linspace(0, mem_width, Nx)
        y = np.linspace(0, mem_height, Ny)
        X, Y = np.meshgrid(x, y, indexing='ij') # 2D grids for (x,y)
        phi = np.zeros((max_mode,max_mode,Nx,Ny)) # 4th-order tensor
        for m in range(1, max_mode+1):
            for n in range(1, max_mode+1):
                phi[m-1,n-1,:,:] = np.sin(m*np.pi*X / mem_width) * np.sin(n*np.pi*Y / mem_height)
        w1_space_freq = np.zeros((Nomega, Nx, Ny), dtype=complex) # Displacement of top membrane (w_1)

    if outputType == OutputType.PRESSURE:
        p5_freq = np.zeros(Nomega, dtype=complex)
        k_prime_sq = (M * np.pi / mem_width)**2 + (N * np.pi / mem_height)**2
        phi_listener = (
            np.sin(M * np.pi * x_listen / mem_width)
            * np.sin(N * np.pi * y_listen / mem_height)
        )

    for k, omega in enumerate(tqdm(omegas, desc=f"Solving system", disable=not progressBar)):
        if k == 0:
            continue
        x_sol = solve_system(omega, k)

        # Keep track of displacement
        if outputType == OutputType.DISPLACEMENT:
            w1mn_modes = x_sol[:,:,0]
            w1_space_freq[k, :, :] = np.einsum("mn,mnij->ij", w1mn_modes, phi) # Fill up a layer of the w1 solution (layer for this k-th frequency)

        # Keep track of pressure
        if outputType == OutputType.PRESSURE:
            prop = 1j * omega / c_air
            k_mn = np.sqrt(-prop**2 - k_prime_sq + 0j)
            k_mn = np.where(k_mn.imag < 0, -k_mn, k_mn)
            p5_freq[k] = np.sum(x_sol[:, :, 2] * np.exp(1j * z_listen * k_mn) * phi_listener)

    if outputType == OutputType.PRESSURE:
        return freqs_hz, p5_freq

    if outputType == OutputType.DISPLACEMENT:
        w1_space_time = np.fft.irfft(w1_space_freq, axis=0)
        nt = w1_space_time.shape[0]
        t = np.arange(nt) / fs
        return t, X, Y, w1_space_time

    return None # Fallback