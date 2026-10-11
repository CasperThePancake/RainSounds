import numpy as np
import matplotlib.pyplot as plt

# RAIN PROPERTIES

density_rainwater = 1000 # Density of water in the raindrop (rho_r)
radius_raindrop = 2.5E-3 # Radius of the raindrop (r)
speed_raindrop = 7 # Rainfall velocity (v_d)

# Model 1: simple sawtooth (Toyoda et al.)

alpha = density_rainwater * np.pi * radius_raindrop**2 * speed_raindrop**2 # Eq. from Toyoda paper
beta = 3/8 * density_rainwater * np.pi * radius_raindrop * speed_raindrop**3 # Eq. from Toyoda paper
T = alpha/beta

def model1(omega):
    if omega == 0:
        return 0
    return alpha / (omega*1j) * (np.exp(1j*omega*T) - 1) - beta/(omega**2) * (1 - np.exp(1j*omega*T) * (1-1j*omega*T) )

dfreq = np.linspace(0,44100,5000)
dOmega = 2*np.pi*dfreq
dq1 = [model1(omega) for omega in dOmega]

# Model 2: smoothed sawtooth (Mitchell et al.)

c = (10*np.sqrt(np.pi)/9)*density_rainwater*(2*radius_raindrop)**2*speed_raindrop**2
tau = (3/10)*(2*radius_raindrop)/speed_raindrop

def model2(omega):
    return c*tau*np.sqrt(np.pi) / (2*(1-1j*omega*tau)**(3/2))

dq2 = [model2(omega) for omega in dOmega]

plt.figure()
plt.title("Comparison of raindrop impact force models (frequency domain, amplitude)")
plt.loglog(dfreq, np.abs(dq1), label="Model 1 (Toyoda)")
plt.loglog(dfreq, np.abs(dq2), label="Model 2 (Mitchell)")
plt.xlabel("f (Hz)")
plt.ylabel("|F(ω)|")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()

plt.figure()
plt.title("Comparison of raindrop impact force models (frequency domain, phase)")
plt.loglog(dfreq, np.angle(dq1), label="Model 1 (Toyoda)")
plt.loglog(dfreq, np.angle(dq2), label="Model 2 (Mitchell)")
plt.xlabel("f (Hz)")
plt.ylabel("∠F(ω)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()