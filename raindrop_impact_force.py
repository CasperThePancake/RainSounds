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

def model1(t):
    if t < 0 or t > T:
        return 0
    else:
        return alpha-beta*t

dt = np.linspace(-T/2,T+T/2,1000)
df1 = [model1(t) for t in dt]

# Model 2: smoothed sawtooth (Mitchell et al.)

c = (10*np.sqrt(np.pi)/9)*density_rainwater*(2*radius_raindrop)**2*speed_raindrop**2
tau = (3/10)*(2*radius_raindrop)/speed_raindrop

def model2(t):
    if t < 0:
        return 0
    else:
        return c*np.sqrt(t/tau)*np.exp(-t/tau)

df2 = [model2(t) for t in dt]

plt.figure()
plt.plot(dt,df1,label="Model 1 (Toyoda)")
plt.plot(dt,df2,label="Model 2 (Mitchell)")
plt.xlabel("t (s)")
plt.ylabel("F(t)")
plt.legend()
plt.show()