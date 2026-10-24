import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad
from math import gamma
from time import sleep

# PARAMETERS

rainfall_rate = 0.4 # Rainfall rate (mm/h)
min_diameter = 1 # Minimum raindrop diameter to simulate (mm)
max_diameter = 10 # Maximum raindrop diameter to simulate (mm)
mean_diameter = 2.5 # Mean raindrop diameter (climate-based, determines other parameters) (mm)
bins = 10 # Amount of bins to simulate for (more bins = more accurate)
membrane_width = 1 # Width of membrane upon which rain impacts (m)
membrane_height = 1 # Height of membrane upon which rain impacts (m)

# DETERMINING RESULTING PARAMETERS

# Area

area = membrane_width*membrane_height

# Mu

def D(mu):
    return (4+mu)/(0.0365*mu**2 + 0.735*mu + 1.935)

res_mu = root_scalar(lambda mu: D(mu) - mean_diameter, bracket=[-3.0, 20.0], method="brentq")
var_mu = res_mu.root

print(f"Solved mu: {var_mu}")

# Lambda

var_lambda = 0.0365*var_mu**2 + 0.735*var_mu + 1.935

# N0

def R(N0): # Equation (19)
    return 6*np.pi*1E-4*N0*gamma(4+var_mu)*(9.65/var_lambda**(4+var_mu) - 10.3 / (var_lambda+0.6)**(4+var_mu))

var_N0 = root_scalar(lambda N0: R(N0) - rainfall_rate, x0=5000.0, x1=10000.0, method="secant").root

print(f"Solved N0: {var_N0}")

# DEFINE N(D) FUNCTION

def N(D):
    return var_N0 * D**var_mu * np.exp(-var_lambda*D)

# DEFINE TERMINAL VELOCITY FUNCTION

def v(D):
    return 9.65 - 10.3*np.exp(-0.6*D)

# ARRIVAL RATE PER BIN (s⁻¹)

bin_rates = np.zeros(bins)
bin_countdowns = np.zeros(bins)

for i in range(bins):
    bin_start = min_diameter + (max_diameter-min_diameter)/bins * i
    bin_end = min_diameter + (max_diameter-min_diameter)/bins * (i+1)

    def integrand(D):
        return N(D) * area * v(D)

    result, abs_err = quad(integrand, bin_start, bin_end)
    bin_rates[i] = result

    bin_countdowns[i] = np.random.exponential(scale=1/bin_rates[i])

# LIVE SIMULATION

total_drops = 0

while True:
    next_bin = int(np.argmin(bin_countdowns)) # Find next bin to be triggered

    sleep(bin_countdowns[next_bin]) # Sleep until next bin triggered

    total_drops += 1

    print(f"Raindrop #{total_drops} with diameter {np.random.uniform(low=min_diameter + (max_diameter-min_diameter)/bins * next_bin,high=min_diameter + (max_diameter-min_diameter)/bins * (next_bin+1))} mm at ({np.random.uniform(low=0,high=membrane_width)},{np.random.uniform(low=0,high=membrane_height)})") # Trigger the bin (raindrop impacts)

    bin_countdowns -= bin_countdowns[next_bin]

    bin_countdowns[next_bin] = np.random.exponential(scale=1/bin_rates[next_bin]) # Regenerate triggered bin