# LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
from models import solve_raindrop_membrane, OutputType

# MATERIAL PROPERTIES

def max_displacement(tension=250):
    times, X, Y, sol = solve_raindrop_membrane(1,1,tension,0.25E-3,1750,0.52E9,0.45,25,0,None,None,OutputType.DISPLACEMENT,max_mode=30)

    absolute_peak = np.max(sol)
    print(absolute_peak)

    return absolute_peak

tensions = np.linspace(10,1000,5)
maxdisplacements = []

for tension in tensions:
    maxdisplacements.append(max_displacement(tension))

plt.figure()
plt.plot(tensions,maxdisplacements)
plt.xlabel("Tension (N/m)")
plt.ylabel("Maximum displacement (m)")
plt.show()