# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from models import solve_raindrop_membrane, OutputType

times, X, Y, sol = solve_raindrop_membrane(1,1,0,0.25E-3,1750,0.52E9,0.45,0,0,None,None,OutputType.DISPLACEMENT,mem_tension_x=250,mem_tension_y=0,max_mode=30)

# W1 DISPLACEMENT (GEMINI)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# 4. Prevent clipping: Set z limits to true absolute max + 15% head room
absolute_peak = np.abs(sol).max()
z_max = absolute_peak * 1.15 if absolute_peak > 0 else 1e-6

# Initialize surface plot
color_range = absolute_peak * 0.20  # Boosts color contrast for small waves
surf = [ax.plot_surface(X, Y, sol[0], cmap="RdBu_r",
                        vmin=-color_range, vmax=color_range,
                        rstride=2, cstride=2, antialiased=True)]

# Axis bounds and labels
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(-z_max, z_max)

ax.set_xlabel("x (m)", labelpad=10)
ax.set_ylabel("y (m)", labelpad=10)
ax.set_zlabel("Displacement (m)", labelpad=10)

# Set initial camera view for optimal 3D perspective
ax.view_init(elev=28, azim=-125)

frame_step = 8
frame_indices = np.arange(0, min(len(times), 2000), frame_step)

def update(i):
    idx = frame_indices[i]
    surf[0].remove()
    surf[0] = ax.plot_surface(X, Y, sol[idx], cmap="RdBu_r",
                              vmin=-color_range, vmax=color_range,
                              rstride=2, cstride=2, antialiased=True)
    ax.set_title(f"t = {times[idx]*1000:.3f} ms")

anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=15, blit=False)

# interval=15ms for high FPS playback (~60 fps)
anim = FuncAnimation(fig, update, frames=len(times), interval=15, blit=False)
plt.show()