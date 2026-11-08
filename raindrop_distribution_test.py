import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad
from math import gamma
from time import sleep

# PARAMETERS

rainfall_rate = 0.2 # Rainfall rate (mm/h)
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

import os
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.spatial import cKDTree

SOUND_FOLDER = "cached_rain"
RUN_TIME = 600      # seconds
DROP_GAIN = 0.5    # per-drop volume; the soft limiter handles pile-ups
BLOCKSIZE = 1024   # raise to 2048 if you still hear dropouts
FADE_OUT_MS = 30   # raise if you still hear clicks at the ends
FADE_IN_MS = 2

# --- Load all sounds as float32 stereo arrays ---

sounds, params = [], []
sample_rate = None

for fname in os.listdir(SOUND_FOLDER):
    if not fname.lower().endswith(".wav"):
        continue
    x, y, r = (float(p) for p in os.path.splitext(fname)[0].split("_"))
    data, sr = sf.read(os.path.join(SOUND_FOLDER, fname), dtype="float32", always_2d=True)
    if sample_rate is None:
        sample_rate = sr
    elif sr != sample_rate:
        raise ValueError(f"{fname} is {sr} Hz, expected {sample_rate} Hz")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)   # mono -> stereo
        data = data[:, :2].copy()

    # Fade edges so the audio always starts and ends at exactly zero
    fade_out = min(int(sample_rate * FADE_OUT_MS / 1000), len(data))
    fade_in = min(int(sample_rate * FADE_IN_MS / 1000), len(data))
    data[-fade_out:] *= np.cos(np.linspace(0, np.pi / 2, fade_out, dtype="float32"))[:, None] ** 2
    data[:fade_in] *= np.sin(np.linspace(0, np.pi / 2, fade_in, dtype="float32"))[:, None] ** 2

    data = np.ascontiguousarray(data * DROP_GAIN)
    params.append((x, y, r))
    sounds.append(data)

params = np.array(params)
p_min = params.min(axis=0)
p_range = params.max(axis=0) - p_min
p_range[p_range == 0] = 1.0

def normalise(p):
    return (np.asarray(p) - p_min) / p_range

tree = cKDTree(normalise(params))
print(f"Loaded {len(sounds)} impact sounds at {sample_rate} Hz.")

# --- Software mixer ---

voices = []               # each voice is [audio_array, play_position]
voices_lock = threading.Lock()

def audio_callback(outdata, frames, time_info, status):
    outdata.fill(0)
    with voices_lock:
        still_playing = []
        for v in voices:
            data, pos = v
            n = min(frames, len(data) - pos)
            outdata[:n] += data[pos:pos + n]
            v[1] += n
            if v[1] < len(data):
                still_playing.append(v)
        voices[:] = still_playing
    np.tanh(outdata, out=outdata)   # soft limiter: no hard clipping

stream = sd.OutputStream(
    samplerate=sample_rate,
    channels=2,
    dtype="float32",
    blocksize=BLOCKSIZE,
    latency="high",
    callback=audio_callback,
)
stream.start()

def play_impact(x, y, radius_m):
    _, idx = tree.query(normalise((x, y, radius_m)))
    with voices_lock:
        voices.append([sounds[idx], 0])

# --- Simulation loop ---

total_drops = 0
uptime = 0

while True:
    next_bin = int(np.argmin(bin_countdowns))

    uptime += bin_countdowns[next_bin]
    if uptime > RUN_TIME:
        break

    sleep(bin_countdowns[next_bin])

    total_drops += 1

    event_diameter = np.random.uniform(
        low=min_diameter + (max_diameter - min_diameter) / bins * next_bin,
        high=min_diameter + (max_diameter - min_diameter) / bins * (next_bin + 1),
    )
    event_x0 = np.random.uniform(low=0, high=membrane_width)
    event_y0 = np.random.uniform(low=0, high=membrane_height)
    event_radius_m = (event_diameter / 2) / 1000  # mm -> m

    play_impact(event_x0, event_y0, event_radius_m)

    bin_countdowns -= bin_countdowns[next_bin]
    bin_countdowns[next_bin] = np.random.exponential(scale=1 / bin_rates[next_bin])

sleep(2)  # let the last drops finish
stream.stop()
stream.close()
print("Simulation finished!")