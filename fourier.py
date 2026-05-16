import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 1. Setup Signal Parameters
sampling_rate = 1000  # Hz (Samples per second)
duration = 1.0        # Seconds
N = int(sampling_rate * duration) # Total samples
t = np.linspace(0, duration, N, endpoint=False) # Time vector

# 2. Create a Composite Signal (50Hz and 120Hz sine waves)
# We add two waves with different amplitudes (1.0 and 0.5)
signal = 1.0 * np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# 3. Perform Fourier Transform
# 'fft' computes the transform, 'fftfreq' calculates frequency bins
yf = fft(signal)
xf = fftfreq(N, 1 / sampling_rate)

# 4. Filter for Positive Frequencies
# The FFT output is symmetric; we typically only plot the positive half
xf_positive = xf[:N//2]
yf_magnitude = 2.0/N * np.abs(yf[:N//2]) # Normalize amplitude

# 5. Plotting
plt.figure(figsize=(10, 6))

# Time Domain Plot
plt.subplot(2, 1, 1)
plt.plot(t[:200], signal[:200]) # Plot first 200 samples for clarity
plt.title("Time Domain (Original Signal)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Frequency Domain Plot
plt.subplot(2, 1, 2)
plt.plot(xf_positive, yf_magnitude)
plt.title("Frequency Domain (Fourier Transform)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()