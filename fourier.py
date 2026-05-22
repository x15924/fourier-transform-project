import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import librosa
import soundfile as sf

signal, sr = librosa.load("song.mp3", sr=None)
duration = librosa.get_duration(path="song.mp3")
N = int(sr * duration)
t = np.linspace(0, duration, N, endpoint=False)

yf = fft(signal)
xf = fftfreq(N, 1 / sr)

xf_positive = xf[:N//2]
yf_magnitude = 2.0/N * np.abs(yf[:N//2]) # Normalize amplitude

plt.figure(figsize=(10, 12))

plt.subplot(4, 1, 1)
plt.plot(t[:len(signal)], signal[:len(signal)])
plt.title("Time Domain (Original Signal)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(xf_positive, yf_magnitude)
plt.title("Frequency Domain (Fourier Transform of Original)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()

freq_shift = 20000
shift = np.exp(2j * np.pi * freq_shift * t[:len(signal)])
signal_shifted = np.real(signal*shift)
yf_shifted = fft(signal_shifted)
yf_shifted_magnitude = 2.0/N * np.abs(yf_shifted[:N//2])

plt.subplot(4, 1, 3)
plt.plot(t[:len(signal_shifted)], signal_shifted[:len(signal_shifted)])
plt.title("Time Domain (Shifted Signal)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(xf_positive, yf_shifted_magnitude)
plt.title("Frequency Domain (Fourier Transform of Shifted Signal)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.tight_layout()
plt.savefig("plot.png")

sf.write("output.wav", signal_shifted, sr)
# print(len(t), len(signal))
# print(N, duration)