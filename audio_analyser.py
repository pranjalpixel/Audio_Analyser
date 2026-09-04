#importing libraries
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog

#mp3 file input
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select the song you want to analyse",
    filetypes=[
        ("Audio Files", "*.mp3 *.wav")
    ]
)

#checking if the user selected a file
if file_path == "":
    print("No audio file selected.")
    exit()



#data printing
print("Does the file exist?", os.path.exists(file_path))
print("File path:", os.path.abspath(file_path))
song_name = os.path.basename(file_path)
audio, sample_rate = librosa.load(file_path)
print("\nAudio loaded successfully!")
print("Selected File:", os.path.basename(file_path))
print("Sample Rate:", sample_rate)
print("Number of Samples:", len(audio))


#graph plotting 
duration = len(audio) / sample_rate

#plotting frequency magnitude spectrum
fft = np.fft.fft(audio)

magnitude = np.abs(fft)
frequencies = np.fft.fftfreq(len(audio), 1/sample_rate)

positive_frequencies = frequencies[:len(frequencies)//2]
positive_magnitude = magnitude[:len(magnitude)//2]

bass = (positive_frequencies >= 20) & (positive_frequencies < 250)
mid = (positive_frequencies >= 250) & (positive_frequencies < 4000)
treble = (positive_frequencies >= 4000) & (positive_frequencies <= 10000)

bass_energy = np.mean(positive_magnitude[bass])
mid_energy = np.mean(positive_magnitude[mid])
treble_energy = np.mean(positive_magnitude[treble])

print("\n______Audio Analysis Results______")
print("Bass Energy:", bass_energy)
print("Mid Energy:", mid_energy)
print("Treble Energy:", treble_energy)


#plot graphs for bass, mid, and treble frequencies
if bass_energy > mid_energy and bass_energy > treble_energy:
    dominant = "BASS "

elif mid_energy > bass_energy and mid_energy > treble_energy:
    dominant = "MID "

else:
    dominant = "TREBLE "


print(f"Dominant Frequency Range: {dominant}")

#spectrogram plotting
D = librosa.stft(audio)
spectrogram = librosa.amplitude_to_db(abs(D), ref=np.max)



#plot all graphs

#1st graph
fig = plt.figure(figsize=(16, 13))
fig.suptitle(f"Audio Analysis for {song_name}", fontsize=20, fontweight="bold")
grid = fig.add_gridspec(
    3,
    2,
    height_ratios=[1, 1, 0.7]
)
ax_waveform = fig.add_subplot(grid[0, 0])
ax_frequency = fig.add_subplot(grid[0, 1])
ax_energy = fig.add_subplot(grid[1, 0])
ax_spectrogram = fig.add_subplot(grid[1, 1])
ax_report = fig.add_subplot(grid[2, :])
librosa.display.waveshow(
    audio,
    sr=sample_rate,
    ax=ax_waveform
)
ax_waveform.set_title("Audio Waveform")
ax_waveform.set_xlabel("Time (seconds)")
ax_waveform.set_ylabel("Amplitude")


#2nd graph
ax_frequency.plot(
    positive_frequencies,
    positive_magnitude
)
ax_frequency.set_title("Frequency Spectrum")
ax_frequency.set_xlabel("Frequency (Hz)")
ax_frequency.set_ylabel("Magnitude")
ax_frequency.set_xlim(0, 10000)

#3rd graph
categories = ["Bass", "Mid", "Treble"]
energy = [
    bass_energy,
    mid_energy,
    treble_energy
]
ax_energy.bar(
    categories,
    energy
)
ax_energy.set_title("Bass vs Mid vs Treble")
ax_energy.set_xlabel("Frequency Range")
ax_energy.set_ylabel("Average Magnitude")

#4th graph
image = librosa.display.specshow(
    spectrogram,
    sr=sample_rate,
    x_axis="time",
    y_axis="hz",
    ax=ax_spectrogram
)
ax_spectrogram.set_title("Audio Spectrogram")
ax_spectrogram.set_xlabel("Time")
ax_spectrogram.set_ylabel("Frequency")
fig.colorbar(
    image,
    ax=ax_spectrogram,
    format="%+2.0f dB"
)

report = f"""
AUDIO ANALYSIS REPORT
Duration: {duration:.2f} seconds
Sample Rate: {sample_rate} Hz
Bass Energy: {bass_energy:.2f}
Mid Energy: {mid_energy:.2f}
Treble Energy: {treble_energy:.2f}
Dominant Range: {dominant}
"""
ax_report.axis("off")
ax_report.text(0.5, 0.5, report, fontsize=14, ha="center", va="center")
plt.subplots_adjust(
    hspace=0.35,
    wspace=0.12
)
plt.show()