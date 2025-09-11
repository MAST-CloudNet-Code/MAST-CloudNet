import numpy as np
from scipy.io import wavfile
from scipy.signal import welch, butter, filtfilt
import matplotlib.pyplot as plt

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def analyze_filtered_psd(wav_file_path, low, high):
    try:
        sample_rate, data = wavfile.read(wav_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{wav_file_path}' was not found.")
        return

    if data.ndim > 1:
        data = data[:, 0]

    filtered_data = bandpass_filter(data, sample_rate, low, high)

    # Calculate PSD on the filtered data
    frequencies, psd = welch(filtered_data, sample_rate, nperseg=16384)

    # Find the peak
    min_freq = low
    max_freq = high
    freq_range = (frequencies >= min_freq) & (frequencies <= max_freq)

    if np.any(freq_range):
        peak_index = np.argmax(psd[freq_range])
        peak_frequency = frequencies[freq_range][peak_index]
        peak_psd = psd[freq_range][peak_index]

        # Plotting the result
        plt.figure(figsize=(12, 6))
        plt.plot(frequencies, psd, color='#4dbbd5', alpha=0.9)
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (V^2/Hz)')

        # Focus the plot on the required range
        plt.xlim(min_freq, max_freq)
        plt.ylim(psd[freq_range].min(), psd[freq_range].max()*5)

        # Annotate the peak
        plt.plot(peak_frequency, peak_psd, 'o', color='#e64b35', markersize=8)
        plt.annotate(f'Peak: {peak_frequency:.2f} Hz', xy=(peak_frequency, peak_psd),
                    xytext=(peak_frequency + 10, peak_psd), arrowprops=dict(facecolor='#e64b35', shrink=0.05))

        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()
    else:
        print(f"No frequencies found in the specified range ({min_freq}-{max_freq} Hz).")


if __name__ == '__main__':
    # Replace with file path
    wav_file = './sweep.wav'
    # plot spectrum
    analyze_filtered_psd(wav_file, low=100, high=900)
