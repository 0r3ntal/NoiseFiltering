import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import pywt
import matplotlib.pyplot as plt


class NoiseFilter:
    def __init__(self, time, signal):
        """
        Initialize the NoiseFilter class.

        :param time: Pandas datetime64 object representing the time data.
        :param signal: Numpy array or pandas Series representing the signal data.
        """
        self.time = time
        self.signal = signal

    def savitzky_golay(self, window_length, polyorder, deriv=0):
        """
        Apply the Savitzky-Golay filter to the signal.

        :param window_length: The length of the filter window (positive odd integer).
        :param polyorder: The order of the polynomial used to fit the samples (integer).
        :param deriv: The order of the derivative to compute (default is 0, which means only smoothing).
        :return: Filtered time and signal as two separate arrays.
        """
        filtered_signal = savgol_filter(self.signal, window_length, polyorder, deriv=deriv)
        return self.time, filtered_signal

    def moving_average(self, window_length):
        """
        Apply the moving average filter to the signal.

        :param window_length: The length of the moving average window (positive integer).
        :return: Filtered time and signal as two separate arrays.
        """
        cumsum = np.cumsum(np.insert(self.signal, 0, 0))
        filtered_signal = (cumsum[window_length:] - cumsum[:-window_length]) / window_length
        return self.time[window_length - 1:], filtered_signal

    def wavelet_denoising(self, wavelet='db8', level=2, mode='soft'):
        """
        Apply wavelet denoising to the signal.

        :param wavelet: The type of wavelet to use (default is 'db8').
        :param level: The level of wavelet decomposition (default is 2).
        :param mode: The thresholding mode to use (default is 'soft', can also be 'hard').
        :return: Filtered time and signal as two separate arrays.
        """
        coeffs = pywt.wavedec(self.signal, wavelet, level=level)
        threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(self.signal)))

        if mode == 'soft':
            def soft_threshold(x, threshold):
                return pywt.threshold(x, threshold, mode='soft')

            coeffs[1:] = [soft_threshold(c, threshold) for c in coeffs[1:]]
        else:
            coeffs[1:] = [pywt.threshold(c, threshold, mode='hard') for c in coeffs[1:]]

        filtered_signal = pywt.waverec(coeffs, wavelet)
        return self.time, filtered_signal

if __name__ == '__main__':
    # Generate sample data
    time = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
    signal = np.random.normal(0, 1, len(time))

    # Create a NoiseFilter instance
    filter_instance = NoiseFilter(time, signal)

    # Apply Savitzky-Golay filter
    time_filtered_sg, signal_filtered_sg = filter_instance.savitzky_golay(window_length=7, polyorder=3)

    # Apply moving average filter
    time_filtered_ma, signal_filtered_ma = filter_instance.moving_average(window_length=5)

    # Apply wavelet denoising
    time_filtered_wd, signal_filtered_wd = filter_instance.wavelet_denoising(wavelet='db8', level=2, mode='soft')

    # Plot the results
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(time, signal, label='Original Signal', alpha=0.5)
    plt.plot(time_filtered_sg, signal_filtered_sg, label='Savitzky-Golay Filtered', linewidth=2)
    plt.plot(time_filtered_ma, signal_filtered_ma, label='Moving Average Filtered', linewidth=2)
    plt.plot(time_filtered_wd, signal_filtered_wd, label='Wavelet Denoised', linewidth=2)
    plt.legend(loc='best')

    ax = plt.gca()
    ax.set_xlabel('Time', color='blue', fontsize=14)
    ax.set_ylabel('Signal', color='blue', fontsize=14)
    ax.xaxis.label.set_color('blue')
    ax.yaxis.label.set_color('blue')
    ax.tick_params(axis='x', colors='blue', labelsize=12)
    ax.tick_params(axis='y', colors='blue', labelsize=12)
    plt.title('Noise Filtering Techniques', fontsize=16)
    plt.savefig('noise_filter_example.png')
    plt.show()



