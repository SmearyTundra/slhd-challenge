import scipy.signal
import numpy as np

class Features():

    # def autocorr(self, signal):
    #     """Computes autocorrelation of the signal."""
    #     signal = np.array(signal)
    #     return float(np.correlate(signal, signal))

    # def negative_turning(self, signal):
    #     """Computes number of negative turning points of the signal."""
    #     diff_sig = np.diff(signal)
    #     array_signal = np.arange(len(diff_sig[:-1]))
    #     negative_turning_pts = np.where((diff_sig[array_signal] < 0) & (diff_sig[array_signal + 1] > 0))[0]

    #     return len(negative_turning_pts)

    # def positive_turning(self, signal):
    #     """Computes number of positive turning points of the signal."""
    #     diff_sig = np.diff(signal)

    #     array_signal = np.arange(len(diff_sig[:-1]))

    #     positive_turning_pts = np.where((diff_sig[array_signal + 1] < 0) & (diff_sig[array_signal] > 0))[0]

    #     return len(positive_turning_pts)

    def mean_abs_diff(self, signal):
        """Computes mean absolute differences of the signal."""
        return np.mean(np.abs(np.diff(signal)))

    def mean_diff(self, signal):
        """Computes mean of differences of the signal."""
        return np.mean(np.diff(signal))

    def median_abs_diff(self, signal):
        """Computes median absolute differences of the signal."""
        return np.median(np.abs(np.diff(signal)))

    def median_diff(self, signal):
        """Computes median of differences of the signal."""
        return np.median(np.diff(signal))

    # def distance(self, signal): # è la lunghezza bro
    #     """Calculates the total distance traveled by the signal
    #     using the hipotenusa between 2 datapoints."""
    #     diff_sig = np.diff(signal).astype(float)
    #     return np.sum([np.sqrt(1 + diff_sig ** 2)])

    def sum_abs_diff(self, signal):
        """Computes sum of absolute differences of the signal."""
        return np.sum(np.abs(np.diff(signal)))

    def zero_cross(self, signal):
        """Computes Zero-crossing rate of the signal."""
        return len(np.where(np.diff(np.sign(signal)))[0])


    def total_energy(self, signal, fs = 200):
        """Computes the total energy of the signal."""
        time = np.arange(0, len(signal))/fs
        return np.sum(np.array(signal) ** 2) / (time[-1] - time[0])

    def slope(self, signal):
        """Slope is computed by fitting a linear equation to the observed data."""
        t = np.linspace(0, len(signal) - 1, len(signal))
        return np.polyfit(t, signal, 1)[0]
    
    def auc(self, signal, fs = 200):
        """Computes the area under the curve of the signal computed with trapezoid rule."""
        t = np.arange(0, len(signal))/fs
        return np.sum(0.5 * np.diff(t) * np.abs(np.array(signal[:-1]) + np.array(signal[1:])))

    def pk_pk_distance(self, signal):
        """Computes the peak to peak distance."""
        return np.abs(np.max(signal) - np.min(signal))

    # def interq_range(self, signal):
    #     """Computes interquartile range of the signal."""
    #     return np.percentile(signal, 75) - np.percentile(signal, 25)

    def kurtosis(self, signal):
        """Computes kurtosis of the signal."""
        return scipy.stats.kurtosis(signal)
    
    def calc_max(self, signal):
        """Computes the maximum value of the signal."""
        return np.max(signal)

    def calc_min(self, signal):
        """Computes the minimum value of the signal."""
        return np.min(signal)
    
    def calc_mean(self, signal):
        """Computes mean value of the signal."""
        return np.mean(signal)
    
    def calc_median(self, signal):
        """Computes median of the signal."""
        return np.median(signal)
    
    def mean_abs_deviation(self, signal):
        """Computes mean absolute deviation of the signal."""
        return np.mean(np.abs(signal - np.mean(signal, axis=0)), axis=0)
    
    def median_abs_deviation(self, signal):
        """Computes median absolute deviation of the signal."""
        return scipy.stats.median_abs_deviation(signal, scale=1)
    
    def calc_std(self, signal):
        """Computes standard deviation (std) of the signal."""
        return np.std(signal)
    
    def calc_var(self, signal):
        """Computes variance of the signal."""
        return np.var(signal)

    def fundamental_frequency(self, signal, fs = 200):
        """Computes fundamental frequency of the signal.
        The fundamental frequency integer multiple best explain
        the content of the signal spectrum.
        """
        signal = signal - np.mean(signal)
        fmag = np.abs(np.fft.fft(signal))
        f = np.linspace(0, fs // 2, len(signal) // 2)
        f, fmag = f[:len(signal) // 2].copy(), fmag[:len(signal) // 2].copy()

        # Finding big peaks, not considering noise peaks with low amplitude

        bp = scipy.signal.find_peaks(fmag, height=max(fmag) * 0.3)[0]

        # # Condition for offset removal, since the offset generates a peak at frequency zero
        bp = bp[bp != 0]
        if not list(bp):
            f0 = 0
        else:
            # f0 is the minimum big peak frequency
            f0 = f[min(bp)]

        return f0
