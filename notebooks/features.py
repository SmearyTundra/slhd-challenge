import scipy.signal
import numpy as np

class Features():

    # def calc_fft(self, signal, fs = 200):
    #     """ This functions computes the fft of a signal."""
    #     print(self)
    #     print(signal)
    #     fmag = np.abs(np.fft.fft(signal))
    #     f = np.linspace(0, fs // 2, len(signal) // 2)

     #   return f[:len(signal) // 2].copy(), fmag[:len(signal) // 2].copy()

    def autocorr(self, signal):
        """Computes autocorrelation of the signal."""
        signal = np.array(signal)
        return float(np.correlate(signal, signal))

    # skip
    # def calc_centroid(signal, fs):
    #     """Computes the centroid along the time axis."""
    #     time = compute_time(signal, fs)
    #     energy = np.array(signal) ** 2
    #     t_energy = np.dot(np.array(time), np.array(energy))
    #     energy_sum = np.sum(energy)
    #     if energy_sum == 0 or t_energy == 0:
    #         centroid = 0
    #     else:
    #         centroid = t_energy / energy_sum
    #     return centroid

    def negative_turning(self, signal):
        """Computes number of negative turning points of the signal."""
        diff_sig = np.diff(signal)
        array_signal = np.arange(len(diff_sig[:-1]))
        negative_turning_pts = np.where((diff_sig[array_signal] < 0) & (diff_sig[array_signal + 1] > 0))[0]

        return len(negative_turning_pts)

    def positive_turning(self, signal):
        """Computes number of positive turning points of the signal."""
        diff_sig = np.diff(signal)

        array_signal = np.arange(len(diff_sig[:-1]))

        positive_turning_pts = np.where((diff_sig[array_signal + 1] < 0) & (diff_sig[array_signal] > 0))[0]

        return len(positive_turning_pts)

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

    def distance(self, signal): # è la lunghezza bro
        """Calculates the total distance traveled by the signal
        using the hipotenusa between 2 datapoints."""
        diff_sig = np.diff(signal).astype(float)
        return np.sum([np.sqrt(1 + diff_sig ** 2)])

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
        print(signal)
        t = np.arange(0, len(signal))/fs
        return np.sum(0.5 * np.diff(t) * np.abs(np.array(signal[:-1]) + np.array(signal[1:])))

    def pk_pk_distance(self, signal):
        """Computes the peak to peak distance."""
        return np.abs(np.max(signal) - np.min(signal))

    # def entropy(signal, prob='standard'):
    #     """Computes the entropy of the signal using the Shannon Entropy."""

    #     if prob == 'standard':
    #         value, counts = np.unique(signal, return_counts=True)
    #         p = counts / counts.sum()
    #     elif prob == 'kde':
    #         p = kde(signal)
    #     elif prob == 'gauss':
    #         p = gaussian(signal)

    #     if np.sum(p) == 0:
    #         return 0.0

    #     # Handling zero probability values
    #     p = p[np.where(p != 0)]

    #     # If probability all in one value, there is no entropy
    #     if np.log2(len(signal)) == 1:
    #         return 0.0
    #     elif np.sum(p * np.log2(p)) / np.log2(len(signal)) == 0:
    #         return 0.0
    #     else:
    #         return - np.sum(p * np.log2(p)) / np.log2(len(signal))

    # def neighbourhood_peaks(signal, n=10):
    #     """Computes the number of peaks from a defined neighbourhood of the signal."""
    #     signal = np.array(signal)
    #     subsequence = signal[n:-n]
    #     # initial iteration
    #     peaks = ((subsequence > np.roll(signal, 1)[n:-n]) & (subsequence > np.roll(signal, -1)[n:-n]))
    #     for i in range(2, n + 1):
    #         peaks &= (subsequence > np.roll(signal, i)[n:-n])
    #         peaks &= (subsequence > np.roll(signal, -i)[n:-n])
    #     return np.sum(peaks)

    def interq_range(self, signal):
        """Computes interquartile range of the signal."""
        return np.percentile(signal, 75) - np.percentile(signal, 25)

    def kurtosis(self, signal):
        """Computes kurtosis of the signal."""
        return scipy.stats.kurtosis(signal)
    
    # def skewness(self, signal):
    #     """Computes skewness of the signal."""
    #     return scipy.stats.skew(signal)
    
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
    
    #def ecdf(self, signal, d=10):
    #    """Computes the values of ECDF (empirical cumulative distribution function) along the time axis."""
    #    _, y = np.sort(signal), np.arange(1, len(signal)+1)/len(signal)
    #    if len(signal) <= d:
    #        return tuple(y)
    #    else:
    #        return tuple(y[:d])
    
    #def ecdf_percentile(self, signal, percentile=[0.2, 0.8]):
    #    """Computes the percentile values of the ECDF."""
    #    signal = np.array(signal)
    #    if isinstance(percentile, str):
    #        percentile = eval(percentile)
    #    if isinstance(percentile, (float, int)):
    #        percentile = [percentile]
#
    #    # calculate ecdf
    #    x, y = np.sort(signal), np.arange(1, len(signal)+1)/len(signal)
#
    #    if len(percentile) > 1:
    #        # check if signal is constant
    #        if np.sum(np.diff(signal)) == 0:
    #            return tuple(np.repeat(signal[0], len(percentile)))
    #        else:
    #            return tuple([x[y <= p].max() for p in percentile])
    #    else:
    #        # check if signal is constant
    #        if np.sum(np.diff(signal)) == 0:
    #            return signal[0]
    #        else:
    #            return x[y <= percentile].max()

    
    # def spectral_distance(signal, fs):
    #     """Computes the signal spectral distance."""
    #     f, fmag = calc_fft(signal, fs)

    #     cum_fmag = np.cumsum(fmag)

    #     # Computing the linear regression
    #     points_y = np.linspace(0, cum_fmag[-1], len(cum_fmag))

    #     return np.sum(points_y - cum_fmag)

    # TODO: reinserire
    # def fundamental_frequency(signal, fs):
    #     """Computes fundamental frequency of the signal.
    #     The fundamental frequency integer multiple best explain
    #     the content of the signal spectrum.
    #     """
    #     signal = signal - np.mean(signal)
    #     f, fmag = calc_fft(signal, fs)

    #     # Finding big peaks, not considering noise peaks with low amplitude

    #     bp = scipy.signal.find_peaks(fmag, height=max(fmag) * 0.3)[0]

    #     # # Condition for offset removal, since the offset generates a peak at frequency zero
    #     bp = bp[bp != 0]
    #     if not list(bp):
    #         f0 = 0
    #     else:
    #         # f0 is the minimum big peak frequency
    #         f0 = f[min(bp)]

    #     return f0
    
    # def max_frequency(signal, fs):
    #     """Computes maximum frequency of the signal."""
    #     f, fmag = calc_fft(signal, fs)
    #     cum_fmag = np.cumsum(fmag)

    #     try:
    #         ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.95)[0][0]
    #     except IndexError:
    #         ind_mag = np.argmax(cum_fmag)

    #     return f[ind_mag]
    
    # def median_frequency(signal, fs):
    #     """Computes median frequency of the signal."""
    #     f, fmag = calc_fft(signal, fs)
    #     cum_fmag = np.cumsum(fmag)
    #     try:
    #         ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.50)[0][0]
    #     except IndexError:
    #         ind_mag = np.argmax(cum_fmag)
    #     f_median = f[ind_mag]

    #     return f_median
    
    # def spectral_decrease(signal, fs):
    #     """Represents the amount of decreasing of the spectra amplitude."""
    #     f, fmag = calc_fft(signal, fs)

    #     fmag_band = fmag[1:]
    #     len_fmag_band = np.arange(2, len(fmag) + 1)

    #     # Sum of numerator
    #     soma_num = np.sum((fmag_band - fmag[0]) / (len_fmag_band - 1), axis=0)

    #     if not np.sum(fmag_band):
    #         return 0
    #     else:
    #         # Sum of denominator
    #         soma_den = 1 / np.sum(fmag_band)

    #         # Spectral decrease computing
    #         return soma_den * soma_num

    
    # def spectral_variation(signal, fs):
    #     """Computes the amount of variation of the spectrum along time.

    #     Spectral variation is computed from the normalized cross-correlation between two consecutive amplitude spectra.

    #     Description and formula in Article:
    #     The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    #     Authors Peeters G., Giordano B., Misdariis P., McAdams S.
    #     """
    #     f, fmag = calc_fft(signal, fs)

    #     sum1 = np.sum(np.array(fmag)[:-1] * np.array(fmag)[1:])
    #     sum2 = np.sum(np.array(fmag)[1:] ** 2)
    #     sum3 = np.sum(np.array(fmag)[:-1] ** 2)

    #     if not sum2 or not sum3:
    #         variation = 1
    #     else:
    #         variation = 1 - (sum1 / ((sum2 ** 0.5) * (sum3 ** 0.5)))

    #     return variation
    
    # def human_range_energy(signal, fs):
    #     """Computes the human range energy ratio.
    #     The human range energy ratio is given by the ratio between the energy
    #     in frequency 0.6-2.5Hz and the whole energy band.
    #     """
    #     f, fmag = calc_fft(signal, fs)

    #     allenergy = np.sum(fmag ** 2)

    #     if allenergy == 0:
    #         # For handling the occurrence of Nan values
    #         return 0.0

    #     hr_energy = np.sum(fmag[np.argmin(np.abs(0.6 - f)):np.argmin(np.abs(2.5 - f))] ** 2)

    #     ratio = hr_energy / allenergy

    #    return ratio
    
    #def fft_mean_coeff(self, signal, fs=200, nfreq=256):
    #    """Computes the mean value of each spectrogram frequency.
    #    nfreq can not be higher than half signal length plus one.
    #    When it does, it is automatically set to half signal length plus one.
    #    """
    #    if nfreq > len(signal) // 2 + 1:
    #        nfreq = len(signal) // 2 + 1
#
    #    fmag_mean = scipy.signal.spectrogram(signal, fs, nperseg=nfreq * 2 - 2)[2].mean(1)
#
    #    return tuple(fmag_mean)
