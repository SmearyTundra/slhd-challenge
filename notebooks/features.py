import scipy.signal
import numpy as np


def compute_time(signal, fs):
    """Creates the signal correspondent time array."""
    return np.arange(0, len(signal))/fs

def kde(features):
    """Computes the probability density function of the input signal using a Gaussian KDE (Kernel Density Estimate)"""
    features_ = np.copy(features)
    xx = create_xx(features_)

    if min(features_) == max(features_):
        noise = np.random.randn(len(features_)) * 0.0001
        features_ = np.copy(features_ + noise)

    kernel = scipy.stats.gaussian_kde(features_, bw_method='silverman')

    return np.array(kernel(xx) / np.sum(kernel(xx)))

def create_xx(features):
    """Computes the range of features amplitude for the probability density function calculus."""

    features_ = np.copy(features)

    if max(features_) < 0:
        max_f = - max(features_)
        min_f = min(features_)
    else:
        min_f = min(features_)
        max_f = max(features_)

    if min(features_) == max(features_):
        xx = np.linspace(min_f, min_f + 10, len(features_))
    else:
        xx = np.linspace(min_f, max_f, len(features_))

    return xx

def gaussian(features):
    """Computes the probability density function of the input signal using a Gaussian function"""

    features_ = np.copy(features)

    xx = create_xx(features_)
    std_value = np.std(features_)
    mean_value = np.mean(features_)

    if std_value == 0:
        return 0.0
    pdf_gauss = scipy.stats.norm.pdf(xx, mean_value, std_value)

    return np.array(pdf_gauss / np.sum(pdf_gauss))

def calc_fft(signal, fs):
    """ This functions computes the fft of a signal."""

    fmag = np.abs(np.fft.fft(signal))
    f = np.linspace(0, fs // 2, len(signal) // 2)

    return f[:len(signal) // 2].copy(), fmag[:len(signal) // 2].copy()

def wavelet(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT (continuous wavelet transform) of the signal."""

    if isinstance(function, str):
        function = eval(function)

    if isinstance(widths, str):
        widths = eval(widths)

    cwt = scipy.signal.cwt(signal, function, widths)

    return cwt

def create_symmetric_matrix(acf, order=11):
    """Computes a symmetric matrix."""

    smatrix = np.empty((order, order))
    xx = np.arange(order)
    j = np.tile(xx, order)
    i = np.repeat(xx, order)
    smatrix[i, j] = acf[np.abs(i - j)]

    return smatrix


def calc_ecdf(signal):
    """Computes the ECDF of the signal."""
    return np.sort(signal), np.arange(1, len(signal)+1)/len(signal)


def autocorr(signal):
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

def negative_turning(signal):
    """Computes number of negative turning points of the signal."""
    diff_sig = np.diff(signal)
    array_signal = np.arange(len(diff_sig[:-1]))
    negative_turning_pts = np.where((diff_sig[array_signal] < 0) & (diff_sig[array_signal + 1] > 0))[0]

    return len(negative_turning_pts)

def positive_turning(signal):
    """Computes number of positive turning points of the signal."""
    diff_sig = np.diff(signal)

    array_signal = np.arange(len(diff_sig[:-1]))

    positive_turning_pts = np.where((diff_sig[array_signal + 1] < 0) & (diff_sig[array_signal] > 0))[0]

    return len(positive_turning_pts)

def mean_abs_diff(signal):
    """Computes mean absolute differences of the signal."""
    return np.mean(np.abs(np.diff(signal)))

def mean_diff(signal):
    """Computes mean of differences of the signal."""
    return np.mean(np.diff(signal))

def median_abs_diff(signal):
    """Computes median absolute differences of the signal."""
    return np.median(np.abs(np.diff(signal)))

def median_diff(signal):
    """Computes median of differences of the signal."""
    return np.median(np.diff(signal))

def distance(signal): # è la lunghezza bro
    """Calculates the total distance traveled by the signal
    using the hipotenusa between 2 datapoints."""
    diff_sig = np.diff(signal).astype(float)
    return np.sum([np.sqrt(1 + diff_sig ** 2)])

def sum_abs_diff(signal):
    """Computes sum of absolute differences of the signal."""
    return np.sum(np.abs(np.diff(signal)))

def zero_cross(signal):
    """Computes Zero-crossing rate of the signal."""
    return len(np.where(np.diff(np.sign(signal)))[0])


def total_energy(signal, fs):
    """Computes the total energy of the signal."""
    time = compute_time(signal, fs)
    return np.sum(np.array(signal) ** 2) / (time[-1] - time[0])

def slope(signal):
    """Slope is computed by fitting a linear equation to the observed data."""
    t = np.linspace(0, len(signal) - 1, len(signal))
    return np.polyfit(t, signal, 1)[0]
 
def auc(signal, fs):
    """Computes the area under the curve of the signal computed with trapezoid rule."""
    t = compute_time(signal, fs)
    return np.sum(0.5 * np.diff(t) * np.abs(np.array(signal[:-1]) + np.array(signal[1:])))

def pk_pk_distance(signal):
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

def interq_range(signal):
    """Computes interquartile range of the signal."""
    return np.percentile(signal, 75) - np.percentile(signal, 25)

def kurtosis(signal):
    """Computes kurtosis of the signal."""
    return scipy.stats.kurtosis(signal)
 
# def skewness(signal):
#     """Computes skewness of the signal."""
#     return scipy.stats.skew(signal)
 
def calc_max(signal):
    """Computes the maximum value of the signal."""
    return np.max(signal)

def calc_min(signal):
    """Computes the minimum value of the signal."""
    return np.min(signal)
 
def calc_mean(signal):
    """Computes mean value of the signal."""
    return np.mean(signal)
 
def calc_median(signal):
    """Computes median of the signal."""
    return np.median(signal)
 
def mean_abs_deviation(signal):
    """Computes mean absolute deviation of the signal."""
    return np.mean(np.abs(signal - np.mean(signal, axis=0)), axis=0)
 
def median_abs_deviation(signal):
    """Computes median absolute deviation of the signal."""
    return scipy.stats.median_abs_deviation(signal, scale=1)
 
def calc_std(signal):
    """Computes standard deviation (std) of the signal."""
    return np.std(signal)
 
def calc_var(signal):
    """Computes variance of the signal."""
    return np.var(signal)
 
def ecdf(signal, d=10):
    """Computes the values of ECDF (empirical cumulative distribution function) along the time axis."""
    _, y = calc_ecdf(signal)
    if len(signal) <= d:
        return tuple(y)
    else:
        return tuple(y[:d])
 
def ecdf_percentile(signal, percentile=[0.2, 0.8]):
    """Computes the percentile values of the ECDF."""
    signal = np.array(signal)
    if isinstance(percentile, str):
        percentile = eval(percentile)
    if isinstance(percentile, (float, int)):
        percentile = [percentile]

    # calculate ecdf
    x, y = calc_ecdf(signal)

    if len(percentile) > 1:
        # check if signal is constant
        if np.sum(np.diff(signal)) == 0:
            return tuple(np.repeat(signal[0], len(percentile)))
        else:
            return tuple([x[y <= p].max() for p in percentile])
    else:
        # check if signal is constant
        if np.sum(np.diff(signal)) == 0:
            return signal[0]
        else:
            return x[y <= percentile].max()

 
# def spectral_distance(signal, fs):
#     """Computes the signal spectral distance."""
#     f, fmag = calc_fft(signal, fs)

#     cum_fmag = np.cumsum(fmag)

#     # Computing the linear regression
#     points_y = np.linspace(0, cum_fmag[-1], len(cum_fmag))

#     return np.sum(points_y - cum_fmag)


def fundamental_frequency(signal, fs):
    """Computes fundamental frequency of the signal.
    The fundamental frequency integer multiple best explain
    the content of the signal spectrum.
    """
    signal = signal - np.mean(signal)
    f, fmag = calc_fft(signal, fs)

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
 
def fft_mean_coeff(signal, fs, nfreq=256):
    """Computes the mean value of each spectrogram frequency.
    nfreq can not be higher than half signal length plus one.
    When it does, it is automatically set to half signal length plus one.
    """
    if nfreq > len(signal) // 2 + 1:
        nfreq = len(signal) // 2 + 1

    fmag_mean = scipy.signal.spectrogram(signal, fs, nperseg=nfreq * 2 - 2)[2].mean(1)

    return tuple(fmag_mean)
 
# def wavelet_abs_mean(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
#     """Computes CWT absolute mean value of each wavelet scale."""
#     return tuple(np.abs(np.mean(wavelet(signal, function, widths), axis=1)))

# def lpc(signal, n_coeff=12):
#     """Computes the linear prediction coefficients.

#     Implementation details and description in:
#     https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf
#     """

#     if signal.ndim > 1:
#         raise ValueError("Only 1 dimensional arrays are valid")
#     if n_coeff > signal.size:
#         raise ValueError("Input signal must have a length >= n_coeff")

#     # Calculate the order based on the number of coefficients
#     order = n_coeff - 1

#     # Calculate LPC with Yule-Walker
#     acf = np.correlate(signal, signal, 'full')

#     r = np.zeros(order+1, 'float32')
#     # Assuring that works for all type of input lengths
#     nx = np.min([order+1, len(signal)])
#     r[:nx] = acf[len(signal)-1:len(signal)+order]

#     smatrix = create_symmetric_matrix(r[:-1], order)

#     if np.sum(smatrix) == 0:
#         return tuple(np.zeros(order+1))

#     lpc_coeffs = np.dot(np.linalg.inv(smatrix), -r[1:])

#     return tuple(np.concatenate(([1.], lpc_coeffs)))

# def lpcc(signal, n_coeff=12):
#     """Computes the linear prediction cepstral coefficients.

#     Implementation details and description in:
#     http://www.practicalcryptography.com/miscellaneous/machine-learning/tutorial-cepstrum-and-lpccs/
#     """
#     # 12-20 cepstral coefficients are sufficient for speech recognition
#     lpc_coeffs = lpc(signal, n_coeff)

#     if np.sum(lpc_coeffs) == 0:
#         return tuple(np.zeros(len(lpc_coeffs)))

#     # Power spectrum
#     powerspectrum = np.abs(np.fft.fft(lpc_coeffs)) ** 2
#     lpcc_coeff = np.fft.ifft(np.log(powerspectrum))

#     return tuple(np.abs(lpcc_coeff))

# def filterbank(signal, fs, pre_emphasis=0.97, nfft=512, nfilt=40):
#     """Computes the MEL-spaced filterbank.

#     It provides the information about the power in each frequency band.

#     Implementation details and description on:
#     https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
#     https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fnref:1
#     """

#     # Signal is already a window from the original signal, so no frame is needed.
#     # According to the references it is needed the application of a window function such as
#     # hann window. However if the signal windows don't have overlap, we will lose information,
#     # as the application of a hann window will overshadow the windows signal edges.

#     # pre-emphasis filter to amplify the high frequencies

#     emphasized_signal = np.append(np.array(signal)[0], np.array(signal[1:]) - pre_emphasis * np.array(signal[:-1]))

#     # Fourier transform and Power spectrum
#     mag_frames = np.absolute(np.fft.rfft(emphasized_signal, nfft))  # Magnitude of the FFT

#     pow_frames = ((1.0 / nfft) * (mag_frames ** 2))  # Power Spectrum

#     low_freq_mel = 0
#     high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
#     mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
#     hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
#     filter_bin = np.floor((nfft + 1) * hz_points / fs)

#     fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
#     for m in range(1, nfilt + 1):

#         f_m_minus = int(filter_bin[m - 1])  # left
#         f_m = int(filter_bin[m])  # center
#         f_m_plus = int(filter_bin[m + 1])  # right

#         for k in range(f_m_minus, f_m):
#             fbank[m - 1, k] = (k - filter_bin[m - 1]) / (filter_bin[m] - filter_bin[m - 1])
#         for k in range(f_m, f_m_plus):
#             fbank[m - 1, k] = (filter_bin[m + 1] - k) / (filter_bin[m + 1] - filter_bin[m])

#     # Area Normalization
#     # If we don't normalize the noise will increase with frequency because of the filter width.
#     enorm = 2.0 / (hz_points[2:nfilt + 2] - hz_points[:nfilt])
#     fbank *= enorm[:, np.newaxis]

#     filter_banks = np.dot(pow_frames, fbank.T)
#     filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
#     filter_banks = 20 * np.log10(filter_banks)  # dB

#     return filter_banks

# def mfcc(signal, fs, pre_emphasis=0.97, nfft=512, nfilt=40, num_ceps=12, cep_lifter=22):
#     """Computes the MEL cepstral coefficients.

#     It provides the information about the power in each frequency band.

#     Implementation details and description on:
#     https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
#     https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fnref:1
#     """
#     filter_banks = filterbank(signal, fs, pre_emphasis, nfft, nfilt)

#     mel_coeff = scipy.fft.dct(filter_banks, type=2, axis=0, norm='ortho')[1:(num_ceps + 1)]  # Keep 2-13

#     mel_coeff -= (np.mean(mel_coeff, axis=0) + 1e-8)

#     # liftering
#     ncoeff = len(mel_coeff)
#     n = np.arange(ncoeff)
#     lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)  # cep_lifter = 22 from python_speech_features library

#     mel_coeff *= lift

#     return tuple(mel_coeff)
