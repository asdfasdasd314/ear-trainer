from typing import List
import numpy as np
from constants import HARMONIC_UPPER_BOUND, HARMONIC_LOWER_BOUND

# Based on the harmonic series:
harmonic_errors = [None, None, 0.5, 0.333, 0.25, 0.2, 0.166, 0.142, 0.125]

def remove_harmonics(peaks: List[int]) -> List[int]:
    """
    Remove the first 8 harmonics of each note
    """
    for i in range(len(peaks) - 1):
        for harmonic in range(2, 9):
            approx_center = peaks[i][1] * harmonic
            upper_bound = approx_center * HARMONIC_UPPER_BOUND
            lower_bound = approx_center * HARMONIC_LOWER_BOUND

            for j in range(i + 1, len(peaks)):
                if peaks[j][1] > lower_bound and peaks[j][1] < upper_bound:
                    peaks[j][0] -= peaks[i][0] * harmonic_errors[harmonic]

    return peaks    

def locate_peaks(spectra: np.ndarray, freqs: np.ndarray, threshold: float) -> np.ndarray:
    peaks = []
    for i in range(len(spectra)):
        if spectra[i] > threshold:
            peaks.append((spectra[i], freqs[i], i))
    return np.array(peaks)


def log_scaled_distance(freq1: float, freq2: float) -> float:
    """
    Notes in music are log-scaled by 2^(1/12), at least in Western music, so use this to adjust the distance between two frequencies

    # F2 = F1 * 2^(k/12)
    # F2 / F1 = 2^(k/12)
    # log2(F2 / F1) = k/12
    # k = 12 * log2(F2 / F1)
    """
    k = 12 * np.log2(freq2 / freq1)
    return k


def cluster_peaks(peaks: np.ndarray, proximity_threshold: float) -> np.ndarray:
    clustered_peaks = []
    curr_cluster = []
    for peak in peaks:
        if len(curr_cluster) == 0 or log_scaled_distance(curr_cluster[-1][1], peak[1]) < proximity_threshold:
            curr_cluster.append(peak)
        else:
            clustered_peaks.append(np.array(curr_cluster))
            curr_cluster = [peak]
    if len(curr_cluster) > 0:
        clustered_peaks.append(np.array(curr_cluster))

    # Perform a weighted average using f1^w1 * f2^w2 * ... * fn^wn as the weighted average
    peaks = []
    for cluster in clustered_peaks:
        # central_peak = cluster.max(axis=0)
        # peaks.append(np.array([central_peak[0], central_peak[1], central_peak[2]]))
        new_amplitude = np.sum(cluster[:, 0]) # Simply add the amplitudes to form the new one
        weights = cluster[:, 0] / new_amplitude
        new_frequency = np.prod(cluster[:, 1] ** weights)
        peaks.append(np.array([new_amplitude, new_frequency, min(cluster[:, 2]), max(cluster[:, 2])]))
    return np.array(peaks)


def peaks_to_midi(peaks: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Use the last two entries to determine the closest MIDI note
    """
    midis = []
    for peak in peaks:
        midi = None
        min_midi = peak[2]
        max_midi = peak[3]
        for i in range(int(min_midi), int(max_midi - 1)):
            if peak[1] > freqs[i] and peak[1] < freqs[i + 1]:
                midi = i
                break

        if midi is None:
            midi = max_midi

        midis.append(midi)

    return np.array(midis)