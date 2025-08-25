import matplotlib.pyplot as plot
import numpy as np
import time
import librosa
from constants import IDX_TO_KEY, IDX_TO_NOTE

def plot_sound_waves(y: np.ndarray):
    plot.plot(y)
    plot.xlabel('Time')
    plot.ylabel('Amplitude')
    plot.title('Sound Waves')
    plot.show()