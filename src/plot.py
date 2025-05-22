import matplotlib.pyplot as plot
import numpy as np

def plot_spectrogram(S: np.ndarray):
    plot.imshow(S, origin='lower', aspect='auto', cmap='magma')
    plot.colorbar(label='Log-scaled Magnitude')
    plot.xlabel('Time')
    plot.ylabel('Frequency')
    plot.title('Spectrogram')
    plot.show()
