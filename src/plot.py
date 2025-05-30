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


def plot_spectrogram(S: np.ndarray):
    # Create the plot only once
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    fig, ax = plot.subplots()
    line, = ax.plot(S_db[0])  # initial plot

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')
    ax.set_title('Spectrogram Frame-by-Frame')

    plot.ion()  # turn on interactive mode
    plot.show()

    for i in range(1, S_db.shape[0]):
        line.set_ydata(S_db[i])  # update data
        ax.set_ylim(S_db.min(), S_db.max())  # adjust y-axis if needed
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)  # control frame rate (e.g., 20 FPS)

    plot.ioff()  # turn off interactive mode
    plot.close()


def plot_model_training(losses: list[float]):
    plot.plot(losses)
    plot.xlabel('Epoch')
    plot.ylabel('Loss')
    plot.title('Model Training Loss')
    plot.show()


def plot_tonal_center_predictions(predictions: np.ndarray, labels: np.ndarray):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plot.subplots(2, 1, figsize=(10, 8))
    
    # Convert indices to note names for y-axis
    note_names = [IDX_TO_NOTE[i] for i in range(12)]
    
    # Plot predictions
    ax1.plot(predictions, label='Predictions')
    ax1.set_ylim(-1, 12)
    ax1.set_yticks(range(12))
    ax1.set_yticklabels(note_names)
    ax1.set_title('Predictions')
    ax1.legend()
    
    # Plot labels
    ax2.plot(labels, label='Labels', color='red')
    ax2.set_ylim(-1, 12)
    ax2.set_yticks(range(12))
    ax2.set_yticklabels(note_names)
    ax2.set_title('Labels')
    ax2.legend()
    
    # Adjust layout to prevent overlap
    plot.tight_layout()
    plot.show()