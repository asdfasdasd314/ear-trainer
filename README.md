# Chord Classifier

## Overview

This is a program I created to classify the chord symbols in a piece of music, kind of like Shazam but for the chords.

## Features

- Custom chord symbol classification "entropy" system
- Audio preprocessing to remove noise + harmonics
- Real-time classification via microphone

## Tech Stack

**Languages**:
- Python

**Libraries**:
- Matplotlib (visualization)
- Numpy (FFT computation & general math)

## Architecture

Audio frames are read in a separate thread (`input_thread`) and passed to a queue ensuring thread safety.

The main loop creates a local copy from which frequencies can be computed. A sequence of preprocessing functions are called to remove noise and harmonics in the sound. The frequencies left over are passed to a chord classification algorithm giving the end chord symbol which is then printed to the console.

As the FFT relies on a non-zero amount of audio to determine frequencies, there will always be some amount of lag in the microphone input and chord symbol output.

### Usage

1. Install dependiences from `requirements.txt` (if `numpy` and `pyaudio` are installed this step can be skipped)
    - `pip3 install -r requirements.txt` or `pip install -r requirements.txt`
2. Run the main file from the project root
    - `python3 src/main.py` or `python src/main.py`
3. Play a song or instrument with clearly voiced chords and the program will classify them
    - ***The program will (most likely) output some incorrect symbols, so generally the latest symbol outputted is most accurate***
    - ***The best results occur with chords that are held for a long time (reduced noise when held) and when no other instruments/voices interfere***