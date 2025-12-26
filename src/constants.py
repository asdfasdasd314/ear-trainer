import pyaudio
import numpy as np

CHUNK = 2048
FORMAT = pyaudio.paInt16
MAX_VALUE = 32768
CHANNELS = 1
RATE = 44100
HOP_SIZE = 6400
BUCKET_SIZE = 44100 # 32768
MIN_FREQ = 20
MAX_FREQ = 8400 # 4200 is the highest note on a piano with 88 keys
HARMONIC_REDUCTION = 100 # constant `C` of A1 = A0 * (1 + C/F)
STRENGTH_THRESHOLD_1 = 20 # The initial threshold to have a peak
STRENGTH_THRESHOLD_2 = 40 # The threshold to have a peak after removing harmonics
REGULAR_HISTORY_LENGTH = 2
ATTACK_THRESHOLD_1 = 80 # If any peaks are above this we have likely detected the "attack" of a chord, so don't use the first threshold
ATTACK_THRESHOLD_2 = 130
ATTACK_HISTORY_LENGTH = 4
PROXIMITY_THRESHOLD = 0.87 # Basically the number of keys between two frequencies before they are the "same"
HARMONIC_UPPER_BOUND = 2 ** (np.log2(440 * 2 ** (PROXIMITY_THRESHOLD / 12)) - np.log2(440)) # This times a central frequency gives the upper bound to check for harmonics
HARMONIC_LOWER_BOUND = 1 / HARMONIC_UPPER_BOUND # This times a central frequency gives the lower bound to check for harmonics

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
KEYS = ["C minor", "C major", "Db minor", "Db major", "D minor", "D major", "Eb minor", "Eb major", "E minor", "E major", "F minor", "F major", "Gb minor", "Gb major", "G minor", "G major", "Ab minor", "Ab major", "A minor", "A major", "Bb minor", "Bb major", "B minor", "B major"]

CHROMA_TO_NOTE = {0: "C4", 1: "C#4", 2: "D4", 3: "E-4", 4: "E4", 5: "F4", 6: "F#4", 7: "G4", 8: "A-4", 9: "A4", 10: "B-4", 11: "B4"}

SHARP_TO_ENHARMONIC_FLAT = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}

KEY_TO_IDX = {key: i for i, key in enumerate(KEYS)}
IDX_TO_KEY = {i: key for i, key in enumerate(KEYS)}

NOTE_TO_IDX = {note: i for i, note in enumerate(NOTES)}
IDX_TO_NOTE = {i: note for i, note in enumerate(NOTES)}