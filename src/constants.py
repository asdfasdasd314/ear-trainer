import pyaudio

CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
KEYS = ["C minor", "C major", "Db minor", "Db major", "D minor", "D major", "Eb minor", "Eb major", "E minor", "E major", "F minor", "F major", "Gb minor", "Gb major", "G minor", "G major", "Ab minor", "Ab major", "A minor", "A major", "Bb minor", "Bb major", "B minor", "B major"]

KEY_TO_IDX = {key: i for i, key in enumerate(KEYS)}
IDX_TO_KEY = {i: key for i, key in enumerate(KEYS)}

NOTE_TO_IDX = {note: i for i, note in enumerate(NOTES)}
IDX_TO_NOTE = {i: note for i, note in enumerate(NOTES)}