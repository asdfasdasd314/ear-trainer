import pyaudio
CHUNK = 2048
FORMAT = pyaudio.paInt16
MAX_VALUE = 32768
CHANNELS = 1
RATE = 44100
BUCKET_SIZE = int(RATE * 1) # 1 second
MIN_FREQ = 20
MAX_FREQ = 12544 # MIDI 127

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
KEYS = ["C minor", "C major", "Db minor", "Db major", "D minor", "D major", "Eb minor", "Eb major", "E minor", "E major", "F minor", "F major", "Gb minor", "Gb major", "G minor", "G major", "Ab minor", "Ab major", "A minor", "A major", "Bb minor", "Bb major", "B minor", "B major"]

CHROMA_TO_NOTE = {0: "C4", 1: "C#4", 2: "D4", 3: "E-4", 4: "E4", 5: "F4", 6: "F#4", 7: "G4", 8: "A-4", 9: "A4", 10: "B-4", 11: "B4"}

SHARP_TO_ENHARMONIC_FLAT = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}

KEY_TO_IDX = {key: i for i, key in enumerate(KEYS)}
IDX_TO_KEY = {i: key for i, key in enumerate(KEYS)}

NOTE_TO_IDX = {note: i for i, note in enumerate(NOTES)}
IDX_TO_NOTE = {i: note for i, note in enumerate(NOTES)}