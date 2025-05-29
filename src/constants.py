import pyaudio

CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
KEYS = ["C minor", "C major", "Db minor", "Db major", "D minor", "D major", "Eb minor", "Eb major", "E minor", "E major", "F minor", "F major", "Gb minor", "Gb major", "G minor", "G major", "Ab minor", "Ab major", "A minor", "A major", "Bb minor", "Bb major", "B minor", "B major"]