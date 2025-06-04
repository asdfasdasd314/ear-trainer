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

BLUES_PROGRESSIONS = [
    # Dominant 7ths
    [["I-M-7", "I-M-7", "I-M-7", "I-M-7"], ["IV-M-7", "IV-M-7", "I-M-7", "I-M-7"], ["V-M-7", "IV-M-7", "I-M-7", "I-M-7"]],
    [["I-M-7", "I-M-7", "I-M-7", "I-M-7"], ["IV-M-7", "IV-M-7", "I-M-7", "I-M-7"], ["V-M-7", "IV-M-7", "I-M-7", "IV-M-7"]],
    [["I-M-7", "I-M-7", "I-M-7", "I-M-7"], ["IV-M-7", "IV-M-7", "I-M-7", "I-M-7"], ["V-M-7", "IV-M-7", "I-M-7", ["I-M-7", "IV-M-7"]]],
    [["I-M-7", "IV-M-7", "I-M-7", "I-M-7"], ["IV-M-7", "IV-M-7", "I-M-7", "I-M-7"], ["V-M-7", "IV-M-7", "I-M-7", "I-M-7"]],
    [["I-M-7", "IV-M-7", "I-M-7", "I-M-7"], ["IV-M-7", "IV-M-7", "I-M-7", "I-M-7"], ["V-M-7", "IV-M-7", "I-M-7", "IV-M-7"]],
    [["I-M-7", "IV-M-7", "I-M-7", "I-M-7"], ["IV-M-7", "IV-M-7", "I-M-7", "I-M-7"], ["V-M-7", "IV-M-7", "I-M-7", ["I-M-7", "IV-M-7"]]],

    # Minor 7ths
    [["I-m-7", "I-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-m-7", "IV-m-7", "I-m-7", "I-m-7"]],
    [["I-m-7", "I-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-m-7", "IV-m-7", "I-m-7", "IV-m-7"]],
    [["I-m-7", "I-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-m-7", "IV-m-7", "I-m-7", ["I-m-7", "IV-m-7"]]],
    [["I-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-m-7", "IV-m-7", "I-m-7", "I-m-7"]],
    [["I-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-m-7", "IV-m-7", "I-m-7", "IV-m-7"]],
    [["I-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-m-7", "IV-m-7", "I-m-7", ["I-m-7", "IV-m-7"]]],

    # Mixed major and minor 7ths
    [["I-m-7", "I-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-m-7", "IV-m-7", "I-m-7", "V-M-7"]],
    [["I-m-7", "I-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-M-7", "IV-m-7", "I-m-7", "I-m-7"]],
    [["I-m-7", "I-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-M-7", "IV-m-7", "I-m-7", "IV-m-7"]],
    [["I-m-7", "I-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-M-7", "IV-m-7", "I-m-7", "V-M-7"]],
    [["I-m-7", "I-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-M-7", "IV-m-7", "I-m-7", ["I-m-7", "IV-m-7"]]],
    [["I-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-m-7", "IV-m-7", "I-m-7", "V-M-7"]],
    [["I-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-M-7", "IV-m-7", "I-m-7", "I-m-7"]],
    [["I-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-M-7", "IV-m-7", "I-m-7", "IV-m-7"]],
    [["I-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-M-7", "IV-m-7", "I-m-7", "V-M-7"]],
    [["I-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["V-M-7", "IV-m-7", "I-m-7", ["I-m-7", "IV-m-7"]]],

    # The Thrill is Gone
    [["I-m-7", "I-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["bVI-M-maj7", "V-M-7", "I-m-7", "I-m-7"]],
    [["I-m-7", "I-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["bVI-M-7", "V-M-7", "I-m-7", "I-m-7"]],
    [["I-m-7", "I-m-7", "I-m-7", "I-m-7"], ["IV-m-7", "IV-m-7", "I-m-7", "I-m-7"], ["bVI-M-7", "V-M-7", "I-m-7", "V-M-7"]],

    # 8 Bar Blues
    [["I-M-7", "V-M-7", "IV-M-7", "IV-M-7"], ["I-M-7", ["V-M-7", "IV-M-7"], "I-M-7", "V-M-7"]],
    [["I-M-7", "I-M-7", "IV-M-7", "IV-M-7"], ["I-M-7", "V-M-7", ["I-M-7", "IV-M-7"], ["I-M-7", "V-M-7"]]],
]

CHORD_NUMERALS = ["I", "bII", "II", "bIII", "III", "IV", "bV", "V", "bVI", "VI", "bVII", "VII"]
NUMERAL_TO_IDX = {numeral: idx for idx, numeral in enumerate(CHORD_NUMERALS)}