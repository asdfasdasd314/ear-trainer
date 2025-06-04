from typing import Optional, Tuple, List
import random
from enum import Enum

from constants import IDX_TO_NOTE, NOTE_TO_IDX, BLUES_PROGRESSIONS, NUMERAL_TO_IDX

class Seventh(Enum):
    NONE = "none"
    DIMINISHED = "dim7"
    DOMINANT = "7"
    MAJOR = "maj7"

class Third(Enum):
    NONE = "none"
    MAJOR = "M"
    MINOR = "m"
    DIMINISHED = "dim"
    AUGMENTED = "aug"
    SUS2 = "sus2"
    SUS4 = "sus4"

class Chord:
    def __init__(self, numeral: str, tonal_center_idx: int, third: Third, seventh: Seventh):
        self.numeral = numeral
        self.tonal_center_idx = tonal_center_idx
        self.third = third
        self.seventh = seventh


    def to_efficient_format(self) -> str:
        root = IDX_TO_NOTE[(self.tonal_center_idx + NUMERAL_TO_IDX[self.numeral]) % 12]
        symbol = f"{root}"
        if self.third != Third.NONE:
            symbol += f"-{self.third.value}"
        if self.seventh != Seventh.NONE:
            symbol += f"-{self.seventh.value}"
        return symbol
    

    def to_standard_format(self) -> str:
        pass


class BluesProgression:
    def __init__(self, tonal_center_idx: int, progression_idx: int, loops: int, tempo: int):
        self.tonal_center_idx = tonal_center_idx
        self.num_loops = loops
        self.tempo = tempo

        progression = BLUES_PROGRESSIONS[progression_idx]
        self.progression = []
        for bars in progression:
            for bar in bars:
                if isinstance(bar, list):
                    chords = []
                    for chord in bar:
                        chords.append(self.parse_chord_numeral(chord))
                    self.progression.append(chords)

                else:
                    self.progression.append(self.parse_chord_numeral(bar))

    
    def parse_chord_numeral(self, chord: str) -> Chord:
        third = Third.NONE
        seventh = Seventh.NONE
        numeral = ""
        if "-" in chord:
            parts = chord.split("-")
            numeral = parts[0]
            if len(parts) > 1:
                third = parts[1]
                third = Third(third)
            if len(parts) > 2:
                seventh = parts[2]
                seventh = Seventh(seventh)
        else:
            numeral = chord

        return Chord(numeral, self.tonal_center_idx, third, seventh)


    def output_to_file(self, file_path: str) -> str:
        seconds_per_bar = 60 * 4 / self.tempo
        start_time = 0.00000
        with open(file_path, "w") as f:
            for i in range(self.num_loops):
                for bar in self.progression:
                    if isinstance(bar, list):
                        num_chords = len(bar)
                        seconds_per_chord = seconds_per_bar / num_chords
                        for c in bar:
                            f.write(f"{start_time:.4f} {start_time + seconds_per_chord:.4f} {c.to_efficient_format()}\n")
                            start_time += seconds_per_chord
                    else:
                        f.write(f"{start_time:.4f} {start_time + seconds_per_bar:.4f} {bar.to_efficient_format()}\n")
                        start_time += seconds_per_bar