import numpy as np
from typing import List

from constants import IDX_TO_NOTE

# Map an interval to its entropy
ENTROPY_MAP = {
    None: 0, # Easier calculation

    "Is Bass": -3, # If the root is the bass note, for ambiguous chords it often clarifies what the chord is

    "Minor 3rd": 0,
    "Major 3rd": 0,
    "Perfect 5th": 0,

    "Dominant 7th": 1,

    "Major 7th": 2, 

    "Diminished 7th": 3,
    "6th": 3,

    "9th": 4,
    "11th": 4,

    "13th": 5,

    "Sharp 9th": 6,
    "Flat 9th": 6,

    "Diminished 5th": 7,
    "Sharp 11th": 7,

    "Augmented 5th": 8,
    "Flat 13th": 8,

    "Doubled 7th": 100, # This never happens
    "Doubled 9th": 100, # This never happens (b9 + #9 is okay though)
    "Doubled 11th": 100, # This never happens
    "Doubled 13th": 100, # This never happens
}

SYMBOL_MAP = {
    None: "",
    "Major 3rd": "",
    "Minor 3rd": "m",
    "Perfect 5th": "",
    "Diminished 5th": "b5",
    "Augmented 5th": "#5",
    "6th": "6",
    "Diminished 7th": "dim7",
    "Dominant 7th": "7",
    "Major 7th": "Maj7",
    "Flat 9th": "b9",
    "9th": "9",
    "Sharp 9th": "#9",
    "11th": "11",
    "Sharp 11th": "#11",
    "13th": "13",
    "Flat 13th": "b13",
}

def compute_entropy(inpt: np.ndarray, is_bass: bool=False) -> float:
    """
    Input is a single representation of the notes present, where the first index should be filled and is the root note
    """
    entropy = 0

    is_bass = "Is Bass" if is_bass else None

    # Identify tertiary harmony
    third = None
    if inpt[4] == 1:
        third = "Major 3rd"
    elif inpt[3] == 1:
        third = "Minor 3rd"

    fifth = None
    if inpt[7] == 1:
        fifth = "Perfect 5th"
    elif inpt[6] == 1:
        fifth = "Diminished 5th"
    elif inpt[8] == 1:
        fifth = "Augmented 5th"
    
    seventh = None
    if inpt[10] == 1:
        seventh = "Dominant 7th"
    if inpt[11] == 1:
        if seventh != None:
            seventh = "Doubled 7th"
        else:
            seventh = "Major 7th"
    if inpt[9] == 1:
        if seventh != None:
            seventh = "Doubled 7th"
        elif fifth == "Diminished 5th":
            seventh = "Diminished 7th"
        else:
            seventh = "6th"

    flat_ninth = None
    ninth = None
    if inpt[14 % 12] == 1:
        ninth = "9th"
    if inpt[13 % 12] == 1:
        if ninth == "9th":
            ninth = "Doubled 9th"
        else:
            flat_ninth = "Flat 9th"
    if inpt[15 % 12] == 1 and third != "Minor 3rd": # Don't double count minor 3rds
        if flat_ninth == "Flat 9th" or ninth == None:
            ninth = "Sharp 9th"

    eleventh = None
    if inpt[17 % 12] == 1:
        eleventh = "11th"
    if inpt[18 % 12] == 1 and ninth != "Flat 9th" and fifth != "Diminished 5th": # Don't double count diminished 5ths
        if eleventh == "11th":
            eleventh = "Doubled 11th"
        else:
            eleventh = "Sharp 11th"

    thirteenth = None
    if inpt[21 % 12] == 1 and seventh != "6th": # Don't double count 6ths
        thirteenth = "13th"
    if inpt[20 % 12] == 1 and fifth != "Augmented 5th": # Don't double count augmented 5ths
        if thirteenth == "13th":
            thirteenth = "Doubled 13th"
        else:
            thirteenth = "Flat 13th"

    entropy += ENTROPY_MAP[is_bass] + ENTROPY_MAP[third] + ENTROPY_MAP[fifth] + ENTROPY_MAP[seventh] + ENTROPY_MAP[flat_ninth] + ENTROPY_MAP[ninth] + ENTROPY_MAP[eleventh] + ENTROPY_MAP[thirteenth]
    return entropy


def compute_starting_points(inpt: np.ndarray) -> List[int]:
    starting_points = []
    for i in range(len(inpt)):
        if inpt[i] == 1:
            starting_points.append(i)
    return starting_points


def determine_root(inpt: np.ndarray, bass: int=None) -> int:
    starting_points = compute_starting_points(inpt)
    min_entropy = float('inf')
    min_starting_point = None
    for starting_point in starting_points:
        adjusted = np.roll(inpt, -starting_point)
        entropy = compute_entropy(adjusted, bass == starting_point)
        if entropy < min_entropy:
            min_entropy = entropy
            min_starting_point = starting_point

    # if a decent chord symbol can't be constructed, a chord symbol isn't really applicable
    if min_entropy > 100:
        return None
    return min_starting_point


def construct_chord_symbol(root: int or None, midi: List[int]) -> str:
    if root is None:
        return "N/A"

    symbol = ""
    symbol += IDX_TO_NOTE[root]
    inpt = np.roll(midi, -root)

    # Identify tertiary harmony (same structure as entropy computation)
    third = None
    if inpt[4] == 1:
        third = "Major 3rd"
    elif inpt[3] == 1:
        third = "Minor 3rd"

    fifth = None
    if inpt[7] == 1:
        fifth = "Perfect 5th"
    elif inpt[6] == 1:
        fifth = "Diminished 5th"
    elif inpt[8] == 1:
        fifth = "Augmented 5th"
    
    seventh = None
    if inpt[10] == 1:
        seventh = "Dominant 7th"
    if inpt[11] == 1:
        seventh = "Major 7th"
    if inpt[9] == 1:
        if fifth == "Diminished 5th" and third == "Minor 3rd":
            seventh = "Diminished 7th"
        else:
            seventh = "6th"

    flat_ninth = None
    ninth = None
    if inpt[14 % 12] == 1:
        ninth = "9th"
    if inpt[13 % 12] == 1:
        flat_ninth = "Flat 9th"
    if inpt[15 % 12] == 1 and third != "Minor 3rd": # Don't double count minor 3rds
        ninth = "Sharp 9th"

    eleventh = None
    if inpt[17 % 12] == 1:
        eleventh = "11th"
    if inpt[18 % 12] == 1 and fifth != "Diminished 5th": # Don't double count diminished 5ths
        eleventh = "Sharp 11th"

    thirteenth = None
    if inpt[21 % 12] == 1 and seventh != "6th": # Don't double count 6ths
        thirteenth = "13th"
    if inpt[20 % 12] == 1 and fifth != "Augmented 5th": # Don't double count augmented 5ths
        thirteenth = "Flat 13th"

    if third == "Minor 3rd" and fifth == "Diminished 5th" and seventh == "Diminished 7th":
        symbol += "dim7"
        # Don't double count alterations
        fifth = None
        seventh = None
    elif third == "Major 3rd" and fifth == "Augmented 5th" and seventh == None:
        symbol += "aug"
        # Don't double count alterations
        fifth = None
    else:
        symbol += SYMBOL_MAP[third] + SYMBOL_MAP[seventh]

    extensions = []
    if fifth != "Perfect 5th" and fifth != None:
        extensions.append(SYMBOL_MAP[fifth])
    if flat_ninth != None:
        extensions.append(SYMBOL_MAP[flat_ninth])
    if ninth != None:
        extensions.append(SYMBOL_MAP[ninth])
    if eleventh != None:
        extensions.append(SYMBOL_MAP[eleventh])
    if thirteenth != None:
        extensions.append(SYMBOL_MAP[thirteenth])

    if len(extensions) > 0:
        symbol += "("
        symbol += ",".join(extensions)
        symbol += ")"
    return symbol


def determine_chord_symbol(inpt: np.ndarray, bass: int=None) -> str:
    root = determine_root(inpt, bass=bass)
    return construct_chord_symbol(root, inpt)