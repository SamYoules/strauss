"""The :obj:`notes` submodule: translating musical note representations

This submodule contains functions for translating between different
representations of musical notes or musical chords, and representative
sound frequencies and MIDI notes.

Attributes:
  tuneC0 (:obj:`float`): The frequency in Hz of the ``C0`` musical
    note
  notecount (:obj:`int`): Semitone offset above C in an octave
  notesharps (:obj:`list`): Names of musical notes using sharp notation
  noteflats (:obj:`list`): Names of musical notes using flat notation.
  semitone_dict (:obj:`dict`): Dictionary of note names to semitone
    offsets above C.
"""

import numpy as np
import pychord as chrd
import re

# Tune note system to A440 standard
# Equal temperement

tuneC0 = 440 * pow(2, (-9./12)-4) 

notecount = np.arange(12)
notesharps  = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
noteflats = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
semitone_dict = {**dict(zip(notesharps, notecount)),
                 **dict(zip(noteflats, notecount))}

def parse_note(notename):
    """ 
    Takes scientific pitch name and returns frequency in Hz.
    Flat and sharp values supported. Assumes equal temperament
    and A4 = 440 Hz tuning (ISO 16)

    Args:
      notename (:obj:`str`): scientific pitch name, in format
        <note><octave>, e.g. 'Ab4', 'E3' or 'F#2'

    Returns:
      out (numerical): Frequency of note in Hertz
    """
    # Regex to capture note (letter, optional #/b) and octave (digits)
    # Ensures the entire string matches this pattern.
    match = re.fullmatch(r"([a-zA-Z][#b]?)([0-9]+)", notename)
    if not match:
        raise ValueError(f"Invalid note format: '{notename}'")

    raw_note_part = match.group(1)
    octave_str = match.group(2)

    if len(raw_note_part) > 1: # Handles "C#", "Db", etc.
        note_component = raw_note_part[0].upper() + raw_note_part[1]
    else: # Handles "C", "A", etc.
        note_component = raw_note_part.upper()

    if note_component not in semitone_dict:
        raise ValueError(f"Invalid note name: '{note_component}' in '{notename}'")

    semi = semitone_dict[note_component]/12.
    octv  = int(octave_str)
    return tuneC0*pow(2.,semi+octv)

def parse_chord(chordname, rootoct=3):
    """
    Takes name of a chord and root octave to generate a valid
    chord voicing as an array of frequencies in Hz, using the
    `pychord` library

    Args:
      chordname (:obj:`str`): Standard chord name, e.g. `'A7'`
        or `'Dm7add9'` etc.
      rootoct (:obj:`int`): Octave number

    Returns:
      out (:obj:`ndarray`) array of frequencies constituting
        chord
    """
    chord = chrd.Chord(chordname)
    rootoct = int(rootoct) # Ensure rootoct is an integer
    notes = chord.components_with_pitch(rootoct)
    frqs = []
    for n in notes:
        frqs.append(parse_note(n))
    return np.array(frqs)

def chord_notes(chordname, rootoct=3):
    """
    Takes name of a chord and root octave to generate a valid
    chord voicing as a list of note names, using the `pychord`
    library

    Args:
      chordname (:obj:`str`): Standard chord name, e.g. `'A7'`
        or `'Dm7add9'` etc.
      rootoct (:obj:`int`): Octave number

    Returns:
      out (:obj:`list`): list of note names constituting chord
    """
    chord = chrd.Chord(chordname)
    notes = chord.components_with_pitch(int(rootoct))
    return notes

def mkey_to_note(val):
    """
    Take MIDI key value and return the note name in scientific
    notation

    Args:
      val (:obj:`int`): MIDI key value

    Returns:
      out (:obj:`str`): scientific pitch name, in format
        `<note><octave>`, e.g. `'E3'` or `'F#2'`
    """
    from strauss.notes import notesharps
    octv = val // 12 - 1
    semi = val % 12
    return f'{notesharps[semi]}{octv}'
