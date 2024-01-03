from MIDI_data_extractor import MIDI_data_extractor

print(MIDI_data_extractor(r"C:\Users\ilove\CODING\PYStuff\MusicNet\Midi2Numpy\MIDI-Generator-with-Transformers\Bach MIDIs\18 Leipzig Chorale Preludes for Organ\bwv651.mid"))
"""
Outputs: 2D array of MIDI data:
        Size: (num_messages, 11)

        index 0: type of data (start/end token, note_on, control_change, program_change, etc.)
        index 1-6: data values (extra indexes included for future expansion)
        index 7: time (relative or absolute)
        index 8: current instrument type (removable)
        index 9: instrument number (2nd violin for example)
        index 10: original instrument type

    Ordering of importance in values (for model):
        0: start/end token
        1: program change
        2: control change
        3: time signature
        4: key signature
        5: tempo
        6: note_on
"""