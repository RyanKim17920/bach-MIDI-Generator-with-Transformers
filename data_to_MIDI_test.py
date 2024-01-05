from Data_Extraction.MIDI_data_extractor import MIDI_data_extractor
from data_to_MIDI import data_to_MIDI

data = MIDI_data_extractor(r"C:\Users\ilove\CODING\PYStuff\MusicNet\Midi2Numpy\MIDI-Generator-with-Transformers\Bach MIDIs\18 Leipzig Chorale Preludes for Organ\bwv651.mid")
print(data[0:100])
print(data[-100:])
data_to_MIDI(data, r"bwv651.mid", relative_time=True)

#things work yay