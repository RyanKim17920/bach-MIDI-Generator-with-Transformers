from Data_Extraction.MIDI_data_extractor import MIDI_data_extractor
from data_to_MIDI import data_to_MIDI

data = MIDI_data_extractor(r"C:\Users\ilove\CODING\PYStuff\MusicNet\Midi2Numpy\MIDI-Generator-with-Transformers\Bach MIDIs\Brandenburg Concerto No. 1 in F Major - BWV 1046\bburg1_1.mid")
print(data[:,0]==6)
data_to_MIDI(data, r"test.mid", relative_time=True)

#things work yay