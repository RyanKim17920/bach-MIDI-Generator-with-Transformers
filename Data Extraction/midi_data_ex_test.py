from MIDI_data_extractor import MIDI_data_extractor

data = MIDI_data_extractor(r"../Bach MIDIs\18 Leipzig Chorale Preludes for Organ/bwv656.mid", 0, True)
print(data)
#print(np.unique(data[:, -4]))