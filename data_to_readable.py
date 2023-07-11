from mido import tick2second
from MIDI_data_extractor import MIDI_data_extractor

# Mapping from MIDI note numbers to note names
note_mapping = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B"
}
INSTRUMENTS = [
    'Acoustic Grand Piano',
    'Bright Acoustic Piano',
    'Electric Grand Piano',
    'Honky-tonk Piano',
    'Electric Piano 1',
    'Electric Piano 2',
    'Harpsichord',
    'Clavi',
    'Celesta',
    'Glockenspiel',
    'Music Box',
    'Vibraphone',
    'Marimba',
    'Xylophone',
    'Tubular Bells',
    'Dulcimer',
    'Drawbar Organ',
    'Percussive Organ',
    'Rock Organ',
    'Church Organ',
    'Reed Organ',
    'Accordion',
    'Harmonica',
    'Tango Accordion',
    'Acoustic Guitar (nylon)',
    'Acoustic Guitar (steel)',
    'Electric Guitar (jazz)',
    'Electric Guitar (clean)',
    'Electric Guitar (muted)',
    'Overdriven Guitar',
    'Distortion Guitar',
    'Guitar harmonics',
    'Acoustic Bass',
    'Electric Bass (finger)',
    'Electric Bass (pick)',
    'Fretless Bass',
    'Slap Bass 1',
    'Slap Bass 2',
    'Synth Bass 1',
    'Synth Bass 2',
    'Violin',
    'Viola',
    'Cello',
    'Contrabass',
    'Tremolo Strings',
    'Pizzicato Strings',
    'Orchestral Harp',
    'Timpani',
    'String Ensemble 1',
    'String Ensemble 2',
    'SynthStrings 1',
    'SynthStrings 2',
    'Choir Aahs',
    'Voice Oohs',
    'Synth Voice',
    'Orchestra Hit',
    'Trumpet',
    'Trombone',
    'Tuba',
    'Muted Trumpet',
    'French Horn',
    'Brass Section',
    'SynthBrass 1',
    'SynthBrass 2',
    'Soprano Sax',
    'Alto Sax',
    'Tenor Sax',
    'Baritone Sax',
    'Oboe',
    'English Horn',
    'Bassoon',
    'Clarinet',
    'Piccolo',
    'Flute',
    'Recorder',
    'Pan Flute',
    'Blown Bottle',
    'Shakuhachi',
    'Whistle',
    'Ocarina',
    'Lead 1 (square)',
    'Lead 2 (sawtooth)',
    'Lead 3 (calliope)',
    'Lead 4 (chiff)',
    'Lead 5 (charang)',
    'Lead 6 (voice)',
    'Lead 7 (fifths)',
    'Lead 8 (bass + lead)',
    'Pad 1 (new age)',
    'Pad 2 (warm)',
    'Pad 3 (polysynth)',
    'Pad 4 (choir)',
    'Pad 5 (bowed)',
    'Pad 6 (metallic)',
    'Pad 7 (halo)',
    'Pad 8 (sweep)',
    'FX 1 (rain)',
    'FX 2 (soundtrack)',
    'FX 3 (crystal)',
    'FX 4 (atmosphere)',
    'FX 5 (brightness)',
    'FX 6 (goblins)',
    'FX 7 (echoes)',
    'FX 8 (sci-fi)',
    'Sitar',
    'Banjo',
    'Shamisen',
    'Koto',
    'Kalimba',
    'Bag pipe',
    'Fiddle',
    'Shanai',
    'Tinkle Bell',
    'Agogo',
    'Steel Drums',
    'Woodblock',
    'Taiko Drum',
    'Melodic Tom',
    'Synth Drum',
    'Reverse Cymbal',
    'Guitar Fret Noise',
    'Breath Noise',
    'Seashore',
    'Bird Tweet',
    'Telephone Ring',
    'Helicopter',
    'Applause',
    'Gunshot'
]
key_sig_dict = {0: 'A', 1: 'A#m', 2: 'Ab', 3: 'Abm', 4: 'Am', 5: 'B', 6: 'Bb', 7: 'Bbm', 8: 'Bm',
                                 9: 'C', 10: 'C#', 11: 'C#m', 12: 'Cb', 13: 'Cm', 14: 'D', 15: 'D#m', 16: 'Db',
                                 17: 'Dm', 18: 'E', 19: 'Eb', 20: 'Ebm', 21: 'Em', 22: 'F', 23: 'F#', 24: 'F#m',
                                 25: 'Fm', 26: 'G', 27: 'G#m', 28: 'Gb', 29: 'Gm'}
'''[note_on_note, note_on_velocity, note_off_note, note_off_velocity,
       control_change_control, control_change_value, program_change_program,
       end_of_track, set_tempo_tempo,
       time_sig_num, itme_sig_den, time_sig_clocksperclick, time_sig_notated_32nd,
       key_sig(turn into numbers), [time (only shown during tests)], instrument_number, instrument_type, orig_instrument_number]'''


def matrix_to_readable_data(matrix):
    readable_data = []
    for row in matrix:
        note_on_note = row[0]
        note_on_velocity = row[1]
        note_off_note = row[2]
        note_off_velocity = row[3]
        control_change_control = row[4]
        control_change_value = row[5]
        program_change_program = row[6]
        end_of_track = row[7]
        set_tempo_tempo = row[8]
        time_sig_num = row[9]
        time_sig_den = row[10]
        time_sig_clocksperclick = row[11]
        time_sig_notated_32nd = row[12]
        key_sig = row[13]
        time_ticks = row[14]
        instrument_number = row[15]
        instrument_type = row[16]
        orig_instrument_number = row[17]

        event_data = []

        if note_on_note != -1:
            note_name = note_mapping[note_on_note % 12]
            octave = note_on_note // 12 - 1
            event_data.append(f"Note On: {note_name}{octave}, Velocity: {note_on_velocity}")

        if note_off_note != -1:
            note_name = note_mapping[note_off_note % 12]
            octave = note_off_note // 12 - 1
            event_data.append(f"Note Off: {note_name}{octave}, Velocity: {note_off_velocity}")

        if control_change_control != -1:
            event_data.append(f"Control Change: Control {control_change_control}, Value: {control_change_value}")

        if program_change_program != -1:
            event_data.append(f"Program Change: {INSTRUMENTS[program_change_program - 1]}")

        if end_of_track != -1:
            event_data.append("End of Track")

        if set_tempo_tempo != -1:
            event_data.append(f"Set Tempo: {set_tempo_tempo}")

        if time_sig_num != -1:
            event_data.append(f"Time Signature: {time_sig_num}/{time_sig_den}, Clocks per Click: {time_sig_clocksperclick}, Notated 32nd: {time_sig_notated_32nd}")

        if key_sig != -1:
            event_data.append(f"Key Signature: {key_sig_dict[key_sig]}")

        if time_ticks != -1:
            time_seconds = tick2second(time_ticks, 480, 500000)
            #DEFINATELY WRONG --> but allows for relative time

            event_data.append(f"Time: {time_seconds} seconds")

        if instrument_number != -1:
            event_data.append(f"Instrument Number: {INSTRUMENTS[instrument_number - 1]}")

        if instrument_type != -1:
            event_data.append(f"Instrument Type: {instrument_type}")

        if orig_instrument_number != -1:
            event_data.append(f"Original Instrument Number: {INSTRUMENTS[orig_instrument_number - 1]}")

        readable_data.append(event_data)

    return readable_data


matrix = MIDI_data_extractor(r"C:\Users\ilove\Downloads\Passacaglia_-_The_Impossible_Duet.mid")
readable_data = matrix_to_readable_data(matrix)

for i in readable_data:
    print(i)