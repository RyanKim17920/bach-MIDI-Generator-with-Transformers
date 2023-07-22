from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from MIDI_data_extractor import MIDI_data_extractor
from tqdm import tqdm


def MIDI_merger(input, output):
    matrix = MIDI_data_extractor(input)
    midi_file = MidiFile()
    tracks = {}
    tracks_t_time = {}
    # print(matrix)
    '''[note_on_note, note_on_velocity, note_off_note, note_off_velocity,
        control_change_control, control_change_value, program_change_program,
        end_of_track, set_tempo_tempo,
        time_sig_num, itme_sig_den, time_sig_clocksperclick, time_sig_notated_32nd,
        key_sig(turn into numbers), [time], instrument_type, instrument_num, orig_instrument_type]'''
    for i in tqdm(range(len(matrix))):
        cur_name = f"o{matrix[i][-1]}"
        # print(matrix[i])
        if cur_name == f"o-1":
            cur_name = list(tracks.keys())[0]
            time = matrix[i][-4] - tracks_t_time[cur_name]
            tracks_t_time[cur_name] = matrix[i][-4]
            if matrix[i][6] != -1:
                tracks[cur_name].append(MetaMessage('set_tempo', tempo=bpm2tempo(20 + matrix[i][6]), time=time))
            if matrix[i][7] != -1:
                tracks[cur_name].append(MetaMessage('time_signature', numerator=matrix[i][7], denominator=matrix[i][8],
                                                    clocks_per_click=matrix[i][9],
                                                    notated_32nd_notes_per_beat=matrix[i][10], time=time))
        elif cur_name not in tracks:
            tracks_t_time[cur_name] = 0
            time = 0
            tracks[cur_name] = MidiTrack()
            tracks[cur_name].append(Message('program_change', program=matrix[i][-1], time=time))

        else:
            time = matrix[i][-4] - tracks_t_time[cur_name]
            tracks_t_time[cur_name] = matrix[i][-4]
            if matrix[i][0] != -1:
                tracks[cur_name].append(Message('note_on', note=matrix[i][0], velocity=matrix[i][1], time=time))
            if matrix[i][2] != -1:
                tracks[cur_name].append(Message('control_change', control=matrix[i][2], value=matrix[i][3], time=time))
            if matrix[i][4] != -1:
                tracks[cur_name].append(Message('program_change', program=matrix[i][4], time=time))
            if matrix[i][5] != -1:
                tracks[cur_name].append(MetaMessage('end_of_track', time=time))
            if matrix[i][11] != -1:
                key_sig_dict = {0: 'A', 1: 'A#m', 2: 'Ab', 3: 'Abm', 4: 'Am', 5: 'B', 6: 'Bb', 7: 'Bbm', 8: 'Bm',
                                9: 'C', 10: 'C#', 11: 'C#m', 12: 'Cb', 13: 'Cm', 14: 'D', 15: 'D#m', 16: 'Db',
                                17: 'Dm', 18: 'E', 19: 'Eb', 20: 'Ebm', 21: 'Em', 22: 'F', 23: 'F#', 24: 'F#m',
                                25: 'Fm', 26: 'G', 27: 'G#m', 28: 'Gb', 29: 'Gm'}
                tracks[cur_name].append(MetaMessage('key_signature', key=key_sig_dict[matrix[i][11]], time=time))
    for track in tracks:
        midi_file.tracks.append(tracks[track])
    # print(midi_file)
    # print(tracks.keys())
    midi_file.save(output)

