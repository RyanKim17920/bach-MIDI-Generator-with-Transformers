from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from tqdm import tqdm

key_sig_dict = {0: 'A', 1: 'A#m', 2: 'Ab', 3: 'Abm', 4: 'Am', 5: 'B', 6: 'Bb', 7: 'Bbm', 8: 'Bm',
                9: 'C', 10: 'C#', 11: 'C#m', 12: 'Cb', 13: 'Cm', 14: 'D', 15: 'D#m', 16: 'Db',
                17: 'Dm', 18: 'E', 19: 'Eb', 20: 'Ebm', 21: 'Em', 22: 'F', 23: 'F#', 24: 'F#m',
                25: 'Fm', 26: 'G', 27: 'G#m', 28: 'Gb', 29: 'Gm'}


def data_to_MIDI(matrix, midi_file_path, relative_time=True, relativity_to_instrument=False):
    midi_file = MidiFile()
    tracks = {}
    tracks_t_time = {}
    total_time = 0
    if len(matrix) > 2:
        for i in tqdm(range(1, len(matrix) - 1)):
            cur_name = f"o{matrix[i][-1]}i{matrix[i][-2]}"
            event_type = matrix[i][0]
            if event_type == 0:
                # start or stop token
                if matrix[i][1] == 0:
                    continue
                else:
                    break
            if cur_name == f"o-1i-1":
                # any meta messages (not based on instrument)
                try:
                    cur_name = list(tracks.keys())[0]
                    if not relative_time:
                        time = (matrix[i][7] - tracks_t_time[cur_name])
                        tracks_t_time[cur_name] = matrix[i][7]
                    else:
                        if relativity_to_instrument:
                            time = matrix[i][7]
                            # this is because messages are based from previous instruments, and if input is already relative, then it will be relative to the previous instrument, no need to change
                        else:
                            total_time += matrix[i][7]
                            # total time counter
                            time = total_time - tracks_t_time[cur_name]
                            # total time - last time of instrument = time since last instrument
                            tracks_t_time[cur_name] = total_time

                    if event_type == 5:
                        tracks[cur_name].append(MetaMessage('set_tempo', tempo=bpm2tempo(matrix[i][1]), time=time))
                    if event_type == 3:
                        tracks[cur_name].append(
                            MetaMessage('time_signature',
                                        numerator=matrix[i][1],
                                        denominator=matrix[i][2],
                                        clocks_per_click=matrix[i][3],
                                        notated_32nd_notes_per_beat=matrix[i][4], time=time))
                except:
                    continue
            elif cur_name not in tracks:
                tracks_t_time[cur_name] = 0
                time = 0
                tracks[cur_name] = MidiTrack()
                tracks[cur_name].append(Message('program_change', program=matrix[i][1], time=time))

            else:
                if not relative_time:
                    time = (matrix[i][7] - tracks_t_time[cur_name])
                    tracks_t_time[cur_name] = matrix[i][7]
                else:
                    if relativity_to_instrument:
                        time = matrix[i][7]
                        # this is because messages are based from previous instruments, and if input is already relative, then it will be relative to the previous instrument, no need to change
                    else:
                        total_time += matrix[i][7]
                        # total time counter
                        time = total_time - tracks_t_time[cur_name]
                        # total time - last time of instrument = time since last instrument
                        tracks_t_time[cur_name] = total_time

                if event_type == 6:
                    tracks[cur_name].append(Message('note_on', note=matrix[i][1], velocity=matrix[i][2], time=time))
                if event_type == 7:
                    tracks[cur_name].append(Message('note_off', note=matrix[i][1], velocity=matrix[i][2], time=time))
                if event_type == 2:
                    tracks[cur_name].append(
                        Message('control_change', control=matrix[i][1], value=matrix[i][2], time=time))
                if event_type == 1:
                    tracks[cur_name].append(Message('program_change', program=matrix[i][1], time=time))
                if event_type == 4:
                    tracks[cur_name].append(MetaMessage('key_signature', key=key_sig_dict[matrix[i][1]], time=time))
    for track in tracks:
        midi_file.tracks.append(tracks[track])
    # print(midi_file)
    # print(tracks.keys())
    midi_file.save(midi_file_path)


'''
input_file_path = r"Bach MIDIs/Sonatas and Partitas for Solo Violin/Partita No. 2 in D minor - BWV 1004/vp2-5cha.mid"
output_file_path = r"cha.mid"
data_0 = MIDI_data_extractor(input_file_path, relative_time=True)
print(data_0)
data_to_MIDI(data_0, output_file_path, relative_time=True)
# print(MIDI_data_extractor(output_file_path))
'''
