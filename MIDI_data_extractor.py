from mido import MidiFile
import numpy as np
from index_based_matrix_appender import index_based_matrix_appender
from add_column_to_2d_array import add_column_to_2d_array
from tqdm import tqdm
def MIDI_data_extractor(midi_file_path, time_inc=True):
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    midi_file = MidiFile(midi_file_path)
    matrix = np.array([], dtype=np.int16)
    used_instruments = np.zeros(128)
    track_rp = -1
    for i, track in tqdm(enumerate(midi_file.tracks)):
        print('Track {}: {}'.format(i, track.name))
        track_matrix = np.array([], dtype=np.int16)
        program_matrix = np.array([], dtype=np.int64)
        msg_counter = 0
        cur_time = 0
        orig_instr = -1
        if track_rp != -1:
            program_matrix = np.append(program_matrix, [[0], [track_rp]])
            program_matrix = np.reshape(program_matrix, (-1, 2))
            msg_array = np.full(15, -1)
            msg_array[-1] = 0
            msg_array[6] = track_rp
            track_matrix = np.append(track_matrix, [msg_array])
            orig_instr = track_rp
            track_rp = -1
            #print(program_matrix)

        for msg in track:
            msg_counter += 1
            msg_array = np.full(15, -1)
            cur_time += msg.time
            msg_array[-1] = cur_time
            if msg.type == 'note_on':
                msg_array[0:2] = msg.note, msg.velocity
            elif msg.type == 'note_off':
                msg_array[2:4] = msg.note, msg.velocity
            elif msg.type == 'control_change':
                msg_array[4:6] = msg.control, msg.value
            elif msg.type == 'program_change':
                msg_array[6] = msg.program
                #print(msg.program)
                program_matrix = np.append(program_matrix, [[msg_counter], [msg.program]])
                program_matrix = np.reshape(program_matrix, (-1, 2))
                if orig_instr == -1:
                    orig_instr = msg.program
                if msg.program <= 8 and track_rp == -1:
                    track_rp = msg.program
            elif msg.type == 'end_of_track':
                eot_array = np.full(18, -1)
                eot_array[-4] = msg.time
                eot_array[7] = 1
                matrix = np.append(matrix, [eot_array])
            elif msg.type == 'set_tempo':
                st_array = np.full(18, -1)
                st_array[-4] = msg.time
                st_array[8] = msg.tempo
                matrix = np.append(matrix, [st_array])
            elif msg.type == 'time_signature':
                ts_array = np.full(18, -1)
                ts_array[-4] = msg.time
                ts_array[9:13] = msg.numerator, msg.denominator, msg.clocks_per_click, msg.notated_32nd_notes_per_beat
                matrix = np.append(matrix, [ts_array])
            elif msg.type == 'key_signature':
                key_sig_dict = {'A': 0, 'A#m': 1, 'Ab': 2, 'Abm': 3, 'Am': 4, 'B': 5,
                                'Bb': 6, 'Bbm': 7, 'Bm': 8, 'C': 9, 'C#': 10, 'C#m': 11, 'Cb': 12, 'Cm': 13,
                                'D': 14, 'D#m': 15, 'Db': 16, 'Dm': 17, 'E': 18, 'Eb': 19, 'Ebm': 20, 'Em': 21, 'F': 22,
                                'F#': 23, 'F#m': 24, 'Fm': 25, 'G': 26, 'G#m': 27, 'Gb': 28, 'Gm': 29}
                msg_array[13] = key_sig_dict[msg.key]

            if not np.all(msg_array[0:-1] == -1):
                track_matrix = np.append(track_matrix, [msg_array])
        track_matrix = track_matrix.astype(np.int64)
        if (len(program_matrix) > 0):
            used_instruments[int(program_matrix[0][1])] += 1
            instr_num = used_instruments[int(program_matrix[0][1])]
            #print(instr_num)
            track_matrix = track_matrix.reshape((-1, 15))
            track_matrix = index_based_matrix_appender(track_matrix, program_matrix)
            track_matrix = add_column_to_2d_array(track_matrix, instr_num)
            track_matrix = add_column_to_2d_array(track_matrix, orig_instr)
            matrix = np.append(matrix, track_matrix)
            matrix = matrix.reshape((-1, 18))
            track_matrix = track_matrix.astype(np.int64)
        #print(track_matrix[0:100])
    matrix = matrix.reshape((-1, 18))
    matrix = np.unique(matrix, axis=0)
    matrix = matrix.reshape((-1, 18))
    matrix = matrix.astype(np.int64)
    matrix = matrix[np.lexsort((-matrix[:, 4], -matrix[:, 13], -matrix[:, 9], -matrix[:, 8], -matrix[:, 6], matrix[:, -4]))]
    # order from first to last: Time, Program_change(Instrument), Tempo, time_sig, key_sig, control_change

    if not time_inc:
        matrix = np.delete(matrix, -4, axis=1)
    '''[note_on_note, note_on_velocity, note_off_note, note_off_velocity,
        control_change_control, control_change_value, program_change_program,
        end_of_track, set_tempo_tempo,
        time_sig_num, itme_sig_den, time_sig_clocksperclick, time_sig_notated_32nd,
        key_sig(turn into numbers), [time (only shown during tests)], instrument_number, instrument_type, orig_instrument_number]'''
    return matrix

print(MIDI_data_extractor(r""))