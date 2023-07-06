from mido import MidiFile
import numpy as np
# import pretty_midi
from index_based_matrix_appender import index_based_matrix_appender
from add_column_to_2d_array import add_column_to_2d_array


def MIDI_data_extractor(midi_file_path):
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    midi_file = MidiFile(midi_file_path)
    matrix = np.array([], dtype=np.int16)
    empty_matrixes = np.array([])
    used_instruments = np.zeros(128)
    # midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    # print(midi_data.instruments)
    for i, track in enumerate(midi_file.tracks):
        # print('Track {}: {}'.format(i, track.name))
        track_matrix = np.array([], dtype=np.int16)
        instr_type = 0
        if used_instruments[0] % 2 == 1:
            instr_type = 0
        # print(track)
        program_matrix = np.array([], dtype=np.int64)
        msg_counter = 0
        for msg in track:
            msg_counter += 1
            msg_array = np.full(15, -1)
            msg_array[-1] = msg.time
            if msg.type == 'note_on':
                msg_array[0:2] = msg.note, msg.velocity
            elif msg.type == 'note_off':
                msg_array[2:4] = msg.note, msg.velocity
            elif msg.type == 'control_change':
                msg_array[4:6] = msg.control, msg.value
            elif msg.type == 'program_change':
                msg_array[6] = msg.program
                program_matrix = np.append(program_matrix, [[msg_counter], [msg.program]])
                program_matrix = np.reshape(program_matrix, (-1, 2))
            elif msg.type == 'end_of_track':
                eot_array = np.full(17, -1)
                eot_array[-3] = msg.time
                eot_array[7] = 1
                matrix = np.append(matrix, [eot_array])
            elif msg.type == 'set_tempo':
                st_array = np.full(17, -1)
                st_array[-3] = msg.time
                st_array[8] = msg.tempo
                matrix = np.append(matrix, [st_array])
            elif msg.type == 'time_signature':
                ts_array = np.full(17, -1)
                ts_array[-3] = msg.time
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
        if (len(program_matrix) > 0):
            # print(program_matrix[0][1])
            used_instruments[int(program_matrix[0][1])] += 1
            instr_num = used_instruments[int(program_matrix[0][1])]
            track_matrix = track_matrix.reshape((-1, 15))
            # print(track_matrix.shape)
            # print(track.name)
            # print(program_matrix)
            # print(track_matrix.shape)
            # print(program_matrix)
            track_matrix = index_based_matrix_appender(track_matrix, program_matrix)
            # print(track_matrix.shape)
            track_matrix = add_column_to_2d_array(track_matrix, instr_num)

            # track_matrix = track_matrix[(track_matrix[:, -1] == track_matrix[:, 6]) | (track_matrix[:, 6] == -1)]

            matrix = np.append(matrix, track_matrix)
            matrix = matrix.reshape((-1, 17))
            # print(matrix)
    matrix = matrix.reshape((-1, 17))
    matrix = matrix.astype(np.int64)

    # matrix_non_zero_7th = matrix[matrix[:, 6] != -1]
    # matrix_zero_7th = matrix[matrix[:, 6] == -1]

    # matrix_zero_7th = matrix_zero_7th[matrix_zero_7th[:, -3].argsort()]

    # matrix = np.concatenate((matrix_non_zero_7th, matrix_zero_7th), axis=0)

    matrix = matrix[
        np.lexsort((-matrix[:, 4], -matrix[:, 13], -matrix[:, 9], -matrix[:, 8], -matrix[:, 6], matrix[:, -3]))]
    # order from first to last: Time, Program_change(Instrument), Tempo, time_sig, key_sig, control_change
    # print(matrix)
    # matrix = np.delete(matrix, -3, axis=1)
    print(matrix)
    '''[note_on_note, note_on_velocity, note_off_note, note_off_velocity,
        control_change_control, control_change_value, program_change_program,
        end_of_track, set_tempo_tempo,
        time_sig_num, itme_sig_den, time_sig_clocksperclick, time_sig_notated_32nd,
        key_sig(turn into numbers), [time (only shown during tests)], instrument_number, instrument_type]'''

    return matrix


MIDI_data_extractor('bach/partitas/all2.mid')
# this MIDI is very corrupted
