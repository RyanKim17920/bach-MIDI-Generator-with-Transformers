from mido import MidiFile, tempo2bpm
import numpy as np
from index_based_matrix_appender import index_based_matrix_appender
from add_column_to_2d_array import add_column_to_2d_array
from tqdm import tqdm


def MIDI_data_extractor(midi_file_path, verbose=0, relative_time=False):
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    midi_file = MidiFile(midi_file_path)
    matrix = np.array([], dtype=np.int16)
    used_instruments = np.zeros(128)
    track_rp = -1
    organ_count = 0
    for i, track in (tqdm(enumerate(midi_file.tracks)) if verbose >= 1 else enumerate(midi_file.tracks)):
        if verbose == 2:
            print('Track {}: {}'.format(i, track.name))
        track_matrix = np.array([], dtype=np.int16)
        program_matrix = np.array([], dtype=np.int64)
        msg_counter = 0
        cur_time = 0
        orig_instr = -1
        if track_rp != -1:
            program_matrix = np.append(program_matrix, [[0], [track_rp]])
            program_matrix = np.reshape(program_matrix, (-1, 2))
            msg_array = np.full(13, -1)
            msg_array[-1] = 0
            msg_array[4] = track_rp
            track_matrix = np.append(track_matrix, [msg_array])
            orig_instr = track_rp
            if track_rp <= 8:
                track_rp = -1
            elif 17 <= track_rp <= 24:
                organ_count -= 1
                if organ_count == 0:
                    track_rp = -1
        for msg in track:
            msg_counter += 1
            msg_array = np.full(13, -1)
            cur_time += msg.time
            msg_array[-1] = cur_time
            # print(cur_time)
            # print(msg.type)
            if msg.type == 'note_on':
                msg_array[0:2] = msg.note, msg.velocity
            elif msg.type == 'note_off':
                msg_array[0:2] = msg.note, 0
                # note_off is unnecessary and should be removed
            # deleted 2-4 hmm
            elif msg.type == 'control_change':
                msg_array[2:4] = msg.control, msg.value
            elif msg.type == 'program_change':
                msg_array[4] = msg.program
                # print(msg.program)
                program_matrix = np.append(program_matrix, [[msg_counter], [msg.program]])
                program_matrix = np.reshape(program_matrix, (-1, 2))
                if orig_instr == -1:
                    orig_instr = msg.program
                if msg.program <= 8 and track_rp == -1:
                    track_rp = msg.program
                if msg.program >= 17 and msg.program <= 24:
                    organ_count += 2
                    track_rp = msg.program
            # elif msg.type == 'end_of_track':
            # msg_array[5] = 0
            elif msg.type == 'set_tempo':
                st_array = np.full(16, -1)
                st_array[-4] = cur_time
                st_array[6] = tempo2bpm(msg.tempo) - 20
                matrix = np.append(matrix, [st_array])
            elif msg.type == 'time_signature':
                ts_array = np.full(16, -1)
                ts_array[-4] = cur_time
                ts_array[7:11] = msg.numerator, msg.denominator, msg.clocks_per_click, msg.notated_32nd_notes_per_beat
                matrix = np.append(matrix, [ts_array])
            elif msg.type == 'key_signature':
                key_sig_dict = {'A': 0, 'A#m': 1, 'Ab': 2, 'Abm': 3, 'Am': 4, 'B': 5,
                                'Bb': 6, 'Bbm': 7, 'Bm': 8, 'C': 9, 'C#': 10, 'C#m': 11, 'Cb': 12, 'Cm': 13,
                                'D': 14, 'D#m': 15, 'Db': 16, 'Dm': 17, 'E': 18, 'Eb': 19, 'Ebm': 20, 'Em': 21, 'F': 22,
                                'F#': 23, 'F#m': 24, 'Fm': 25, 'G': 26, 'G#m': 27, 'Gb': 28, 'Gm': 29}
                msg_array[11] = key_sig_dict[msg.key]
            if not np.all(msg_array[0:-1] == -1):
                track_matrix = np.append(track_matrix, [msg_array])
        track_matrix = track_matrix.astype(np.int64)
        if len(program_matrix) > 0:
            used_instruments[int(program_matrix[0][1])] += 1
            instr_num = used_instruments[int(program_matrix[0][1])]
            # print(instr_num)
            track_matrix = track_matrix.reshape((-1, 13))
            track_matrix = index_based_matrix_appender(track_matrix, program_matrix)
            track_matrix = add_column_to_2d_array(track_matrix, instr_num)
            track_matrix = add_column_to_2d_array(track_matrix, orig_instr)
            matrix = np.append(matrix, track_matrix)
            matrix = matrix.reshape((-1, 16))
        # print(track_matrix[0:100])

    matrix = matrix.reshape((-1, 16))
    matrix = np.unique(matrix, axis=0)
    matrix = matrix.astype(np.int64)
    # deprecated order: from first to last: Time, Program_change(Instrument), Tempo, time_sig, key_sig, control_change
    cols_to_sort = [4, 6, 7, 11]

    # Initialize an empty list to hold the new rows
    new_rows = []

    # Iterate over each row in your matrix
    for row in matrix:
        # Create a flag to check if a new row has been added
        new_row_added = False
        # Iterate over the columns you want to split
        for col in cols_to_sort:
            # Skip the iteration if the column is 12 (or -4)
            if col == 12 or col == -4:
                continue
            # Create a new row that is a copy of the original row
            new_row = row.copy()
            # Check if the value in the column is not -1
            if new_row[col] != -1:
                this_time = new_row[-4]
                # Set all other columns in cols_to_sort to -1
                for i in cols_to_sort:
                    if i != col:
                        new_row[i] = -1
                new_row[-4] = this_time
                # Append the new row to the list of new rows
                new_rows.append(new_row)
                # Set the flag to True
                new_row_added = True
        # If no new row was added, append the original row
        if not new_row_added:
            new_rows.append(row)

    # Use numpy.vstack to stack the list of new rows into a new matrix
    matrix = np.vstack(new_rows)

    # masking for the 5th index
    mask = matrix[:, 5] == -1

    matrix = np.concatenate((matrix[mask], matrix[~mask]))
    # Now sort by column -4 (time) in ascending order
    index_time = np.argsort(matrix[:, -4], kind='stable')
    matrix = matrix[index_time]

    # Then sort by column 4 (value 4) - put all -1 at the end
    index_value_4 = np.argsort(matrix[:, 4] == -1, kind='stable')
    matrix = matrix[index_value_4]

    end_track = np.full(16, -1)
    end_track[-4] = matrix[len(matrix) - 1][-4]
    end_track[5] = 0
    matrix = np.append(matrix, [end_track])
    matrix = matrix.reshape((-1, 16))

    if relative_time:
        tracks_t_time = {}
        for i in tqdm(range(len(matrix))):
            cur_name = f"o{matrix[i][-1]}i{matrix[i][-2]}"
            if cur_name == f"o-1i-1":
                try:
                    cur_name = list(tracks_t_time.keys())[0]
                except IndexError:
                    print(matrix)
                time = matrix[i][-4] - tracks_t_time[cur_name]
                tracks_t_time[cur_name] = matrix[i][-4]
                matrix[i][-4] = time
            else:
                if cur_name not in tracks_t_time:
                    tracks_t_time[cur_name] = 0
                time = matrix[i][-4] - tracks_t_time[cur_name]
                tracks_t_time[cur_name] = matrix[i][-4]
                matrix[i][-4] = time
    '''[note_on_note, note_on_velocity,
        control_change_control, control_change_value, program_change_program,
        end_marking, set_tempo_tempo,
        time_sig_num, itme_sig_den, time_sig_clocksperclick, time_sig_notated_32nd,
        key_sig(turn into numbers), [time], instrument_type, instrument_num, orig_instrument_type]'''

    return matrix
