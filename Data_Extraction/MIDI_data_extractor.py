from mido import MidiFile, tempo2bpm
import numpy as np
from tqdm import tqdm

np.printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)


# helper functions
def index_based_matrix_appender(matrix1, matrix2):
    new_matrix = np.full((matrix1.shape[0], matrix1.shape[1] + 1), 0)
    new_matrix[:, :-1] = matrix1

    for i, arr in enumerate(matrix2):
        start_index = arr[0]
        if i == 0:
            start_index = 0
        value = arr[1]

        if i < len(matrix2) - 1:
            end_index = matrix2[i + 1][0]
        else:
            end_index = len(matrix1)

        new_matrix[start_index:end_index, -1] = value

    return new_matrix


def add_column_to_2d_array(array, number):
    # Create a column with the same number of rows as the original array
    column = np.full((array.shape[0], 1), number)
    # Add the column to the array
    return np.append(array, column, axis=1)


key_sig_dict = {'A': 0, 'A#m': 1, 'Ab': 2, 'Abm': 3, 'Am': 4, 'B': 5,
                'Bb': 6, 'Bbm': 7, 'Bm': 8, 'C': 9, 'C#': 10, 'C#m': 11, 'Cb': 12, 'Cm': 13,
                'D': 14, 'D#m': 15, 'Db': 16, 'Dm': 17, 'E': 18, 'Eb': 19, 'Ebm': 20, 'Em': 21, 'F': 22,
                'F#': 23, 'F#m': 24, 'Fm': 25, 'G': 26, 'G#m': 27, 'Gb': 28, 'Gm': 29}


# key signature dictionary mapping


# Creation of a new, more efficient, more readable, and more compact data extractor

def MIDI_data_extractor(midi_file_path,
                        verbose=0,
                        relative_time=True,
                        relativity_to_instrument=False,
                        include_start=True,
                        include_end=True,
                        include_instr_type=True):
    """'
    Inputs: midi_file_path: path to MIDI file
            verbose: 0 for no output, 1 for track names, 2 for track names and messages
            relative_time: True for relative time, False for absolute time
            relativity_to_instrument: True for relative time to the instrument, False for relative time to any previous event
            include_start: True for start token, False for no start token
            include_end: True for end token, False for no end token
            include_instr_type: True for instrument type, False for no instrument type

    Outputs: 2D array of MIDI data:
        Size: (num_messages, 11)

        index 0: type of data (
            0: start/end token,
            1: program_change
            2: control_change
            3: time signature
            4: key signature
            5: tempo
            6: note_on)
        index 1-6: data values (extra indexes included for future expansion):
            start/end token:
                index 1: 0 for start token, 1 for end token
            program_change:
                index 1: program number
            control_change:
                index 1: control number
                index 2: control value
            time signature:
                index 1: numerator
                index 2: denominator
                index 3: clocks per click
                index 4: notated 32nd notes per beat
            key signature:
                index 1: key signature number
            tempo:
                index 1: tempo in bpm
            note_on:
                index 1: note number
                index 2: velocity
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

        Notes:
        --> adding new importance values is significantly easier than the old method
        --> note_off is treated as note_off with velocity 0 to reduce data size

        todo: add a toggle for note_off with velocity 0 or not,
            config system to make much simpler
    """

    midi_file = MidiFile(midi_file_path)
    matrix = np.array([], dtype=np.int16)
    used_instruments = np.zeros(128)
    track_rp = -1
    organ_count = 0

    for i, track in tqdm(enumerate(midi_file.tracks), disable=False if verbose >= 1 else True):
        # tqdm progress bar if verbose
        if verbose == 2:
            print('Track {}: {}'.format(i, track.name))
        track_matrix = np.array([], dtype=np.int16)
        program_matrix = np.array([], dtype=np.int64)
        msg_counter = 0
        cur_time = 0
        orig_instr = -1
        for msg in track:
            msg_counter += 1
            msg_array = np.full(8, -1)
            cur_time += msg.time
            msg_array[-1] = cur_time
            # print(cur_time)
            # print(msg.type)
            match msg.type:
                case 'program_change':
                    msg_array[0] = 1
                    # this is set to 1 due to its importance in the model
                    # the programs (instruments) have to be put almost first in order for generation to function
                    msg_array[1] = msg.program
                    # print(msg.program)
                    program_matrix = np.append(program_matrix, [[msg_counter], [msg.program]])
                    program_matrix = np.reshape(program_matrix, (-1, 2))
                    if orig_instr == -1:
                        orig_instr = msg.program
                        # tracking original_instrument
                    if msg.program <= 8 and track_rp == -1:
                        track_rp = msg.program
                        # tracking track_rp, which includes possible multi-handed instruments
                        # these can, for some reason, have no program_changes for one of the hand
                        # thus, it is worth tracking the program_change for future usage if necessary
                    if 17 <= msg.program <= 24:
                        organ_count += 2
                        track_rp = msg.program
                        # same here, but for organs, which can have up to three hands
                case 'control_change':
                    msg_array[0] = 2
                    # control changes are very important for the model
                    msg_array[1:3] = msg.control, msg.value
                case 'time_signature':
                    # time signatures are global and also have importance
                    ts_array = np.full(11, -1)
                    ts_array[-4] = cur_time
                    ts_array[0] = 3
                    ts_array[1:5] = msg.numerator, msg.denominator, msg.clocks_per_click, msg.notated_32nd_notes_per_beat
                    matrix = np.append(matrix, [ts_array])
                    # global impact occurs here too
                case 'key_signature':
                    msg_array[0] = 4
                    msg_array[1] = key_sig_dict[msg.key]
                case 'set_tempo':
                    st_array = np.full(11, -1)
                    st_array[-4] = cur_time
                    st_array[0] = 5
                    st_array[1] = tempo2bpm(msg.tempo)
                    matrix = np.append(matrix, [st_array])
                    # appending to matrix, not track matrix, because this has a global impact (impacts all tracks)
                case 'note_on':
                    msg_array[0] = 6
                    # this is set to this value (not like 1 or 0) due to order of importance in the model
                    msg_array[1:3] = msg.note, msg.velocity
                case 'note_off':
                    msg_array[0] = 6
                    msg_array[1:3] = msg.note, 0
                    # note_off is often considered a note with zero velocity and can be removed for conciseness
                    # implement later: toggle on/off for this

            if not np.all(msg_array[0:-1] == -1):
                # checks if it's not emptpy, if it isn't, add to track_matrix
                track_matrix = np.append(track_matrix, [msg_array])

        track_matrix = track_matrix.astype(np.int64)

        if program_matrix.size == 0:
            # if there are no programs, use the tracked instrument number for multi-hand instruments
            if track_rp != -1:
                # if track_rp exists, use that
                program_matrix = np.append(program_matrix, [[0], [track_rp]])
                program_matrix = np.reshape(program_matrix, (-1, 2))
                msg_array = np.full(8, -1)
                msg_array[-1] = 0
                msg_array[0] = 1
                msg_array[1] = track_rp
                track_matrix = np.append(track_matrix, [msg_array])
                orig_instr = track_rp
                # remove track_rp if used up
                if track_rp <= 8:
                    track_rp = -1
                elif 17 <= track_rp <= 24:
                    organ_count -= 1
                    if organ_count == 0:
                        track_rp = -1
            else:
                # if doesn't exist just default to program 0 (piano)
                program_matrix = np.append(program_matrix, [[0], [0]])
                program_matrix = np.reshape(program_matrix, (-1, 2))
                msg_array = np.full(8, -1)
                msg_array[-1] = 0
                msg_array[0] = 1
                msg_array[1] = 0
                track_matrix = np.append(track_matrix, [msg_array])
                orig_instr = 0
        # increment respective used_instruments and use that instrument number
        used_instruments[int(program_matrix[0][1])] += 1
        instr_num = used_instruments[int(program_matrix[0][1])]

        track_matrix = track_matrix.reshape((-1, 8))
        # reshape for appending
        track_matrix = index_based_matrix_appender(track_matrix, program_matrix)
        track_matrix = add_column_to_2d_array(track_matrix, instr_num)
        track_matrix = add_column_to_2d_array(track_matrix, orig_instr)
        # add program_matrix onto track_matrix for current instrument based on the message
        # add instrument number and original instrument number to track_matrix
        matrix = np.append(matrix, track_matrix)
        # append track_matrix to matrix
        matrix = matrix.reshape((-1, 11))

    # print(track_matrix[0:100])

    # remove duplicates
    matrix = np.unique(matrix, axis=0)
    matrix = matrix.astype(np.int64)
    # set to respective type

    # sorting starts here
    # sort by time, then by index 0 within the times in increasing order while preserving order
    matrix = matrix[np.lexsort((matrix[:, 0], matrix[:, -4]))]

    if relative_time:
        if relativity_to_instrument:
            tracks_t_time = {}
            # track time for each track
            for i in tqdm(range(len(matrix)), disable=False if verbose >= 1 else True):
                try:
                    cur_name = f"o{matrix[i][-1]}i{matrix[i][-2]}"
                    # get the name of the instrument based on original instrument and instrument number
                    if cur_name == f"o-1i-1":
                        # if it's a global message, just set it to the first instrument that exists
                        cur_name = list(tracks_t_time.keys())[0]
                    else:
                        # if it's not global message, set it to the instrument name
                        # if it doesn't already exist, create it with initialized time of 0
                        if cur_name not in tracks_t_time:
                            tracks_t_time[cur_name] = 0

                    time = matrix[i][-4] - tracks_t_time[cur_name]
                    # get the time difference between the current time and the last time
                    tracks_t_time[cur_name] = matrix[i][-4]
                    # set the last time to the current time
                    matrix[i][-4] = time
                    # set the time to the time difference
                except IndexError:
                    pass
        else:
            # if not relative to instrument, just set it to the time difference between the current time and the last time
            cur = 0
            for i in tqdm(range(len(matrix)), disable=False if verbose >= 1 else True):
                try:
                    time = matrix[i][-4] - cur
                    cur = matrix[i][-4]
                    matrix[i][-4] = time
                except IndexError:
                    pass
    if include_end:
        end_track = np.full(11, -1)
        end_track[-4] = 0 if relative_time else matrix[len(matrix) - 1][-4]
        # 0 if it is relative time (right after last event), else the last time
        end_track[1] = 1
        end_track[0] = 0
        # greatest importance!
        matrix = np.append(matrix, [end_track])
        # append to end of matrix
        matrix = matrix.reshape((-1, 11))
    if include_start:
        start_track = np.full(11, -1)
        start_track[-4] = 0
        # occurs at time 0 regardless of relative time or not
        start_track[1] = 0
        start_track[0] = 0
        matrix = np.append([start_track], matrix)
        # append to start of matrix
        matrix = matrix.reshape((-1, 11))
    if not include_instr_type:
        matrix = np.delete(matrix, 8, 1)
    return matrix

