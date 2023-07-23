import numpy as np
from MIDI_data_extractor import MIDI_data_extractor
from tqdm import tqdm

#deprecated

'''[note_on_note, note_on_velocity, note_off_note, note_off_velocity,
        control_change_control, control_change_value, program_change_program,
        end_of_track, set_tempo_tempo,
        time_sig_num, itme_sig_den, time_sig_clocksperclick, time_sig_notated_32nd,
        key_sig(turn into numbers), [time], instrument_type, instrument_num, orig_instrument_type]'''


# for note_on_note ==> 129, note_on_velocity ==> 129 = 129^2 (-1 --> 127)
# for control_change_control ==> 129, control_change_value ==> 129 = 129^2 (-1 --> 127)
# for program_change_program ==> 129 = 129 (-1 --> 127)
# for end_of_track ==> 2 (0,1)
# for set_tempo_tempo ==> 200 (20 -> 220)
# for time_sig_num ==> 32, time_sig_den ==> 32 time_sig_clocksperclick ==> 128, time_sig_notated_32nd ==> 64 = 32*32*128*64
# key_sig = 30
# time = 3000 (use relative time)
# instrument_type = 129, instrument_num = 5, orig_instrument_number = 129 = 5 * 129^2
# [129,129,129,129,129,2,200,32,32,128,64,30,3000,129,5,129]
# 2, 2, 1, 1, 1, 4, 1
def relative_time(matrix):
    tracks_t_time = {}
    for i in tqdm(range(len(matrix))):
        cur_name = f"o{matrix[i][-1]}i{matrix[i][-2]}"
        if cur_name == f"o-1i-1":
            cur_name = list(tracks_t_time.keys())[0]
            time = matrix[i][-4] - tracks_t_time[cur_name]
            tracks_t_time[cur_name] = matrix[i][-4]
            matrix[i][-4] = time
        elif cur_name not in tracks_t_time:
            tracks_t_time[cur_name] = 0
            time = 0
            matrix[i][-4] = time
        else:
            time = matrix[i][-4] - tracks_t_time[cur_name]
            tracks_t_time[cur_name] = matrix[i][-4]
            matrix[i][-4] = time
    # print(tracks_t_time)
    # Remove the time column and replace it with relative_time
    '''diff = np.diff(data[:, -4])
    diff = np.insert(diff, 0, 0)
    data[:, -4] = diff
    return data'''
    return matrix

print(np.max(relative_time(MIDI_data_extractor(r"C:\Users\ilove\Downloads\bwv988.mid"))[:,-4]))




'''


max_values = np.array([130, 130, 0, 0, 130, 130, 130, 3, 16777218, 258, 258, 258, 258, 32, 1002, 131, 19, 131])

def convert_row_to_number(row, max_values):
    number = 0
    multiplier = 1
    # Only consider the last four entries
    for value, max_value in zip(row[-4:], max_values[-4:]):
        # Adjust the value because the range starts at -1
        adjusted_value = value + 1
        number += adjusted_value * multiplier
        multiplier *= (max_value + 2)  # Update multiplier based on max_value + 2
    return number

def convert_matrix_to_number(matrix, max_values):
    numbers = []
    for row in matrix:
        numbers.append(convert_row_to_number(row, max_values))
    return numbers

def convert_number_to_row(number, max_values):
    row = [-1] * (len(max_values) - 4)  # Initialize row with -1s
    for max_value in reversed(max_values[-4:]):  # Only consider the last four max_values
        value = number % (max_value + 2) - 1  # Add 2 because the range is -1 to max_value
        row.append(value)
        number //= (max_value + 2)  # Update divisor based on max_value + 2
    return row

def convert_numbers_to_matrix(numbers, max_values):
    matrix = []
    for number in numbers:
        matrix.append(convert_number_to_row(number, max_values[::-1]))  # Reverse the max_values
    return np.array(matrix)





numbers_matrix = convert_matrix_to_number(matrx, max_values)
'''