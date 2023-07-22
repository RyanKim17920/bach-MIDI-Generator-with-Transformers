import numpy as np
import pickle
from tqdm import tqdm

'''[note_on_note, note_on_velocity, 
        control_change_control, control_change_value, program_change_program,
        end_of_track, set_tempo_tempo,
        time_sig_num, itme_sig_den, time_sig_clocksperclick, time_sig_notated_32nd,
        key_sig(turn into numbers), [time], instrument_type, instrument_num, orig_instrument_type]'''

#DEF NOT WORKING, what am I doing

note_on_note_l = 128
note_on_velocity_l = 128
control_change_control_l = 128
control_change_value_l = 128
program_change_program_l = 80#(cut down)
end_of_track = 1 #(-1,0)
set_tempo_tempo_l = 140 #(21 -> 160)  (but acts like 1-->140)
time_sig_num_l = 12
time_sig_den_l = 12
time_sig_clocksperclick_l = 96
time_sig_notated_32nd_l = 64
key_sig_l = 30
time_l = 1500 #(use relative time)
instrument_type_l = 80 #(cut down),
instrument_num_l = 5
orig_instrument_number_l = 80 #(cut down)
# [129,129,129,129,129,2,200,32,32,128,64,30,3000,129,5,129]
# 2, 2, 1, 1, 1, 4, 1
numpy_to_number = {}
number_to_numpy = {}
count = 0
for time in tqdm(range(time_l)):
    for instrument_type in tqdm(range(instrument_type_l)):
        for instrument_num in tqdm(range(instrument_num_l)):
            for orig_instrument_number in tqdm(range(orig_instrument_number_l)):
                for note_on_note in range(note_on_note_l):
                    for note_on_velocity in range(note_on_velocity_l):
                        numpy_array = np.array([note_on_note, note_on_velocity,
                                                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                time, instrument_type, instrument_num, orig_instrument_number])
                        number_to_numpy[count] = numpy_array
                        numpy_to_number[str(numpy_array)] = count
                        count += 1
                for control_change_control in range(control_change_control_l):
                    for control_change_value in range(control_change_value_l):
                        numpy_array = np.array([-1, -1,
                                                control_change_control, control_change_value, -1, -1, -1, -1, -1, -1,
                                                -1, -1,
                                                time, instrument_type, instrument_num, orig_instrument_number])
                        number_to_numpy[count] = numpy_array
                        numpy_to_number[str(numpy_array)] = count
                        count += 1
                for program_change_program in range(program_change_program_l):
                    numpy_array = np.array([-1, -1,
                                            -1, -1, program_change_program, -1, -1, -1, -1, -1, -1, -1,
                                            time, instrument_type, instrument_num, orig_instrument_number])
                    number_to_numpy[count] = numpy_array
                    numpy_to_number[str(numpy_array)] = count
                    count += 1
                for key_sig in range(0, key_sig_l):
                    numpy_array = np.array([-1, -1,
                                            -1, -1, -1, -1, -1, -1, -1, -1, -1, key_sig,
                                            time, instrument_type, instrument_num, orig_instrument_number])
                    number_to_numpy[count] = numpy_array
                    numpy_to_number[str(numpy_array)] = count
                    count += 1


    # end of track
    numpy_array = np.array([-1, -1,
                            -1, -1, -1, 0, -1, -1, -1, -1, -1, -1,
                            time, -1, -1, -1])
    number_to_numpy[count] = numpy_array
    numpy_to_number[str(numpy_array)] = count
    count += 1

    for set_tempo_tempo in range(set_tempo_tempo_l):
        numpy_array = np.array([-1, -1,
                                -1, -1, -1, -1, set_tempo_tempo, -1, -1, -1, -1, -1,
                                time, -1, -1, -1])
        number_to_numpy[count] = numpy_array
        numpy_to_number[str(numpy_array)] = count
        count += 1
    for time_sig_num in tqdm(range(1, time_sig_num_l+1)):
        for time_sig_den in range(1, time_sig_den_l+1):
            for time_sig_clocksperclick in range(1, time_sig_clocksperclick_l):
                for time_sig_notated_32nd in range(1, time_sig_notated_32nd_l):
                    numpy_array = np.array([-1, -1,
                                            -1, -1, -1, -1, -1, time_sig_num, time_sig_den,
                                            time_sig_clocksperclick, time_sig_notated_32nd, -1,
                                            time, -1, -1, -1])
                    number_to_numpy[count] = numpy_array
                    numpy_to_number[str(numpy_array)] = count
                    count += 1

with open('numpy_to_number.pkl', 'wb') as f:
    pickle.dump(numpy_to_number, f)
with open('number_to_numpy.pkl', 'wb') as f:
    pickle.dump(number_to_numpy, f)