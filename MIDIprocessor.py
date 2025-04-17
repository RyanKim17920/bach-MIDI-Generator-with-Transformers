import os
from typing import List, Dict, Tuple, Any, Optional
import pretty_midi
import numpy as np

PAD_TOKEN_VALUE = -100 # Usually used for masking loss, not as a token index
PAD_IDX = 0
START_IDX = 1
END_IDX = 2
MAX_LOCAL_INSTRUMENTS = 16 # Max concurrent instruments *per file* to track
DRUM_PROGRAM_TOKEN_OFFSET = 128 # Relative offset for the drum token
DEFAULT_VELOCITY = 64 # Default velocity if none is specified before a note
TIME_EPSILON = 0.001 # Small offset to ensure event ordering (was 0.01, reduced for less impact)

class MIDIProcessor:
    """
    Processes MIDI files into token sequences and converts token sequences back
    to MIDI or human-readable text. Uses specific program tokens and local
    instance tokens. Now includes START/END tokens in the sequence.
    """

    def __init__(self,
                 time_step: float = 0.01,
                 velocity_bins: int = 32,
                 max_time_shift_seconds: float = 10.0,
                 max_local_instruments: int = MAX_LOCAL_INSTRUMENTS):

        self.time_step = time_step
        self.velocity_bins = velocity_bins
        self.max_time_shift_seconds = max_time_shift_seconds
        self.max_local_instruments = max_local_instruments
        self.pad_token_idx = PAD_IDX

        # --- Vocabulary Definition
        _current_offset = 0
        self.PAD = PAD_IDX; _current_offset +=1
        self.START = START_IDX; _current_offset +=1
        self.END = END_IDX; _current_offset +=1
        # ... (rest of vocab definition remains the same) ...
        self.program_token_offset = _current_offset
        self.num_program_tokens = 128 + 1
        self.drum_program_token_idx = self.program_token_offset + DRUM_PROGRAM_TOKEN_OFFSET
        self.program_token_range = (self.program_token_offset, self.program_token_offset + self.num_program_tokens)
        _current_offset += self.num_program_tokens
        self.local_instance_offset = _current_offset
        self.local_instance_range = (self.local_instance_offset, self.local_instance_offset + self.max_local_instruments)
        _current_offset += self.max_local_instruments
        self.NOTE_ON = _current_offset; _current_offset += 1
        self.NOTE_OFF = _current_offset; _current_offset += 1
        self.TIME_SHIFT = _current_offset; _current_offset += 1
        self.VELOCITY = _current_offset; _current_offset += 1
        self.CONTROL_CHANGE = _current_offset; _current_offset += 1
        self.PROGRAM_CHANGE = _current_offset; _current_offset += 1
        self.PEDAL_ON = _current_offset; _current_offset += 1
        self.PEDAL_OFF = _current_offset; _current_offset += 1
        self.note_value_offset = _current_offset
        self.note_range = 128; _current_offset += self.note_range
        self.time_shift_value_offset = _current_offset
        self.time_shift_steps = int(self.max_time_shift_seconds / self.time_step)
        _current_offset += self.time_shift_steps
        self.velocity_value_offset = _current_offset
        _current_offset += self.velocity_bins
        self.cc_number_offset = _current_offset
        self.cc_range = 128; _current_offset += self.cc_range
        self.cc_value_offset = _current_offset
        _current_offset += self.cc_range
        self.program_value_offset = _current_offset
        self.program_range = 128; _current_offset += self.program_range
        self.vocab_size = _current_offset
        # --- End Vocabulary Definition ---

        # --- Build Reverse Mapping for Text Conversion ---
        self.token_to_name_map = {}
        self._build_reverse_map() # Definition assumed unchanged

        print(f"Total Vocab Size: {self.vocab_size}")


    # --- Assume other methods like _build_reverse_map, _get_program_token, etc. exist ---
    def _build_reverse_map(self):
        """Builds the token index to human-readable name map."""
        self.token_to_name_map = {
            self.PAD: "PAD", self.START: "START", self.END: "END",
            self.NOTE_ON: "NOTE_ON", self.NOTE_OFF: "NOTE_OFF", self.TIME_SHIFT: "TIME_SHIFT",
            self.VELOCITY: "VELOCITY", self.CONTROL_CHANGE: "CONTROL_CHANGE",
            self.PROGRAM_CHANGE: "PROGRAM_CHANGE", self.PEDAL_ON: "PEDAL_ON", self.PEDAL_OFF: "PEDAL_OFF"
        }
        # Instrument Programs
        for i in range(128):
            self.token_to_name_map[self.program_token_offset + i] = f"PROG({i})"
        self.token_to_name_map[self.drum_program_token_idx] = "PROG(DRUMS)"
        # Local Instances
        for i in range(self.max_local_instruments):
            self.token_to_name_map[self.local_instance_offset + i] = f"LOCAL_INST({i})"
        # Note Values
        for i in range(self.note_range):
            self.token_to_name_map[self.note_value_offset + i] = f"NOTE_VAL({i})"
        # Time Shift Values
        for i in range(self.time_shift_steps):
            self.token_to_name_map[self.time_shift_value_offset + i] = f"TIME_SHIFT_VAL({i})"
        # Velocity Values
        for i in range(self.velocity_bins):
            self.token_to_name_map[self.velocity_value_offset + i] = f"VEL_VAL({i})"
        # CC Numbers
        for i in range(self.cc_range):
            self.token_to_name_map[self.cc_number_offset + i] = f"CC_NUM({i})"
        # CC Values
        for i in range(self.cc_range):
            self.token_to_name_map[self.cc_value_offset + i] = f"CC_VAL({i})"
        # Program Change Values
        for i in range(self.program_range):
             self.token_to_name_map[self.program_value_offset + i] = f"PROG_VAL({i})"

    def _get_program_token(self, program_number: int, is_drum: bool) -> Optional[int]:
        if is_drum: return self.drum_program_token_idx
        elif 0 <= program_number <= 127: return self.program_token_offset + program_number
        else: return None
    def _get_local_instance_token(self, instrument_index: int) -> int:
        local_idx = instrument_index % self.max_local_instruments
        return self.local_instance_offset + local_idx
    def _quantize_velocity(self, velocity: int) -> int:
        return min(int(velocity * self.velocity_bins / 128), self.velocity_bins - 1)
    def _unquantize_velocity(self, velocity_bin: int) -> int:
        return min(127, max(0, int((velocity_bin + 0.5) * 128.0 / self.velocity_bins)))
    def _quantize_time_shift(self, time_diff: float) -> int:
        if time_diff < self.time_step / 2.0: return 0
        return min(int(round(time_diff / self.time_step)), self.time_shift_steps - 1)
    def _extract_tempo_changes(self, midi_data: pretty_midi.PrettyMIDI) -> List[Dict[str, float]]:
        # (Implementation assumed unchanged)
        changes = []
        try:
            tempo_times, tempo_values = midi_data.get_tempo_changes()
            for t, q in zip(tempo_times, tempo_values):
                if t >= 0 and q > 0: changes.append({'time': float(t), 'tempo': float(q)})
            if not changes or changes[0]['time'] > 1e-6:
                 initial_tempo = 120.0
                 if len(tempo_values) > 0 and tempo_values[0] > 0: initial_tempo = float(tempo_values[0])
                 changes.insert(0, {'time': 0.0, 'tempo': initial_tempo})
        except Exception: changes = [{'time': 0.0, 'tempo': 120.0}]
        return sorted(changes, key=lambda x: x['time'])
    def _extract_time_signatures(self, midi_data: pretty_midi.PrettyMIDI) -> List[Dict[str, Any]]:
        # (Implementation assumed unchanged)
        sigs = []
        try:
            sigs = [{'time': float(ts.time), 'numerator': int(ts.numerator), 'denominator': int(ts.denominator)}
                    for ts in midi_data.time_signature_changes if ts.time >= 0 and ts.numerator > 0 and ts.denominator > 0]
            if not sigs or sigs[0]['time'] > 1e-6: sigs.insert(0, {'time': 0.0, 'numerator': 4, 'denominator': 4})
        except Exception: sigs = [{'time': 0.0, 'numerator': 4, 'denominator': 4}]
        unique_sigs = []; last_t = -1.0
        for sig in sorted(sigs, key=lambda x: x['time']):
            if sig['time'] > last_t + 1e-6: unique_sigs.append(sig); last_t = sig['time']
        return unique_sigs
    def _extract_key_signatures(self, midi_data: pretty_midi.PrettyMIDI) -> List[Dict[str, Any]]:
        # (Implementation assumed unchanged)
        keys = []
        try:
            keys = [{'time': float(ks.time), 'key_number': int(ks.key_number)}
                    for ks in midi_data.key_signature_changes if ks.time >= 0]
            if not keys or keys[0]['time'] > 1e-6: keys.insert(0, {'time': 0.0, 'key_number': 0})
        except Exception: keys = [{'time': 0.0, 'key_number': 0}]
        unique_keys = []; last_t = -1.0
        for key in sorted(keys, key=lambda x: x['time']):
             if key['time'] > last_t + 1e-6: unique_keys.append(key); last_t = key['time']
        return unique_keys
    def _extract_sorted_events(self, midi_data: pretty_midi.PrettyMIDI, instrument_token_map: Dict[int, Tuple[int, int]]) -> List[Dict[str, Any]]:
         # (Implementation assumed unchanged)
        events = []; last_velocity_bin = {}
        for original_instrument_idx, instrument in enumerate(midi_data.instruments):
            token_pair = instrument_token_map.get(original_instrument_idx);
            if token_pair is None: continue
            program_token_idx, local_instance_token_idx = token_pair; instance_key = token_pair
            if instance_key not in last_velocity_bin: last_velocity_bin[instance_key] = -1
            first_event_time = min(([n.start for n in instrument.notes] if instrument.notes else [float('inf')]) + ([c.time for c in instrument.control_changes] if instrument.control_changes else [float('inf')]))
            if first_event_time != float('inf') and not instrument.is_drum:
                 prog_change_time = max(0.0, first_event_time - 0.001)
                 events.append({'type': 'program_change_marker', 'time': prog_change_time, 'program_token_idx': program_token_idx, 'local_instance_token_idx': local_instance_token_idx, 'program': instrument.program})
            common_meta = {'program_token_idx': program_token_idx, 'local_instance_token_idx': local_instance_token_idx}
            for note in instrument.notes:
                 events.append({'type': 'note_on', 'time': note.start, 'note': note.pitch, 'velocity': note.velocity, **common_meta})
                 events.append({'type': 'note_off', 'time': note.end, 'note': note.pitch, **common_meta})
            for control in instrument.control_changes:
                if control.number == 64: events.append({'type': 'pedal_on' if control.value >= 64 else 'pedal_off', 'time': control.time, **common_meta})
                else: events.append({'type': 'control_change', 'time': control.time, 'control_number': control.number, 'control_value': control.value, **common_meta})
        events.sort(key=lambda x: x['time'])
        timed_events = []; last_time = 0.0
        for event in events:
            time_diff = event['time'] - last_time; steps = self._quantize_time_shift(time_diff)
            if steps > 0: timed_events.append({'type': 'time_shift', 'steps': steps}); last_time += steps * self.time_step
            if event['type'] == 'note_on':
                instance_key = (event['program_token_idx'], event['local_instance_token_idx']); velocity_bin = self._quantize_velocity(event['velocity'])
                if velocity_bin != last_velocity_bin.get(instance_key, -1):
                    timed_events.append({'type': 'velocity', 'program_token_idx': event['program_token_idx'], 'local_instance_token_idx': event['local_instance_token_idx'], 'velocity_bin': velocity_bin})
                    last_velocity_bin[instance_key] = velocity_bin
            timed_events.append(event); last_time = max(last_time, event['time'])
        return timed_events
    def _tokenize_events(self, events: List[Dict[str, Any]]) -> List[int]:
        # (Implementation assumed unchanged)
        tokens = []
        for event in events:
            event_type = event['type']
            if event_type in ['note_on','note_off','velocity','control_change','program_change_marker','pedal_on','pedal_off']:
                prog_token = event.get('program_token_idx'); local_token = event.get('local_instance_token_idx')
                if prog_token is None or local_token is None: continue
                tokens.append(prog_token); tokens.append(local_token)
                if event_type == 'note_on': tokens.extend([self.NOTE_ON, self.note_value_offset + event['note']])
                elif event_type == 'note_off': tokens.extend([self.NOTE_OFF, self.note_value_offset + event['note']])
                elif event_type == 'velocity': tokens.extend([self.VELOCITY, self.velocity_value_offset + event['velocity_bin']])
                elif event_type == 'control_change': tokens.extend([self.CONTROL_CHANGE, self.cc_number_offset + event['control_number'], self.cc_value_offset + event['control_value']])
                elif event_type == 'program_change_marker': tokens.extend([self.PROGRAM_CHANGE, self.program_value_offset + event['program']])
                elif event_type == 'pedal_on': tokens.append(self.PEDAL_ON)
                elif event_type == 'pedal_off': tokens.append(self.PEDAL_OFF)
            elif event_type == 'time_shift': tokens.extend([self.TIME_SHIFT, self.time_shift_value_offset + event['steps']])
        return tokens
    # --- End Helper Methods ---


    # --- MIDI File Processing (Extraction, Tokenization) ---
    def process_midi_file(self, midi_path: str) -> Dict[str, Any] | None:
        """
        Processes a MIDI file into a token sequence, including START and END tokens.
        """
        try: midi_data = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e: print(f"Error loading MIDI {midi_path}: {e}"); return None

        # --- Instrument Mapping (unchanged) ---
        instrument_token_map = {}; instrument_metadata_map = {}; valid_instrument_count = 0
        for i, inst in enumerate(midi_data.instruments):
            prog_token = self._get_program_token(inst.program, inst.is_drum)
            if prog_token is not None:
                local_token = self._get_local_instance_token(valid_instrument_count)
                instrument_token_map[i] = (prog_token, local_token)
                instrument_metadata_map[f"original_inst_{i}"] = {
                    "program": inst.program, "is_drum": inst.is_drum, "name": inst.name,
                    "program_token": prog_token, "local_instance_token": local_token}
                valid_instrument_count += 1
        if not instrument_token_map: print(f"Warning: No valid instruments mapped in {midi_path}")

        # --- Metadata Extraction (unchanged) ---
        try:
            metadata = {'filename': os.path.basename(midi_path),
                        'tempo_changes': self._extract_tempo_changes(midi_data),
                        'time_signature_changes': self._extract_time_signatures(midi_data),
                        'key_signatures': self._extract_key_signatures(midi_data),
                        'total_time': midi_data.get_end_time(),
                        'instrument_mapping': instrument_metadata_map}
        except Exception as e: print(f"Error extracting metadata: {e}"); metadata = {'filename': os.path.basename(midi_path)}

        # --- Event Extraction and Tokenization ---
        try:
            events = self._extract_sorted_events(midi_data, instrument_token_map)
            core_tokens = self._tokenize_events(events)

            # --- MODIFICATION: Add START and END tokens ---
            final_tokens = [self.START] + core_tokens + [self.END]
            # --- End Modification ---

            return {'metadata': metadata, 'tokens': np.array(final_tokens)} # Store as numpy array
        except Exception as e: print(f"Error processing events: {e}"); return None


    # --- Tokens to MIDI Conversion ---
    def tokens_to_midi(self, tokens: List[int], vocab_start_offset: int = 0, default_tempo: float = 120.0, save_path=None) -> pretty_midi.PrettyMIDI:
        """
        Converts a sequence of tokens back into a pretty_midi.PrettyMIDI object.
        Ignores START and END tokens if present in the sequence.

        Args:
            tokens: The list of integer tokens.
            vocab_start_offset: If the tokens have an offset (e.g., from adding
                                other tokens like bos/eos outside this processor),
                                specify it here. Usually 0.
            default_tempo: Tempo to use if no tempo information is generated
                           (or if initial tempo is needed).
             save_path: Optional path to save the generated MIDI file.

        Returns:
            A pretty_midi.PrettyMIDI object.
        """
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=default_tempo)
        instrument_tracks: Dict[Tuple[int, int], pretty_midi.Instrument] = {} # (prog_token, local_token) -> Instrument
        active_notes: Dict[Tuple[int, int, int], Tuple[float, int]] = {} # (prog_token, local_token, pitch) -> (start_time, velocity)
        current_velocities: Dict[Tuple[int, int], int] = {} # (prog_token, local_token) -> velocity (0-127)
        current_time = 0.0

        i = 0
        while i < len(tokens):
            token = tokens[i] - vocab_start_offset # Adjust for any external offset

            # Identify the type of event
            event_processed = False

            # --- MODIFICATION: Explicitly handle START/END tokens by skipping ---
            if token == self.START or token == self.END or token == self.PAD:
                i += 1
                event_processed = True # Mark as processed to prevent falling into the 'unknown' case
            # --- End Modification ---

            # --- Global Events (TIME_SHIFT) ---
            elif token == self.TIME_SHIFT:
                if i + 1 < len(tokens):
                    steps_token = tokens[i+1] - vocab_start_offset
                    if self.time_shift_value_offset <= steps_token < self.time_shift_value_offset + self.time_shift_steps:
                        steps = steps_token - self.time_shift_value_offset
                        current_time += steps * self.time_step
                        i += 2 # Consumed TIME_SHIFT + value
                        event_processed = True
                    else:
                        # Invalid time shift value, skip only TIME_SHIFT token
                        # print(f"Warning: Invalid TIME_SHIFT value token {steps_token} at index {i+1}")
                        i += 1
                else:
                     # Incomplete time shift event
                    # print(f"Warning: Incomplete TIME_SHIFT event at index {i}")
                    i += 1

            # --- Instrument-Specific Events ---
            # Check if the current token is a program token
            elif self.program_token_range[0] <= token < self.program_token_range[1]:
                # Expect program_token, local_instance_token, event_marker, [value(s)]
                if i + 2 < len(tokens): # Need at least prog, local, marker
                    prog_token = token
                    local_token = tokens[i+1] - vocab_start_offset
                    event_marker = tokens[i+2] - vocab_start_offset

                    # Check if local_token is valid
                    if not (self.local_instance_range[0] <= local_token < self.local_instance_range[1]):
                        # print(f"Warning: Invalid LOCAL_INSTANCE token {local_token} following program token {prog_token} at index {i+1}")
                        i += 1 # Invalid local token, skip program token
                        continue # Move to the next token

                    instance_key = (prog_token, local_token)

                    # --- Get or Create Instrument ---
                    if instance_key not in instrument_tracks:
                        is_drum = (prog_token == self.drum_program_token_idx)
                        program_num = 0 # Default for drums
                        if not is_drum:
                             program_num = prog_token - self.program_token_offset

                        instrument_tracks[instance_key] = pretty_midi.Instrument(
                            program=program_num,
                            is_drum=is_drum,
                            name=f"{'Drum' if is_drum else f'Inst {program_num}'}_{local_token - self.local_instance_offset}"
                        )
                        midi_obj.instruments.append(instrument_tracks[instance_key])
                        # Initialize default velocity for this new instance
                        current_velocities[instance_key] = DEFAULT_VELOCITY

                    track = instrument_tracks[instance_key]
                    # Use current_time directly for program changes, add epsilon for others
                    # (This logic seems reversed from the original, let's keep original epsilon use for note events etc.)
                    event_time = current_time

                    # --- Process Specific Instrument Events ---
                    if event_marker == self.PROGRAM_CHANGE:
                         # Program change should happen exactly at current_time (no epsilon)
                         if i + 3 < len(tokens):
                            program_val_token = tokens[i+3] - vocab_start_offset
                            if self.program_value_offset <= program_val_token < self.program_value_offset + self.program_range:
                                program_num = program_val_token - self.program_value_offset
                                if not track.is_drum: # Don't change program for drum track
                                     track.program = program_num
                                     # Optionally add a Program Change MIDI event if needed for strict representation
                                     # pc_event = pretty_midi.ProgramChange(program=program_num, time=event_time)
                                     # track.program_changes.append(pc_event) # Note: pretty_midi doesn't directly support this list
                            # else: print(f"Warning: Invalid PROG_VAL token {program_val_token}")
                            i += 4; event_processed = True
                         else: i += 3 # Incomplete program change

                    # Apply epsilon offset for time-sensitive events *after* program change check
                    event_time += TIME_EPSILON

                    if not event_processed and event_marker == self.VELOCITY:
                         if i + 3 < len(tokens):
                            vel_val_token = tokens[i+3] - vocab_start_offset
                            if self.velocity_value_offset <= vel_val_token < self.velocity_value_offset + self.velocity_bins:
                                vel_bin = vel_val_token - self.velocity_value_offset
                                current_velocities[instance_key] = self._unquantize_velocity(vel_bin)
                            # else: print(f"Warning: Invalid VEL_VAL token {vel_val_token}")
                            i += 4; event_processed = True
                         else: i += 3 # Incomplete velocity event

                    elif not event_processed and event_marker == self.NOTE_ON:
                        if i + 3 < len(tokens):
                            note_val_token = tokens[i+3] - vocab_start_offset
                            if self.note_value_offset <= note_val_token < self.note_value_offset + self.note_range:
                                pitch = note_val_token - self.note_value_offset
                                velocity = current_velocities.get(instance_key, DEFAULT_VELOCITY)
                                note_key = (*instance_key, pitch)
                                # End previous note if overlapping
                                if note_key in active_notes:
                                    start_time, start_vel = active_notes.pop(note_key)
                                    end_time_prev = max(start_time + TIME_EPSILON / 2, event_time - TIME_EPSILON / 2)
                                    if end_time_prev > start_time:
                                        note_obj = pretty_midi.Note(velocity=start_vel, pitch=pitch, start=start_time, end=end_time_prev)
                                        track.notes.append(note_obj)
                                # Start the new note
                                active_notes[note_key] = (event_time, velocity)
                            # else: print(f"Warning: Invalid NOTE_VAL token {note_val_token}")
                            i += 4; event_processed = True
                        else: i += 3 # Incomplete note on

                    elif not event_processed and event_marker == self.NOTE_OFF:
                        if i + 3 < len(tokens):
                            note_val_token = tokens[i+3] - vocab_start_offset
                            if self.note_value_offset <= note_val_token < self.note_value_offset + self.note_range:
                                pitch = note_val_token - self.note_value_offset
                                note_key = (*instance_key, pitch)
                                if note_key in active_notes:
                                    start_time, start_vel = active_notes.pop(note_key)
                                    end_time = max(start_time + TIME_EPSILON, event_time) # Ensure end > start
                                    note_obj = pretty_midi.Note(velocity=start_vel, pitch=pitch, start=start_time, end=end_time)
                                    track.notes.append(note_obj)
                                # else: print(f"Warning: NOTE_OFF for inactive note {pitch}")
                            # else: print(f"Warning: Invalid NOTE_VAL token {note_val_token}")
                            i += 4; event_processed = True
                        else: i += 3 # Incomplete note off

                    elif not event_processed and event_marker == self.PEDAL_ON:
                        cc = pretty_midi.ControlChange(number=64, value=100, time=event_time)
                        track.control_changes.append(cc)
                        i += 3; event_processed = True # Consumed prog, local, marker

                    elif not event_processed and event_marker == self.PEDAL_OFF:
                        cc = pretty_midi.ControlChange(number=64, value=0, time=event_time)
                        track.control_changes.append(cc)
                        i += 3; event_processed = True # Consumed prog, local, marker

                    elif not event_processed and event_marker == self.CONTROL_CHANGE:
                         if i + 4 < len(tokens): # Need marker, num, val
                             cc_num_token = tokens[i+3] - vocab_start_offset
                             cc_val_token = tokens[i+4] - vocab_start_offset
                             valid_num = self.cc_number_offset <= cc_num_token < self.cc_number_offset + self.cc_range
                             valid_val = self.cc_value_offset <= cc_val_token < self.cc_value_offset + self.cc_range
                             if valid_num and valid_val:
                                 cc_num = cc_num_token - self.cc_number_offset
                                 cc_val = cc_val_token - self.cc_value_offset
                                 if cc_num != 64: # Avoid adding pedal CCs here if handled separately
                                     cc = pretty_midi.ControlChange(number=cc_num, value=cc_val, time=event_time)
                                     track.control_changes.append(cc)
                             # else: print(f"Warning: Invalid CC num ({cc_num_token}) or val ({cc_val_token}) token")
                             i += 5; event_processed = True
                         else: i+= 3 # Incomplete CC event

                    # If event marker wasn't recognized or event was incomplete
                    if not event_processed:
                        # print(f"Warning: Unrecognized or incomplete event marker {event_marker} at index {i+2}")
                        i += 3 # Skip prog, local, marker

                else: # Not enough tokens for a full instrument event
                    # print(f"Warning: Incomplete instrument event start at index {i}")
                    i += 1 # Skip program token

            # --- Other/Unknown Tokens ---
            # This case should now only be hit for truly unknown tokens if START/END/PAD and handled tokens cover everything else
            if not event_processed:
                # print(f"Warning: Skipping unknown token {token} at index {i}")
                i += 1

        # --- Final Cleanup (unchanged) ---
        # After processing all tokens, turn off any remaining active notes
        for (prog_token, local_token, pitch), (start_time, start_vel) in active_notes.items():
             instance_key = (prog_token, local_token)
             if instance_key in instrument_tracks:
                 track = instrument_tracks[instance_key]
                 end_time = max(start_time + TIME_EPSILON, current_time + self.time_step)
                 note_obj = pretty_midi.Note(velocity=start_vel, pitch=pitch, start=start_time, end=end_time)
                 track.notes.append(note_obj)

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
                midi_obj.write(save_path)
                print(f"Saved processed MIDI to: {save_path}")
            except Exception as e:
                print(f"Error saving MIDI to {save_path}: {e}")
        return midi_obj

    # --- tokens_to_text (unchanged, as it already handles START/END via map) ---
    def tokens_to_text(self, tokens: List[int], vocab_start_offset: int = 0) -> str:
        # (Implementation assumed unchanged)
        text_events = []
        i = 0
        while i < len(tokens):
            token = tokens[i] - vocab_start_offset
            token_name = self.token_to_name_map.get(token, f"UNK({token})")
            current_event_str = ""
            consumed = 1 # How many tokens this logical event consumed

            # --- Global Events ---
            if token == self.TIME_SHIFT:
                if i + 1 < len(tokens):
                    val_token = tokens[i+1] - vocab_start_offset
                    val_name = self.token_to_name_map.get(val_token, f"UNK_VAL({val_token})")
                    current_event_str = f"[{token_name} {val_name}]"
                    consumed = 2
                else:
                    current_event_str = f"[{token_name} (incomplete)]"
                    consumed = 1

            # --- Instrument-Specific Events ---
            elif self.program_token_range[0] <= token < self.program_token_range[1]:
                prog_name = token_name # Already formatted like PROG(x)
                if i + 2 < len(tokens): # Need prog, local, marker
                    local_token = tokens[i+1] - vocab_start_offset
                    local_name = self.token_to_name_map.get(local_token, f"UNK_LOCAL({local_token})")
                    marker_token = tokens[i+2] - vocab_start_offset
                    marker_name = self.token_to_name_map.get(marker_token, f"UNK_MARKER({marker_token})")
                    consumed = 3
                    event_parts = [prog_name, local_name, marker_name]

                    # Check for value tokens based on marker
                    if marker_token in [self.VELOCITY, self.NOTE_ON, self.NOTE_OFF]:
                        if i + 3 < len(tokens):
                            val_token = tokens[i+3] - vocab_start_offset
                            val_name = self.token_to_name_map.get(val_token, f"UNK_VAL({val_token})")
                            event_parts.append(val_name)
                            consumed = 4
                        else: event_parts.append("(incomplete_val)")
                    elif marker_token == self.PROGRAM_CHANGE:
                         if i + 3 < len(tokens):
                             val_token = tokens[i+3] - vocab_start_offset
                             val_name = self.token_to_name_map.get(val_token, f"UNK_PROGVAL({val_token})")
                             event_parts.append(val_name)
                             consumed = 4
                         else: event_parts.append("(incomplete_prog_val)")
                    elif marker_token == self.CONTROL_CHANGE:
                        if i + 4 < len(tokens):
                            num_token = tokens[i+3] - vocab_start_offset
                            val_token = tokens[i+4] - vocab_start_offset
                            num_name = self.token_to_name_map.get(num_token, f"UNK_CCNUM({num_token})")
                            val_name = self.token_to_name_map.get(val_token, f"UNK_CCVAL({val_token})")
                            event_parts.extend([num_name, val_name])
                            consumed = 5
                        else: event_parts.append("(incomplete_cc)")
                    # PEDAL_ON/OFF have no values, consumed is already 3

                    current_event_str = f"[{' '.join(event_parts)}]"

                else: # Incomplete instrument event start
                    current_event_str = f"[{prog_name} (incomplete)]"
                    consumed = min(len(tokens) - i, 2) # Consume prog and maybe local if present

            # --- Other Tokens (Includes START, END, PAD) ---
            else:
                current_event_str = f"[{token_name}]" # Handle special tokens like START/END or unknowns
                consumed = 1

            text_events.append(current_event_str)
            i += consumed

        return "\n".join(text_events) # Or join with spaces: " ".join(text_events)