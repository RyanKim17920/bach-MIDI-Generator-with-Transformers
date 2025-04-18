import os
from typing import List, Dict, Tuple, Any, Optional
import pretty_midi
import numpy as np
import math # Added for ceil

PAD_TOKEN_VALUE = -100 # Usually used for masking loss, not as a token index
PAD_IDX = 0
START_IDX = 1
END_IDX = 2
MAX_LOCAL_INSTRUMENTS = 16 # Max concurrent instruments *per file* to track
DRUM_PROGRAM_TOKEN_OFFSET = 128 # Relative offset for the drum token
DEFAULT_VELOCITY = 64 # Default velocity if none is specified before a note
TIME_EPSILON = 0.001 # Small offset to ensure event ordering
MIN_NOTE_DURATION_SECONDS = 0.005 # Minimum duration for a note when reconstructing

class MIDIProcessor:
    """
    Processes MIDI files into token sequences and converts token sequences back
    to MIDI or human-readable text. Uses specific program tokens and local
    instance tokens. Includes START/END tokens.

    Offers two note representation modes:
    1. use_note_off=True: Explicit NOTE_ON and NOTE_OFF events.
    2. use_note_off=False: NOTE_ON followed by NOTE_DURATION and duration value.
    """

    def __init__(self,
                 time_step: float = 0.01,
                 velocity_bins: int = 32,
                 max_time_shift_seconds: float = 10.0,
                 max_local_instruments: int = MAX_LOCAL_INSTRUMENTS,
                 use_note_off: bool = True, # <<< New parameter
                 max_note_duration_seconds: Optional[float] = None # <<< New parameter
                ):

        self.time_step = time_step
        self.velocity_bins = velocity_bins
        self.max_time_shift_seconds = max_time_shift_seconds
        # If max_note_duration not specified, tie it to max_time_shift
        self.max_note_duration_seconds = max_note_duration_seconds if max_note_duration_seconds is not None else max_time_shift_seconds
        self.max_local_instruments = max_local_instruments
        self.use_note_off = use_note_off # Store the mode
        self.pad_token_idx = PAD_IDX

        # --- Vocabulary Definition
        _current_offset = 0
        self.PAD = PAD_IDX; _current_offset +=1
        self.START = START_IDX; _current_offset +=1
        self.END = END_IDX; _current_offset +=1

        self.program_token_offset = _current_offset
        self.num_program_tokens = 128 + 1 # 128 standard + 1 drum meta
        self.drum_program_token_idx = self.program_token_offset + DRUM_PROGRAM_TOKEN_OFFSET
        self.program_token_range = (self.program_token_offset, self.program_token_offset + self.num_program_tokens)
        _current_offset += self.num_program_tokens

        self.local_instance_offset = _current_offset
        self.local_instance_range = (self.local_instance_offset, self.local_instance_offset + self.max_local_instruments)
        _current_offset += self.max_local_instruments

        # Event Markers
        self.NOTE_ON = _current_offset; _current_offset += 1
        self.NOTE_OFF = _current_offset; _current_offset += 1 # Always defined, but maybe unused
        self.TIME_SHIFT = _current_offset; _current_offset += 1
        self.VELOCITY = _current_offset; _current_offset += 1
        self.CONTROL_CHANGE = _current_offset; _current_offset += 1
        self.PROGRAM_CHANGE = _current_offset; _current_offset += 1
        self.PEDAL_ON = _current_offset; _current_offset += 1
        self.PEDAL_OFF = _current_offset; _current_offset += 1
        self.NOTE_DURATION = _current_offset; _current_offset += 1 # New marker

        # Value Ranges
        self.note_value_offset = _current_offset
        self.note_range = 128; _current_offset += self.note_range

        self.time_shift_value_offset = _current_offset
        self.time_shift_steps = self._calculate_steps(self.max_time_shift_seconds, self.time_step)
        _current_offset += self.time_shift_steps

        self.velocity_value_offset = _current_offset
        _current_offset += self.velocity_bins

        self.cc_number_offset = _current_offset
        self.cc_range = 128; _current_offset += self.cc_range
        self.cc_value_offset = _current_offset
        _current_offset += self.cc_range

        self.program_value_offset = _current_offset
        self.program_range = 128; _current_offset += self.program_range

        self.note_duration_value_offset = _current_offset # New value range
        self.duration_steps = self._calculate_steps(self.max_note_duration_seconds, self.time_step)
        _current_offset += self.duration_steps

        self.vocab_size = _current_offset
        # --- End Vocabulary Definition ---

        # --- Build Reverse Mapping for Text Conversion ---
        self.token_to_name_map = {}
        self._build_reverse_map()

        print(f"MIDI Processor Config:")
        print(f"  - use_note_off: {self.use_note_off}")
        print(f"  - time_step: {self.time_step}s")
        print(f"  - max_time_shift: {self.max_time_shift_seconds}s ({self.time_shift_steps} steps)")
        if not self.use_note_off:
            print(f"  - max_note_duration: {self.max_note_duration_seconds}s ({self.duration_steps} steps)")
        print(f"Total Vocab Size: {self.vocab_size}")

    def _calculate_steps(self, max_seconds: float, step_size: float) -> int:
        """Calculates the number of steps for time or duration."""
        # Add 1 because the range includes 0 steps
        return max(1, int(math.ceil(max_seconds / step_size)) + 1)

    def _build_reverse_map(self):
        """Builds the token index to human-readable name map."""
        self.token_to_name_map = {
            self.PAD: "PAD", self.START: "START", self.END: "END",
            self.NOTE_ON: "NOTE_ON", self.NOTE_OFF: "NOTE_OFF", self.TIME_SHIFT: "TIME_SHIFT",
            self.VELOCITY: "VELOCITY", self.CONTROL_CHANGE: "CONTROL_CHANGE",
            self.PROGRAM_CHANGE: "PROGRAM_CHANGE", self.PEDAL_ON: "PEDAL_ON",
            self.PEDAL_OFF: "PEDAL_OFF", self.NOTE_DURATION: "NOTE_DURATION"
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
        # Note Duration Values
        for i in range(self.duration_steps):
             self.token_to_name_map[self.note_duration_value_offset + i] = f"DUR_VAL({i})"

    # --- Quantization Helpers ---
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
    def _quantize_time_or_duration(self, time_diff: float, max_steps: int) -> int:
        """Quantizes time difference or duration into steps."""
        if time_diff < self.time_step / 2.0: return 0
        # Use round instead of ceil for potentially better accuracy near step boundaries
        steps = int(round(time_diff / self.time_step))
        return min(steps, max_steps - 1)

    def _unquantize_steps(self, steps: int) -> float:
        """Converts steps back to seconds."""
        return steps * self.time_step

    # --- Metadata Extraction Helpers (unchanged) ---
    def _extract_tempo_changes(self, midi_data: pretty_midi.PrettyMIDI) -> List[Dict[str, float]]:
        # (Implementation assumed unchanged from original)
        changes = []
        try:
            tempo_times, tempo_values = midi_data.get_tempo_changes()
            for t, q in zip(tempo_times, tempo_values):
                if t >= 0 and q > 0: changes.append({'time': float(t), 'tempo': float(q)})
            if not changes or changes[0]['time'] > 1e-6:
                 initial_tempo = 120.0
                 # Use midi_data's estimate if available
                 if midi_data.estimate_tempo() > 0: initial_tempo = midi_data.estimate_tempo()
                 elif len(tempo_values) > 0 and tempo_values[0] > 0: initial_tempo = float(tempo_values[0])
                 changes.insert(0, {'time': 0.0, 'tempo': initial_tempo})
        except Exception: changes = [{'time': 0.0, 'tempo': 120.0}]
        return sorted(changes, key=lambda x: x['time'])

    def _extract_time_signatures(self, midi_data: pretty_midi.PrettyMIDI) -> List[Dict[str, Any]]:
        # (Implementation assumed unchanged from original)
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
        # (Implementation assumed unchanged from original)
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
         """Extracts and sorts all MIDI events, handling note representation based on self.use_note_off."""
         events = []
         last_velocity_bin = {} # (prog_token, local_token) -> last_vel_bin

         for original_instrument_idx, instrument in enumerate(midi_data.instruments):
            token_pair = instrument_token_map.get(original_instrument_idx)
            if token_pair is None: continue # Skip instrument if not mapped

            program_token_idx, local_instance_token_idx = token_pair
            instance_key = token_pair # Use (prog_token, local_token) as the key

            # Initialize last velocity for this instance if not seen before
            if instance_key not in last_velocity_bin:
                last_velocity_bin[instance_key] = -1 # Use -1 to indicate no velocity set yet

            # Add program change marker slightly before the first event of non-drum instruments
            first_event_time = min(([n.start for n in instrument.notes] if instrument.notes else [float('inf')]) +
                                   ([c.time for c in instrument.control_changes] if instrument.control_changes else [float('inf')]))

            if first_event_time != float('inf') and not instrument.is_drum:
                 # Place program change marker slightly before the first event
                 prog_change_time = max(0.0, first_event_time - TIME_EPSILON * 2) # Ensure non-negative and distinct time
                 events.append({
                     'type': 'program_change_marker',
                     'time': prog_change_time,
                     'program_token_idx': program_token_idx,
                     'local_instance_token_idx': local_instance_token_idx,
                     'program': instrument.program # Store original program number
                 })

            common_meta = {
                'program_token_idx': program_token_idx,
                'local_instance_token_idx': local_instance_token_idx
            }

            # --- Note Events ---
            for note in instrument.notes:
                # Skip zero or negative duration notes as they cause issues
                if note.end <= note.start:
                    # print(f"Warning: Skipping note with non-positive duration (start={note.start}, end={note.end}, pitch={note.pitch})")
                    continue

                if self.use_note_off:
                    # Mode 1: Explicit NOTE_ON / NOTE_OFF
                    events.append({'type': 'note_on', 'time': note.start, 'note': note.pitch, 'velocity': note.velocity, **common_meta})
                    events.append({'type': 'note_off', 'time': note.end, 'note': note.pitch, **common_meta})
                else:
                    # Mode 2: NOTE_ON + Duration
                    duration_seconds = note.end - note.start
                    duration_steps = self._quantize_time_or_duration(duration_seconds, self.duration_steps)
                    events.append({
                        'type': 'note_on_with_duration',
                        'time': note.start,
                        'note': note.pitch,
                        'velocity': note.velocity,
                        'duration_steps': duration_steps,
                        **common_meta
                    })

            # --- Control Change Events ---
            for control in instrument.control_changes:
                # Specifically handle sustain pedal (CC 64)
                if control.number == 64:
                    event_type = 'pedal_on' if control.value >= 64 else 'pedal_off'
                    events.append({'type': event_type, 'time': control.time, **common_meta})
                else:
                    # Handle other CCs
                    events.append({
                        'type': 'control_change',
                        'time': control.time,
                        'control_number': control.number,
                        'control_value': control.value,
                        **common_meta
                    })

         # --- Sort all events globally by time ---
         # Use a stable sort if needed, but primary sort is time. Add secondary sort keys if necessary.
         events.sort(key=lambda x: (x['time'], self._event_sort_priority(x['type'])))


         # --- Insert Time Shifts and Velocity Changes ---
         timed_events = []
         last_time = 0.0
         for event in events:
             # Ensure time doesn't go backwards due to precision or ordering issues
             event_time = max(last_time, event['time'])

             # Calculate and add time shift if needed
             time_diff = event_time - last_time
             if time_diff > TIME_EPSILON / 2: # Only add if difference is significant
                 time_shift_steps = self._quantize_time_or_duration(time_diff, self.time_shift_steps)
                 if time_shift_steps > 0:
                     timed_events.append({'type': 'time_shift', 'steps': time_shift_steps})
                     # Update last_time based on the quantized shift
                     last_time += self._unquantize_steps(time_shift_steps)
                     # Re-adjust event_time if quantization changed it significantly relative to original
                     event_time = last_time # Align subsequent events to the quantized time grid


             # Add velocity event if changed (only for note_on types)
             if event['type'] == 'note_on' or event['type'] == 'note_on_with_duration':
                 instance_key = (event['program_token_idx'], event['local_instance_token_idx'])
                 velocity_bin = self._quantize_velocity(event['velocity'])

                 if velocity_bin != last_velocity_bin.get(instance_key, -1):
                     # Insert velocity *before* the note event it applies to
                     timed_events.append({
                         'type': 'velocity',
                         'program_token_idx': event['program_token_idx'],
                         'local_instance_token_idx': event['local_instance_token_idx'],
                         'velocity_bin': velocity_bin
                     })
                     last_velocity_bin[instance_key] = velocity_bin # Update last known velocity

             # Add the original event itself
             timed_events.append(event)

             # Update last_time based on the processed event's time
             # For note_off, time is note end. For others, it's the event occurrence time.
             # Make sure last_time reflects the *actual* time grid position.
             last_time = max(last_time, event_time) # Use event_time which might have been adjusted by time shift


         return timed_events

    def _event_sort_priority(self, event_type: str) -> int:
        """Assigns priority for sorting events occurring at the same time."""
        # Lower numbers sort earlier
        if event_type == 'program_change_marker': return 0
        if event_type == 'control_change': return 1
        if event_type == 'pedal_off': return 2 # Pedal off slightly before notes
        if event_type == 'note_off': return 3   # Note off before note on
        if event_type == 'velocity': return 4   # Velocity change just before note on
        if event_type == 'note_on': return 5
        if event_type == 'note_on_with_duration': return 5
        if event_type == 'pedal_on': return 6 # Pedal on slightly after notes
        return 10 # Default priority

    def _tokenize_events(self, events: List[Dict[str, Any]]) -> List[int]:
        """Converts the intermediate event list into a sequence of tokens."""
        tokens = []
        for event in events:
            event_type = event['type']

            # Events associated with an instrument instance
            if event_type in ['note_on', 'note_off', 'note_on_with_duration', 'velocity',
                              'control_change', 'program_change_marker', 'pedal_on', 'pedal_off']:
                prog_token = event.get('program_token_idx')
                local_token = event.get('local_instance_token_idx')
                if prog_token is None or local_token is None:
                    # print(f"Warning: Skipping event missing program/local token: {event}")
                    continue # Should not happen with proper extraction

                # Prepend instrument identifiers
                tokens.append(prog_token)
                tokens.append(local_token)

                # Append event marker and value(s)
                if event_type == 'note_on':
                    tokens.extend([self.NOTE_ON, self.note_value_offset + event['note']])
                elif event_type == 'note_off':
                     # Only add NOTE_OFF if using that mode
                     if self.use_note_off:
                         tokens.extend([self.NOTE_OFF, self.note_value_offset + event['note']])
                     # If not using note_off, this event type shouldn't exist here, but we handle defensively
                elif event_type == 'note_on_with_duration':
                     # Only generated if not use_note_off
                     if not self.use_note_off:
                         tokens.extend([
                             self.NOTE_ON, self.note_value_offset + event['note'],
                             self.NOTE_DURATION, self.note_duration_value_offset + event['duration_steps']
                         ])
                     # If use_note_off is True, this event type shouldn't exist
                elif event_type == 'velocity':
                    tokens.extend([self.VELOCITY, self.velocity_value_offset + event['velocity_bin']])
                elif event_type == 'control_change':
                    tokens.extend([self.CONTROL_CHANGE,
                                   self.cc_number_offset + event['control_number'],
                                   self.cc_value_offset + event['control_value']])
                elif event_type == 'program_change_marker':
                    # Use PROGRAM_CHANGE marker, value is original program
                    tokens.extend([self.PROGRAM_CHANGE, self.program_value_offset + event['program']])
                elif event_type == 'pedal_on':
                    tokens.append(self.PEDAL_ON)
                elif event_type == 'pedal_off':
                    tokens.append(self.PEDAL_OFF)

            # Global Events (currently only time shift)
            elif event_type == 'time_shift':
                tokens.extend([self.TIME_SHIFT, self.time_shift_value_offset + event['steps']])

            # Handle potential unknown event types defensively
            # else:
            #    print(f"Warning: Skipping unknown event type during tokenization: {event_type}")

        return tokens

    # --- MIDI File Processing (Extraction, Tokenization) ---
    def process_midi_file(self, midi_path: str) -> Dict[str, Any] | None:
        """
        Processes a MIDI file into a token sequence, including START and END tokens.
        Uses the note representation mode specified in __init__ (use_note_off).
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            print(f"Error loading MIDI {midi_path}: {e}")
            return None

        # --- Instrument Mapping (unchanged) ---
        instrument_token_map = {} # Map original index -> (prog_token, local_token)
        instrument_metadata_map = {}
        valid_instrument_count = 0
        for i, inst in enumerate(midi_data.instruments):
            prog_token = self._get_program_token(inst.program, inst.is_drum)
            if prog_token is not None:
                # Ensure we don't exceed max concurrent instruments tracked
                if valid_instrument_count < self.max_local_instruments:
                    local_token = self._get_local_instance_token(valid_instrument_count)
                    instrument_token_map[i] = (prog_token, local_token)
                    instrument_metadata_map[f"original_inst_{i}"] = {
                        "program": inst.program, "is_drum": inst.is_drum, "name": inst.name,
                        "program_token": prog_token, "local_instance_token": local_token}
                    valid_instrument_count += 1
                # else:
                #     print(f"Warning: Exceeded max_local_instruments ({self.max_local_instruments}). Skipping instrument {i} ('{inst.name}') in {midi_path}")


        if not instrument_token_map:
            # print(f"Warning: No valid/trackable instruments mapped in {midi_path}. Skipping file.")
            return None # Cannot process file with no mapped instruments

        # --- Metadata Extraction (unchanged) ---
        try:
            metadata = {
                'filename': os.path.basename(midi_path),
                'tempo_changes': self._extract_tempo_changes(midi_data),
                'time_signature_changes': self._extract_time_signatures(midi_data),
                'key_signatures': self._extract_key_signatures(midi_data),
                'total_time': midi_data.get_end_time(),
                'instrument_mapping': instrument_metadata_map,
                'processing_mode': {'use_note_off': self.use_note_off} # Store mode used
            }
        except Exception as e:
            print(f"Error extracting metadata from {midi_path}: {e}")
            metadata = {'filename': os.path.basename(midi_path),
                        'processing_mode': {'use_note_off': self.use_note_off}}

        # --- Event Extraction and Tokenization ---
        try:
            # Use the modified extraction function
            events = self._extract_sorted_events(midi_data, instrument_token_map)
            # Use the modified tokenization function
            core_tokens = self._tokenize_events(events)

            # Add START and END tokens
            final_tokens = [self.START] + core_tokens + [self.END]

            return {'metadata': metadata, 'tokens': np.array(final_tokens, dtype=np.int32)} # Use int32 for numpy
        except Exception as e:
            print(f"Error processing events/tokenizing {midi_path}: {e}")
            import traceback
            traceback.print_exc() # Print stack trace for debugging
            return None


    # --- Tokens to MIDI Conversion ---
    def tokens_to_midi(self, tokens: List[int], vocab_start_offset: int = 0, default_tempo: float = 120.0, save_path=None) -> pretty_midi.PrettyMIDI:
        """
        Converts a sequence of tokens back into a pretty_midi.PrettyMIDI object.
        Handles both NOTE_OFF and NOTE_DURATION modes based on self.use_note_off.
        Ignores START, END, and PAD tokens.

        Args:
            tokens: The list of integer tokens.
            vocab_start_offset: Offset if tokens use a different base index.
            default_tempo: Initial tempo if not otherwise specified.
            save_path: Optional path to save the generated MIDI file.

        Returns:
            A pretty_midi.PrettyMIDI object.
        """
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=default_tempo)
        instrument_tracks: Dict[Tuple[int, int], pretty_midi.Instrument] = {} # (prog_token, local_token) -> Instrument
        # Active notes only needed for use_note_off=True mode
        active_notes: Dict[Tuple[int, int, int], Tuple[float, int]] = {} # (prog_token, local_token, pitch) -> (start_time, velocity)
        current_velocities: Dict[Tuple[int, int], int] = {} # (prog_token, local_token) -> velocity (0-127)
        current_time = 0.0

        i = 0
        while i < len(tokens):
            token = tokens[i] - vocab_start_offset # Adjust for external offset

            # --- Skip Special Tokens ---
            if token in [self.START, self.END, self.PAD]:
                i += 1
                continue

            event_processed = False # Flag to check if token was handled

            # --- Global Events (TIME_SHIFT) ---
            if token == self.TIME_SHIFT:
                if i + 1 < len(tokens):
                    steps_token = tokens[i+1] - vocab_start_offset
                    if self.time_shift_value_offset <= steps_token < self.time_shift_value_offset + self.time_shift_steps:
                        steps = steps_token - self.time_shift_value_offset
                        current_time += self._unquantize_steps(steps)
                        i += 2; event_processed = True
                    else: i += 1 # Invalid value, skip only TIME_SHIFT
                else: i += 1 # Incomplete

            # --- Instrument-Specific Events ---
            # Check if the current token is a program token
            elif self.program_token_range[0] <= token < self.program_token_range[1]:
                # Expect: program_token, local_instance_token, event_marker, [value(s)]...
                # Need at least 3 tokens for basic events (prog, local, marker)
                if i + 2 < len(tokens):
                    prog_token = token
                    local_token_raw = tokens[i+1] - vocab_start_offset

                    # Validate local instance token
                    if not (self.local_instance_range[0] <= local_token_raw < self.local_instance_range[1]):
                        # print(f"Warning: Invalid LOCAL_INSTANCE token {local_token_raw} at index {i+1}. Skipping.")
                        i += 1 # Skip only program token if local is invalid
                        continue

                    local_token = local_token_raw # Validated
                    instance_key = (prog_token, local_token)

                    # Get or create instrument track
                    if instance_key not in instrument_tracks:
                        is_drum = (prog_token == self.drum_program_token_idx)
                        program_num = 0 if is_drum else (prog_token - self.program_token_offset)
                        instrument_tracks[instance_key] = pretty_midi.Instrument(
                            program=program_num,
                            is_drum=is_drum,
                            name=f"{'Drum' if is_drum else f'Inst {program_num}'}_{local_token - self.local_instance_offset}"
                        )
                        midi_obj.instruments.append(instrument_tracks[instance_key])
                        current_velocities[instance_key] = DEFAULT_VELOCITY # Initialize default velocity

                    track = instrument_tracks[instance_key]
                    event_marker = tokens[i+2] - vocab_start_offset
                    # Most events happen slightly *after* the current time step starts
                    event_time = current_time + TIME_EPSILON

                    # Determine required tokens and process event
                    consumed_tokens = 0 # How many tokens are consumed by this event block

                    # --- Process based on event_marker ---
                    if event_marker == self.PROGRAM_CHANGE:
                        if i + 3 < len(tokens):
                            program_val_token = tokens[i+3] - vocab_start_offset
                            if self.program_value_offset <= program_val_token < self.program_value_offset + self.program_range:
                                program_num = program_val_token - self.program_value_offset
                                if not track.is_drum: track.program = program_num
                                # Program changes happen exactly at current_time (no epsilon)
                                # Note: Pretty MIDI doesn't directly store program changes this way,
                                # setting track.program is the main effect.
                            consumed_tokens = 4
                        else: consumed_tokens = 3 # Incomplete

                    elif event_marker == self.VELOCITY:
                        if i + 3 < len(tokens):
                            vel_val_token = tokens[i+3] - vocab_start_offset
                            if self.velocity_value_offset <= vel_val_token < self.velocity_value_offset + self.velocity_bins:
                                vel_bin = vel_val_token - self.velocity_value_offset
                                current_velocities[instance_key] = self._unquantize_velocity(vel_bin)
                            consumed_tokens = 4
                        else: consumed_tokens = 3 # Incomplete

                    elif event_marker == self.NOTE_ON:
                        if self.use_note_off: # --- Mode 1: NOTE_ON / NOTE_OFF ---
                            if i + 3 < len(tokens):
                                note_val_token = tokens[i+3] - vocab_start_offset
                                if self.note_value_offset <= note_val_token < self.note_value_offset + self.note_range:
                                    pitch = note_val_token - self.note_value_offset
                                    velocity = current_velocities.get(instance_key, DEFAULT_VELOCITY)
                                    note_key = (*instance_key, pitch)

                                    # End previous note of same pitch if still active (handle overlap)
                                    if note_key in active_notes:
                                        start_time_prev, start_vel_prev = active_notes.pop(note_key)
                                        # Ensure end time is slightly before new start, but after old start
                                        end_time_prev = max(start_time_prev + TIME_EPSILON / 2, event_time - TIME_EPSILON / 2)
                                        if end_time_prev > start_time_prev:
                                             note_obj = pretty_midi.Note(velocity=start_vel_prev, pitch=pitch, start=start_time_prev, end=end_time_prev)
                                             track.notes.append(note_obj)

                                    # Record the start of the new note
                                    active_notes[note_key] = (event_time, velocity)
                                consumed_tokens = 4
                            else: consumed_tokens = 3 # Incomplete NOTE_ON

                        else: # --- Mode 2: NOTE_ON + NOTE_DURATION ---
                            # Expect: PROG, LOCAL, NOTE_ON, NOTE_VAL, NOTE_DURATION, DUR_VAL
                            if i + 5 < len(tokens):
                                note_val_token = tokens[i+3] - vocab_start_offset
                                duration_marker_token = tokens[i+4] - vocab_start_offset
                                duration_val_token = tokens[i+5] - vocab_start_offset

                                # Validate tokens
                                valid_note = self.note_value_offset <= note_val_token < self.note_value_offset + self.note_range
                                valid_dur_marker = (duration_marker_token == self.NOTE_DURATION)
                                valid_dur_val = self.note_duration_value_offset <= duration_val_token < self.note_duration_value_offset + self.duration_steps

                                if valid_note and valid_dur_marker and valid_dur_val:
                                    pitch = note_val_token - self.note_value_offset
                                    duration_steps = duration_val_token - self.note_duration_value_offset
                                    duration_sec = self._unquantize_steps(duration_steps)
                                    velocity = current_velocities.get(instance_key, DEFAULT_VELOCITY)

                                    start_time = event_time
                                    # Ensure minimum duration, end time must be after start time
                                    end_time = start_time + max(duration_sec, MIN_NOTE_DURATION_SECONDS)

                                    note_obj = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
                                    track.notes.append(note_obj)
                                    consumed_tokens = 6
                                else:
                                    # Invalid sequence for duration note
                                    # print(f"Warning: Invalid NOTE_ON+DURATION sequence at index {i}. Skipping.")
                                    consumed_tokens = 3 # Skip prog, local, NOTE_ON marker only
                            else:
                                consumed_tokens = 3 # Incomplete duration note sequence

                    elif event_marker == self.NOTE_OFF:
                         if self.use_note_off: # --- Only process in NOTE_OFF mode ---
                             if i + 3 < len(tokens):
                                 note_val_token = tokens[i+3] - vocab_start_offset
                                 if self.note_value_offset <= note_val_token < self.note_value_offset + self.note_range:
                                     pitch = note_val_token - self.note_value_offset
                                     note_key = (*instance_key, pitch)
                                     if note_key in active_notes:
                                         start_time, start_vel = active_notes.pop(note_key)
                                         # Ensure end time is strictly after start time
                                         end_time = max(start_time + TIME_EPSILON, event_time)
                                         note_obj = pretty_midi.Note(velocity=start_vel, pitch=pitch, start=start_time, end=end_time)
                                         track.notes.append(note_obj)
                                     # else: print(f"Warning: NOTE_OFF for inactive note {pitch} at time {event_time:.3f}")
                                 consumed_tokens = 4
                             else: consumed_tokens = 3 # Incomplete NOTE_OFF
                         else:
                             # Ignore NOTE_OFF tokens if not in use_note_off mode
                             consumed_tokens = 3 # Skip prog, local, NOTE_OFF marker

                    elif event_marker == self.PEDAL_ON:
                        cc = pretty_midi.ControlChange(number=64, value=100, time=event_time)
                        track.control_changes.append(cc)
                        consumed_tokens = 3

                    elif event_marker == self.PEDAL_OFF:
                        cc = pretty_midi.ControlChange(number=64, value=0, time=event_time)
                        track.control_changes.append(cc)
                        consumed_tokens = 3

                    elif event_marker == self.CONTROL_CHANGE:
                         if i + 4 < len(tokens): # Need marker, num, val
                             cc_num_token = tokens[i+3] - vocab_start_offset
                             cc_val_token = tokens[i+4] - vocab_start_offset
                             valid_num = self.cc_number_offset <= cc_num_token < self.cc_number_offset + self.cc_range
                             valid_val = self.cc_value_offset <= cc_val_token < self.cc_value_offset + self.cc_range
                             if valid_num and valid_val:
                                 cc_num = cc_num_token - self.cc_number_offset
                                 cc_val = cc_val_token - self.cc_value_offset
                                 # Avoid adding pedal CCs here if handled by PEDAL_ON/OFF
                                 if cc_num != 64:
                                     cc = pretty_midi.ControlChange(number=cc_num, value=cc_val, time=event_time)
                                     track.control_changes.append(cc)
                             consumed_tokens = 5
                         else: consumed_tokens = 3 # Incomplete CC event

                    # --- Handle unrecognized marker or incomplete sequence ---
                    if consumed_tokens == 0:
                        # print(f"Warning: Unrecognized or incomplete instrument event sequence starting at index {i}. Marker: {event_marker}. Skipping 3 tokens.")
                        consumed_tokens = 3 # Default skip: prog, local, marker

                    i += consumed_tokens
                    event_processed = True

                else: # Not enough tokens for a full instrument event (prog, local, marker)
                    # print(f"Warning: Incomplete instrument event start at index {i}. Skipping.")
                    i += 1 # Skip only the program token

            # --- Fallback for Unhandled Tokens ---
            if not event_processed:
                # This case handles tokens that are not TIME_SHIFT and not program tokens
                # print(f"Warning: Skipping unexpected token {token} ({self.token_to_name_map.get(token, 'UNK')}) at index {i}")
                i += 1

        # --- Final Cleanup (Only relevant for use_note_off=True) ---
        if self.use_note_off:
            # Turn off any remaining active notes at the very end
            final_event_time = current_time + self.time_step # A bit after the last event time
            for (prog_token, local_token, pitch), (start_time, start_vel) in active_notes.items():
                 instance_key = (prog_token, local_token)
                 if instance_key in instrument_tracks:
                     track = instrument_tracks[instance_key]
                     # Ensure end time is strictly after start time
                     end_time = max(start_time + MIN_NOTE_DURATION_SECONDS, final_event_time)
                     note_obj = pretty_midi.Note(velocity=start_vel, pitch=pitch, start=start_time, end=end_time)
                     track.notes.append(note_obj)
                     # print(f"Warning: Auto-closing active note {pitch} at end time {end_time:.3f}")

        # Sort notes within each track (important for some players/DAWs)
        for inst in midi_obj.instruments:
            inst.notes.sort(key=lambda n: n.start)
            inst.control_changes.sort(key=lambda cc: cc.time)

        # Save if requested
        if save_path:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                midi_obj.write(save_path)
                print(f"Saved processed MIDI to: {save_path}")
            except Exception as e:
                print(f"Error saving MIDI to {save_path}: {e}")

        return midi_obj

    # --- Tokens to Text Conversion ---
    def tokens_to_text(self, tokens: List[int], vocab_start_offset: int = 0) -> str:
        """Converts a token sequence to a human-readable text representation."""
        text_events = []
        i = 0
        while i < len(tokens):
            token = tokens[i] - vocab_start_offset
            token_name = self.token_to_name_map.get(token, f"UNK({token})")
            current_event_str = ""
            consumed = 1 # Default consumption

            # --- Special Tokens ---
            if token in [self.START, self.END, self.PAD]:
                current_event_str = f"[{token_name}]"
                consumed = 1

            # --- Global Events ---
            elif token == self.TIME_SHIFT:
                if i + 1 < len(tokens):
                    val_token = tokens[i+1] - vocab_start_offset
                    val_name = self.token_to_name_map.get(val_token, f"UNK_VAL({val_token})")
                    current_event_str = f"[{token_name} {val_name}]"
                    consumed = 2
                else: current_event_str = f"[{token_name} (incomplete)]"; consumed = 1

            # --- Instrument-Specific Events ---
            elif self.program_token_range[0] <= token < self.program_token_range[1]:
                prog_name = token_name # e.g., PROG(0)
                if i + 2 < len(tokens): # Need at least prog, local, marker
                    local_token = tokens[i+1] - vocab_start_offset
                    local_name = self.token_to_name_map.get(local_token, f"UNK_LOCAL({local_token})")
                    marker_token = tokens[i+2] - vocab_start_offset
                    marker_name = self.token_to_name_map.get(marker_token, f"UNK_MARKER({marker_token})")
                    consumed = 3 # Base consumption for instr event
                    event_parts = [prog_name, local_name, marker_name]

                    # Check for values based on marker
                    if marker_token == self.VELOCITY:
                        if i + 3 < len(tokens):
                            val_token = tokens[i+3] - vocab_start_offset; val_name = self.token_to_name_map.get(val_token, "UNK_VAL")
                            event_parts.append(val_name); consumed = 4
                        else: event_parts.append("(incomplete_val)")
                    elif marker_token == self.NOTE_ON:
                        if i + 3 < len(tokens):
                            note_val_token = tokens[i+3] - vocab_start_offset; note_val_name = self.token_to_name_map.get(note_val_token, "UNK_NOTE")
                            event_parts.append(note_val_name); consumed = 4
                            # Check for duration if not using NOTE_OFF
                            if not self.use_note_off:
                                if i + 5 < len(tokens):
                                    dur_marker_token = tokens[i+4] - vocab_start_offset; dur_marker_name = self.token_to_name_map.get(dur_marker_token, "UNK_DUR_MARKER")
                                    dur_val_token = tokens[i+5] - vocab_start_offset; dur_val_name = self.token_to_name_map.get(dur_val_token, "UNK_DUR_VAL")
                                    if dur_marker_token == self.NOTE_DURATION: # Check marker is correct
                                        event_parts.append(dur_marker_name); event_parts.append(dur_val_name); consumed = 6
                                    else: event_parts.append("(invalid_dur_marker)"); consumed=4 # Fallback if marker wrong
                                else: event_parts.append("(incomplete_dur)"); consumed=4 # Fallback if seq too short
                        else: event_parts.append("(incomplete_note)")
                    elif marker_token == self.NOTE_OFF: # Only relevant if use_note_off=True, but show anyway
                         if i + 3 < len(tokens):
                             note_val_token = tokens[i+3] - vocab_start_offset; note_val_name = self.token_to_name_map.get(note_val_token, "UNK_NOTE")
                             event_parts.append(note_val_name); consumed = 4
                         else: event_parts.append("(incomplete_note)")
                    elif marker_token == self.PROGRAM_CHANGE:
                         if i + 3 < len(tokens):
                             val_token = tokens[i+3] - vocab_start_offset; val_name = self.token_to_name_map.get(val_token, "UNK_PROGVAL")
                             event_parts.append(val_name); consumed = 4
                         else: event_parts.append("(incomplete_prog_val)")
                    elif marker_token == self.CONTROL_CHANGE:
                        if i + 4 < len(tokens): # Need marker, num, val
                            num_token = tokens[i+3] - vocab_start_offset; num_name = self.token_to_name_map.get(num_token, "UNK_CCNUM")
                            val_token = tokens[i+4] - vocab_start_offset; val_name = self.token_to_name_map.get(val_token, "UNK_CCVAL")
                            event_parts.extend([num_name, val_name]); consumed = 5
                        else: event_parts.append("(incomplete_cc)")
                    # PEDAL_ON/OFF have no values, consumed is already 3

                    current_event_str = f"[{' '.join(event_parts)}]"

                else: # Incomplete instrument event start (missing local/marker)
                    current_event_str = f"[{prog_name} (incomplete)]"
                    consumed = min(len(tokens) - i, 2) # Consume prog and maybe local if present

            # --- Fallback for other unknown tokens ---
            else:
                current_event_str = f"[{token_name}]" # Handles unknown tokens
                consumed = 1

            text_events.append(current_event_str)
            i += consumed

        return "\n".join(text_events)
