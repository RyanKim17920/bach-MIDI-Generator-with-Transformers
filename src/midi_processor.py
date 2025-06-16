import os
import logging
from typing import List, Dict, Tuple, Any, Optional
import pretty_midi
import numpy as np
import math
import traceback

# Use relative imports for modules within the same package (src)
try:
    from .constants import PAD_IDX, START_IDX, END_IDX
except ImportError:
    # Fallback for scenarios where the script might be run directly
    # logging.error("Failed relative import of constants. Ensure running as part of the 'src' package.")
    try:
        from constants import PAD_IDX, START_IDX, END_IDX
    except ImportError:
        # logging.critical("Cannot import PAD_IDX, START_IDX, END_IDX from constants.")
        # Using placeholder values for standalone execution
        PAD_IDX, START_IDX, END_IDX = 0, 1, 2
        # raise

# Get logger for this module
logger = logging.getLogger(__name__)
# Basic config for logging if no handler is set
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Default values used if not provided in config/init
DEFAULT_MAX_LOCAL_INSTRUMENTS = 32
DEFAULT_DRUM_PROGRAM_TOKEN_OFFSET = 128
DEFAULT_VELOCITY_VALUE = 64
DEFAULT_TIME_EPSILON = 0.001
DEFAULT_MIN_NOTE_DURATION_SECONDS = 0.005

class MIDIProcessor:
    """
    Processes MIDI files into token sequences and converts token sequences back
    to MIDI or human-readable text. Uses specific program tokens and local
    instance tokens. Includes START/END tokens.

    Offers two note representation modes:
    1. use_note_off=True: Explicit NOTE_ON and NOTE_OFF events.
    2. use_note_off=False: NOTE_ON followed by NOTE_DURATION and duration value.
    
    Supports selective exclusion of token types via include_* parameters.
    """
    
    def __init__(self,
                 time_step: float = 0.01,
                 velocity_bins: int = 32,
                 max_time_shift_seconds: float = 10.0,
                 max_local_instruments: int = DEFAULT_MAX_LOCAL_INSTRUMENTS,
                 use_note_off: bool = True,
                 max_note_duration_seconds: Optional[float] = None,
                 # Token exclusion parameters
                 include_velocity: bool = True,
                 include_control_changes: bool = True,
                 include_program_changes: bool = True,
                 include_pedal_events: bool = True,
                 # Velocity tokenization mode
                 velocity_before_note: bool = True,
                 # Internal constants
                 drum_program_token_offset: int = DEFAULT_DRUM_PROGRAM_TOKEN_OFFSET,
                 default_velocity: int = DEFAULT_VELOCITY_VALUE,
                 time_epsilon: float = DEFAULT_TIME_EPSILON,
                 min_note_duration_seconds: float = DEFAULT_MIN_NOTE_DURATION_SECONDS,
                 min_sequence_length: int = None,
                 reorder_specialized_tokens: bool = False,
                 verbose: bool = False
                ):

        # Set Logger Level based on verbose flag
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        self.reorder_specialized_tokens = reorder_specialized_tokens

        # Store exclusion parameters
        self.include_velocity = include_velocity
        self.include_control_changes = include_control_changes
        self.include_program_changes = include_program_changes
        self.include_pedal_events = include_pedal_events
        # The `velocity_before_note` mode is only meaningful if velocity is included.
        self.velocity_before_note = velocity_before_note if self.include_velocity else False

        # --- Validate Inputs ---
        if not isinstance(time_step, (float, int)) or time_step <= 0:
            raise ValueError("time_step must be a positive number")
        if not isinstance(velocity_bins, int) or velocity_bins <= 0:
            raise ValueError("velocity_bins must be a positive integer")
        if not isinstance(max_time_shift_seconds, (float, int)) or max_time_shift_seconds <= 0:
            raise ValueError("max_time_shift_seconds must be a positive number")
        if not isinstance(max_local_instruments, int) or max_local_instruments <= 0:
            raise ValueError("max_local_instruments must be a positive integer")
        if not isinstance(drum_program_token_offset, int) or drum_program_token_offset < 0:
             raise ValueError("drum_program_token_offset must be a non-negative integer")

        self.time_step = time_step
        self.velocity_bins = velocity_bins
        self.max_time_shift_seconds = max_time_shift_seconds
        self.max_note_duration_seconds = max_note_duration_seconds if max_note_duration_seconds is not None else max_time_shift_seconds
        self.max_local_instruments = max_local_instruments
        self.use_note_off = use_note_off

        # Store internal constants/parameters
        self.drum_program_token_offset_val = drum_program_token_offset
        self.default_velocity_val = default_velocity
        self.time_epsilon_val = time_epsilon
        self.min_note_duration_seconds_val = min_note_duration_seconds
        self.min_sequence_length = min_sequence_length

        # Calculate sizes of various token groups first
        self.time_shift_steps = self._calculate_steps(self.max_time_shift_seconds, self.time_step)
        self.duration_steps = self._calculate_steps(self.max_note_duration_seconds, self.time_step)
        self.note_range = 128
        self.cc_range = 128
        self.program_range = 128

        # --- Vocabulary Definition ---
        _current_offset = 0
        self.PAD = PAD_IDX; _current_offset = max(_current_offset, self.PAD + 1)
        self.START = START_IDX; _current_offset = max(_current_offset, self.START + 1)
        self.END = END_IDX; _current_offset = max(_current_offset, self.END + 1)
        _current_offset = max(PAD_IDX, START_IDX, END_IDX) + 1

        self.program_token_offset = _current_offset
        self.num_program_tokens = 128 + 1 # 128 standard + 1 drum meta
        
        # Ensure drum token offset is valid
        if self.drum_program_token_offset_val >= self.num_program_tokens:
             logger.warning(f"drum_program_token_offset ({self.drum_program_token_offset_val}) is too large. Setting to {self.num_program_tokens - 1}")
             self.drum_program_token_idx = self.program_token_offset + self.num_program_tokens - 1
        else:
             self.drum_program_token_idx = self.program_token_offset + self.drum_program_token_offset_val

        self.program_token_range = (self.program_token_offset, self.program_token_offset + self.num_program_tokens)
        _current_offset += self.num_program_tokens

        self.local_instance_offset = _current_offset
        self.local_instance_range = (self.local_instance_offset, self.local_instance_offset + self.max_local_instruments)
        _current_offset += self.max_local_instruments

        # Event Markers - conditionally defined
        self.NOTE_ON = _current_offset; _current_offset += 1
        self.NOTE_OFF = _current_offset if self.use_note_off else None
        if self.NOTE_OFF is not None: _current_offset += 1
        
        self.TIME_SHIFT = _current_offset; _current_offset += 1
        
        # VELOCITY marker is only needed in the "old" mode.
        self.VELOCITY = _current_offset if self.include_velocity and not self.velocity_before_note else None
        if self.VELOCITY is not None: _current_offset += 1
        
        self.CONTROL_CHANGE = _current_offset if self.include_control_changes else None
        if self.CONTROL_CHANGE is not None: _current_offset += 1
        
        self.PROGRAM_CHANGE = _current_offset if self.include_program_changes else None
        if self.PROGRAM_CHANGE is not None: _current_offset += 1
        
        self.PEDAL_ON = _current_offset if self.include_pedal_events else None
        if self.PEDAL_ON is not None: _current_offset += 1
        self.PEDAL_OFF = _current_offset if self.include_pedal_events else None
        if self.PEDAL_OFF is not None: _current_offset += 1

        self.NOTE_DURATION = _current_offset if not self.use_note_off else None
        if self.NOTE_DURATION is not None: _current_offset += 1

        # This marks the end of "marker" tokens.
        self.categorical_vocab_size = _current_offset

        # This block defines the start of value ranges.
        # It respects the reordering flag.
        def define_value_ranges(reorder):
            nonlocal _current_offset
            value_offsets = {}
            if reorder:
                # CC and Program values come first
                if self.include_control_changes:
                    value_offsets['cc_number'] = _current_offset; _current_offset += self.cc_range
                    value_offsets['cc_value'] = _current_offset; _current_offset += self.cc_range
                if self.include_program_changes:
                    value_offsets['program_value'] = _current_offset; _current_offset += self.program_range
                
                # Specialized values come last
                value_offsets['note_value'] = _current_offset; _current_offset += self.note_range
                value_offsets['time_shift_value'] = _current_offset; _current_offset += self.time_shift_steps
                if self.include_velocity:
                    value_offsets['velocity_value'] = _current_offset; _current_offset += self.velocity_bins
                if not self.use_note_off:
                    value_offsets['note_duration_value'] = _current_offset; _current_offset += self.duration_steps
            else:
                # Original order
                value_offsets['note_value'] = _current_offset; _current_offset += self.note_range
                value_offsets['time_shift_value'] = _current_offset; _current_offset += self.time_shift_steps
                if self.include_velocity:
                    value_offsets['velocity_value'] = _current_offset; _current_offset += self.velocity_bins
                if self.include_control_changes:
                    value_offsets['cc_number'] = _current_offset; _current_offset += self.cc_range
                    value_offsets['cc_value'] = _current_offset; _current_offset += self.cc_range
                if self.include_program_changes:
                    value_offsets['program_value'] = _current_offset; _current_offset += self.program_range
                if not self.use_note_off:
                    value_offsets['note_duration_value'] = _current_offset; _current_offset += self.duration_steps
            return value_offsets

        value_offsets = define_value_ranges(self.reorder_specialized_tokens)
        self.note_value_offset = value_offsets.get('note_value')
        self.time_shift_value_offset = value_offsets.get('time_shift_value')
        self.velocity_value_offset = value_offsets.get('velocity_value')
        self.cc_number_offset = value_offsets.get('cc_number')
        self.cc_value_offset = value_offsets.get('cc_value')
        self.program_value_offset = value_offsets.get('program_value')
        self.note_duration_value_offset = value_offsets.get('note_duration_value')
            
        self.vocab_size = _current_offset

        # Build Reverse Mapping for Text Conversion
        self.token_to_name_map = {}
        self._build_reverse_map()

        logger.info("MIDI Processor Initialized:")
        logger.info(f"  - Categorical Vocab Size: {self.categorical_vocab_size}")
        logger.info(f"  - Vocab Size: {self.vocab_size}")
        logger.info(f"  - Tokenization mode: use_note_off={self.use_note_off}")
        logger.info(f"  - Token Inclusion: Velocity={self.include_velocity}, CC={self.include_control_changes}, Program={self.include_program_changes}, Pedal={self.include_pedal_events}")
        if self.include_velocity:
            logger.info(f"  - Velocity Mode: {'VEL_VAL before NOTE_ON' if self.velocity_before_note else 'Separate VELOCITY event'}")
        
    def _calculate_steps(self, max_seconds: float, step_size: float) -> int:
        """Calculates the number of steps for time or duration."""
        if step_size <= 0:
            logger.error("Cannot calculate steps: step_size must be positive.")
            return 1
        return max(1, int(math.ceil(max_seconds / step_size)) + 1)

    def _build_reverse_map(self):
        """Builds the token index to human-readable name map."""
        self.token_to_name_map = {
            self.PAD: "PAD", self.START: "START", self.END: "END",
            self.NOTE_ON: "NOTE_ON", self.TIME_SHIFT: "TIME_SHIFT",
        }
        # Conditionally add event markers that might be None
        for marker_name in ['NOTE_OFF', 'VELOCITY', 'CONTROL_CHANGE', 'PROGRAM_CHANGE', 'PEDAL_ON', 'PEDAL_OFF', 'NOTE_DURATION']:
            marker_val = getattr(self, marker_name)
            if marker_val is not None:
                self.token_to_name_map[marker_val] = marker_name
            
        # Instrument Programs
        for i in range(self.num_program_tokens):
            idx = self.program_token_offset + i
            if idx == self.drum_program_token_idx:
                self.token_to_name_map[idx] = "PROG(DRUMS)"
            else:
                # This logic correctly maps program numbers around the drum token
                prog_num = i if i < self.drum_program_token_offset_val else i - 1
                self.token_to_name_map[idx] = f"PROG({prog_num})"

        # Local Instances
        for i in range(self.max_local_instruments):
            self.token_to_name_map[self.local_instance_offset + i] = f"LOCAL_INST({i})"
        
        # Value ranges
        if self.note_value_offset is not None:
            for i in range(self.note_range): self.token_to_name_map[self.note_value_offset + i] = f"NOTE_VAL({i})"
        if self.time_shift_value_offset is not None:
            for i in range(self.time_shift_steps): self.token_to_name_map[self.time_shift_value_offset + i] = f"TIME_SHIFT_VAL({i})"
        if self.velocity_value_offset is not None:
            for i in range(self.velocity_bins): self.token_to_name_map[self.velocity_value_offset + i] = f"VEL_VAL({i})"
        if self.cc_number_offset is not None:
            for i in range(self.cc_range): self.token_to_name_map[self.cc_number_offset + i] = f"CC_NUM({i})"
        if self.cc_value_offset is not None:
            for i in range(self.cc_range): self.token_to_name_map[self.cc_value_offset + i] = f"CC_VAL({i})"
        if self.program_value_offset is not None:
            for i in range(self.program_range): self.token_to_name_map[self.program_value_offset + i] = f"PROG_VAL({i})"
        if self.note_duration_value_offset is not None:
            for i in range(self.duration_steps): self.token_to_name_map[self.note_duration_value_offset + i] = f"DUR_VAL({i})"

    def _get_program_token(self, program_number: int, is_drum: bool) -> Optional[int]:
        if is_drum:
            return self.drum_program_token_idx
        elif 0 <= program_number <= 127:
            # Shift the index if it's at or after the drum token's position in the range
            if program_number >= self.drum_program_token_offset_val:
                return self.program_token_offset + program_number + 1
            else:
                return self.program_token_offset + program_number
        logger.warning(f"Invalid program number encountered: {program_number}")
        return None
            
    def _get_local_instance_token(self, instrument_index: int) -> int:
        return self.local_instance_offset + (instrument_index % self.max_local_instruments)
        
    def _quantize_velocity(self, velocity: int) -> int:
        return min(int(max(0, min(127, velocity)) * self.velocity_bins / 128), self.velocity_bins - 1)
        
    def _unquantize_velocity(self, velocity_bin: int) -> int:
        return min(127, max(0, int((max(0, min(self.velocity_bins - 1, velocity_bin)) + 0.5) * 128.0 / self.velocity_bins)))
        
    def _quantize_time_or_duration(self, time_diff: float, max_steps: int) -> int:
        if time_diff < self.time_step / 2.0: return 0
        return min(int(round(time_diff / self.time_step)), max_steps - 1)

    def _unquantize_steps(self, steps: int) -> float:
        return max(0, steps) * self.time_step

    def _event_sort_priority(self, event_type: str) -> int:
        """Assigns priority for sorting events at the same time. Lower numbers are earlier."""
        return {
            'program_change_marker': 0, 'control_change': 1, 'pedal_off': 2, 'note_off': 3,
            'velocity': 4, 'note_on': 5, 'note_on_with_duration': 5, 'pedal_on': 6
        }.get(event_type, 10)
    
    # --- Metadata and Event Extraction ---

    def _extract_sorted_events(self, midi_data: pretty_midi.PrettyMIDI, instrument_token_map: Dict[int, Tuple[int, int]]) -> List[Dict[str, Any]]:
        events = []
        for original_instrument_idx, instrument in enumerate(midi_data.instruments):
            token_pair = instrument_token_map.get(original_instrument_idx)
            if token_pair is None: continue

            program_token_idx, local_instance_token_idx = token_pair
            common_meta = {'program_token_idx': program_token_idx, 'local_instance_token_idx': local_instance_token_idx}

            # Program Change Marker
            if self.include_program_changes and not instrument.is_drum:
                first_event_time = min([n.start for n in instrument.notes] + [c.time for c in instrument.control_changes], default=None)
                if first_event_time is not None:
                    events.append({
                        'type': 'program_change_marker', 'time': max(0.0, first_event_time - self.time_epsilon_val),
                        'program': instrument.program, **common_meta
                    })
            
            # Note Events
            for note in instrument.notes:
                if note.end - note.start < self.min_note_duration_seconds_val: continue
                if self.use_note_off:
                    events.append({'type': 'note_on', 'time': note.start, 'note': note.pitch, 'velocity': note.velocity, **common_meta})
                    events.append({'type': 'note_off', 'time': note.end, 'note': note.pitch, **common_meta})
                else:
                    duration_steps = self._quantize_time_or_duration(note.end - note.start, self.duration_steps)
                    events.append({'type': 'note_on_with_duration', 'time': note.start, 'note': note.pitch, 'velocity': note.velocity, 'duration_steps': duration_steps, **common_meta})

            # Control Change Events
            for control in instrument.control_changes:
                if control.number == 64 and self.include_pedal_events: # Sustain Pedal
                    events.append({'type': 'pedal_on' if control.value >= 64 else 'pedal_off', 'time': control.time, **common_meta})
                elif control.number != 64 and self.include_control_changes:
                    events.append({'type': 'control_change', 'time': control.time, 'control_number': control.number, 'control_value': control.value, **common_meta})

        events.sort(key=lambda x: (x['time'], self._event_sort_priority(x['type'])))

        timed_events = []
        last_time = 0.0
        last_velocity_bin = {}
        for event in events:
            event_time = max(last_time, event['time'])
            time_diff = event_time - last_time
            if time_diff > self.time_epsilon_val / 2:
                time_shift_steps = self._quantize_time_or_duration(time_diff, self.time_shift_steps)
                if time_shift_steps > 0:
                    timed_events.append({'type': 'time_shift', 'steps': time_shift_steps})
                    last_time += self._unquantize_steps(time_shift_steps)
            
            # Add velocity event only if needed (not using velocity_before_note)
            if event['type'] in ('note_on', 'note_on_with_duration') and self.include_velocity and not self.velocity_before_note:
                instance_key = (event['program_token_idx'], event['local_instance_token_idx'])
                velocity_bin = self._quantize_velocity(event['velocity'])
                if velocity_bin != last_velocity_bin.get(instance_key, -1):
                    timed_events.append({'type': 'velocity', 'velocity_bin': velocity_bin, **event})
                    last_velocity_bin[instance_key] = velocity_bin

            timed_events.append(event)
        return timed_events

    def _tokenize_events(self, events: List[Dict[str, Any]]) -> List[int]:
        tokens = []
        for event in events:
            event_type = event['type']
            prog_token = event.get('program_token_idx')
            local_token = event.get('local_instance_token_idx')
            
            event_tokens = []
            # Global event
            if event_type == 'time_shift':
                event_tokens.extend([self.TIME_SHIFT, self.time_shift_value_offset + event['steps']])
            # Instrument-specific events
            elif prog_token is not None and local_token is not None:
                inst_prefix = [prog_token, local_token]
                
                if event_type == 'note_on' or event_type == 'note_on_with_duration':
                    note_tokens = []
                    if self.velocity_before_note:
                        vel_bin = self._quantize_velocity(event['velocity'])
                        note_tokens.append(self.velocity_value_offset + vel_bin)
                    
                    note_tokens.extend([self.NOTE_ON, self.note_value_offset + event['note']])

                    if event_type == 'note_on_with_duration':
                         note_tokens.extend([self.NOTE_DURATION, self.note_duration_value_offset + event['duration_steps']])
                    event_tokens = inst_prefix + note_tokens

                elif event_type == 'note_off':
                    if self.NOTE_OFF: event_tokens = inst_prefix + [self.NOTE_OFF, self.note_value_offset + event['note']]
                elif event_type == 'velocity':
                    if self.VELOCITY: event_tokens = inst_prefix + [self.VELOCITY, self.velocity_value_offset + event['velocity_bin']]
                elif event_type == 'program_change_marker':
                    if self.PROGRAM_CHANGE: event_tokens = inst_prefix + [self.PROGRAM_CHANGE, self.program_value_offset + event['program']]
                elif event_type == 'pedal_on':
                    if self.PEDAL_ON: event_tokens = inst_prefix + [self.PEDAL_ON]
                elif event_type == 'pedal_off':
                    if self.PEDAL_OFF: event_tokens = inst_prefix + [self.PEDAL_OFF]
                elif event_type == 'control_change':
                    if self.CONTROL_CHANGE: event_tokens = inst_prefix + [self.CONTROL_CHANGE, self.cc_number_offset + event['control_number'], self.cc_value_offset + event['control_value']]
            
            tokens.extend(event_tokens)
        return tokens

    def process_midi_file(self, midi_path: str) -> Dict[str, Any] | None:
        filename = os.path.basename(midi_path)
        logger.info(f"Processing MIDI file: {filename}")
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            logger.error(f"Error loading MIDI {filename}: {e}")
            return None

        instrument_token_map, instrument_metadata_map = {}, {}
        valid_instrument_count = 0
        for i, inst in enumerate(midi_data.instruments):
            if not inst.notes and not inst.control_changes: continue # Skip empty instruments
            prog_token = self._get_program_token(inst.program, inst.is_drum)
            if prog_token and valid_instrument_count < self.max_local_instruments:
                local_token = self._get_local_instance_token(valid_instrument_count)
                instrument_token_map[i] = (prog_token, local_token)
                instrument_metadata_map[f"original_inst_{i}"] = {"program": inst.program, "is_drum": inst.is_drum, "name": inst.name}
                valid_instrument_count += 1
        
        if not instrument_token_map:
            logger.warning(f"No valid instruments in {filename}. Skipping.")
            return None

        events = self._extract_sorted_events(midi_data, instrument_token_map)
        core_tokens = self._tokenize_events(events)
        
        if self.min_sequence_length and len(core_tokens) < self.min_sequence_length:
            logger.warning(f"Skipping {filename}: token length {len(core_tokens)} < min {self.min_sequence_length}")
            return None
        
        final_tokens = [self.START] + core_tokens + [self.END]
        metadata = {'filename': filename, 'instrument_mapping': instrument_metadata_map}
        return {'metadata': metadata, 'tokens': np.array(final_tokens, dtype=np.int32)}

    # ======================================================================
    # FIXED: tokens_to_midi (Re-implemented with robust, stateful parsing)
    # ======================================================================
    def tokens_to_midi(self, tokens: List[int], vocab_start_offset: int = 0, default_tempo: float = 120.0, save_path: Optional[str] = None) -> Optional[pretty_midi.PrettyMIDI]:
        """Converts a sequence of tokens back into a pretty_midi.PrettyMIDI object."""
        logger.info(f"Converting {len(tokens)} tokens to MIDI...")
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=default_tempo)
        instrument_tracks: Dict[Tuple[int, int], pretty_midi.Instrument] = {}
        active_notes: Dict[Tuple[int, int, int], Tuple[float, int]] = {}
        current_velocities: Dict[Tuple[int, int], int] = {}
        current_time = 0.0

        i = 0
        while i < len(tokens):
            token = tokens[i] - vocab_start_offset
            if token in [self.START, self.END, self.PAD]:
                i += 1
                continue

            # --- Global Event: TIME_SHIFT ---
            if token == self.TIME_SHIFT:
                if i + 1 < len(tokens):
                    steps_token = tokens[i+1] - vocab_start_offset
                    if self.time_shift_value_offset <= steps_token < self.time_shift_value_offset + self.time_shift_steps:
                        steps = steps_token - self.time_shift_value_offset
                        current_time += self._unquantize_steps(steps)
                    i += 2
                else: i += 1 # Incomplete
                continue

            # --- Instrument Event ---
            # Must start with a Program Token
            if not (self.program_token_range[0] <= token < self.program_token_range[1]):
                logger.warning(f"Expected PROG token, but found {self.token_to_name_map.get(token, token)} at index {i}. Skipping.")
                i += 1
                continue

            prog_token = token
            # Must have a Local Instance token next
            if i + 1 >= len(tokens) or not (self.local_instance_range[0] <= tokens[i+1] - vocab_start_offset < self.local_instance_range[1]):
                logger.warning(f"Incomplete instrument event at index {i}. Skipping.")
                i += 1
                continue
            local_token = tokens[i+1] - vocab_start_offset
            
            instance_key = (prog_token, local_token)
            
            # Get or create PrettyMIDI Instrument
            if instance_key not in instrument_tracks:
                is_drum = (prog_token == self.drum_program_token_idx)
                if is_drum:
                    program_num = 0
                else:
                    prog_idx = prog_token - self.program_token_offset
                    program_num = prog_idx if prog_idx < self.drum_program_token_offset_val else prog_idx - 1
                
                track_name = f"{'Drum' if is_drum else f'Inst {program_num}'}_{local_token - self.local_instance_offset}"
                instrument_tracks[instance_key] = pretty_midi.Instrument(program=program_num, is_drum=is_drum, name=track_name)
                midi_obj.instruments.append(instrument_tracks[instance_key])
                current_velocities[instance_key] = self.default_velocity_val

            track = instrument_tracks[instance_key]
            
            # --- Parse Event Body ---
            ptr = i + 2 # Pointer to the start of the event body
            velocity_for_this_note = current_velocities.get(instance_key, self.default_velocity_val)

            # Check for `velocity_before_note` mode
            if self.velocity_before_note:
                if ptr < len(tokens):
                    next_token = tokens[ptr] - vocab_start_offset
                    if self.velocity_value_offset is not None and self.velocity_value_offset <= next_token < self.velocity_value_offset + self.velocity_bins:
                        vel_bin = next_token - self.velocity_value_offset
                        velocity_for_this_note = self._unquantize_velocity(vel_bin)
                        current_velocities[instance_key] = velocity_for_this_note
                        ptr += 1 # Consume velocity value

            if ptr >= len(tokens):
                i = ptr; continue # End of sequence
            
            marker = tokens[ptr] - vocab_start_offset
            ptr += 1

            # --- Handle Event by Marker ---
            if marker == self.NOTE_ON:
                if ptr < len(tokens):
                    note_val = tokens[ptr] - vocab_start_offset - self.note_value_offset
                    ptr += 1
                    if 0 <= note_val <= 127:
                        if self.use_note_off:
                            note_key = (*instance_key, note_val)
                            # End previous note of same pitch if overlapping
                            if note_key in active_notes:
                                start_time, start_vel = active_notes.pop(note_key)
                                end_time = max(start_time + self.min_note_duration_seconds_val, current_time)
                                track.notes.append(pretty_midi.Note(velocity=start_vel, pitch=note_val, start=start_time, end=end_time))
                            active_notes[note_key] = (current_time, velocity_for_this_note)
                        else: # Use NOTE_DURATION
                            if ptr + 1 < len(tokens) and (tokens[ptr] - vocab_start_offset) == self.NOTE_DURATION:
                                dur_val = tokens[ptr+1] - vocab_start_offset - self.note_duration_value_offset
                                ptr += 2
                                duration = max(self._unquantize_steps(dur_val), self.min_note_duration_seconds_val)
                                track.notes.append(pretty_midi.Note(velocity=velocity_for_this_note, pitch=note_val, start=current_time, end=current_time + duration))
            
            elif marker == self.NOTE_OFF:
                if ptr < len(tokens):
                    note_val = tokens[ptr] - vocab_start_offset - self.note_value_offset
                    ptr += 1
                    note_key = (*instance_key, note_val)
                    if note_key in active_notes:
                        start_time, start_vel = active_notes.pop(note_key)
                        end_time = max(start_time + self.min_note_duration_seconds_val, current_time)
                        track.notes.append(pretty_midi.Note(velocity=start_vel, pitch=note_val, start=start_time, end=end_time))

            elif marker == self.VELOCITY: # Only if not velocity_before_note
                if ptr < len(tokens):
                    vel_bin = tokens[ptr] - vocab_start_offset - self.velocity_value_offset
                    ptr += 1
                    current_velocities[instance_key] = self._unquantize_velocity(vel_bin)

            elif marker == self.PEDAL_ON:
                track.control_changes.append(pretty_midi.ControlChange(64, 100, current_time))
            elif marker == self.PEDAL_OFF:
                track.control_changes.append(pretty_midi.ControlChange(64, 0, current_time))
            
            elif marker == self.PROGRAM_CHANGE:
                 if ptr < len(tokens) and not track.is_drum:
                    track.program = tokens[ptr] - vocab_start_offset - self.program_value_offset
                    ptr += 1

            elif marker == self.CONTROL_CHANGE:
                if ptr + 1 < len(tokens):
                    cc_num = tokens[ptr] - vocab_start_offset - self.cc_number_offset
                    cc_val = tokens[ptr+1] - vocab_start_offset - self.cc_value_offset
                    ptr += 2
                    if cc_num != 64: # Avoid redundant pedal events
                        track.control_changes.append(pretty_midi.ControlChange(cc_num, cc_val, current_time))

            i = ptr # Move main counter to end of processed event
        
        # Clean up any remaining active notes
        for note_key, (start_time, start_vel) in active_notes.items():
            prog, local, pitch = note_key
            track = instrument_tracks[(prog, local)]
            end_time = max(start_time + self.min_note_duration_seconds_val, current_time)
            track.notes.append(pretty_midi.Note(velocity=start_vel, pitch=pitch, start=start_time, end=end_time))
        
        if save_path:
            try:
                # Only create directory if save_path contains a directory part
                dir_path = os.path.dirname(save_path)
                if dir_path:  # Only create directory if dir_path is not empty
                    os.makedirs(dir_path, exist_ok=True)
                midi_obj.write(save_path)
                logger.info(f"Successfully saved MIDI to: {save_path}")
            except Exception as e:
                logger.error(f"Error saving MIDI to {save_path}: {e}")

        return midi_obj

    # ======================================================================
    # FIXED: tokens_to_text (Re-implemented with event grouping)
    # ======================================================================
    def tokens_to_text(self, tokens: List[int], vocab_start_offset: int = 0) -> str:
        """Converts a token sequence to a human-readable text representation."""
        text_events = []
        i = 0
        while i < len(tokens):
            token = tokens[i] - vocab_start_offset
            token_name = self.token_to_name_map.get(token, f"UNK({token})")
            consumed = 1
            
            if token in [self.START, self.END, self.PAD]:
                text_events.append(f"[{token_name}]")

            elif token == self.TIME_SHIFT:
                if i + 1 < len(tokens):
                    val_token = tokens[i+1] - vocab_start_offset
                    val_name = self.token_to_name_map.get(val_token, f"UNK_VAL({val_token})")
                    text_events.append(f"[{token_name} {val_name}]")
                    consumed = 2
                else: text_events.append(f"[{token_name} (incomplete)]")

            elif self.program_token_range[0] <= token < self.program_token_range[1]:
                event_parts = [token_name]
                ptr = i + 1
                
                # Consume local instance
                if ptr < len(tokens):
                    local_name = self.token_to_name_map.get(tokens[ptr] - vocab_start_offset, "UNK_LOCAL")
                    event_parts.append(local_name)
                    ptr += 1
                
                # Consume event body
                if ptr < len(tokens):
                    # Velocity-first mode
                    if self.velocity_before_note:
                        vel_cand = tokens[ptr] - vocab_start_offset
                        if self.velocity_value_offset is not None and self.velocity_value_offset <= vel_cand < self.velocity_value_offset + self.velocity_bins:
                            event_parts.append(self.token_to_name_map.get(vel_cand, "UNK_VEL"))
                            ptr += 1
                    
                    if ptr < len(tokens):
                        marker_cand = tokens[ptr] - vocab_start_offset
                        marker_name = self.token_to_name_map.get(marker_cand, "UNK_MARKER")
                        event_parts.append(marker_name)
                        ptr += 1

                        # Consume values based on marker
                        # Note: This is a simplified representation, a full parser would be more complex
                        if marker_cand in [self.NOTE_ON, self.NOTE_OFF, self.VELOCITY, self.PROGRAM_CHANGE]:
                            if ptr < len(tokens):
                                event_parts.append(self.token_to_name_map.get(tokens[ptr] - vocab_start_offset, "UNK_VAL"))
                                ptr += 1
                        
                        if marker_cand == self.NOTE_ON and not self.use_note_off:
                             if ptr + 1 < len(tokens):
                                event_parts.append(self.token_to_name_map.get(tokens[ptr] - vocab_start_offset, "UNK_DUR_MARKER"))
                                event_parts.append(self.token_to_name_map.get(tokens[ptr+1] - vocab_start_offset, "UNK_DUR_VAL"))
                                ptr += 2
                        
                        if marker_cand == self.CONTROL_CHANGE:
                            if ptr + 1 < len(tokens):
                                event_parts.append(self.token_to_name_map.get(tokens[ptr] - vocab_start_offset, "UNK_CC_NUM"))
                                event_parts.append(self.token_to_name_map.get(tokens[ptr+1] - vocab_start_offset, "UNK_CC_VAL"))
                                ptr += 2
                
                text_events.append(f"[{' '.join(event_parts)}]")
                consumed = ptr - i
            
            else: # Fallback for unexpected tokens
                text_events.append(f"[{token_name}]")
            
            i += consumed
        return "\n".join(text_events)