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
    logging.error("Failed relative import of constants. Ensure running as part of the 'src' package.")
    try:
        from constants import PAD_IDX, START_IDX, END_IDX
    except ImportError:
        logging.critical("Cannot import PAD_IDX, START_IDX, END_IDX from constants.")
        raise

# Get logger for this module
logger = logging.getLogger(__name__)

# Default values used if not provided in config/init
DEFAULT_MAX_LOCAL_INSTRUMENTS = 16
DEFAULT_DRUM_PROGRAM_TOKEN_OFFSET = 12832
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
    """

    def __init__(self,
                 time_step: float = 0.01,
                 velocity_bins: int = 32,
                 max_time_shift_seconds: float = 10.0,
                 max_local_instruments: int = DEFAULT_MAX_LOCAL_INSTRUMENTS,
                 use_note_off: bool = True,
                 max_note_duration_seconds: Optional[float] = None,
                 # Internal constants can also be passed if needed, but defaults are fine
                 drum_program_token_offset: int = DEFAULT_DRUM_PROGRAM_TOKEN_OFFSET,
                 default_velocity: int = DEFAULT_VELOCITY_VALUE,
                 time_epsilon: float = DEFAULT_TIME_EPSILON,
                 min_note_duration_seconds: float = DEFAULT_MIN_NOTE_DURATION_SECONDS,
                 min_sequence_length: int = None,
                ):

        # --- Validate Inputs ---
        if not isinstance(time_step, (float, int)) or time_step <= 0:
            raise ValueError("time_step must be a positive number")
        if not isinstance(velocity_bins, int) or velocity_bins <= 0:
            raise ValueError("velocity_bins must be a positive integer")
        if not isinstance(max_time_shift_seconds, (float, int)) or max_time_shift_seconds <= 0:
            raise ValueError("max_time_shift_seconds must be a positive number")
        if not isinstance(max_local_instruments, int) or max_local_instruments <= 0:
            raise ValueError("max_local_instruments must be a positive integer")
        # ... add validation for other params if necessary ...

        self.time_step = time_step
        self.velocity_bins = velocity_bins
        self.max_time_shift_seconds = max_time_shift_seconds
        # If max_note_duration not specified, tie it to max_time_shift
        self.max_note_duration_seconds = max_note_duration_seconds if max_note_duration_seconds is not None else max_time_shift_seconds
        self.max_local_instruments = max_local_instruments
        self.use_note_off = use_note_off # Store the mode

        # Store internal constants/parameters
        self.drum_program_token_offset_val = drum_program_token_offset
        self.default_velocity_val = default_velocity
        self.time_epsilon_val = time_epsilon
        self.min_note_duration_seconds_val = min_note_duration_seconds

        self.min_sequence_length = min_sequence_length 
        # --- Vocabulary Definition using constants ---
        _current_offset = 0
        self.PAD = PAD_IDX; _current_offset = max(_current_offset, self.PAD + 1)
        self.START = START_IDX; _current_offset = max(_current_offset, self.START + 1)
        self.END = END_IDX; _current_offset = max(_current_offset, self.END + 1)
        # Ensure offset starts after the highest special index
        _current_offset = max(PAD_IDX, START_IDX, END_IDX) + 1

        self.program_token_offset = _current_offset
        self.num_program_tokens = 128 + 1 # 128 standard + 1 drum meta
        self.drum_program_token_idx = self.program_token_offset + self.drum_program_token_offset_val
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

        # Log configuration using logger
        logger.info("MIDI Processor Initialized:")
        logger.info(f"  - use_note_off: {self.use_note_off}")
        logger.info(f"  - time_step: {self.time_step}s")
        logger.info(f"  - velocity_bins: {self.velocity_bins}")
        logger.info(f"  - max_time_shift: {self.max_time_shift_seconds}s ({self.time_shift_steps} steps)")
        if not self.use_note_off:
            logger.info(f"  - max_note_duration: {self.max_note_duration_seconds}s ({self.duration_steps} steps)")
        logger.info(f"  - max_local_instruments: {self.max_local_instruments}")
        logger.info(f"  - Vocab Size: {self.vocab_size}")

    def _calculate_steps(self, max_seconds: float, step_size: float) -> int:
        """Calculates the number of steps for time or duration."""
        if step_size <= 0:
            logger.error("Cannot calculate steps: step_size must be positive.")
            return 1 # Avoid division by zero, return minimum steps
        # Add 1 because the range includes 0 steps
        return max(1, int(math.ceil(max_seconds / step_size)) + 1)

    def _build_reverse_map(self):
        """Builds the token index to human-readable name map."""
        # (Implementation remains the same as before, using self attributes)
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
    # (Implementation remains the same, using self attributes like self.velocity_bins)
    def _get_program_token(self, program_number: int, is_drum: bool) -> Optional[int]:
        if is_drum: return self.drum_program_token_idx
        elif 0 <= program_number <= 127: return self.program_token_offset + program_number
        else:
            logger.warning(f"Invalid program number encountered: {program_number}")
            return None
    def _get_local_instance_token(self, instrument_index: int) -> int:
        local_idx = instrument_index % self.max_local_instruments
        return self.local_instance_offset + local_idx
    def _quantize_velocity(self, velocity: int) -> int:
        # Ensure velocity is within MIDI range 0-127 before quantizing
        velocity = max(0, min(127, velocity))
        return min(int(velocity * self.velocity_bins / 128), self.velocity_bins - 1)
    def _unquantize_velocity(self, velocity_bin: int) -> int:
        # Ensure bin is within valid range
        velocity_bin = max(0, min(self.velocity_bins - 1, velocity_bin))
        # Center the value within the bin
        return min(127, max(0, int((velocity_bin + 0.5) * 128.0 / self.velocity_bins)))
    def _quantize_time_or_duration(self, time_diff: float, max_steps: int) -> int:
        """Quantizes time difference or duration into steps."""
        if time_diff < self.time_step / 2.0: return 0
        # Use round for potentially better accuracy near step boundaries
        steps = int(round(time_diff / self.time_step))
        # Clamp to the maximum allowed steps (max_steps - 1 is the highest index)
        return min(steps, max_steps - 1)

    def _unquantize_steps(self, steps: int) -> float:
        """Converts steps back to seconds."""
        # Ensure steps are non-negative
        steps = max(0, steps)
        return steps * self.time_step

    # --- Metadata Extraction Helpers ---
    # (Implementations assumed unchanged, but could add logging)
    def _extract_tempo_changes(self, midi_data: pretty_midi.PrettyMIDI) -> List[Dict[str, float]]:
        changes = []
        try:
            tempo_times, tempo_values = midi_data.get_tempo_changes()
            for t, q in zip(tempo_times, tempo_values):
                if t >= 0 and q > 0: changes.append({'time': float(t), 'tempo': float(q)})
            if not changes or changes[0]['time'] > self.time_epsilon_val: # Use epsilon
                 initial_tempo = 120.0
                 try: # Estimate tempo can sometimes fail
                     est_tempo = midi_data.estimate_tempo()
                     if est_tempo > 0: initial_tempo = est_tempo
                 except Exception: pass # Ignore estimation error, use default
                 if not changes and len(tempo_values) > 0 and tempo_values[0] > 0:
                     initial_tempo = float(tempo_values[0]) # Use first if available
                 changes.insert(0, {'time': 0.0, 'tempo': initial_tempo})
                 logger.debug("Inserted initial tempo change.")
        except Exception as e:
            logger.warning(f"Could not extract tempo changes: {e}. Using default 120bpm.")
            changes = [{'time': 0.0, 'tempo': 120.0}]
        return sorted(changes, key=lambda x: x['time'])

    def _extract_time_signatures(self, midi_data: pretty_midi.PrettyMIDI) -> List[Dict[str, Any]]:
        sigs = []
        try:
            sigs = [{'time': float(ts.time), 'numerator': int(ts.numerator), 'denominator': int(ts.denominator)}
                    for ts in midi_data.time_signature_changes if ts.time >= 0 and ts.numerator > 0 and ts.denominator > 0]
            if not sigs or sigs[0]['time'] > self.time_epsilon_val: # Use epsilon
                sigs.insert(0, {'time': 0.0, 'numerator': 4, 'denominator': 4})
                logger.debug("Inserted initial 4/4 time signature.")
        except Exception as e:
            logger.warning(f"Could not extract time signatures: {e}. Using default 4/4.")
            sigs = [{'time': 0.0, 'numerator': 4, 'denominator': 4}]
        # Ensure uniqueness by time
        unique_sigs = []; last_t = -1.0
        for sig in sorted(sigs, key=lambda x: x['time']):
            if sig['time'] > last_t + self.time_epsilon_val: # Use epsilon
                unique_sigs.append(sig); last_t = sig['time']
        return unique_sigs

    def _extract_key_signatures(self, midi_data: pretty_midi.PrettyMIDI) -> List[Dict[str, Any]]:
        keys = []
        try:
            keys = [{'time': float(ks.time), 'key_number': int(ks.key_number)}
                    for ks in midi_data.key_signature_changes if ks.time >= 0]
            if not keys or keys[0]['time'] > self.time_epsilon_val: # Use epsilon
                keys.insert(0, {'time': 0.0, 'key_number': 0}) # Default C Major
                logger.debug("Inserted initial C Major key signature.")
        except Exception as e:
            logger.warning(f"Could not extract key signatures: {e}. Using default C Major.")
            keys = [{'time': 0.0, 'key_number': 0}]
        # Ensure uniqueness by time
        unique_keys = []; last_t = -1.0
        for key in sorted(keys, key=lambda x: x['time']):
             if key['time'] > last_t + self.time_epsilon_val: # Use epsilon
                 unique_keys.append(key); last_t = key['time']
        return unique_keys

    def _extract_sorted_events(self, midi_data: pretty_midi.PrettyMIDI, instrument_token_map: Dict[int, Tuple[int, int]]) -> List[Dict[str, Any]]:
         """Extracts and sorts all MIDI events, handling note representation based on self.use_note_off."""
         events = []
         last_velocity_bin = {} # (prog_token, local_token) -> last_vel_bin

         for original_instrument_idx, instrument in enumerate(midi_data.instruments):
            token_pair = instrument_token_map.get(original_instrument_idx)
            if token_pair is None:
                logger.debug(f"Skipping instrument index {original_instrument_idx} ('{instrument.name}') as it's not mapped.")
                continue # Skip instrument if not mapped

            program_token_idx, local_instance_token_idx = token_pair
            instance_key = token_pair # Use (prog_token, local_token) as the key
            logger.debug(f"Processing instrument {original_instrument_idx} ('{instrument.name}') -> Prog:{program_token_idx}, Local:{local_instance_token_idx}")


            # Initialize last velocity for this instance if not seen before
            if instance_key not in last_velocity_bin:
                last_velocity_bin[instance_key] = -1 # Use -1 to indicate no velocity set yet

            # Add program change marker slightly before the first event of non-drum instruments
            try:
                note_starts = [n.start for n in instrument.notes] if instrument.notes else []
                cc_times = [c.time for c in instrument.control_changes] if instrument.control_changes else []
                all_times = [t for t in note_starts + cc_times if t is not None]
                first_event_time = min(all_times) if all_times else float('inf')
            except Exception as e:
                 logger.warning(f"Could not determine first event time for instrument {original_instrument_idx}: {e}. Skipping program change marker.")
                 first_event_time = float('inf')


            if first_event_time != float('inf') and not instrument.is_drum:
                 # Place program change marker slightly before the first event
                 prog_change_time = max(0.0, first_event_time - self.time_epsilon_val * 2) # Ensure non-negative and distinct time
                 events.append({
                     'type': 'program_change_marker',
                     'time': prog_change_time,
                     'program_token_idx': program_token_idx,
                     'local_instance_token_idx': local_instance_token_idx,
                     'program': instrument.program # Store original program number
                 })
                 logger.debug(f"Added program change marker for inst {original_instrument_idx} at time {prog_change_time:.4f}")


            common_meta = {
                'program_token_idx': program_token_idx,
                'local_instance_token_idx': local_instance_token_idx
            }

            # --- Note Events ---
            notes_processed = 0
            for note in instrument.notes:
                # Skip zero or negative duration notes as they cause issues
                if note.end <= note.start:
                    logger.debug(f"Skipping note with non-positive duration (start={note.start}, end={note.end}, pitch={note.pitch}) for inst {original_instrument_idx}")
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
                notes_processed += 1
            logger.debug(f"Processed {notes_processed} notes for inst {original_instrument_idx}.")

            # --- Control Change Events ---
            ccs_processed = 0
            pedals_processed = 0
            for control in instrument.control_changes:
                # Specifically handle sustain pedal (CC 64)
                if control.number == 64:
                    event_type = 'pedal_on' if control.value >= 64 else 'pedal_off'
                    events.append({'type': event_type, 'time': control.time, **common_meta})
                    pedals_processed += 1
                else:
                    # Handle other CCs
                    events.append({
                        'type': 'control_change',
                        'time': control.time,
                        'control_number': control.number,
                        'control_value': control.value,
                        **common_meta
                    })
                    ccs_processed += 1
            logger.debug(f"Processed {ccs_processed} CCs and {pedals_processed} pedal events for inst {original_instrument_idx}.")


         # --- Sort all events globally by time ---
         logger.debug(f"Sorting {len(events)} extracted events...")
         # Use a stable sort if needed, but primary sort is time. Add secondary sort keys if necessary.
         try:
             events.sort(key=lambda x: (x['time'], self._event_sort_priority(x['type'])))
         except Exception as e:
             logger.error(f"Error sorting events: {e}. Event list might be inconsistent.")
             # Consider returning or raising here depending on desired robustness


         # --- Insert Time Shifts and Velocity Changes ---
         logger.debug("Inserting time shifts and velocity changes...")
         timed_events = []
         last_time = 0.0
         time_shifts_added = 0
         velocity_events_added = 0

         for event in events:
             # Ensure time doesn't go backwards due to precision or ordering issues
             event_time = max(last_time, event['time'])

             # Calculate and add time shift if needed
             time_diff = event_time - last_time
             if time_diff > self.time_epsilon_val / 2: # Only add if difference is significant
                 time_shift_steps = self._quantize_time_or_duration(time_diff, self.time_shift_steps)
                 if time_shift_steps > 0:
                     timed_events.append({'type': 'time_shift', 'steps': time_shift_steps})
                     time_shifts_added += 1
                     # Update last_time based on the quantized shift
                     quantized_diff = self._unquantize_steps(time_shift_steps)
                     last_time += quantized_diff
                     # Re-adjust event_time if quantization changed it significantly relative to original
                     # Align subsequent events to the quantized time grid
                     event_time = last_time
                     logger.debug(f"Added TIME_SHIFT({time_shift_steps}) -> {quantized_diff:.4f}s. New last_time: {last_time:.4f}")


             # Add velocity event if changed (only for note_on types)
             if event['type'] == 'note_on' or event['type'] == 'note_on_with_duration':
                 instance_key = (event['program_token_idx'], event['local_instance_token_idx'])
                 velocity_bin = self._quantize_velocity(event['velocity'])

                 # Check if velocity actually changed for this instance
                 if velocity_bin != last_velocity_bin.get(instance_key, -1):
                     # Insert velocity *before* the note event it applies to
                     timed_events.append({
                         'type': 'velocity',
                         'program_token_idx': event['program_token_idx'],
                         'local_instance_token_idx': event['local_instance_token_idx'],
                         'velocity_bin': velocity_bin
                     })
                     last_velocity_bin[instance_key] = velocity_bin # Update last known velocity
                     velocity_events_added += 1
                     logger.debug(f"Added VELOCITY({velocity_bin}) for instance {instance_key}")


             # Add the original event itself
             timed_events.append(event)

             # Update last_time based on the processed event's time
             # Make sure last_time reflects the *actual* time grid position.
             last_time = max(last_time, event_time) # Use event_time which might have been adjusted by time shift

         logger.debug(f"Finished inserting events. Added {time_shifts_added} time shifts, {velocity_events_added} velocity events. Total timed events: {len(timed_events)}")
         return timed_events

    def _event_sort_priority(self, event_type: str) -> int:
        """Assigns priority for sorting events occurring at the same time."""
        # Lower numbers sort earlier
        priority_map = {
            'program_change_marker': 0,
            'control_change': 1,
            'pedal_off': 2,
            'note_off': 3,
            'velocity': 4,
            'note_on': 5,
            'note_on_with_duration': 5,
            'pedal_on': 6,
            'time_shift': -1 # Should ideally not happen at same time, but place early if it does
        }
        return priority_map.get(event_type, 10) # Default priority for unknown types


    def _tokenize_events(self, events: List[Dict[str, Any]]) -> List[int]:
        """Converts the intermediate event list into a sequence of tokens."""
        tokens = []
        logger.debug(f"Tokenizing {len(events)} timed events...")
        skipped_events = 0
        for event in events:
            event_type = event['type']
            event_tokens = [] # Tokens for this specific event

            # Events associated with an instrument instance
            if event_type in ['note_on', 'note_off', 'note_on_with_duration', 'velocity',
                              'control_change', 'program_change_marker', 'pedal_on', 'pedal_off']:
                prog_token = event.get('program_token_idx')
                local_token = event.get('local_instance_token_idx')
                if prog_token is None or local_token is None:
                    logger.warning(f"Skipping event missing program/local token: {event}")
                    skipped_events += 1
                    continue # Should not happen with proper extraction

                # Prepend instrument identifiers
                event_tokens.append(prog_token)
                event_tokens.append(local_token)

                # Append event marker and value(s)
                try:
                    if event_type == 'note_on':
                        event_tokens.extend([self.NOTE_ON, self.note_value_offset + event['note']])
                    elif event_type == 'note_off':
                         # Only add NOTE_OFF if using that mode
                         if self.use_note_off:
                             event_tokens.extend([self.NOTE_OFF, self.note_value_offset + event['note']])
                         # If not using note_off, this event type shouldn't exist here, but we handle defensively
                    elif event_type == 'note_on_with_duration':
                         # Only generated if not use_note_off
                         if not self.use_note_off:
                             event_tokens.extend([
                                 self.NOTE_ON, self.note_value_offset + event['note'],
                                 self.NOTE_DURATION, self.note_duration_value_offset + event['duration_steps']
                             ])
                         # If use_note_off is True, this event type shouldn't exist
                    elif event_type == 'velocity':
                        event_tokens.extend([self.VELOCITY, self.velocity_value_offset + event['velocity_bin']])
                    elif event_type == 'control_change':
                        # Ensure CC number and value are within valid range (0-127)
                        cc_num = max(0, min(127, event['control_number']))
                        cc_val = max(0, min(127, event['control_value']))
                        event_tokens.extend([self.CONTROL_CHANGE,
                                       self.cc_number_offset + cc_num,
                                       self.cc_value_offset + cc_val])
                    elif event_type == 'program_change_marker':
                        # Use PROGRAM_CHANGE marker, value is original program
                        prog_num = max(0, min(127, event['program']))
                        event_tokens.extend([self.PROGRAM_CHANGE, self.program_value_offset + prog_num])
                    elif event_type == 'pedal_on':
                        event_tokens.append(self.PEDAL_ON)
                    elif event_type == 'pedal_off':
                        event_tokens.append(self.PEDAL_OFF)
                except KeyError as e:
                     logger.warning(f"Missing expected key '{e}' in event during tokenization: {event}. Skipping event.")
                     skipped_events += 1
                     continue # Skip this event if data is missing

            # Global Events (currently only time shift)
            elif event_type == 'time_shift':
                try:
                    event_tokens.extend([self.TIME_SHIFT, self.time_shift_value_offset + event['steps']])
                except KeyError as e:
                     logger.warning(f"Missing expected key '{e}' in time_shift event: {event}. Skipping event.")
                     skipped_events += 1
                     continue # Skip this event

            # Handle potential unknown event types defensively
            else:
               logger.warning(f"Skipping unknown event type during tokenization: {event_type}")
               skipped_events += 1
               continue # Skip unknown event types

            # Add the generated tokens for this event to the main list
            tokens.extend(event_tokens)

        logger.debug(f"Tokenization complete. Generated {len(tokens)} tokens. Skipped {skipped_events} events.")
        return tokens

    # --- MIDI File Processing (Extraction, Tokenization) ---
    def process_midi_file(self, midi_path: str) -> Dict[str, Any] | None:
        """
        Processes a MIDI file into a token sequence, including START and END tokens.
        Uses the note representation mode specified in __init__ (use_note_off).
        """
        filename = os.path.basename(midi_path)
        logger.info(f"Processing MIDI file: {filename}")
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            logger.debug(f"Successfully loaded MIDI: {filename}")
        except Exception as e:
            logger.error(f"Error loading MIDI {filename} ({midi_path}): {e}")
            return None

        # --- Instrument Mapping ---
        instrument_token_map = {} # Map original index -> (prog_token, local_token)
        instrument_metadata_map = {}
        valid_instrument_count = 0
        skipped_instruments = 0
        logger.debug(f"Mapping instruments for {filename}...")
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
                else:
                    logger.warning(f"Exceeded max_local_instruments ({self.max_local_instruments}). Skipping instrument {i} ('{inst.name}') in {filename}")
                    skipped_instruments += 1
            else:
                 logger.warning(f"Could not get program token for instrument {i} ('{inst.name}', program={inst.program}, is_drum={inst.is_drum}). Skipping.")
                 skipped_instruments += 1

        logger.info(f"Mapped {valid_instrument_count} instruments, skipped {skipped_instruments} for {filename}.")

        if not instrument_token_map:
            logger.warning(f"No valid/trackable instruments mapped in {filename}. Skipping file.")
            return None # Cannot process file with no mapped instruments

        # --- Metadata Extraction ---
        logger.debug(f"Extracting metadata for {filename}...")
        try:
            metadata = {
                'filename': filename,
                'tempo_changes': self._extract_tempo_changes(midi_data),
                'time_signature_changes': self._extract_time_signatures(midi_data),
                'key_signatures': self._extract_key_signatures(midi_data),
                'total_time': midi_data.get_end_time(),
                'instrument_mapping': instrument_metadata_map,
                'processing_mode': {'use_note_off': self.use_note_off} # Store mode used
            }
            logger.debug(f"Metadata extracted successfully for {filename}.")
        except Exception as e:
            logger.error(f"Error extracting metadata from {filename}: {e}", exc_info=True)
            metadata = {'filename': filename,
                        'processing_mode': {'use_note_off': self.use_note_off}} # Minimal metadata on error

        # --- Event Extraction and Tokenization ---
        logger.debug(f"Extracting and tokenizing events for {filename}...")
        try:
            # Use the modified extraction function
            events = self._extract_sorted_events(midi_data, instrument_token_map)
            # Use the modified tokenization function
            core_tokens = self._tokenize_events(events)

            # Add START and END tokens
            final_tokens = [self.START] + core_tokens + [self.END]
            logger.info(f"Successfully processed {filename}. Total tokens: {len(final_tokens)}")
            if self.min_sequence_length is not None and len(final_tokens) < self.min_sequence_length:
                logger.warning(
                    f"Skipping {filename}: token length {len(final_tokens)} < min_sequence_length "
                    f"{self.min_sequence_length}"
                )
                return None
            # Return as numpy array for consistency with dataset loading
            return {'metadata': metadata, 'tokens': np.array(final_tokens, dtype=np.int32)}
        except Exception as e:
            logger.error(f"Error during event processing/tokenization for {filename}: {e}", exc_info=True)
            # Optionally log traceback: logger.error(traceback.format_exc())
            return None


    # --- Tokens to MIDI Conversion ---
    def tokens_to_midi(self, tokens: List[int], vocab_start_offset: int = 0, default_tempo: float = 120.0, save_path: Optional[str] = None) -> Optional[pretty_midi.PrettyMIDI]:
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
            A pretty_midi.PrettyMIDI object, or None if conversion fails badly.
        """
        logger.info(f"Converting {len(tokens)} tokens to MIDI...")
        try:
            midi_obj = pretty_midi.PrettyMIDI(initial_tempo=default_tempo)
            instrument_tracks: Dict[Tuple[int, int], pretty_midi.Instrument] = {} # (prog_token, local_token) -> Instrument
            # Active notes only needed for use_note_off=True mode
            active_notes: Dict[Tuple[int, int, int], Tuple[float, int]] = {} # (prog_token, local_token, pitch) -> (start_time, velocity)
            current_velocities: Dict[Tuple[int, int], int] = {} # (prog_token, local_token) -> velocity (0-127)
            current_time = 0.0
            notes_added = 0
            ccs_added = 0
            events_processed = 0
            skipped_tokens = 0

            i = 0
            while i < len(tokens):
                token = tokens[i] - vocab_start_offset # Adjust for external offset

                # --- Skip Special Tokens ---
                if token in [self.START, self.END, self.PAD]:
                    logger.debug(f"Skipping special token {self.token_to_name_map.get(token)} at index {i}")
                    i += 1
                    skipped_tokens += 1
                    continue

                event_processed = False # Flag to check if token was handled this iteration

                # --- Global Events (TIME_SHIFT) ---
                if token == self.TIME_SHIFT:
                    if i + 1 < len(tokens):
                        steps_token = tokens[i+1] - vocab_start_offset
                        if self.time_shift_value_offset <= steps_token < self.time_shift_value_offset + self.time_shift_steps:
                            steps = steps_token - self.time_shift_value_offset
                            time_increase = self._unquantize_steps(steps)
                            current_time += time_increase
                            logger.debug(f"Processed TIME_SHIFT({steps}) -> +{time_increase:.4f}s. New time: {current_time:.4f}")
                            i += 2; event_processed = True; events_processed += 1
                        else:
                            logger.warning(f"Invalid TIME_SHIFT value token {steps_token} at index {i+1}. Skipping TIME_SHIFT marker.")
                            i += 1 # Skip only TIME_SHIFT marker
                            skipped_tokens += 1
                    else:
                        logger.warning(f"Incomplete TIME_SHIFT event at end of sequence (index {i}).")
                        i += 1 # Skip incomplete marker
                        skipped_tokens += 1

                # --- Instrument-Specific Events ---
                # Check if the current token is a program token
                elif self.program_token_range[0] <= token < self.program_token_range[1]:
                    # Expect: program_token, local_instance_token, event_marker, [value(s)]...
                    if i + 2 < len(tokens): # Need at least prog, local, marker
                        prog_token = token
                        local_token_raw = tokens[i+1] - vocab_start_offset

                        # Validate local instance token
                        if not (self.local_instance_range[0] <= local_token_raw < self.local_instance_range[1]):
                            logger.warning(f"Invalid LOCAL_INSTANCE token {local_token_raw} following PROG token {prog_token} at index {i+1}. Skipping PROG token.")
                            i += 1 # Skip only program token if local is invalid
                            skipped_tokens += 1
                            continue # Move to next token

                        local_token = local_token_raw # Validated
                        instance_key = (prog_token, local_token)

                        # Get or create instrument track
                        if instance_key not in instrument_tracks:
                            is_drum = (prog_token == self.drum_program_token_idx)
                            program_num = 0 if is_drum else (prog_token - self.program_token_offset)
                            track_name = f"{'Drum' if is_drum else f'Inst {program_num}'}_{local_token - self.local_instance_offset}"
                            instrument_tracks[instance_key] = pretty_midi.Instrument(
                                program=program_num,
                                is_drum=is_drum,
                                name=track_name
                            )
                            midi_obj.instruments.append(instrument_tracks[instance_key])
                            current_velocities[instance_key] = self.default_velocity_val # Initialize default velocity
                            logger.debug(f"Created track for instance {instance_key} ('{track_name}')")


                        track = instrument_tracks[instance_key]
                        event_marker = tokens[i+2] - vocab_start_offset
                        # Most events happen slightly *after* the current time step starts
                        event_time = current_time + self.time_epsilon_val

                        # Determine required tokens and process event
                        consumed_tokens = 0 # How many tokens are consumed by this event block (prog, local, marker + values)

                        # --- Process based on event_marker ---
                        try:
                            if event_marker == self.PROGRAM_CHANGE:
                                if i + 3 < len(tokens):
                                    program_val_token = tokens[i+3] - vocab_start_offset
                                    if self.program_value_offset <= program_val_token < self.program_value_offset + self.program_range:
                                        program_num = program_val_token - self.program_value_offset
                                        if not track.is_drum:
                                            track.program = program_num
                                            logger.debug(f"Processed PROGRAM_CHANGE({program_num}) for instance {instance_key} at time {current_time:.4f}")
                                        else:
                                            logger.debug(f"Ignoring PROGRAM_CHANGE for drum track instance {instance_key}")
                                    else: logger.warning(f"Invalid PROGRAM_CHANGE value token {program_val_token} at index {i+3}.")
                                    consumed_tokens = 4
                                else: logger.warning(f"Incomplete PROGRAM_CHANGE event at index {i}."); consumed_tokens = 3

                            elif event_marker == self.VELOCITY:
                                if i + 3 < len(tokens):
                                    vel_val_token = tokens[i+3] - vocab_start_offset
                                    if self.velocity_value_offset <= vel_val_token < self.velocity_value_offset + self.velocity_bins:
                                        vel_bin = vel_val_token - self.velocity_value_offset
                                        current_velocities[instance_key] = self._unquantize_velocity(vel_bin)
                                        logger.debug(f"Processed VELOCITY({vel_bin}) -> {current_velocities[instance_key]} for instance {instance_key}")
                                    else: logger.warning(f"Invalid VELOCITY value token {vel_val_token} at index {i+3}.")
                                    consumed_tokens = 4
                                else: logger.warning(f"Incomplete VELOCITY event at index {i}."); consumed_tokens = 3

                            elif event_marker == self.NOTE_ON:
                                if self.use_note_off: # --- Mode 1: NOTE_ON / NOTE_OFF ---
                                    if i + 3 < len(tokens):
                                        note_val_token = tokens[i+3] - vocab_start_offset
                                        if self.note_value_offset <= note_val_token < self.note_value_offset + self.note_range:
                                            pitch = note_val_token - self.note_value_offset
                                            velocity = current_velocities.get(instance_key, self.default_velocity_val)
                                            note_key = (*instance_key, pitch)

                                            # End previous note of same pitch if still active (handle overlap)
                                            if note_key in active_notes:
                                                start_time_prev, start_vel_prev = active_notes.pop(note_key)
                                                # Ensure end time is slightly before new start, but after old start
                                                end_time_prev = max(start_time_prev + self.time_epsilon_val / 2, event_time - self.time_epsilon_val / 2)
                                                if end_time_prev > start_time_prev:
                                                     note_obj = pretty_midi.Note(velocity=start_vel_prev, pitch=pitch, start=start_time_prev, end=end_time_prev)
                                                     track.notes.append(note_obj)
                                                     notes_added += 1
                                                     logger.debug(f"Overlap NOTE_ON: Closed previous note {pitch} for {instance_key} at {end_time_prev:.4f}")


                                            # Record the start of the new note
                                            active_notes[note_key] = (event_time, velocity)
                                            logger.debug(f"Processed NOTE_ON({pitch}, vel={velocity}) for instance {instance_key} at time {event_time:.4f}")

                                        else: logger.warning(f"Invalid NOTE_ON value token {note_val_token} at index {i+3}.")
                                        consumed_tokens = 4
                                    else: logger.warning(f"Incomplete NOTE_ON event at index {i}."); consumed_tokens = 3

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
                                            velocity = current_velocities.get(instance_key, self.default_velocity_val)

                                            start_time = event_time
                                            # Ensure minimum duration, end time must be after start time
                                            end_time = start_time + max(duration_sec, self.min_note_duration_seconds_val)

                                            note_obj = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
                                            track.notes.append(note_obj)
                                            notes_added += 1
                                            logger.debug(f"Processed NOTE_ON({pitch}, vel={velocity}, dur={duration_sec:.4f}s) for instance {instance_key} at time {start_time:.4f}")
                                            consumed_tokens = 6
                                        else:
                                            logger.warning(f"Invalid NOTE_ON+DURATION sequence at index {i}. Tokens: {tokens[i:i+6]}. Skipping.")
                                            consumed_tokens = 3 # Skip prog, local, NOTE_ON marker only
                                    else:
                                        logger.warning(f"Incomplete NOTE_ON+DURATION sequence at index {i}.")
                                        consumed_tokens = 3 # Skip prog, local, NOTE_ON marker only

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
                                                 end_time = max(start_time + self.time_epsilon_val, event_time)
                                                 note_obj = pretty_midi.Note(velocity=start_vel, pitch=pitch, start=start_time, end=end_time)
                                                 track.notes.append(note_obj)
                                                 notes_added += 1
                                                 logger.debug(f"Processed NOTE_OFF({pitch}) for instance {instance_key} at time {end_time:.4f}")
                                             else:
                                                 logger.debug(f"Ignoring NOTE_OFF for inactive note {pitch} for instance {instance_key} at time {event_time:.4f}")
                                         else: logger.warning(f"Invalid NOTE_OFF value token {note_val_token} at index {i+3}.")
                                         consumed_tokens = 4
                                     else: logger.warning(f"Incomplete NOTE_OFF event at index {i}."); consumed_tokens = 3
                                 else:
                                     # Ignore NOTE_OFF tokens if not in use_note_off mode
                                     logger.debug(f"Ignoring NOTE_OFF token at index {i+2} because use_note_off is False.")
                                     consumed_tokens = 3 # Skip prog, local, NOTE_OFF marker

                            elif event_marker == self.PEDAL_ON:
                                cc = pretty_midi.ControlChange(number=64, value=100, time=event_time)
                                track.control_changes.append(cc)
                                ccs_added += 1
                                logger.debug(f"Processed PEDAL_ON for instance {instance_key} at time {event_time:.4f}")
                                consumed_tokens = 3

                            elif event_marker == self.PEDAL_OFF:
                                cc = pretty_midi.ControlChange(number=64, value=0, time=event_time)
                                track.control_changes.append(cc)
                                ccs_added += 1
                                logger.debug(f"Processed PEDAL_OFF for instance {instance_key} at time {event_time:.4f}")
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
                                             ccs_added += 1
                                             logger.debug(f"Processed CONTROL_CHANGE(num={cc_num}, val={cc_val}) for instance {instance_key} at time {event_time:.4f}")
                                         else:
                                             logger.debug(f"Ignoring CONTROL_CHANGE for pedal (num=64) as PEDAL_ON/OFF is used.")
                                     else: logger.warning(f"Invalid CONTROL_CHANGE num/val tokens {cc_num_token}/{cc_val_token} at index {i+3}/{i+4}.")
                                     consumed_tokens = 5
                                 else: logger.warning(f"Incomplete CONTROL_CHANGE event at index {i}."); consumed_tokens = 3

                            # --- Handle unrecognized marker or incomplete sequence ---
                            if consumed_tokens == 0:
                                marker_name = self.token_to_name_map.get(event_marker, f"UNK({event_marker})")
                                logger.warning(f"Unrecognized or incomplete instrument event sequence starting at index {i}. Marker: {marker_name}. Skipping 3 tokens (prog, local, marker).")
                                consumed_tokens = 3 # Default skip: prog, local, marker

                        except Exception as e_inner:
                             # Catch errors during event processing for robustness
                             logger.error(f"Error processing event block at index {i}: {e_inner}", exc_info=True)
                             consumed_tokens = 3 # Skip the basic block on error

                        i += consumed_tokens
                        event_processed = True
                        events_processed += 1

                    else: # Not enough tokens for a full instrument event (prog, local, marker)
                        logger.warning(f"Incomplete instrument event start at index {i} (needs prog, local, marker). Skipping.")
                        i += 1 # Skip only the program token
                        skipped_tokens += 1

                # --- Fallback for Unhandled Tokens ---
                if not event_processed:
                    # This case handles tokens that are not TIME_SHIFT and not program tokens
                    token_name = self.token_to_name_map.get(token, f"UNK({token})")
                    logger.warning(f"Skipping unexpected/unhandled token {token} ('{token_name}') at index {i}")
                    i += 1
                    skipped_tokens += 1

            # --- Final Cleanup (Only relevant for use_note_off=True) ---
            if self.use_note_off:
                # Turn off any remaining active notes at the very end
                final_event_time = current_time + self.time_step # A bit after the last event time
                if active_notes:
                    logger.debug(f"Closing {len(active_notes)} remaining active notes at final time {final_event_time:.4f}")
                    # Iterate over a copy of items as we modify the dict
                    for note_key, (start_time, start_vel) in list(active_notes.items()):
                         prog_token, local_token, pitch = note_key
                         instance_key = (prog_token, local_token)
                         if instance_key in instrument_tracks:
                             track = instrument_tracks[instance_key]
                             # Ensure end time is strictly after start time
                             end_time = max(start_time + self.min_note_duration_seconds_val, final_event_time)
                             note_obj = pretty_midi.Note(velocity=start_vel, pitch=pitch, start=start_time, end=end_time)
                             track.notes.append(note_obj)
                             notes_added += 1
                             logger.debug(f"Auto-closing active note {pitch} for {instance_key} at end time {end_time:.4f}")
                         # Remove from active notes even if track somehow disappeared
                         del active_notes[note_key]


            # Sort notes and CCs within each track (important for some players/DAWs)
            logger.debug("Sorting notes and CCs within final tracks...")
            for inst in midi_obj.instruments:
                inst.notes.sort(key=lambda n: n.start)
                inst.control_changes.sort(key=lambda cc: cc.time)

            logger.info(f"MIDI conversion complete. Processed {events_processed} events, added {notes_added} notes, {ccs_added} CCs. Skipped {skipped_tokens} tokens.")

            # Save if requested
            if save_path:
                logger.info(f"Attempting to save MIDI to: {save_path}")
                try:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    midi_obj.write(save_path)
                    logger.info(f"Successfully saved MIDI to: {save_path}")
                except Exception as e:
                    logger.error(f"Error saving MIDI to {save_path}: {e}", exc_info=True)

            return midi_obj

        except Exception as e_outer:
            # Catch any top-level error during conversion
            logger.critical(f"Fatal error during tokens_to_midi conversion: {e_outer}", exc_info=True)
            return None


    # --- Tokens to Text Conversion ---
    def tokens_to_text(self, tokens: List[int], vocab_start_offset: int = 0) -> str:
        """Converts a token sequence to a human-readable text representation."""
        # (Implementation remains largely the same, but could add logging for warnings)
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
                                    else:
                                        event_parts.append(f"(invalid_dur_marker:{dur_marker_name})"); consumed=4 # Fallback if marker wrong
                                        logger.debug(f"Invalid duration marker found at index {i+4} in text conversion.")
                                else:
                                    event_parts.append("(incomplete_dur)"); consumed=4 # Fallback if seq too short
                                    logger.debug(f"Incomplete duration sequence found at index {i+4} in text conversion.")
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
                    logger.debug(f"Incomplete instrument event start at index {i} in text conversion.")


            # --- Fallback for other unknown tokens ---
            else:
                current_event_str = f"[{token_name}]" # Handles unknown tokens
                consumed = 1
                logger.debug(f"Encountered unknown token {token} at index {i} in text conversion.")


            text_events.append(current_event_str)
            i += consumed

        return "\n".join(text_events)