import os
import logging # Import logging
import numpy as np
import pretty_midi
import pytest
import math # Import math

# Import constants directly
from src.constants import PAD_IDX, START_IDX, END_IDX
from src.midi_processor import MIDIProcessor

# Configure logging for tests (optional, but helpful for debugging)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Fixtures ---

@pytest.fixture(scope="module")
def default_processor():
    """Provides a default MIDIProcessor instance."""
    return MIDIProcessor(time_step=0.1, velocity_bins=4, max_time_shift_seconds=2.0, use_note_off=True)

@pytest.fixture(scope="module")
def duration_processor():
    """Provides a MIDIProcessor instance using note duration."""
    return MIDIProcessor(time_step=0.1, velocity_bins=4, max_time_shift_seconds=2.0, use_note_off=False, max_note_duration_seconds=2.0)

@pytest.fixture
def complex_midi_file(tmp_path):
    """Creates a more complex MIDI file for testing."""
    midi_path = tmp_path / "complex.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)

    # Instrument 1: Piano
    piano = pretty_midi.Instrument(program=0, name="Piano")
    piano.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.4)) # C4
    piano.notes.append(pretty_midi.Note(velocity=90, pitch=64, start=0.5, end=0.9)) # E4
    piano.control_changes.append(pretty_midi.ControlChange(number=64, value=100, time=0.1)) # Pedal On
    piano.control_changes.append(pretty_midi.ControlChange(number=64, value=0, time=0.8))  # Pedal Off
    pm.instruments.append(piano)

    # Instrument 2: Drums
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1)) # Kick
    drums.notes.append(pretty_midi.Note(velocity=70, pitch=42, start=0.5, end=0.6)) # Hi-hat Closed
    pm.instruments.append(drums)

    # Instrument 3: Bass (will likely exceed max_local_instruments if default is low)
    bass = pretty_midi.Instrument(program=33, name="Bass")
    bass.notes.append(pretty_midi.Note(velocity=75, pitch=48, start=0.0, end=0.9)) # C3
    pm.instruments.append(bass)

    pm.write(str(midi_path))
    return str(midi_path)

# --- Basic Helper Tests ---

def test_calculate_steps(default_processor):
    assert default_processor._calculate_steps(1.0, 0.5) == 3
    assert default_processor._calculate_steps(0.9, 0.5) == 3
    # 0.4/0.5 = 0.8 → ceil(0.8) + 1 = 1 + 1 = 2
    assert default_processor._calculate_steps(0.4, 0.5) == 2
    # Let's re-evaluate _calculate_steps logic slightly based on implementation
    # steps = int(round(time_diff / self.time_step)) -> min(steps, max_steps - 1)
    # max_steps = ceil(max_seconds / step_size) + 1
    # For max_time=1.0, step=0.5 -> max_steps = ceil(2)+1 = 3. Indices 0, 1, 2.
    # time_diff=1.0 -> round(1.0/0.5)=2. Index 2. Correct.
    # time_diff=0.9 -> round(0.9/0.5)=round(1.8)=2. Index 2. Correct.
    # time_diff=0.4 -> round(0.4/0.5)=round(0.8)=1. Index 1. Correct.
    # time_diff=0.1 -> round(0.1/0.5)=round(0.2)=0. Index 0. Correct.
    assert default_processor._quantize_time_or_duration(1.0, default_processor.time_shift_steps) == 10 # round(1.0/0.1)=10. max_steps=ceil(2.0/0.1)+1=21. Indices 0-20.
    assert default_processor._quantize_time_or_duration(0.0, default_processor.time_shift_steps) == 0
    assert default_processor._quantize_time_or_duration(0.04, default_processor.time_shift_steps) == 0 # < time_step/2 (0.05)
    assert default_processor._quantize_time_or_duration(0.06, default_processor.time_shift_steps) == 1 # round(0.06/0.1)=1
    # Test clamping
    assert default_processor._quantize_time_or_duration(3.0, default_processor.time_shift_steps) == 20 # max_steps-1 = 21-1=20

def test_quantize_velocity_and_unquantize(default_processor):
    proc = default_processor
    assert proc._quantize_velocity(0) == 0
    assert proc._quantize_velocity(127) == 3
    # 64*4/128 = 2 → int(2) = 2
    assert proc._quantize_velocity(64) == 2

    # Unquantize - should roughly center in the bin
    assert proc._unquantize_velocity(0) == 16 # (0+0.5)*128/4 = 0.5*32 = 16
    assert proc._unquantize_velocity(1) == 48 # (1+0.5)*32 = 1.5*32 = 48
    assert proc._unquantize_velocity(2) == 80 # (2+0.5)*32 = 2.5*32 = 80
    assert proc._unquantize_velocity(3) == 112 # (3+0.5)*32 = 3.5*32 = 112

# --- Tokenization and Detokenization Tests ---

@pytest.mark.parametrize("processor_fixture", ["default_processor", "duration_processor"])
def test_process_complex_midi(processor_fixture, complex_midi_file, request):
    """Tests processing a more complex MIDI file."""
    processor: MIDIProcessor = request.getfixturevalue(processor_fixture)
    result = processor.process_midi_file(complex_midi_file)

    assert result is not None and 'tokens' in result and 'metadata' in result
    tokens = result['tokens']
    metadata = result['metadata']

    assert isinstance(tokens, np.ndarray)
    assert tokens.dtype == np.int32
    assert tokens[0] == START_IDX
    assert tokens[-1] == END_IDX
    assert len(tokens) > 10 # Expect a reasonable number of tokens

    # Check for expected token types
    assert processor.TIME_SHIFT in tokens
    assert processor.NOTE_ON in tokens
    assert processor.VELOCITY in tokens
    # Check for program tokens (Piano=0, Drums=meta, Bass=33)
    assert processor.program_token_offset + 0 in tokens # Piano program
    assert processor.drum_program_token_idx in tokens # Drum meta program
    if processor.max_local_instruments >= 3: # Only expect bass if tracked
         assert processor.program_token_offset + 33 in tokens # Bass program
    # Check for local instance tokens (at least 0 and 1)
    assert processor.local_instance_offset + 0 in tokens
    assert processor.local_instance_offset + 1 in tokens
    if processor.max_local_instruments >= 3:
         assert processor.local_instance_offset + 2 in tokens
    # Check for note values (C4=60, E4=64, Kick=36, HH=42, C3=48)
    assert processor.note_value_offset + 60 in tokens
    assert processor.note_value_offset + 64 in tokens
    assert processor.note_value_offset + 36 in tokens
    assert processor.note_value_offset + 42 in tokens
    if processor.max_local_instruments >= 3:
         assert processor.note_value_offset + 48 in tokens
    # Check for pedal events
    assert processor.PEDAL_ON in tokens
    assert processor.PEDAL_OFF in tokens

    # Check based on mode
    if processor.use_note_off:
        assert processor.NOTE_OFF in tokens
        assert processor.NOTE_DURATION not in tokens[1:-1] # Duration marker shouldn't appear
    else:
        assert processor.NOTE_DURATION in tokens
        assert processor.NOTE_OFF not in tokens[1:-1] # Note off marker shouldn't appear

    # Check metadata structure
    assert 'tempo_changes' in metadata
    assert 'time_signature_changes' in metadata
    assert 'key_signatures' in metadata
    assert 'instrument_mapping' in metadata
    assert len(metadata['instrument_mapping']) == min(3, processor.max_local_instruments) # Piano, Drums, maybe Bass

@pytest.mark.parametrize("processor_fixture", ["default_processor", "duration_processor"])
def test_roundtrip_complex_midi(processor_fixture, complex_midi_file, request, tmp_path):
    """Tests processing and then converting back to MIDI."""
    processor: MIDIProcessor = request.getfixturevalue(processor_fixture)
    result = processor.process_midi_file(complex_midi_file)
    assert result is not None and 'tokens' in result
    tokens = result['tokens']

    # Convert back to MIDI
    output_midi_path = tmp_path / f"roundtrip_{processor_fixture}.mid"
    midi_out = processor.tokens_to_midi(tokens, default_tempo=120.0, save_path=str(output_midi_path))

    assert isinstance(midi_out, pretty_midi.PrettyMIDI)
    assert os.path.exists(output_midi_path)

    # Basic checks on the reconstructed MIDI
    num_instruments_out = len(midi_out.instruments)
    expected_instruments = min(3, processor.max_local_instruments)
    assert num_instruments_out == expected_instruments

    total_notes_out = sum(len(inst.notes) for inst in midi_out.instruments)
    # Expect roughly the original number of notes (5)
    # Might differ slightly due to quantization or minimum duration enforcement
    assert abs(total_notes_out - 5) <= 1

    # Check if drum instrument exists and has notes
    drum_instrument = next((inst for inst in midi_out.instruments if inst.is_drum), None)
    assert drum_instrument is not None
    assert len(drum_instrument.notes) >= 2 # Kick and HH

    # Check if piano instrument exists and has notes and pedal CCs
    piano_instrument = next((inst for inst in midi_out.instruments if not inst.is_drum and inst.program == 0), None)
    assert piano_instrument is not None
    assert len(piano_instrument.notes) >= 2 # C4 and E4
    pedal_ccs = [cc for cc in piano_instrument.control_changes if cc.number == 64]
    assert len(pedal_ccs) >= 2 # Pedal On and Off

def test_process_empty_midi(tmp_path, default_processor):
    """Tests processing an empty MIDI file."""
    midi_path = tmp_path / "empty.mid"
    pm = pretty_midi.PrettyMIDI()
    pm.write(str(midi_path))

    result = default_processor.process_midi_file(str(midi_path))
    # Should return None because no instruments are mapped
    assert result is None

def test_max_local_instruments_limit(tmp_path, complex_midi_file):
    proc_limit_1 = MIDIProcessor(max_local_instruments=1, use_note_off=True)
    result = proc_limit_1.process_midi_file(complex_midi_file)
    assert result is not None

    tokens = result['tokens']
    meta = result['metadata']

    offset = proc_limit_1.local_instance_offset
    # Only one local‐instance token (offset) should ever appear
    local_tokens = [t for t in tokens if offset <= t < offset + proc_limit_1.max_local_instruments]
    assert set(local_tokens) == {offset}

    # Metadata mapping should only include one instrument
    assert len(meta['instrument_mapping']) == 1

    # Round‐trip still yields a single track
    midi_out = proc_limit_1.tokens_to_midi(tokens)
    assert midi_out is not None
    assert len(midi_out.instruments) == 1

def test_tokens_to_text_detailed(default_processor):
    """Tests text conversion with more specific tokens."""
    proc = default_processor
    # Example sequence: START, Prog0, Local0, VEL(bin 2), NOTE_ON(60), TIME_SHIFT(5 steps), NOTE_OFF(60), END
    vel_val = proc.velocity_value_offset + 2
    note_val = proc.note_value_offset + 60
    time_val = proc.time_shift_value_offset + 5
    prog0 = proc.program_token_offset + 0
    local0 = proc.local_instance_offset + 0

    tokens = [
        START_IDX,
        prog0, local0, proc.VELOCITY, vel_val,
        prog0, local0, proc.NOTE_ON, note_val,
        proc.TIME_SHIFT, time_val,
        prog0, local0, proc.NOTE_OFF, note_val,
        END_IDX
    ]
    text = proc.tokens_to_text(tokens)

    expected_lines = [
        "[START]",
        f"[PROG(0) LOCAL_INST(0) VELOCITY VEL_VAL(2)]",
        f"[PROG(0) LOCAL_INST(0) NOTE_ON NOTE_VAL(60)]",
        f"[TIME_SHIFT TIME_SHIFT_VAL(5)]",
        f"[PROG(0) LOCAL_INST(0) NOTE_OFF NOTE_VAL(60)]",
        "[END]"
    ]
    # Split generated text and remove empty lines
    actual_lines = [line for line in text.split('\n') if line]

    assert actual_lines == expected_lines

# --- Error Handling Tests ---
# (Add tests for invalid inputs to __init__ if not already covered by type hints/basic checks)
def test_init_invalid_params():
    with pytest.raises(ValueError):
        MIDIProcessor(time_step=0)
    with pytest.raises(ValueError):
        MIDIProcessor(velocity_bins=0)
    with pytest.raises(ValueError):
        MIDIProcessor(max_local_instruments=-1)