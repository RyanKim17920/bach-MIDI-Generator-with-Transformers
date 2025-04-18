import os
import numpy as np
import pytest
import torch
import pretty_midi # Import pretty_midi
import random # Import random

# Import constants and classes from src
from src.constants import PAD_IDX, START_IDX, END_IDX
from src.base_dataset import BaseChunkingDataset
from src.midi_dataset_preprocessed import MIDIDatasetPreprocessed
from src.midi_dataset import MIDIDataset
from src.midi_processor import MIDIProcessor # Import the real processor

# --- Fixtures ---

@pytest.fixture(scope="module")
def test_processor():
    """Provides a MIDIProcessor instance for dataset tests."""
    # Use settings that generate a predictable, reasonably short token sequence
    return MIDIProcessor(
        time_step=0.2, # Larger time step -> fewer time shift tokens
        velocity_bins=2, # Fewer velocity tokens
        max_time_shift_seconds=1.0, # Limit time shifts
        use_note_off=True,
        max_local_instruments=2 # Limit instruments
    )

@pytest.fixture
def simple_midi_file(tmp_path):
    """Creates a simple MIDI file with a few notes."""
    midi_path = tmp_path / "simple.mid"
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    # Note C4 for 0.5 sec, Note E4 for 0.5 sec
    inst.notes.append(pretty_midi.Note(velocity=64, pitch=60, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=96, pitch=64, start=0.6, end=1.1))
    pm.instruments.append(inst)
    pm.write(str(midi_path))
    return str(midi_path)

@pytest.fixture
def very_short_midi_file(tmp_path):
    """Creates a MIDI file that will likely be shorter than sequence_length."""
    midi_path = tmp_path / "veryshort.mid"
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    # Single short note
    inst.notes.append(pretty_midi.Note(velocity=64, pitch=60, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(midi_path))
    return str(midi_path)

# --- Test Cases ---

def test_mididataset_on_the_fly(simple_midi_file, test_processor):
    """Tests MIDIDataset processing a real MIDI file."""
    sequence_length = 10 # B = 10, chunks need B+1 = 11 tokens
    augmentation_shift = 0

    # Process the file once to know the expected token sequence
    expected_result = test_processor.process_midi_file(simple_midi_file)
    assert expected_result is not None and 'tokens' in expected_result
    expected_tokens = expected_result['tokens']
    A = len(expected_tokens) # Actual length of token sequence
    B = sequence_length

    # Calculate expected number of chunks based on BaseChunkingDataset logic
    expected_chunks = 0
    if A >= B + 1:
        expected_chunks += 1 # Start chunk
        last_possible_start = A - (B + 1)
        k = 1
        while True:
            base_start = k * B
            if base_start <= last_possible_start:
                if base_start > 0: # Avoid duplicating start chunk
                    expected_chunks += 1
                k += 1
            else:
                break
        end_chunk_start = A - (B + 1)
        # Check if end chunk is different from the last added chunk
        last_added_start = (k - 1) * B if k > 1 else 0
        if end_chunk_start > last_added_start:
            expected_chunks += 1
    elif A >= 2: # Can form at least one padded chunk
        expected_chunks = 1

    # Initialize the dataset
    ds = MIDIDataset(
        midi_file_paths=[simple_midi_file],
        processor=test_processor,
        sequence_length=sequence_length,
        augmentation_shift=augmentation_shift
    )

    assert len(ds) == expected_chunks, f"Expected {expected_chunks} chunks, got {len(ds)} for token length {A} and seq_len {B}"

    # Check content of the first chunk (start chunk or padded chunk)
    if len(ds) > 0:
        x0, y0 = ds[0]
        assert isinstance(x0, torch.Tensor)
        assert x0.shape == (sequence_length,)
        assert y0.shape == (sequence_length,)

        # Verify content against expected tokens
        if A >= B + 1: # Start chunk
            expected_x0 = expected_tokens[0:B]
            expected_y0 = expected_tokens[1:B+1]
            assert x0.tolist() == list(expected_x0)
            assert y0.tolist() == list(expected_y0)
        elif A >= 2: # Padded chunk
            expected_x0 = list(expected_tokens[:-1]) + [PAD_IDX] * (B - (A - 1))
            expected_y0 = list(expected_tokens[1:]) + [PAD_IDX] * (B - (A - 1))
            assert x0.tolist() == expected_x0[:B] # Ensure length B
            assert y0.tolist() == expected_y0[:B] # Ensure length B
            assert PAD_IDX in x0.tolist() # Check padding occurred

def test_mididataset_very_short_file(very_short_midi_file, test_processor):
    """Tests MIDIDataset with a file too short for a full sequence."""
    sequence_length = 20 # B=20, needs 21 tokens. The file is unlikely to produce this many.
    ds = MIDIDataset(
        midi_file_paths=[very_short_midi_file],
        processor=test_processor,
        sequence_length=sequence_length,
        augmentation_shift=0
    )

    # Process to check length
    expected_result = test_processor.process_midi_file(very_short_midi_file)
    A = len(expected_result['tokens']) if expected_result else 0

    if A >= 2: # Should produce exactly one padded chunk
        assert len(ds) == 1
        x, y = ds[0]
        assert x.shape == (sequence_length,)
        assert y.shape == (sequence_length,)
        assert PAD_IDX in x.tolist() # Must be padded
        assert PAD_IDX in y.tolist()
        # Check that the initial part matches the tokens
        assert x.tolist()[:A-1] == list(expected_result['tokens'][:-1])
    else: # If processing yields < 2 tokens (e.g., only START/END)
        assert len(ds) == 0

def test_preprocessed_dataset_loading_and_padding(tmp_path):
    """Tests MIDIDatasetPreprocessed with padding."""
    sequence_length = 5  # B=5, needs 6 tokens
    pad_idx = PAD_IDX

    # Short array (3 tokens), needs padding
    short_tokens = np.array([START_IDX, 10, END_IDX], dtype=np.int32)
    # Long array (10 tokens)
    long_tokens = np.array([START_IDX, *range(20, 28), END_IDX], dtype=np.int32)

    sfile = tmp_path / "short.npy"
    lfile = tmp_path / "long.npy"
    np.save(sfile, short_tokens)
    np.save(lfile, long_tokens)

    ds = MIDIDatasetPreprocessed(
        sequence_length=sequence_length,
        preprocessed_dir=str(tmp_path),
        pad_idx=pad_idx,
        augmentation_shift=0
    )

    # We expect 3 chunks: long-start, long-end, short-pad
    assert len(ds) == 3

    # Identify indices by chunk_type + file name
    # ds.file_names = ["long", "short"] (sorted lexically)
    # ds.chunk_info: List[(file_idx, chunk_type, base_start)]
    # Find the pad‐chunk index
    pad_chunk_idx = next(i for i, info in enumerate(ds.chunk_info) if info[1] == 'pad')
    x_pad, y_pad = ds[pad_chunk_idx]
    assert x_pad.shape == (sequence_length,)
    # Expected for short: tokens=[1,10,2] → x=[1,10,2,0,0], y=[10,2,0,0,0]
    assert x_pad.tolist() == [START_IDX, 10, END_IDX, pad_idx, pad_idx]
    assert y_pad.tolist() == [10, END_IDX, pad_idx, pad_idx, pad_idx]

    # Find the long‐start chunk
    start_idx = next(i for i, info in enumerate(ds.chunk_info)
                     if info[1] == 'start' and ds.file_names[info[0]] == 'long')
    x_start, y_start = ds[start_idx]
    assert x_start.tolist() == list(long_tokens[0:sequence_length])
    assert y_start.tolist() == list(long_tokens[1:sequence_length+1])

    # Find the long‐end chunk
    end_idx = next(i for i, info in enumerate(ds.chunk_info)
                   if info[1] == 'end' and ds.file_names[info[0]] == 'long')
    x_end, y_end = ds[end_idx]
    start_end = len(long_tokens) - (sequence_length + 1)  # 10 - 6 = 4
    assert x_end.tolist() == list(long_tokens[start_end : start_end + sequence_length])
    assert y_end.tolist() == list(long_tokens[start_end + 1 : start_end + sequence_length + 1])

def test_preprocessed_dataset_augmentation(tmp_path):
    """Tests augmentation shift in MIDIDatasetPreprocessed."""
    sequence_length = 4 # B=4, needs 5 tokens
    augmentation_shift = 2 # S=2
    pad_idx = PAD_IDX

    # Long enough file to have a middle chunk with room to shift
    # A = 15 tokens. B=4. Needs 5.
    # Start chunk: idx 0
    # Middle chunks: base_k = k*B.
    # k=1, base=4. 4 <= A-(B+1) = 15-5=10. Add middle chunk at base=4.
    # k=2, base=8. 8 <= 10. Add middle chunk at base=8.
    # k=3, base=12. 12 > 10. Stop.
    # End chunk: idx = A-(B+1) = 10.
    # Chunks: (start, 0), (middle, 4), (middle, 8), (end, 10)
    tokens = np.arange(15, dtype=np.int32)
    lfile = tmp_path / "long_for_aug.npy"
    np.save(lfile, tokens)

    ds = MIDIDatasetPreprocessed(
        sequence_length=sequence_length,
        preprocessed_dir=str(tmp_path),
        pad_idx=pad_idx,
        augmentation_shift=augmentation_shift
    )
    assert len(ds) == 4

    # Get the first middle chunk (index 1 in ds, base_start_index=4)
    middle_chunk_idx = 1
    base_start_index = ds.chunk_info[middle_chunk_idx][2]
    assert base_start_index == 4

    # Retrieve the same middle chunk multiple times and check if start index varies
    retrieved_starts = set()
    for _ in range(20): # Retrieve multiple times
        x, y = ds[middle_chunk_idx]
        # Infer the actual start index from the first element of x
        actual_start = x.tolist()[0] # Since tokens are just range(15)
        retrieved_starts.add(actual_start)

        # Check bounds: actual_start = base_start + shift
        # shift range: max(-S, -base_start) to min(S, A - base_start - B - 1)
        # max(-2, -4) = -2
        # min(2, 15 - 4 - 4 - 1) = min(2, 6) = 2
        # Shift range [-2, 2]. Actual start range [4-2, 4+2] = [2, 6]
        assert 2 <= actual_start <= 6

    # Expect more than one start index if augmentation worked
    assert len(retrieved_starts) > 1, "Augmentation did not produce different start indices"

def test_base_chunking_invalid_sequence_length():
    """Tests that BaseChunkingDataset rejects invalid sequence length."""
    with pytest.raises(ValueError, match="sequence_length must be a positive integer"):
        BaseChunkingDataset(sequence_length=0)
    with pytest.raises(ValueError, match="sequence_length must be a positive integer"):
        BaseChunkingDataset(sequence_length=-1)

def test_dataset_empty_input(tmp_path, test_processor):
    """Tests dataset initialization with no valid input files."""
    # MIDIDataset
    ds_live = MIDIDataset(midi_file_paths=[], processor=test_processor, sequence_length=10)
    assert len(ds_live) == 0

    # MIDIDatasetPreprocessed (dir)
    empty_dir = tmp_path / "empty_npy"
    empty_dir.mkdir()
    ds_pre_dir = MIDIDatasetPreprocessed(sequence_length=10, preprocessed_dir=str(empty_dir))
    assert len(ds_pre_dir) == 0

    # MIDIDatasetPreprocessed (list)
    ds_pre_list = MIDIDatasetPreprocessed(sequence_length=10, npy_file_paths=[])
    assert len(ds_pre_list) == 0

    # MIDIDatasetPreprocessed (non-existent dir)
    with pytest.raises(FileNotFoundError):
         MIDIDatasetPreprocessed(sequence_length=10, preprocessed_dir=str(tmp_path / "non_existent_dir"))