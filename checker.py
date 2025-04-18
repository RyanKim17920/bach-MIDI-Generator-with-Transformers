import os
import pretty_midi
from MIDIprocessor import MIDIProcessor # Assuming MIDIprocessor.py is in the same directory or PYTHONPATH

# --- Configuration ---
TEST_MIDI_PATH = f"path_to_midi" # <<< CHANGE THIS to your actual MIDI file path
OUTPUT_DIR = "reconstruction_test_output"

# --- Create output directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper function to load and print basic MIDI info ---
def print_midi_info(midi_obj: pretty_midi.PrettyMIDI, label: str):
    if midi_obj is None:
        print(f"{label}: Could not load MIDI.")
        return
    total_notes = sum(len(inst.notes) for inst in midi_obj.instruments)
    end_time = midi_obj.get_end_time()
    num_instruments = len(midi_obj.instruments)
    print(f"{label}:")
    print(f"  Instruments: {num_instruments}")
    print(f"  Total Notes: {total_notes}")
    print(f"  End Time: {end_time:.3f}s")

# --- Main Test Logic ---
if __name__ == "__main__":
    if not os.path.exists(TEST_MIDI_PATH):
        print(f"Error: Test MIDI file not found at '{TEST_MIDI_PATH}'")
        print("Please update the TEST_MIDI_PATH variable in the script.")
        exit()

    print(f"Loading original MIDI: {TEST_MIDI_PATH}")
    try:
        original_midi = pretty_midi.PrettyMIDI(TEST_MIDI_PATH)
        print_midi_info(original_midi, "Original MIDI")
    except Exception as e:
        print(f"Error loading original MIDI: {e}")
        exit()

    # --- Test 1: use_note_off = True ---
    print("\n--- Testing with use_note_off = True ---")
    processor_note_off = MIDIProcessor(use_note_off=True, time_step=0.01, velocity_bins=32, max_time_shift_seconds=10.0)
    processed_data_note_off = processor_note_off.process_midi_file(TEST_MIDI_PATH)

    if processed_data_note_off:
        tokens_note_off = processed_data_note_off['tokens']
        print(f"Processed into {len(tokens_note_off)} tokens (Note Off Mode).")

        # Convert back to MIDI
        reconstructed_midi_note_off_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(TEST_MIDI_PATH))[0]}_reconstructed_note_off.mid")
        reconstructed_midi_note_off = processor_note_off.tokens_to_midi(
            tokens=tokens_note_off.tolist(), # Convert numpy array to list
            save_path=reconstructed_midi_note_off_path
        )
        print_midi_info(reconstructed_midi_note_off, "Reconstructed MIDI (Note Off Mode)")

        # Optional: Print token sequence as text
        # text_repr_note_off = processor_note_off.tokens_to_text(tokens_note_off.tolist())
        # text_path_note_off = os.path.join(OUTPUT_DIR, "tokens_note_off.txt")
        # with open(text_path_note_off, "w") as f:
        #     f.write(text_repr_note_off)
        # print(f"Saved token text representation to: {text_path_note_off}")

    else:
        print("Failed to process MIDI in Note Off mode.")

    # --- Test 2: use_note_off = False ---
    print("\n--- Testing with use_note_off = False ---")
    # Ensure max_note_duration is reasonable, e.g., same as max_time_shift
    processor_duration = MIDIProcessor(use_note_off=False, time_step=0.01, velocity_bins=32, max_time_shift_seconds=10.0, max_note_duration_seconds=10.0)
    processed_data_duration = processor_duration.process_midi_file(TEST_MIDI_PATH)

    if processed_data_duration:
        tokens_duration = processed_data_duration['tokens']
        print(f"Processed into {len(tokens_duration)} tokens (Duration Mode).")

        # Convert back to MIDI
        reconstructed_midi_duration_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(TEST_MIDI_PATH))[0]}_reconstructed_duration.mid")
        reconstructed_midi_duration = processor_duration.tokens_to_midi(
            tokens=tokens_duration.tolist(), # Convert numpy array to list
            save_path=reconstructed_midi_duration_path
        )
        print_midi_info(reconstructed_midi_duration, "Reconstructed MIDI (Duration Mode)")

        # Optional: Print token sequence as text
        # text_repr_duration = processor_duration.tokens_to_text(tokens_duration.tolist())
        # text_path_duration = os.path.join(OUTPUT_DIR, "tokens_duration.txt")
        # with open(text_path_duration, "w") as f:
        #     f.write(text_repr_duration)
        # print(f"Saved token text representation to: {text_path_duration}")

    else:
        print("Failed to process MIDI in Duration mode.")

    print(f"\nReconstruction complete. Check the '{OUTPUT_DIR}' directory for the generated MIDI files.")
    print("Note: Due to quantization, the reconstructed MIDIs might not be identical to the original.")