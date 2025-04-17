import os
import argparse
import numpy as np
from tqdm import tqdm # For progress bar

# Assuming MIDIProcessor is defined in midi_processor.py (or adjust import)
try:
    from MIDIprocessor import MIDIProcessor
except ImportError:
    print("Error: Make sure MIDIProcessor class is defined or importable (e.g., in midi_processor.py)")
    # You might need to copy the MIDIProcessor class definition here if it's not in a separate file
    exit()

def preprocess_directory(midi_dir, output_dir, processor_config, 
                         walk=False, override=False):
    """
    Processes all MIDI files in midi_dir, saves token sequences as .npy files in output_dir.

    Args:
        midi_dir (str): Path to the directory containing MIDI files.
        output_dir (str): Path to the directory where .npy files will be saved.
        processor_config (dict): Dictionary with parameters for MIDIProcessor initialization.
    """
    if not os.path.isdir(midi_dir):
        print(f"Error: MIDI input directory not found: {midi_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- Initialize MIDIProcessor ---
    # Use the provided configuration
    try:
        processor = MIDIProcessor(
            time_step=processor_config.get('time_step', 0.01),
            velocity_bins=processor_config.get('velocity_bins', 32),
            max_time_shift_seconds=processor_config.get('max_time_shift_seconds', 10.0),
            max_local_instruments=processor_config.get('max_local_instruments', 16)
        )
        print("MIDIProcessor initialized successfully.")
    except Exception as e:
        print(f"Error initializing MIDIProcessor: {e}")
        return

    # --- Find MIDI Files ---
    midi_files = []
    if not walk:
        for filename in os.listdir(midi_dir):
            if filename.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(midi_dir, filename))
    else:
        for root, _, files in os.walk(midi_dir):
            for filename in files:
                if filename.lower().endswith(('.mid', '.midi')):
                    midi_files.append(os.path.join(root, filename))

    if not midi_files:
        print(f"No MIDI files found in {midi_dir}")
        return

    print(f"Found {len(midi_files)} MIDI files to process.")

    # --- Process and Save ---
    skipped_count = 0
    success_count = 0
    for midi_path in tqdm(midi_files, desc="Processing MIDI files"):
        base_filename = os.path.splitext(os.path.basename(midi_path))[0]
        output_npy_path = os.path.join(output_dir, f"{base_filename}.npy")

        # Optional: Skip if already processed
        if not override and os.path.exists(output_npy_path):
             continue

        try:
            processed_data = processor.process_midi_file(midi_path)

            if processed_data and processed_data.get('tokens'):
                tokens = processed_data['tokens']
                if len(tokens) >= 2: # Need at least 2 tokens for a valid sequence pair later
                    # Save as numpy array (int32 is usually sufficient for tokens)
                    np.save(output_npy_path, np.array(tokens, dtype=np.int32))
                    success_count += 1
                else:
                    print(f"Skipping {base_filename}: Too few tokens ({len(tokens)})")
                    skipped_count += 1
            else:
                print(f"Skipping {base_filename}: Processing failed or no tokens.")
                skipped_count += 1
        except Exception as e:
            print(f"\nError processing {base_filename}: {e}") # Print error with newline because of tqdm
            skipped_count += 1

    print("\n--- Pre-processing Summary ---")
    print(f"Successfully processed and saved: {success_count} files.")
    print(f"Skipped (errors, too short, etc.): {skipped_count} files.")
    print(f"Pre-processed sequences saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process MIDI files into token sequences (.npy).")
    parser.add_argument("midi_dir", help="Directory containing input MIDI files.")
    parser.add_argument("output_dir", help="Directory to save the output .npy files.")
    parser.add_argument("--walk", action="store_true", help="Recursively search for MIDI files in subdirectories.", default=False)
    parser.add_argument("--override", action="store_true", help="Override existing .npy files if they exist.", default=False)
    # Add arguments for MIDIProcessor configuration if needed, or use defaults
    parser.add_argument("--time_step", type=float, default=0.01, help="Time step for quantization.")
    parser.add_argument("--velocity_bins", type=int, default=32, help="Number of velocity bins.")
    parser.add_argument("--max_time_shift", type=float, default=10.0, help="Max time shift in seconds.")
    parser.add_argument("--max_local_inst", type=int, default=16, help="Max local instruments per file.")

    args = parser.parse_args()

    # Pack processor config from args
    proc_config = {
        'time_step': args.time_step,
        'velocity_bins': args.velocity_bins,
        'max_time_shift_seconds': args.max_time_shift,
        'max_local_instruments': args.max_local_inst
    }

    preprocess_directory(args.midi_dir, args.output_dir, proc_config, args.walk, args.override)