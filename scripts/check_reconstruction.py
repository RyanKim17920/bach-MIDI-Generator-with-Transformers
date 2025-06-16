import os
import logging
import yaml
import pretty_midi
from src.midi_processor import MIDIProcessor

def load_config(path="config/config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def print_midi_info(midi_obj: pretty_midi.PrettyMIDI, label: str):
    logger = logging.getLogger(__name__)
    if midi_obj is None:
        logger.error(f"{label}: Could not load MIDI.")
        return
    total_notes = sum(len(inst.notes) for inst in midi_obj.instruments)
    end_time = midi_obj.get_end_time()
    num_inst = len(midi_obj.instruments)
    logger.info(f"{label} | Instruments: {num_inst}, Notes: {total_notes}, End Time: {end_time:.3f}s")

if __name__ == "__main__":
    cfg = load_config()
    # Get logging level from config string
    log_level_str = cfg.get('logging_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO) # Default to INFO if invalid

    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    TEST_MIDI = cfg['checker']['test_midi_file']
    OUT_DIR = cfg['checker'].get('output_dir', 'reconstruction_test_output')
    os.makedirs(OUT_DIR, exist_ok=True)

    print_midi_info(pretty_midi.PrettyMIDI(TEST_MIDI), "Original MIDI")

    proc = MIDIProcessor(**cfg['processor'])
    result = proc.process_midi_file(TEST_MIDI)
    if not result:
        logging.error("Processing failed, cannot reconstruct.")
        exit(1)

    tokens = result['tokens']
    reconstructed = proc.tokens_to_midi(tokens)
    if reconstructed:
        out_path = os.path.join(OUT_DIR, os.path.basename(TEST_MIDI))
        reconstructed.write(out_path)
        logging.info(f"Reconstructed MIDI saved to {out_path}")
    print_midi_info(reconstructed, "Reconstructed MIDI")