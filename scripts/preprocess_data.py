import os
import logging
import yaml
from tqdm import tqdm
from src.midi_processor import MIDIProcessor

def load_config(path="config/config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def preprocess_directory(midi_dir: str,
                         output_dir: str,
                         processor: MIDIProcessor,
                         walk: bool = True,
                         override: bool = False,
                         verbose: bool = False):
    logger = logging.getLogger(__name__)
    if not os.path.isdir(midi_dir):
        logger.error(f"Raw MIDI directory not found: {midi_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)

    files = []
    if walk:
        for root, _, fnames in os.walk(midi_dir):
            for fn in fnames:
                if fn.lower().endswith(('.mid', '.midi')):
                    files.append(os.path.join(root, fn))
    else:
        files = [os.path.join(midi_dir, f)
                 for f in os.listdir(midi_dir)
                 if f.lower().endswith(('.mid', '.midi'))]

    logger.debug(f"Processing {len(files)} MIDI files to '{output_dir}'")
    for path in tqdm(files):
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, f"{name}.npy")
        if os.path.exists(out_path) and not override:
            logger.debug(f"Skipping existing: {out_path}")
            continue
        res = processor.process_midi_file(path)
        if res and 'tokens' in res:
            try:
                import numpy as np
                np.save(out_path, res['tokens'])
                logger.debug(f"Saved tokens for {name}")
            except Exception as e:
                logger.error(f"Failed saving {out_path}: {e}")

if __name__ == "__main__":
    # Load config and set log level
    cfg = load_config()
    # New: Get logging level from config string
    log_level_str = cfg.get('logging_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO) # Default to INFO if invalid

    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    # Initialize MIDIProcessor
    raw_dir = cfg['raw_midi_dir']
    pre_dir = cfg['preprocessed_dir']
    walk_cfg = cfg.get('walk', True)
    override_cfg = cfg.get('override', False)
    proc = MIDIProcessor(**cfg['processor'])
    preprocess_directory(raw_dir, pre_dir, proc,
                         walk=walk_cfg,
                         override=override_cfg,
                         verbose=cfg.get('verbose', False))