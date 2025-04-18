import os
import logging
from typing import List
import torch
import numpy as np

# Use relative imports for modules within the same package (src)
try:
    from .midi_processor import MIDIProcessor
    from .base_dataset import BaseChunkingDataset
    from .constants import PAD_IDX
except ImportError:
    # Fallback for scenarios where the script might be run directly or package structure isn't recognized
    logging.error("Failed relative imports. Ensure running as part of the 'src' package or check PYTHONPATH.")
    # Attempt absolute imports as a fallback (less ideal for package structure)
    try:
        from midi_processor import MIDIProcessor
        from base_dataset import BaseChunkingDataset
        from constants import PAD_IDX
    except ImportError:
        logging.critical("Cannot import required modules (MIDIProcessor, BaseChunkingDataset, constants). Exiting.")
        raise # Re-raise the critical import error

# Get logger for this module
logger = logging.getLogger(__name__)

class MIDIDataset(BaseChunkingDataset):
    """
    PyTorch Dataset for loading and chunking MIDI token sequences.

    Processes MIDI files on-the-fly using a provided MIDIProcessor instance.
    Inherits chunking logic from BaseChunkingDataset.
    """
    def __init__(self,
                 midi_file_paths: List[str],
                 processor: MIDIProcessor,
                 sequence_length: int,
                 pad_idx: int = PAD_IDX, # Use constant from src/constants.py
                 augmentation_shift: int = 0):
        """
        Args:
            midi_file_paths: List of paths to MIDI files.
            processor: An initialized instance of the MIDIProcessor.
            sequence_length: The desired context length (B) for model sequences.
            pad_idx: The index used for padding sequences. Defaults to PAD_IDX.
            augmentation_shift: Max random shift (+/-) applied to middle chunks.
        """
        # Initialize base class attributes (sequence_length, pad_idx, augmentation_shift)
        # Pass the pad_idx received (which defaults to the constant)
        super().__init__(sequence_length=sequence_length,
                         pad_idx=pad_idx,
                         augmentation_shift=augmentation_shift)

        if not isinstance(processor, MIDIProcessor):
            # Use logger for errors
            logger.error("Invalid argument: 'processor' must be an instance of MIDIProcessor.")
            raise TypeError("processor must be an instance of MIDIProcessor")
        self.processor = processor

        logger.info(f"Initializing MIDIDataset (on-the-fly processing) for {len(midi_file_paths)} files...")
        # Load data and populate self.file_tokens and self.file_names
        self._load_and_process_files(midi_file_paths)
        # Calculate chunk info based on loaded data using the base class method
        self._calculate_chunk_info()

    def _load_and_process_files(self, midi_file_paths: List[str]):
        """
        Loads MIDI files, processes them using self.processor, and stores
        valid token sequences in self.file_tokens and filenames in self.file_names.
        """
        # Reset lists in case of re-initialization (though typically not done)
        self.file_tokens = []
        self.file_names = []
        processed_count = 0
        skipped_count = 0
        logger.info(f"Starting MIDI file processing...")

        for file_path in midi_file_paths:
            if not isinstance(file_path, str) or not file_path:
                 logger.warning(f"Invalid file path encountered (type: {type(file_path)}), skipping.")
                 skipped_count += 1
                 continue

            if not os.path.exists(file_path):
                logger.warning(f"File not found, skipping: {file_path}")
                skipped_count += 1
                continue

            filename = os.path.basename(file_path)
            try:
                # Use the passed processor instance
                processed_data = self.processor.process_midi_file(file_path)

                if processed_data and 'tokens' in processed_data:
                    tokens = processed_data['tokens'] # list or numpy array

                    # Basic validation of the returned tokens
                    if not hasattr(tokens, '__len__') or not hasattr(tokens, '__getitem__'):
                         logger.warning(f"Processor returned non-sequence data for {filename}, skipping.")
                         skipped_count += 1
                         continue

                    # Need at least 2 tokens to form an input/target pair for training
                    if len(tokens) >= 2:
                        self.file_tokens.append(tokens)
                        self.file_names.append(filename) # Store just the base name
                        processed_count += 1
                        logger.debug(f"Successfully processed {filename} -> {len(tokens)} tokens.")
                    else:
                        logger.warning(f"File processed but too short (< 2 tokens), skipping: {filename} (Length: {len(tokens)})")
                        skipped_count += 1
                else:
                    # Handle cases where processing might return None or an empty dict
                    logger.warning(f"Failed to process or empty tokens returned for: {filename}")
                    skipped_count += 1
            # Catch potential errors during MIDI processing (e.g., corrupted files mentioned in README)
            except Exception as e:
                 logger.error(f"Error processing file {filename} ({file_path}): {e}", exc_info=True) # Log traceback
                 skipped_count += 1

        logger.info(f"Finished processing files. Successfully processed and kept: {processed_count}. Skipped: {skipped_count}.")