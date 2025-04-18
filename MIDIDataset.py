import os
import logging
from typing import List
import torch
import numpy as np
from MIDIprocessor import MIDIProcessor
from base_dataset import BaseChunkingDataset 

# Logging is configured in base_dataset.py

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
                 pad_idx: int = 0,
                 augmentation_shift: int = 0):
        """
        Args:
            midi_file_paths: List of paths to MIDI files.
            processor: An initialized instance of the MIDIProcessor.
            sequence_length: The desired context length (B) for model sequences.
            pad_idx: The index used for padding sequences.
            augmentation_shift: Max random shift (+/-) applied to middle chunks.
        """
        # Initialize base class attributes
        super().__init__(sequence_length, pad_idx, augmentation_shift)

        if not isinstance(processor, MIDIProcessor):
            raise TypeError("processor must be an instance of MIDIProcessor")
        self.processor = processor

        logging.info(f"Initializing MIDIDataset (on-the-fly processing)...")
        self._load_and_process_files(midi_file_paths)
        self._calculate_chunk_info() # Call base class method to calculate chunks

    def _load_and_process_files(self, midi_file_paths: List[str]):
        """Loads MIDI files, processes them using self.processor, and stores token sequences."""
        processed_count = 0
        skipped_count = 0
        logging.info(f"Processing {len(midi_file_paths)} MIDI file paths...")
        for file_path in midi_file_paths:
            if not os.path.exists(file_path):
                logging.warning(f"File not found, skipping: {file_path}")
                skipped_count += 1
                continue
            try:
                processed_data = self.processor.process_midi_file(file_path)
                if processed_data and 'tokens' in processed_data:
                    tokens = processed_data['tokens'] # list or numpy array
                    # Need at least 2 tokens to form an input/target pair
                    if len(tokens) >= 2:
                        self.file_tokens.append(tokens)
                        self.file_names.append(os.path.basename(file_path))
                        processed_count += 1
                    else:
                        logging.warning(f"File processed but too short (< 2 tokens), skipping: {file_path} (Length: {len(tokens)})")
                        skipped_count += 1
                else:
                    logging.warning(f"Failed to process or empty tokens returned for: {file_path}")
                    skipped_count += 1
            except Exception as e:
                 logging.error(f"Error processing file {file_path}: {e}", exc_info=True)
                 skipped_count += 1

        logging.info(f"Finished processing files. Successfully processed and kept: {processed_count}. Skipped: {skipped_count}.")
