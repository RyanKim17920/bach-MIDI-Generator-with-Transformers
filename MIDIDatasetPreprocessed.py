import os
import logging
from typing import List, Optional
import torch
import numpy as np
from base_dataset import BaseChunkingDataset 


class MIDIDatasetPreprocessed(BaseChunkingDataset):
    """
    PyTorch Dataset that loads pre-processed MIDI token sequences (.npy files).
    Inherits chunking logic from BaseChunkingDataset.
    """
    def __init__(self,
                 sequence_length: int,
                 preprocessed_dir: Optional[str] = None,
                 npy_file_paths: Optional[List[str]] = None,
                 pad_idx: int = 0,
                 augmentation_shift: int = 0):
        """
        Args:
            sequence_length: The desired context length (B) for model sequences.
            preprocessed_dir: Path to the directory containing .npy token files.
                               (Either this or npy_file_paths must be provided).
            npy_file_paths: A list of direct paths to .npy files.
                            (Either this or preprocessed_dir must be provided).
            pad_idx: The index used for padding sequences.
            augmentation_shift: Max random shift (+/-) applied to middle chunks.
        """
        # Initialize base class attributes
        super().__init__(sequence_length, pad_idx, augmentation_shift)

        if preprocessed_dir is None and npy_file_paths is None:
            raise ValueError("Must provide either preprocessed_dir or npy_file_paths.")
        if preprocessed_dir is not None and npy_file_paths is not None:
            logging.warning("Both preprocessed_dir and npy_file_paths provided. Using preprocessed_dir.")
            npy_file_paths = None # Prioritize directory if both given
        if preprocessed_dir is not None and not os.path.isdir(preprocessed_dir):
             raise FileNotFoundError(f"Preprocessed data directory not found: {preprocessed_dir}")

        self.preprocessed_dir = preprocessed_dir
        self.npy_file_paths = npy_file_paths

        logging.info(f"Initializing MIDIDatasetPreprocessed...")
        self._load_preprocessed_files() # Load .npy files
        self._calculate_chunk_info()    # Calculate chunks based on loaded data

    def _load_preprocessed_files(self):
        """Loads token sequences from .npy files."""
        loaded_count = 0
        skipped_count = 0
        files_to_load = []

        if self.preprocessed_dir:
            logging.info(f"Scanning directory: {self.preprocessed_dir}")
            try:
                for filename in sorted(os.listdir(self.preprocessed_dir)):
                    if filename.lower().endswith('.npy'):
                        files_to_load.append(os.path.join(self.preprocessed_dir, filename))
            except Exception as e:
                logging.error(f"Error scanning directory {self.preprocessed_dir}: {e}", exc_info=True)
                return # Cannot proceed if directory scan fails
        elif self.npy_file_paths:
            logging.info(f"Using provided list of {len(self.npy_file_paths)} NPY files.")
            files_to_load = self.npy_file_paths
        else:
             logging.error("No data source specified (directory or file list).") # Should be caught by __init__ check
             return

        logging.info(f"Attempting to load {len(files_to_load)} .npy files...")
        for npy_path in files_to_load:
            filename = os.path.basename(npy_path)
            try:
                if not os.path.exists(npy_path):
                     logging.warning(f"File not found, skipping: {npy_path}")
                     skipped_count += 1
                     continue

                tokens = np.load(npy_path)
                # Basic validation
                if isinstance(tokens, np.ndarray) and np.issubdtype(tokens.dtype, np.integer) and tokens.ndim == 1:
                    if len(tokens) >= 2: # Need at least 2 tokens for a pair
                        self.file_tokens.append(tokens)
                        self.file_names.append(os.path.splitext(filename)[0]) # Store base name
                        loaded_count += 1
                    else:
                        logging.warning(f"Skipping {filename}: Too few tokens ({len(tokens)})")
                        skipped_count += 1
                else:
                    logging.warning(f"Skipping {filename}: Invalid format (expected 1D integer numpy array, got {tokens.dtype}, {tokens.ndim}D).")
                    skipped_count += 1
            except Exception as e:
                logging.error(f"Error loading {filename} from {npy_path}: {e}", exc_info=True)
                skipped_count += 1

        logging.info(f"Finished loading preprocessed files. Successfully loaded: {loaded_count}. Skipped: {skipped_count}.")
