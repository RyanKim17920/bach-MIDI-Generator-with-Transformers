import os
import logging
from typing import List, Optional
import torch
import numpy as np

# Use relative imports for modules within the same package (src)
try:
    from .base_dataset import BaseChunkingDataset
    from .constants import PAD_IDX
except ImportError:
    # Fallback for scenarios where the script might be run directly
    logging.error("Failed relative imports. Ensure running as part of the 'src' package.")
    try:
        from base_dataset import BaseChunkingDataset
        from constants import PAD_IDX
    except ImportError:
        logging.critical("Cannot import required modules (BaseChunkingDataset, constants). Exiting.")
        raise

# Get logger for this module
logger = logging.getLogger(__name__)

class MIDIDatasetPreprocessed(BaseChunkingDataset):
    """
    PyTorch Dataset that loads pre-processed MIDI token sequences (.npy files).
    Inherits chunking logic from BaseChunkingDataset.
    """
    def __init__(self,
                 sequence_length: int,
                 preprocessed_dir: Optional[str] = None,
                 npy_file_paths: Optional[List[str]] = None,
                 pad_idx: int = PAD_IDX, # Use constant from src/constants.py
                 augmentation_shift: int = 0):
        """
        Args:
            sequence_length: The desired context length (B) for model sequences.
            preprocessed_dir: Path to the directory containing .npy token files.
                               (Either this or npy_file_paths must be provided).
            npy_file_paths: A list of direct paths to .npy files.
                            (Either this or preprocessed_dir must be provided).
            pad_idx: The index used for padding sequences. Defaults to PAD_IDX.
            augmentation_shift: Max random shift (+/-) applied to middle chunks.
        """
        # Initialize base class attributes (sequence_length, pad_idx, augmentation_shift)
        # Pass the pad_idx received (which defaults to the constant)
        super().__init__(sequence_length=sequence_length,
                         pad_idx=pad_idx,
                         augmentation_shift=augmentation_shift)

        # Validate input sources
        if preprocessed_dir is None and npy_file_paths is None:
            logger.error("Initialization failed: Must provide either preprocessed_dir or npy_file_paths.")
            raise ValueError("Must provide either preprocessed_dir or npy_file_paths.")
        if preprocessed_dir is not None and npy_file_paths is not None:
            logger.warning("Both preprocessed_dir and npy_file_paths provided. Using preprocessed_dir.")
            npy_file_paths = None # Prioritize directory
        if preprocessed_dir is not None and not os.path.isdir(preprocessed_dir):
             logger.error(f"Initialization failed: Preprocessed data directory not found: {preprocessed_dir}")
             raise FileNotFoundError(f"Preprocessed data directory not found: {preprocessed_dir}")

        self.preprocessed_dir = preprocessed_dir
        self.npy_file_paths = npy_file_paths

        logger.info(f"Initializing MIDIDatasetPreprocessed...")
        # Load data and populate self.file_tokens and self.file_names
        self._load_preprocessed_files()
        # Calculate chunk info based on loaded data using the base class method
        self._calculate_chunk_info()

    def _load_preprocessed_files(self):
        """Loads token sequences from .npy files specified by directory or list."""
        # Reset lists in case of re-initialization
        self.file_tokens = []
        self.file_names = []
        loaded_count = 0
        skipped_count = 0
        files_to_load = []

        # Determine the list of files to load
        if self.preprocessed_dir:
            logger.info(f"Scanning directory for .npy files: {self.preprocessed_dir}")
            try:
                # Use list comprehension for potentially better performance on large dirs
                files_to_load = [
                    os.path.join(self.preprocessed_dir, f)
                    for f in os.listdir(self.preprocessed_dir)
                    if os.path.isfile(os.path.join(self.preprocessed_dir, f)) and f.lower().endswith('.npy')
                ]
                files_to_load.sort() # Ensure consistent order
                logger.info(f"Found {len(files_to_load)} .npy files in directory.")
            except Exception as e:
                logger.error(f"Error scanning directory {self.preprocessed_dir}: {e}", exc_info=True)
                # Do not proceed if directory scan fails
                return
        elif self.npy_file_paths:
            logger.info(f"Using provided list of {len(self.npy_file_paths)} NPY file paths.")
            files_to_load = self.npy_file_paths
            # Filter out non-existent or non-file paths from the provided list upfront
            initial_count = len(files_to_load)
            files_to_load = [f for f in files_to_load if isinstance(f, str) and os.path.isfile(f)]
            removed_count = initial_count - len(files_to_load)
            if removed_count > 0:
                logger.warning(f"Removed {removed_count} invalid or non-existent file paths from the provided list.")
        else:
             # This state should not be reachable due to __init__ checks, but handle defensively
             logger.error("No data source specified (directory or file list) for loading.")
             return

        if not files_to_load:
            logger.warning("No .npy files found or specified to load.")
            return

        logger.info(f"Attempting to load data from {len(files_to_load)} .npy files...")
        for npy_path in files_to_load:
            filename = os.path.basename(npy_path)
            try:
                # Load the numpy array
                tokens = np.load(npy_path)

                # --- Validation ---
                if not isinstance(tokens, np.ndarray):
                    logger.warning(f"Skipping {filename}: Data is not a numpy array (type: {type(tokens)}).")
                    skipped_count += 1
                    continue
                if not np.issubdtype(tokens.dtype, np.integer):
                    logger.warning(f"Skipping {filename}: Expected integer dtype, found {tokens.dtype}.")
                    skipped_count += 1
                    continue
                if tokens.ndim != 1:
                    logger.warning(f"Skipping {filename}: Expected 1D array, found {tokens.ndim} dimensions.")
                    skipped_count += 1
                    continue
                if len(tokens) < 2: # Need at least 2 tokens for a training pair
                    logger.warning(f"Skipping {filename}: Too few tokens ({len(tokens)} < 2).")
                    skipped_count += 1
                    continue
                # --- End Validation ---

                # If all checks pass, add the data
                self.file_tokens.append(tokens)
                self.file_names.append(os.path.splitext(filename)[0]) # Store base name without extension
                loaded_count += 1
                logger.debug(f"Successfully loaded {filename} ({len(tokens)} tokens).")

            except FileNotFoundError:
                # Should be caught by initial check if using list, but handle just in case
                logger.warning(f"File not found during load attempt (should have been filtered?): {npy_path}")
                skipped_count += 1
            except Exception as e:
                logger.error(f"Error loading or validating {filename} from {npy_path}: {e}", exc_info=True)
                skipped_count += 1

        logger.info(f"Finished loading preprocessed files. Successfully loaded: {loaded_count}. Skipped: {skipped_count}.")
