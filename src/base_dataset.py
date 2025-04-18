import os
import logging
import random
from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset
import numpy as np

# Use relative import for constants within the same package
try:
    from .constants import PAD_IDX
except ImportError:
    # Fallback for scenarios where the script might be run directly
    logging.error("Failed relative import of constants. Ensure running as part of the 'src' package.")
    try:
        from constants import PAD_IDX
    except ImportError:
        logging.critical("Cannot import PAD_IDX from constants.")
        raise

# Get logger for this module
logger = logging.getLogger(__name__)
# Basic logging config (can be overridden by application-level config)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseChunkingDataset(Dataset):
    """
    Base class for MIDI datasets providing chunking logic.

    Subclasses must implement a loading method (e.g., _load_files)
    that populates self.file_tokens and self.file_names.
    """
    def __init__(self,
                 sequence_length: int,
                 pad_idx: int = PAD_IDX, # Use constant as default
                 augmentation_shift: int = 0):
        """
        Args:
            sequence_length: The desired context length (B) for model sequences.
            pad_idx: The index used for padding sequences. Defaults to PAD_IDX.
            augmentation_shift: Max random shift (+/-) applied to middle chunks.
        """
        if not isinstance(sequence_length, int) or sequence_length <= 0:
             logger.error(f"Invalid sequence_length: {sequence_length}. Must be a positive integer.")
             raise ValueError("sequence_length must be a positive integer")
        if not isinstance(pad_idx, int) or pad_idx < 0:
             logger.error(f"Invalid pad_idx: {pad_idx}. Must be a non-negative integer.")
             raise ValueError("pad_idx must be a non-negative integer")
        if not isinstance(augmentation_shift, int) or augmentation_shift < 0:
             logger.error(f"Invalid augmentation_shift: {augmentation_shift}. Must be a non-negative integer.")
             raise ValueError("augmentation_shift must be a non-negative integer")

        self.sequence_length = sequence_length # B
        self.pad_idx = pad_idx
        self.augmentation_shift = augmentation_shift
        self.apply_augmentation = augmentation_shift > 0

        # Populated by subclass's loading method
        self.file_tokens: List[Union[np.ndarray, List[int]]] = []
        self.file_names: List[str] = []

        # Calculated after loading
        # Stores info needed to retrieve a chunk: (file_index, chunk_type, base_start_index)
        self.chunk_info: List[Tuple[int, str, int]] = []

        logger.info(f"BaseChunkingDataset initialized with seq_len={sequence_length}, pad_idx={pad_idx}, aug_shift={augmentation_shift}")
        # Subclass __init__ should call _calculate_chunk_info() after loading data

    def _calculate_chunk_info(self):
        """
        Calculates the base information for each chunk across all loaded files.
        Chunks represent potential starting points for sequences of length B+1.
        """
        self.chunk_info = []
        B = self.sequence_length # Target sequence length for the model input/output
        total_token_count = 0
        files_contributing = 0

        logger.info("Calculating chunk information...")
        for file_idx, tokens in enumerate(self.file_tokens):
            try:
                A = len(tokens) # Total length of the current file's tokens
            except TypeError:
                logger.warning(f"File at index {file_idx} ('{self.file_names[file_idx]}') has non-sequence tokens ({type(tokens)}). Skipping.")
                continue

            total_token_count += A
            file_chunks_added = 0

            # A sequence pair (input x, target y) requires B+1 tokens.
            if A < B + 1:
                # Case 1: Shorter than sequence length + 1 -> Pad
                # Still need at least 2 tokens (should be ensured by loading logic)
                if A >= 2:
                    self.chunk_info.append((file_idx, 'pad', 0))
                    file_chunks_added += 1
                else:
                    logger.debug(f"File '{self.file_names[file_idx]}' too short ({A} tokens) for even a padded chunk (needs >= 2). Skipping.")
            else:
                # Case 2: Long enough for at least one full sequence
                self.chunk_info.append((file_idx, 'start', 0)) # Fixed start chunk
                file_chunks_added += 1

                # Middle chunks: Iterate through possible base start indices k*B
                # Last possible start index for a full B+1 sequence is A - (B + 1)
                last_possible_start_index = A - (B + 1)
                k = 1
                while True:
                    base_start_index = k * B
                    # Ensure the base start index allows a full sequence
                    if base_start_index <= last_possible_start_index:
                        # Ensure middle chunk start is distinct from fixed start chunk (k=0)
                        if base_start_index > 0:
                             self.chunk_info.append((file_idx, 'middle', base_start_index))
                             file_chunks_added += 1
                        k += 1
                    else:
                        break # No more middle chunks possible starting at k*B

                # Fixed end chunk: Starts exactly B+1 tokens from the end
                end_chunk_start_index = A - (B + 1)

                # Avoid adding duplicate chunk if end overlaps start/middle perfectly
                # This happens if A = B+1, where start and end chunks are the same (index 0).
                # Or if A = 2*B+1, where first middle chunk (k=1, start=B) is same as end chunk.
                last_added_start_for_file = -1
                # Check the start index of the *last* chunk added *for this specific file*
                if file_chunks_added > 0 and self.chunk_info[-1][0] == file_idx:
                    last_added_start_for_file = self.chunk_info[-1][2]

                # Only add the end chunk if its start index is strictly greater than the last one added for this file.
                if end_chunk_start_index > last_added_start_for_file:
                    self.chunk_info.append((file_idx, 'end', end_chunk_start_index))
                    file_chunks_added += 1

            if file_chunks_added > 0:
                files_contributing += 1

        logger.info(f"Finished calculating chunk information.")
        logger.info(f"  Total tokens across {len(self.file_tokens)} files: {total_token_count}")
        logger.info(f"  Files contributing chunks: {files_contributing}")
        logger.info(f"  Total chunks generated: {len(self.chunk_info)}")

        if not self.chunk_info:
            logger.warning("No valid chunks could be generated from the loaded data. Dataset will be empty.")


    def __len__(self) -> int:
        """Returns the total number of chunks available in the dataset."""
        return len(self.chunk_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a specific chunk, applying augmentation or padding as needed,
        and returns the input (x) and target (y) tensors.
        """
        if not 0 <= idx < len(self.chunk_info):
            logger.error(f"Index {idx} out of bounds for chunk_info with length {len(self.chunk_info)}")
            raise IndexError(f"Index {idx} out of bounds for chunk_info with length {len(self.chunk_info)}")

        try:
            file_idx, chunk_type, base_start_index = self.chunk_info[idx]

            if not 0 <= file_idx < len(self.file_tokens):
                logger.error(f"Invalid file_idx {file_idx} found in chunk_info at index {idx}. Max file index: {len(self.file_tokens)-1}")
                # This indicates a bug in _calculate_chunk_info or data corruption
                raise RuntimeError(f"Invalid dataset state: file_idx {file_idx} out of bounds.")

            tokens = self.file_tokens[file_idx] # list, np.ndarray
            filename = self.file_names[file_idx] if file_idx < len(self.file_names) else f"File_{file_idx}"
            A = len(tokens)
            B = self.sequence_length
            S = self.augmentation_shift

            actual_start = base_start_index

            # --- Determine the actual starting index for slicing ---
            if chunk_type == 'middle' and self.apply_augmentation and S > 0:
                # Calculate valid shift range to keep the slice [actual_start, actual_start + B + 1) within bounds [0, A)
                # Constraint 1: actual_start >= 0  => base_start + shift >= 0 => shift >= -base_start
                min_shift = -base_start_index
                # Constraint 2: actual_start + B + 1 <= A => base_start + shift + B + 1 <= A => shift <= A - base_start - B - 1
                max_shift = A - base_start_index - (B + 1)

                # Combine with user-defined augmentation shift limit S
                final_min_shift = max(-S, min_shift)
                final_max_shift = min(S, max_shift)

                if final_max_shift >= final_min_shift:
                    shift = random.randint(final_min_shift, final_max_shift)
                    actual_start = base_start_index + shift
                    logger.debug(f"Augmenting middle chunk {idx} (file {filename}): base={base_start_index}, shift={shift}, actual={actual_start}")
                else:
                    logger.debug(f"No valid augmentation shift for middle chunk {idx} (file {filename}): base={base_start_index}, range=[{final_min_shift}, {final_max_shift}]")


            # --- Extract the chunk of B+1 tokens ---
            final_chunk = None # Initialize
            if chunk_type == 'pad':
                # File is shorter than B+1. Take all available tokens and pad.
                # Convert to list if numpy array for easier padding
                chunk_list = list(tokens) if isinstance(tokens, np.ndarray) else list(tokens[:A])
                needed_padding = (B + 1) - len(chunk_list)
                if needed_padding > 0:
                    chunk_list.extend([self.pad_idx] * needed_padding)
                final_chunk = chunk_list[:B+1] # Ensure length B+1
                logger.debug(f"Padding chunk {idx} (file {filename}): original_len={A}, padded_len={len(final_chunk)}")
            else:
                # For 'start', 'middle', 'end'. A is guaranteed >= B + 1.
                # Clamp actual_start just in case (shouldn't be needed with correct shift calculation)
                actual_start = max(0, min(actual_start, A - (B + 1)))
                end_slice = actual_start + B + 1
                final_chunk = tokens[actual_start : end_slice] # Slice list or numpy array
                logger.debug(f"Extracting chunk {idx} (file {filename}, type {chunk_type}): slice=[{actual_start}:{end_slice}]")


                # Sanity check length and pad if necessary (should only happen if slicing logic fails)
                if len(final_chunk) != B + 1:
                     logger.warning(f"Chunk extraction for {filename} (idx {idx}, type {chunk_type}) resulted in unexpected length {len(final_chunk)} (expected {B+1}). Slice was [{actual_start}:{end_slice}], A={A}. Attempting padding.")
                     chunk_list = list(final_chunk) if isinstance(final_chunk, np.ndarray) else list(final_chunk)
                     needed_padding = (B + 1) - len(chunk_list)
                     if needed_padding > 0:
                          chunk_list.extend([self.pad_idx] * needed_padding)
                     final_chunk = chunk_list[:B+1] # Ensure length B+1

            # --- Create input (x) and target (y) sequences ---
            # Ensure final_chunk is suitable for slicing (list or numpy array)
            if isinstance(final_chunk, (np.ndarray, list)):
                x = final_chunk[:-1]
                y = final_chunk[1:]
            else:
                logger.error(f"Unexpected type for final_chunk: {type(final_chunk)} for chunk {idx} (file {filename}). Cannot create x, y.")
                # Return dummy tensors or raise error
                x = [self.pad_idx] * B
                y = [self.pad_idx] * B


            # --- Convert to tensors ---
            # torch.tensor handles both lists and numpy arrays
            input_tensor = torch.tensor(x, dtype=torch.long)
            target_tensor = torch.tensor(y, dtype=torch.long)

            # Final shape check
            if input_tensor.shape != (B,) or target_tensor.shape != (B,):
                 logger.error(f"Tensor shape mismatch for chunk {idx} (file {filename}). Input: {input_tensor.shape}, Target: {target_tensor.shape}. Expected: ({B},)")
                 # Handle error: return dummy tensors or raise
                 input_tensor = torch.full((B,), self.pad_idx, dtype=torch.long)
                 target_tensor = torch.full((B,), self.pad_idx, dtype=torch.long)

            return input_tensor, target_tensor

        except Exception as e:
            # Catch any unexpected error during item retrieval
            logger.error(f"Unexpected error in __getitem__ for index {idx}: {e}", exc_info=True)
            # Return dummy tensors to prevent crashing the DataLoader in most cases,
            # but log the error critically.
            B = self.sequence_length
            input_tensor = torch.full((B,), self.pad_idx, dtype=torch.long)
            target_tensor = torch.full((B,), self.pad_idx, dtype=torch.long)
            return input_tensor, target_tensor