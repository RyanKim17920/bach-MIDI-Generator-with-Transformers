import os
import logging
import random
from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset
import numpy as np

# Configure logging - could be configured externally in a larger app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseChunkingDataset(Dataset):
    """
    Base class for MIDI datasets providing chunking logic.

    Subclasses must implement a loading method (e.g., _load_files)
    that populates self.file_tokens and self.file_names.
    """
    def __init__(self,
                 sequence_length: int,
                 pad_idx: int = 0,
                 augmentation_shift: int = 0):
        """
        Args:
            sequence_length: The desired context length (B) for model sequences.
            pad_idx: The index used for padding sequences.
            augmentation_shift: Max random shift (+/-) applied to middle chunks.
        """
        self.sequence_length = sequence_length # B
        self.pad_idx = pad_idx
        self.augmentation_shift = augmentation_shift
        self.apply_augmentation = augmentation_shift > 0

        # Populated by subclass's loading method
        self.file_tokens: List[Union[np.ndarray, List[int]]] = []
        self.file_names: List[str] = []

        # Calculated after loading
        self.chunk_info: List[Tuple[int, str, int]] = [] # (file_idx, chunk_type, base_start_idx)

        # Subclass __init__ should call _calculate_chunk_info() after loading data

    def _calculate_chunk_info(self):
        """
        Calculates the base information for each chunk across all loaded files.
        Chunks represent potential starting points for sequences of length B+1.
        """
        self.chunk_info = []
        B = self.sequence_length # Target sequence length for the model input/output
        total_token_count = 0

        logging.info("Calculating chunk information...")
        for file_idx, tokens in enumerate(self.file_tokens):
            A = len(tokens) # Total length of the current file's tokens
            total_token_count += A

            # A sequence pair (input x, target y) requires B+1 tokens.
            if A < B + 1:
                # Case 1: Shorter than sequence length + 1 -> Pad
                # Still need at least 2 tokens (should be ensured by loading logic)
                if A >= 2:
                    self.chunk_info.append((file_idx, 'pad', 0))
                # else: Skip file if less than 2 tokens (should have been skipped during load)
            else:
                # Case 2: Long enough for at least one full sequence
                self.chunk_info.append((file_idx, 'start', 0)) # Fixed start chunk

                # Middle chunks: Iterate through possible base start indices k*B
                last_possible_middle_base_start = A - (B + 1) # Last possible start index overall
                k = 1
                while (k * B) <= last_possible_middle_base_start:
                    # Ensure middle chunk start is distinct from fixed start chunk
                    if k * B > 0:
                         self.chunk_info.append((file_idx, 'middle', k * B))
                    k += 1

                # Fixed end chunk: Starts exactly B+1 tokens from the end
                end_chunk_start_index = A - (B + 1)

                # Avoid adding duplicate chunk if end overlaps start/middle perfectly
                last_added_start_for_file = -1
                if self.chunk_info and self.chunk_info[-1][0] == file_idx:
                    last_added_start_for_file = self.chunk_info[-1][2]

                if end_chunk_start_index > last_added_start_for_file:
                    self.chunk_info.append((file_idx, 'end', end_chunk_start_index))

        logging.info(f"Finished calculating chunk information. Total tokens: {total_token_count}. Total chunks: {len(self.chunk_info)}")
        if not self.chunk_info:
            logging.warning("No valid chunks could be generated from the loaded data. Dataset is empty.")
        else:
            logging.info(f"Dataset initialized. Number of files contributing chunks: {len(self.file_tokens)}. Total chunks: {len(self.chunk_info)}.")


    def __len__(self) -> int:
        """Returns the total number of chunks available in the dataset."""
        return len(self.chunk_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a specific chunk, applying augmentation or padding as needed,
        and returns the input (x) and target (y) tensors.
        """
        if not 0 <= idx < len(self.chunk_info):
            raise IndexError(f"Index {idx} out of bounds for chunk_info with length {len(self.chunk_info)}")

        file_idx, chunk_type, base_start_index = self.chunk_info[idx]

        if not 0 <= file_idx < len(self.file_tokens):
            logging.error(f"Invalid file_idx {file_idx} encountered in chunk_info at index {idx}. Max file index: {len(self.file_tokens)-1}")
            raise RuntimeError(f"Invalid dataset state: file_idx {file_idx} out of bounds.")

        tokens = self.file_tokens[file_idx] # list, np.ndarray
        A = len(tokens)
        B = self.sequence_length
        S = self.augmentation_shift

        actual_start = base_start_index

        # --- Determine the actual starting index for slicing ---
        if chunk_type == 'middle' and self.apply_augmentation and S > 0:
            # Calculate valid shift range
            min_shift = -base_start_index
            max_shift = A - base_start_index - (B + 1) # Ensure actual_start + B + 1 <= A

            final_min_shift = max(-S, min_shift)
            final_max_shift = min(S, max_shift)

            if final_max_shift >= final_min_shift:
                shift = random.randint(final_min_shift, final_max_shift)
                actual_start = base_start_index + shift

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
        else:
            # For 'start', 'middle', 'end'. A is guaranteed >= B + 1.
            actual_start = max(0, min(actual_start, A - (B + 1))) # Clamp start index
            end_slice = actual_start + B + 1
            final_chunk = tokens[actual_start : end_slice] # Slice list or numpy array

            # Sanity check and potential padding if slicing failed (shouldn't happen with clamping)
            if len(final_chunk) != B + 1:
                 logging.warning(f"Chunk extraction resulted in unexpected length. File: {self.file_names[file_idx]}, Type: {chunk_type}, BaseStart: {base_start_index}, ActualStart: {actual_start}, A: {A}, B: {B}. Len(chunk): {len(final_chunk)}. Attempting padding.")
                 chunk_list = list(final_chunk) if isinstance(final_chunk, np.ndarray) else list(final_chunk)
                 needed_padding = (B + 1) - len(chunk_list)
                 if needed_padding > 0:
                      chunk_list.extend([self.pad_idx] * needed_padding)
                 final_chunk = chunk_list[:B+1] # Ensure length B+1

        # --- Create input (x) and target (y) sequences ---
        # Ensure final_chunk is suitable for slicing (list or numpy array)
        if isinstance(final_chunk, np.ndarray):
            x = final_chunk[:-1]
            y = final_chunk[1:]
        elif isinstance(final_chunk, list):
            x = final_chunk[:-1]
            y = final_chunk[1:]
        else:
            # This case should ideally not be reached if final_chunk is always list/ndarray
            logging.error(f"Unexpected type for final_chunk: {type(final_chunk)}. Cannot create x, y.")
            # Return dummy tensors or raise error
            x = [self.pad_idx] * B
            y = [self.pad_idx] * B


        # --- Convert to tensors ---
        try:
            # torch.tensor handles both lists and numpy arrays
            input_tensor = torch.tensor(x, dtype=torch.long)
            target_tensor = torch.tensor(y, dtype=torch.long)
        except Exception as e:
             logging.error(f"Error converting chunk to tensor. File: {self.file_names[file_idx]}, Type: {chunk_type}. Error: {e}", exc_info=True)
             input_tensor = torch.full((B,), self.pad_idx, dtype=torch.long)
             target_tensor = torch.full((B,), self.pad_idx, dtype=torch.long)

        # Final shape check
        if input_tensor.shape != (B,) or target_tensor.shape != (B,):
             logging.error(f"Tensor shape mismatch after processing. File: {self.file_names[file_idx]}, Type: {chunk_type}. Input shape: {input_tensor.shape}, Target shape: {target_tensor.shape}. Expected: ({B},)")
             input_tensor = torch.full((B,), self.pad_idx, dtype=torch.long)
             target_tensor = torch.full((B,), self.pad_idx, dtype=torch.long)

        return input_tensor, target_tensor