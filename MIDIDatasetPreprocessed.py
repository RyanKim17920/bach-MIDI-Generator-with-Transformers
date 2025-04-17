import torch
from torch.utils.data import Dataset, DataLoader
import random
import math
import os
import numpy as np
from typing import List, Dict, Tuple
from MIDIprocessor import MIDIProcessor 

class MIDIDatasetPreprocessed(Dataset): 
    """
    PyTorch Dataset that loads pre-processed MIDI token sequences (.npy files)
    and provides chunks based on the specified strategy.
    """
    def __init__(self,
                 preprocessed_dir: str, # Directory with .npy files
                 sequence_length: int,
                 pad_idx = 0, 
                 augmentation_shift = 0):
        """
        Args:
            preprocessed_dir: Path to the directory containing .npy token files.
            processor: An initialized MIDIProcessor (must match pre-processing settings).
            sequence_length: The desired context length (B) for model sequences.
            apply_augmentation: Whether to apply random shifting to middle chunks.
        """
        self.preprocessed_dir = preprocessed_dir
        self.sequence_length = sequence_length
        self.pad_idx = pad_idx

        # Calculate augmentation shift
        self.augmentation_shift = augmentation_shift
        self.apply_augmentation = (augmentation_shift > 0)
        
        self.file_tokens: List[np.ndarray] = [] # Stores loaded numpy arrays
        self.file_names: List[str] = [] # Stores original base filenames
        self.chunk_info: List[Tuple[int, str, int]] = [] # (file_idx, chunk_type, base_start_idx)

        if not os.path.isdir(preprocessed_dir):
             raise FileNotFoundError(f"Preprocessed data directory not found: {preprocessed_dir}")

        print(f"Initializing MIDIDatasetPreprocessed from: {preprocessed_dir}")
        print(f"Sequence length: {self.sequence_length}, Augmentation shift: {self.augmentation_shift}")

        self._load_preprocessed_files() # Load .npy files
        self._calculate_chunk_info()    # Calculate chunks based on loaded data

        if not self.chunk_info:
            raise ValueError("No valid chunks could be generated from the preprocessed data. Check .npy files and sequence length.")

        print(f"Dataset initialized. Number of loaded files: {len(self.file_tokens)}. Total number of chunks: {len(self.chunk_info)}.")

    def _load_preprocessed_files(self):
        """Loads token sequences from .npy files in the specified directory."""
        loaded_count = 0
        skipped_count = 0
        for filename in sorted(os.listdir(self.preprocessed_dir)): # Sort for consistency
            if filename.lower().endswith('.npy'):
                npy_path = os.path.join(self.preprocessed_dir, filename)
                try:
                    tokens = np.load(npy_path)
                    # Basic validation
                    if isinstance(tokens, np.ndarray) and np.issubdtype(tokens.dtype, np.integer) and tokens.ndim == 1:
                        if len(tokens) >= 2: # Need at least 2 tokens for a pair
                            self.file_tokens.append(tokens)
                            self.file_names.append(os.path.splitext(filename)[0]) # Store base name
                            loaded_count += 1
                        else:
                            # print(f"Skipping {filename}: Too few tokens ({len(tokens)})")
                            skipped_count += 1
                    else:
                        print(f"Skipping {filename}: Invalid format (not 1D integer numpy array).")
                        skipped_count += 1
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    skipped_count += 1

        print(f"Successfully loaded {loaded_count} .npy files.")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} files (errors or too short).")


    def _calculate_chunk_info(self):
        """Calculates the base information for each chunk across all loaded files."""
        # This method logic is IDENTICAL to the previous MIDIDataset version,
        # as it only depends on self.file_tokens and self.sequence_length.
        self.chunk_info = []
        B = self.sequence_length

        for file_idx, tokens in enumerate(self.file_tokens):
            A = len(tokens) # Length of the loaded token array

            if A < B + 1:
                # Case 1: Shorter than sequence length + 1 -> Pad
                self.chunk_info.append((file_idx, 'pad', 0))
            else:
                # Case 2: Long enough for at least one full sequence
                self.chunk_info.append((file_idx, 'start', 0)) # Fixed start chunk

                # Middle chunks
                last_possible_middle_chunk_base_start = A - (B + 1) # Last possible start index overall
                # Iterate based on base starts at k*B, ensuring they don't cause out-of-bounds access for B+1 items
                k = 1
                while (k * B) <= last_possible_middle_chunk_base_start:
                    self.chunk_info.append((file_idx, 'middle', k * B))
                    k += 1
                
                # Fixed end chunk
                end_chunk_start_index = A - (B + 1) # Start exactly B+1 from end
                
                # Avoid adding duplicate chunk if end overlaps start/middle perfectly
                last_added_start = -1
                if self.chunk_info and self.chunk_info[-1][0] == file_idx:
                     last_added_start = self.chunk_info[-1][2]

                if end_chunk_start_index > last_added_start:
                    self.chunk_info.append((file_idx, 'end', end_chunk_start_index))


    def __len__(self) -> int:
        """Returns the total number of chunks available."""
        return len(self.chunk_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a specific chunk, applying augmentation or padding.
        Loads data from pre-processed files.
        """
        # This method logic is IDENTICAL to the previous MIDIDataset version,
        # except self.file_tokens now contains numpy arrays. Indexing/slicing
        # works the same way.
        file_idx, chunk_type, base_start_index = self.chunk_info[idx]
        tokens = self.file_tokens[file_idx] # This is now a numpy array
        A = len(tokens)
        B = self.sequence_length
        S = self.augmentation_shift

        actual_start = base_start_index

        # Apply augmentation to middle chunks
        if chunk_type == 'middle' and self.apply_augmentation and S > 0:
            min_shift = max(-S, -base_start_index)
            max_shift = min(S, A - base_start_index - (B + 1)) # Ensure start + B + 1 <= A
            if max_shift >= min_shift:
                shift = random.randint(min_shift, max_shift)
                actual_start = base_start_index + shift

        # Extract or pad the sequence chunk (B + 1 tokens needed)
        if chunk_type == 'pad':
            # Take all tokens and pad to B+1
            chunk_list = tokens.tolist() # Convert numpy to list for padding
            needed_padding = (B + 1) - A
            if needed_padding > 0:
                chunk_list.extend([self.pad_idx] * needed_padding)
            chunk = chunk_list[:B+1] # Ensure length
        else:
            # Clamp start index and extract B+1 tokens
            actual_start = max(0, min(actual_start, A - (B + 1)))
            chunk = tokens[actual_start : actual_start + B + 1] # Slicing works on numpy array

            # Fallback padding if chunk is unexpectedly short
            if len(chunk) != B + 1:
                 print(f"Warning: Chunk extraction issue (DatasetPreprocessed). File: {self.file_names[file_idx]}, Type: {chunk_type}, BaseStart: {base_start_index}, ActualStart: {actual_start}, A: {A}, B: {B}. Len(chunk): {len(chunk)}. Padding.")
                 chunk_list = chunk.tolist() # Convert numpy to list for padding
                 needed_padding = (B + 1) - len(chunk_list)
                 if needed_padding > 0:
                      chunk_list.extend([self.pad_idx] * needed_padding)
                 chunk = chunk_list[:B+1] # Ensure length


        # Create input (x) and target (y) sequences
        # Ensure 'chunk' is a list or numpy array before splitting
        if isinstance(chunk, np.ndarray):
             x = chunk[:-1]
             y = chunk[1:]
        else: # If it became a list due to padding
             x = chunk[:-1]
             y = chunk[1:]


        # Convert final x, y to tensors
        input_tensor = torch.tensor(x, dtype=torch.long)
        target_tensor = torch.tensor(y, dtype=torch.long)

        return input_tensor, target_tensor