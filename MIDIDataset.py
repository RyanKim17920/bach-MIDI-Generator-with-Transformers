import os
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from MIDIprocessor import MIDIProcessor
import random

class MIDIDataset(Dataset):
    """
    PyTorch Dataset for loading and chunking MIDI token sequences.

    Processes MIDI files using MIDIProcessor and provides chunks based on a
    strategy involving fixed start/end chunks and augmented middle chunks.
    """
    def __init__(self,
                 midi_file_paths: List[str],
                 sequence_length: int,
                 pad_idx: int = 0,
                 augmentation_shift:int = 0,):
        """
        Args:
            midi_file_paths: List of paths to MIDI files.
            processor: An initialized instance of the MIDIProcessor.
            sequence_length: The desired context length (B) for the model sequences.
            apply_augmentation: Whether to apply random shifting to middle chunks.
        """
        self.sequence_length = sequence_length # This is B
        self.pad_idx = pad_idx

        self.augmentation_shift = augmentation_shift
        self.apply_augmentation = augmentation_shift > 0

        
        self.file_tokens: List[List[int]] = [] # Stores token list for each valid file
        self.chunk_info: List[Tuple[int, str, int]] = [] # (file_idx, chunk_type, base_start_idx)

        print(f"Initializing MIDIDataset with sequence length {self.sequence_length} and augmentation shift {self.augmentation_shift}...")
        self._load_and_process_files(midi_file_paths)
        self._calculate_chunk_info()

        if not self.chunk_info:
            raise ValueError("No valid chunks could be generated from the provided MIDI files. Check file validity and sequence length.")

        print(f"Dataset initialized. Number of processable files: {len(self.file_tokens)}. Total number of chunks: {len(self.chunk_info)}.")

    def _load_and_process_files(self, midi_file_paths: List[str]):
        """Loads MIDI files, processes them, and stores token sequences."""
        processed_count = 0
        for file_path in midi_file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File not found, skipping: {file_path}")
                continue
            processed_data = self.processor.process_midi_file(file_path)
            if processed_data and processed_data.get('tokens'):
                tokens = processed_data['tokens']
                # We need at least 2 tokens to form an input/target pair, even if padding
                if len(tokens) >= 2:
                    self.file_tokens.append(tokens)
                    processed_count += 1
                else:
                    print(f"Warning: File processed but too short (< 2 tokens), skipping: {file_path}")
            else:
                 print(f"Warning: Failed to process or empty tokens for: {file_path}")
        print(f"Successfully processed and kept {processed_count} MIDI files.")


    def _calculate_chunk_info(self):
        """Calculates the base information for each chunk across all files."""
        self.chunk_info = []
        B = self.sequence_length

        for file_idx, tokens in enumerate(self.file_tokens):
            A = len(tokens) # Total length of the current file's tokens

            # We need A >= B+1 tokens to extract a full unpadded sequence pair (x, y)
            # But we handle shorter sequences by padding. Min length needed is 2.

            if A < B + 1:
                # --- Case 1: File is shorter than required sequence length + 1 ---
                # Treat as a single chunk that needs padding
                self.chunk_info.append((file_idx, 'pad', 0))
            else:
                # --- Case 2: File is long enough for at least one full sequence ---
                # N = A // B # Number of full blocks of size B
                # Note: A sequence requires B+1 tokens.

                # Add Chunk 0 (Start) - Always fixed
                self.chunk_info.append((file_idx, 'start', 0))

                # Add Middle Chunks (1 to N-1, if they exist)
                # A middle chunk k conceptually covers tokens around [k*B, k*B + B)
                # The actual start index will be augmented.
                # We need to ensure the *base* starting point allows for a full sequence.
                # The last possible base start for a middle chunk k is when k*B + B + 1 <= A
                last_possible_middle_chunk_index = (A - B - 1) // B
                
                for k in range(1, last_possible_middle_chunk_index): # Iterate through valid middle chunk indices
                     base_start_index = k * B
                     self.chunk_info.append((file_idx, 'middle', base_start_index))

                # Add Chunk N (End) - Always fixed at the very end
                # Ensure start index is non-negative (handles cases where A is just B+1)
                end_chunk_start_index = max(0, A - (B + 1)) # Start B+1 tokens from the end
                # Avoid adding duplicate chunk if end chunk overlaps completely with start/middle chunk
                # Check if this end chunk's start index is different from previously added chunks for this file
                last_added_start = -1
                if self.chunk_info and self.chunk_info[-1][0] == file_idx:
                     last_added_start = self.chunk_info[-1][2]
                
                # Only add the end chunk if its start index is strictly greater
                # than the start index of the last added chunk for this file.
                # This prevents adding the same chunk twice if A is small, e.g. A = B+1
                if end_chunk_start_index > last_added_start:
                    self.chunk_info.append((file_idx, 'end', end_chunk_start_index))


    def __len__(self) -> int:
        """Returns the total number of chunks available in the dataset."""
        return len(self.chunk_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a specific chunk, applying augmentation or padding as needed.

        Args:
            idx: Index of the chunk information in self.chunk_info.

        Returns:
            A tuple (input_sequence, target_sequence):
              - input_sequence (x): Tensor of shape (sequence_length,)
              - target_sequence (y): Tensor of shape (sequence_length,)
        """
        file_idx, chunk_type, base_start_index = self.chunk_info[idx]
        tokens = self.file_tokens[file_idx]
        A = len(tokens)
        B = self.sequence_length
        S = self.augmentation_shift

        actual_start = base_start_index

        # Determine the actual starting index based on chunk type
        if chunk_type == 'middle' and self.apply_augmentation and S > 0:
            # Calculate valid shift range for this middle chunk
            # Lower bound for shift: ensure start >= 0. shift >= -base_start_index
            min_shift = max(-S, -base_start_index)

            # Upper bound for shift: ensure start + B + 1 <= A
            # shift <= A - base_start_index - B - 1
            max_shift = min(S, A - base_start_index - B - 1)

            # Choose shift only if range is valid
            if max_shift >= min_shift:
                shift = random.randint(min_shift, max_shift)
                actual_start = base_start_index + shift
            # else: no augmentation possible for this chunk, use base_start_index

        # Extract the sequence chunk (needs B + 1 tokens)
        if chunk_type == 'pad':
            # File is shorter than B+1, take all tokens and pad
            chunk = tokens # Tokens from index 0 up to A-1
            needed_padding = (B + 1) - A
            if needed_padding > 0:
                padding = [self.pad_idx] * needed_padding
                chunk.extend(padding)
            # Ensure chunk has exactly B+1 elements after padding
            chunk = chunk[:B+1]
        else:
            # For start, middle, end chunks (guaranteed to have A >= B + 1)
            # Clamp actual_start just in case augmentation calculation had edge issues
            actual_start = max(0, min(actual_start, A - (B + 1)))
            chunk = tokens[actual_start : actual_start + B + 1]

            # Sanity check: Ensure chunk has the correct length B+1
            if len(chunk) != B + 1:
                 # This might happen if A is exactly B+1 and augmentation causes issues.
                 # Or if clamping logic wasn't perfect. Fallback to padding.
                 print(f"Warning: Chunk extraction issue. File: {file_idx}, Type: {chunk_type}, BaseStart: {base_start_index}, ActualStart: {actual_start}, A: {A}, B: {B}. Len(chunk): {len(chunk)}. Padding.")
                 needed_padding = (B + 1) - len(chunk)
                 if needed_padding > 0:
                      chunk.extend([self.pad_idx] * needed_padding)
                 chunk = chunk[:B+1] # Ensure length

        # Create input (x) and target (y) sequences
        x = chunk[:-1]
        y = chunk[1:]

        # Convert to tensors
        input_tensor = torch.tensor(x, dtype=torch.long)
        target_tensor = torch.tensor(y, dtype=torch.long)

        return input_tensor, target_tensor