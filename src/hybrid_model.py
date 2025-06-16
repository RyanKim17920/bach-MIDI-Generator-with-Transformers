import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math
from x_transformers import TransformerWrapper # Added import
from x_transformers import RotaryEmbedding # Corrected import

# Placeholder for Time2Vec - User should replace with actual implementation
class Time2VecPlaceholder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Simplified: just a linear layer for now
        # Actual Time2Vec would have periodic and non-periodic components
        self.linear = nn.Linear(input_dim, embed_dim)
        print(f"Warning: Using Placeholder Time2Vec. Replace with actual implementation.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be a tensor of time values, e.g., (batch, 1) or (1)
        return self.linear(x)

class CustomTokenEmbeddingModule(nn.Module):
    def __init__(self, processor, embed_dim: int, pad_idx: int):
        super().__init__()
        self.processor = processor
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx # Global PAD index

        # Determine the number of distinct event categories based on processor's mapping
        # This assumes processor.get_local_index_of_token for EVENT_ types returns 0..N-1
        # Example: if processor.get_local_index_of_token for EVENT_X returns 0..8
        # This needs to be robustly derived. For now, let's find max local_idx for events.
        max_event_local_idx = 0
        for p_attr_name in dir(processor):
            if p_attr_name.startswith("EVENT_") and p_attr_name.isupper(): # e.g. EVENT_NOTE_ON
                # This is a bit of a hack to find event markers defined in processor
                # A better way would be a dedicated list in processor or from active_type_names
                # For now, assume get_local_index_of_token handles this.
                # The local indices for events are defined in MIDIProcessor.get_local_index_of_token
                # event_marker_to_local_idx = { "EVENT_NOTE_ON": 0, ..., "EVENT_NOTE_DURATION": 8 }
                # So, num_event_categories = 9
                pass # Max index is known from MIDIProcessor's implementation
        self.num_event_categories = 9 # Based on current MIDIProcessor.get_local_index_of_token

        # Embedding Layers:
        # 1. For VALUE_TIME_SHIFT (Placeholder for Time2Vec)
        # self.time_value_embed = Time2VecPlaceholder(input_dim=1, embed_dim=embed_dim)
        # Using nn.Embedding for time steps for now, user can replace
        self.time_value_embed = nn.Embedding(processor.time_shift_steps, embed_dim)
        print("Note: Using nn.Embedding for TIME_VALUE tokens. Consider replacing with Time2Vec.")

        # 2. For VALUE_NOTE (pitch)
        self.note_pitch_embed = nn.Embedding(processor.note_range, embed_dim)
        print("Note: Using nn.Embedding for NOTE_VALUE tokens. Consider specific NoteEmbedding.")

        # 3. For VALUE_VELOCITY (bins)
        self.velocity_bin_embed = nn.Embedding(processor.velocity_bins, embed_dim)
        print("Note: Using nn.Embedding for VELOCITY_VALUE tokens. Consider specific VelocityEmbedding.")

        # 4. For PROGRAM tokens
        self.program_embed = nn.Embedding(processor.num_program_tokens, embed_dim)
        
        # 5. For LOCAL_INSTANCE tokens
        self.local_instance_embed = nn.Embedding(processor.max_local_instruments, embed_dim)

        # 6. For EVENT marker tokens (categorical)
        self.event_category_embed = nn.Embedding(self.num_event_categories, embed_dim)

        # 7. For SPECIAL tokens (PAD, START, END) - map global to local 0,1,2
        self.special_token_map = {
            processor.PAD: 0,
            processor.START: 1,
            processor.END: 2
        }
        # Assuming PAD maps to local_idx 0 for this embedding layer
        self.special_embed = nn.Embedding(len(self.special_token_map), embed_dim, padding_idx=0)


        # 8. For other VALUE types
        self.cc_num_embed = nn.Embedding(processor.cc_range, embed_dim)
        self.cc_val_embed = nn.Embedding(processor.cc_range, embed_dim) # Assuming 128 CC values
        self.prog_val_embed = nn.Embedding(processor.program_range, embed_dim) # Program *value* for PC event
        
        if hasattr(processor, 'duration_steps'):
            self.duration_val_embed = nn.Embedding(processor.duration_steps, embed_dim)
        
        # Fallback for UNKNOWN or types not explicitly handled (e.g. learnable UNKNOWN embedding)
        self.unknown_embed = nn.Parameter(torch.randn(embed_dim))


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize output tensor
        # Using float32 for embeddings, can be configured
        output_embeddings = torch.zeros(batch_size, seq_len, self.embed_dim, device=device, dtype=torch.float32)

        # This loop is for clarity; for performance, vectorization would be needed.
        # print("Warning: CustomTokenEmbeddingModule.forward uses a loop, consider vectorizing for performance.")
        for b in range(batch_size):
            for s in range(seq_len):
                global_idx = input_ids[b, s].item()
                
                # Handle global PAD index explicitly at the start
                if global_idx == self.pad_idx:
                    # Output for PAD is typically zeros, handled by special_embed's padding_idx if PAD maps to local_idx 0
                    # Ensure special_embed's local_idx 0 corresponds to PAD
                    if self.processor.PAD in self.special_token_map and self.special_token_map[self.processor.PAD] == self.special_embed.padding_idx:
                         # Let nn.Embedding handle padding_idx if applicable
                         # If not, output_embeddings[b,s] remains zero which is often desired for padding
                         pass # Rely on padding_idx of special_embed or default zero
                    # Fallback if PAD is not mapped or padding_idx is not 0
                    # output_embeddings[b, s] = torch.zeros(self.embed_dim, device=device) # Already initialized to zeros
                    # continue # Skip further processing for PAD

                type_name = self.processor.get_type_of_token(global_idx)
                local_idx = self.processor.get_local_index_of_token(global_idx, type_name)
                
                current_embedding_source = None
                
                if local_idx is None and type_name != "UNKNOWN": # Should not happen for valid non-UNKNOWN tokens
                    # print(f"Warning: No local_idx for token {global_idx} (type: {type_name}). Using UNKNOWN embedding.")
                    type_name = "UNKNOWN" # Force unknown handling

                embed_tensor = None

                if type_name == "VALUE_TIME_SHIFT":
                    # Placeholder: using nn.Embedding with local_idx (time step)
                    # Real Time2Vec would take unquantized time_val = self.processor._unquantize_steps(local_idx)
                    # and pass it to self.time_value_embed(torch.tensor([[time_val]], device=device))
                    embed_tensor = self.time_value_embed(torch.tensor(local_idx, device=device))
                elif type_name == "VALUE_NOTE":
                    embed_tensor = self.note_pitch_embed(torch.tensor(local_idx, device=device))
                elif type_name == "VALUE_VELOCITY":
                    embed_tensor = self.velocity_bin_embed(torch.tensor(local_idx, device=device))
                elif type_name == "PROGRAM":
                    embed_tensor = self.program_embed(torch.tensor(local_idx, device=device))
                elif type_name == "LOCAL_INSTANCE":
                    embed_tensor = self.local_instance_embed(torch.tensor(local_idx, device=device))
                elif type_name in ["SPECIAL_PAD", "SPECIAL_START", "SPECIAL_END"]:
                    if global_idx in self.special_token_map:
                        mapped_local_idx = self.special_token_map[global_idx]
                        embed_tensor = self.special_embed(torch.tensor(mapped_local_idx, device=device))
                    else: # Should not happen if map is correct
                        embed_tensor = self.unknown_embed.unsqueeze(0)
                elif type_name.startswith("EVENT_"):
                    # local_idx for events is 0 to num_event_categories-1
                    if 0 <= local_idx < self.num_event_categories:
                         embed_tensor = self.event_category_embed(torch.tensor(local_idx, device=device))
                    else: # Should not happen
                        embed_tensor = self.unknown_embed.unsqueeze(0)
                elif type_name == "VALUE_CC_NUMBER":
                    embed_tensor = self.cc_num_embed(torch.tensor(local_idx, device=device))
                elif type_name == "VALUE_CC_VALUE":
                    embed_tensor = self.cc_val_embed(torch.tensor(local_idx, device=device))
                elif type_name == "VALUE_PROGRAM": # Program value for ProgramChange event
                    embed_tensor = self.prog_val_embed(torch.tensor(local_idx, device=device))
                elif type_name == "VALUE_NOTE_DURATION" and hasattr(self, 'duration_val_embed'):
                    embed_tensor = self.duration_val_embed(torch.tensor(local_idx, device=device))
                elif type_name == "UNKNOWN":
                    embed_tensor = self.unknown_embed.unsqueeze(0)
                else:
                    # print(f"Warning: Unhandled token type '{type_name}' for embedding (global_idx: {global_idx}). Using UNKNOWN embedding.")
                    embed_tensor = self.unknown_embed.unsqueeze(0)

                if embed_tensor is not None:
                    output_embeddings[b, s] = embed_tensor.squeeze() # Squeeze if it was (1, dim)

        return output_embeddings

class MusicHybridTransformer(TransformerWrapper): # Inherit from TransformerWrapper
    def __init__(self,
                 *, # Keyword-only arguments
                 num_tokens: int, # Global vocabulary size
                 max_seq_len: int,
                 attn_layers: nn.Module, # This is the x_transformers.Decoder
                 processor, # Instance of MIDIProcessor
                 embed_dim: Optional[int] = None, # Dimension for all embeddings and transformer
                 emb_dropout: float = 0.1,
                 post_attn_norm: bool = True,
                 # Add any specific params for custom embeddings if needed, e.g. time2vec_style
                 **kwargs): # Catch other TransformerWrapper params
        
        # If embed_dim is not provided, try to infer from attn_layers.dim
        # TransformerWrapper does this: dim = attn_layers.dim if embed_dim is None else embed_dim
        actual_embed_dim = attn_layers.dim if embed_dim is None else embed_dim
        if actual_embed_dim is None:
            raise ValueError("embed_dim must be specified or inferable from attn_layers.dim")

        super().__init__(
            num_tokens=num_tokens, # Still needed for TransformerWrapper's to_logits layer, and for ARW
            max_seq_len=max_seq_len,
            attn_layers=attn_layers,
            emb_dim=actual_embed_dim, # This sets self.dim in TransformerWrapper
            emb_dropout=0., # We will apply dropout *after* our custom embedding + pos embedding
            post_attn_norm=post_attn_norm,
            # We are not using TransformerWrapper's token_emb directly.
            # We are also not using its to_logits directly for hierarchical output.
            **kwargs
        )
        self.emb_dropout_rate = emb_dropout # Store for manual application
        self.custom_emb_dropout = nn.Dropout(emb_dropout)


        self.processor = processor
        self.pad_idx = self.processor.PAD

        # Instantiate our custom embedding module
        # self.dim is set by TransformerWrapper's __init__ based on actual_embed_dim
        self.custom_token_embedding = CustomTokenEmbeddingModule(processor, self.dim, self.pad_idx)

        self._setup_token_classification_and_ranges() # Remains the same
        self._setup_specialized_heads()             # Remains the same

    def _setup_token_classification_and_ranges(self):
        p = self.processor
        # Define an ordered list of all possible type names that MIDIProcessor.get_type_of_token can return.
        # This order will determine the type_id for the classifier.
        # These should match the strings returned by processor.get_type_of_token()
        all_possible_type_names = [
            "SPECIAL_PAD", "SPECIAL_START", "SPECIAL_END",
            "PROGRAM", "LOCAL_INSTANCE",
            "EVENT_NOTE_ON", "EVENT_NOTE_OFF", "EVENT_TIME_SHIFT", "EVENT_VELOCITY",
            "EVENT_CONTROL_CHANGE", "EVENT_PROGRAM_CHANGE", "EVENT_PEDAL_ON", "EVENT_PEDAL_OFF",
            "EVENT_NOTE_DURATION", 
            "VALUE_NOTE", "VALUE_TIME_SHIFT", "VALUE_VELOCITY", 
            "VALUE_CC_NUMBER", "VALUE_CC_VALUE", "VALUE_PROGRAM", 
            "VALUE_NOTE_DURATION",
            "UNKNOWN" # Include UNKNOWN as a potential type from processor
        ]

        # Filter to include only types that are relevant based on processor's current configuration
        # and that have corresponding attributes in the processor.
        self.active_type_names = []
        for type_name in all_possible_type_names:
            if type_name == "SPECIAL_PAD" and hasattr(p, 'PAD'): self.active_type_names.append(type_name)
            elif type_name == "SPECIAL_START" and hasattr(p, 'START'): self.active_type_names.append(type_name)
            elif type_name == "SPECIAL_END" and hasattr(p, 'END'): self.active_type_names.append(type_name)
            elif type_name == "PROGRAM" and hasattr(p, 'program_token_offset'): self.active_type_names.append(type_name)
            elif type_name == "LOCAL_INSTANCE" and hasattr(p, 'local_instance_offset'): self.active_type_names.append(type_name)
            elif type_name == "EVENT_NOTE_ON" and hasattr(p, 'NOTE_ON'): self.active_type_names.append(type_name)
            elif type_name == "EVENT_NOTE_OFF" and hasattr(p, 'NOTE_OFF'): self.active_type_names.append(type_name)
            elif type_name == "EVENT_TIME_SHIFT" and hasattr(p, 'TIME_SHIFT'): self.active_type_names.append(type_name)
            elif type_name == "EVENT_VELOCITY" and hasattr(p, 'VELOCITY'): self.active_type_names.append(type_name)
            elif type_name == "EVENT_CONTROL_CHANGE" and hasattr(p, 'CONTROL_CHANGE'): self.active_type_names.append(type_name)
            elif type_name == "EVENT_PROGRAM_CHANGE" and hasattr(p, 'PROGRAM_CHANGE'): self.active_type_names.append(type_name)
            elif type_name == "EVENT_PEDAL_ON" and hasattr(p, 'PEDAL_ON'): self.active_type_names.append(type_name)
            elif type_name == "EVENT_PEDAL_OFF" and hasattr(p, 'PEDAL_OFF'): self.active_type_names.append(type_name)
            elif type_name == "EVENT_NOTE_DURATION" and hasattr(p, 'NOTE_DURATION'): self.active_type_names.append(type_name)
            elif type_name == "VALUE_NOTE" and hasattr(p, 'note_value_offset'): self.active_type_names.append(type_name)
            elif type_name == "VALUE_TIME_SHIFT" and hasattr(p, 'time_shift_value_offset'): self.active_type_names.append(type_name)
            elif type_name == "VALUE_VELOCITY" and hasattr(p, 'velocity_value_offset'): self.active_type_names.append(type_name)
            elif type_name == "VALUE_CC_NUMBER" and hasattr(p, 'cc_number_offset'): self.active_type_names.append(type_name)
            elif type_name == "VALUE_CC_VALUE" and hasattr(p, 'cc_value_offset'): self.active_type_names.append(type_name)
            elif type_name == "VALUE_PROGRAM" and hasattr(p, 'program_value_offset'): self.active_type_names.append(type_name)
            elif type_name == "VALUE_NOTE_DURATION" and hasattr(p, 'note_duration_value_offset'): self.active_type_names.append(type_name)
            elif type_name == "UNKNOWN": self.active_type_names.append(type_name) # Always include UNKNOWN if processor might return it

        self.num_token_types = len(self.active_type_names)
        self.type_name_to_id = {name: i for i, name in enumerate(self.active_type_names)}
        self.type_id_to_name = {i: name for i, name in enumerate(self.active_type_names)}

        self.token_type_classifier = nn.Linear(self.dim, self.num_token_types)

    def _get_target_token_types(self, target_ids: torch.Tensor) -> torch.Tensor:
        # target_ids: [batch, seq_len]
        target_types = torch.full_like(target_ids, -1, dtype=torch.long, device=target_ids.device)

        for i in range(target_ids.shape[0]):
            for j in range(target_ids.shape[1]):
                global_token_idx = target_ids[i, j].item()
                
                # For PAD tokens, we want them to be ignored in the loss.
                # If PAD has its own type (e.g., "SPECIAL_PAD") and it's active, it will be mapped.
                # Otherwise, it remains -1 and will be ignored by loss functions with ignore_index=-1.
                if global_token_idx == self.pad_idx:
                    pad_type_name = "SPECIAL_PAD"
                    if pad_type_name in self.type_name_to_id:
                        target_types[i, j] = self.type_name_to_id[pad_type_name]
                    else:
                        target_types[i, j] = -1 # Explicitly mark for ignoring if not an active type
                    continue

                type_name = self.processor.get_type_of_token(global_token_idx)
                if type_name and type_name in self.type_name_to_id:
                    target_types[i, j] = self.type_name_to_id[type_name]
                else:
                    # Token's type is not active or unknown beyond what's configured.
                    # Remains -1, to be ignored in loss.
                    # print(f"Warning: Target token {global_token_idx} (type from proc: {type_name}) not mapped to an active type ID. Will be ignored.")
                    pass 
        return target_types

    def _setup_specialized_heads(self):
        p = self.processor
        self.heads = nn.ModuleDict()

        for type_name in self.active_type_names:
            if type_name == "UNKNOWN": continue # No head for unknown type

            output_dim = 0
            if type_name == "SPECIAL_PAD": output_dim = 1
            elif type_name == "SPECIAL_START": output_dim = 1
            elif type_name == "SPECIAL_END": output_dim = 1
            elif type_name == "PROGRAM": output_dim = p.num_program_tokens
            elif type_name == "LOCAL_INSTANCE": output_dim = p.max_local_instruments
            elif type_name.startswith("EVENT_"):
                # Each specific event type (e.g., EVENT_NOTE_ON) has a head predicting its occurrence (dim 1).
                # The local_index for this event (e.g., processor.NOTE_ON) should be 0 for this head.
                output_dim = 1 
            elif type_name == "VALUE_NOTE": output_dim = p.note_range
            elif type_name == "VALUE_TIME_SHIFT": output_dim = p.time_shift_steps
            elif type_name == "VALUE_VELOCITY": output_dim = p.velocity_bins
            elif type_name == "VALUE_CC_NUMBER": output_dim = p.cc_range
            elif type_name == "VALUE_CC_VALUE": output_dim = p.cc_range 
            elif type_name == "VALUE_PROGRAM": output_dim = p.program_range 
            elif type_name == "VALUE_NOTE_DURATION": output_dim = p.duration_steps
            
            if output_dim > 0:
                self.heads[type_name] = nn.Linear(self.dim, output_dim)
            # else:
                # print(f"Warning: No head created for active type: {type_name} as output_dim is 0.")

    def forward(self, input_ids: torch.Tensor, target_ids: Optional[torch.Tensor] = None, **kwargs):
        # input_ids: (batch, seq_len)
        # target_ids: (batch, seq_len), shifted labels from AutoregressiveWrapper
        # kwargs will include `mask` from AutoregressiveWrapper (e.g., attention_mask)

        # 1. Get custom token embeddings
        x = self.custom_token_embedding(input_ids)  # (batch, seq_len, self.dim)

        # 2. Add positional embeddings
        # (Logic adapted from TransformerWrapper's forward)
        if self.has_pos_emb:
            # Rotary embeddings are typically handled inside the attention layers.
            # Other types (Absolute, Sinusoidal) are added here.
            if not isinstance(self.pos_emb, RotaryEmbedding):
                if isinstance(self.pos_emb, nn.Embedding): # e.g. AbsolutePositionalEmbedding
                    # AbsolutePositionalEmbedding expects input (seq_len)
                    seq_len = input_ids.shape[1]
                    positions = torch.arange(seq_len, device=input_ids.device)
                    x = x + self.pos_emb(positions) # Broadcasts over batch
                else: # e.g. ScaledSinusoidalEmbedding that takes the sequence tensor itself
                    x = x + self.pos_emb(x)
        
        # 3. Apply embedding dropout
        x = self.custom_emb_dropout(x) # Use the stored dropout rate

        # 4. Project embeddings if needed (from TransformerWrapper)
        if self.project_emb is not None:
            x = self.project_emb(x)

        # 5. Pass through attention layers (the Decoder)
        # kwargs should include mask if provided by AutoregressiveWrapper
        transformer_hidden_states = self.attn_layers(x, **kwargs) 

        # 6. Apply final norm (from TransformerWrapper, if post_attn_norm=True)
        # self.norm is LayerNorm(self.dim)
        transformer_output = self.norm(transformer_hidden_states) if self.post_attn_norm else transformer_hidden_states
        
        # Hierarchical prediction (remains the same as before)
        type_logits = self.token_type_classifier(transformer_output) # [B, S, num_token_types]

        all_specialized_logits = {}
        for type_name_key in self.active_type_names: # Iterate active types for heads
            if type_name_key in self.heads: # Check if head exists for this active type
                 all_specialized_logits[type_name_key] = self.heads[type_name_key](transformer_output)

        if target_ids is not None:
            # Training/validation path - calculate and return custom loss
            actual_next_tokens = target_ids 
            actual_next_token_types = self._get_target_token_types(actual_next_tokens)
            loss = self._calculate_loss(type_logits, all_specialized_logits, actual_next_tokens, actual_next_token_types)
            return loss
        else:
            # Generation path - reconstruct full logits
            full_logits = self._reconstruct_full_logits(type_logits, all_specialized_logits)
            return full_logits

    def _reconstruct_full_logits(self, type_logits, all_specialized_logits):
        # type_logits: [B, S, num_token_types]
        # all_specialized_logits: dict with key=type_name, value=logits tensor for that type

        # Start with the type logits
        batch_size, seq_len, _ = type_logits.shape

        # Initialize full logits with zeros
        # Assuming num_tokens is the sum of all specialized output dimensions + 1 for UNKNOWN
        num_tokens = sum(head.out_features for head in self.heads.values()) + 1 
        full_logits = torch.zeros(batch_size, seq_len, num_tokens, device=type_logits.device)

        # Fill in the specialized logits
        for type_name, logits in all_specialized_logits.items():
            if logits is not None and type_name in self.type_name_to_id:
                type_id = self.type_name_to_id[type_name]
                full_logits[:, :, type_id] = logits

        # Handle UNKNOWN type - assuming it's the last token in the vocabulary
        if "UNKNOWN" in self.type_name_to_id:
            unknown_id = self.type_name_to_id["UNKNOWN"]
            full_logits[:, :, unknown_id] = 0 # or some other logic for UNKNOWN

        return full_logits

    def _calculate_loss(self, type_logits, all_specialized_logits, target_ids, target_types):
        # type_logits: [B, S, num_token_types]
        # all_specialized_logits: dict with key=type_name, value=logits tensor for that type
        # target_ids: [B, S], actual token IDs
        # target_types: [B, S], actual token types (mapped to active type IDs)

        # 1. Calculate the base loss using type_logits
        # Ignoring padding tokens in the loss calculation
        batch_size, seq_len, num_token_types = type_logits.shape

        # Flatten the tensors for loss calculation
        type_logits_flat = type_logits.view(-1, num_token_types) # [B*S, num_token_types]
        target_types_flat = target_types.view(-1) # [B*S]

        # Base loss - only for the active token types
        base_loss = F.cross_entropy(type_logits_flat, target_types_flat, ignore_index=-1, reduction='mean')

        # 2. Add specialized losses for each active type
        specialized_loss = 0.0
        for type_name, logits in all_specialized_logits.items():
            if logits is not None and type_name in self.type_name_to_id:
                type_id = self.type_name_to_id[type_name]

                # Get the corresponding target values for this type
                target_values = (target_types == type_id).nonzero(as_tuple=True)

                if len(target_values[0]) > 0: # There are some targets for this type
                    # Gather the logits for these targets
                    type_logits_for_targets = logits[target_values]

                    # Calculate the loss for these targets
                    # Using binary cross-entropy for specialized heads
                    specialized_loss += F.binary_cross_entropy_with_logits(type_logits_for_targets, target_values[0].float())

        # Total loss is the base loss plus the specialized losses
        total_loss = base_loss + specialized_loss
        return total_loss
