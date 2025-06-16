import torch
import torch.nn as nn
import math

class MusicHybridEmbedding(nn.Module):
    """Custom embedding module that handles continuous representations for musical tokens"""
    
    # Removed vocab_size from __init__ signature, rely on processor
    def __init__(self, embed_dim: int, processor, use_hybrid: bool = True):
        super().__init__()
        self.processor = processor
        self.embed_dim = embed_dim
        self.use_hybrid = use_hybrid
        
        padding_idx = self.processor.pad_token_id if hasattr(self.processor, 'pad_token_id') else None

        if not use_hybrid:
            # Fallback to standard embedding using the full vocab size
            self.token_embedding = nn.Embedding(self.processor.vocab_size, embed_dim, padding_idx=padding_idx)
            return
        
        # Determine the size for the main token_embedding layer
        if self.processor.reorder_specialized_tokens:
            embedding_vocab_size = self.processor.categorical_vocab_size
        else:
            embedding_vocab_size = self.processor.vocab_size
            
        self.token_embedding = nn.Embedding(embedding_vocab_size, embed_dim, padding_idx=padding_idx)
        
        # Specialized embeddings for continuous values (initialized as before)
        self.time_shift_embedder = Time2Vec(embed_dim, max_steps=self.processor.time_shift_steps)
        self.pitch_embedder = PitchEmbedding(embed_dim) 
        self.velocity_embedder = VelocityEmbedding(embed_dim, max_bins=self.processor.velocity_bins)
        
        if hasattr(self.processor, 'note_duration_value_offset') and hasattr(self.processor, 'duration_steps'):
            self.duration_embedder = Time2Vec(embed_dim, max_steps=self.processor.duration_steps)
        else:
            self.duration_embedder = None
        
        self._setup_token_ranges()
        
    def _setup_token_ranges(self):
        """Setup token range detection based on processor configuration"""
        p = self.processor
        
        # Time shift tokens
        self.time_start = p.time_shift_value_offset
        self.time_end = p.time_shift_value_offset + p.time_shift_steps
        
        # Note value tokens
        self.note_start = p.note_value_offset
        self.note_end = p.note_value_offset + p.note_range
        
        # Velocity tokens
        self.vel_start = p.velocity_value_offset
        self.vel_end = p.velocity_value_offset + p.velocity_bins
        
        # Duration tokens (if using note duration mode)
        if hasattr(p, 'note_duration_value_offset'):
            self.dur_start = p.note_duration_value_offset
            self.dur_end = p.note_duration_value_offset + p.duration_steps
        else:
            self.dur_start = self.dur_end = -1
    
    def forward(self, tokens):
        if not self.use_hybrid:
            return self.token_embedding(tokens)
        
        # Initialize embeddings tensor
        # embeddings = torch.zeros(*tokens.shape, self.embed_dim, device=tokens.device, dtype=self.token_embedding.weight.dtype)
        # Simpler initialization and then fill:
        
        if self.processor.reorder_specialized_tokens:
            # Initialize with zeros, as token_embedding only covers categorical tokens
            embeddings = torch.zeros(*tokens.shape, self.embed_dim, device=tokens.device, dtype=self.token_embedding.weight.dtype)
            
            # Create a mask for tokens that should use the standard nn.Embedding
            # These are tokens with global_id < categorical_vocab_size
            categorical_mask = tokens < self.processor.categorical_vocab_size
            
            # Apply standard embedding only to categorical tokens
            # Ensure tokens passed to self.token_embedding are within its range
            if categorical_mask.any():
                # We need to be careful if padding_idx is within categorical_vocab_size and is used.
                # If tokens[categorical_mask] can be empty, handle that.
                # Create a temporary tensor for valid categorical tokens to avoid out-of-bounds with padding_idx if it's >= categorical_vocab_size
                # However, padding_idx should ideally be low (0, 1, 2).
                # Assuming padding_idx, if used, is < categorical_vocab_size.
                
                # Get the actual token values for the categorical part
                categorical_tokens_to_embed = tokens[categorical_mask]

                # If padding_idx is used and is one of the categorical_tokens_to_embed,
                # nn.Embedding handles it by setting its vector to zeros (by default).
                # If padding_idx is >= categorical_vocab_size (which shouldn't happen with reordering),
                # then it won't be in categorical_tokens_to_embed.
                
                embedded_categorical = self.token_embedding(categorical_tokens_to_embed)
                embeddings[categorical_mask] = embedded_categorical
        else:
            # Original behavior: token_embedding covers the whole vocab_size initially
            embeddings = self.token_embedding(tokens)
            
        # Create masks for different specialized token types (global ranges)
        time_mask = (tokens >= self.time_start) & (tokens < self.time_end)
        note_mask = (tokens >= self.note_start) & (tokens < self.note_end)
        vel_mask = (tokens >= self.vel_start) & (tokens < self.vel_end)
        
        dur_mask = torch.zeros_like(tokens, dtype=torch.bool, device=tokens.device)
        if hasattr(self, 'dur_start') and self.dur_start != -1 and self.duration_embedder is not None: # Check if duration range is valid
            dur_mask = (tokens >= self.dur_start) & (tokens < self.dur_end)
        
        # Apply specialized embeddings, overwriting the initial ones where applicable
        if time_mask.any():
            time_values = tokens[time_mask] - self.time_start
            embeddings[time_mask] = self.time_shift_embedder(time_values)
        
        if note_mask.any():
            note_values = tokens[note_mask] - self.note_start
            embeddings[note_mask] = self.pitch_embedder(note_values)
        
        if vel_mask.any():
            vel_values = tokens[vel_mask] - self.vel_start
            embeddings[vel_mask] = self.velocity_embedder(vel_values)
        
        if dur_mask.any(): 
            dur_values = tokens[dur_mask] - self.dur_start
            embeddings[dur_mask] = self.duration_embedder(dur_values)
        
        return embeddings

class Time2Vec(nn.Module):
    def __init__(self, embed_dim, max_steps): # Added max_steps
        super().__init__()
        self.embed_dim = embed_dim
        self.max_steps = float(max_steps) 
        if embed_dim % 2 != 0:
            raise ValueError("Time2Vec embed_dim must be even.")
        self.linear_proj = nn.Linear(1, embed_dim // 2)
        self.w = nn.Parameter(torch.randn(embed_dim // 2))
        self.b = nn.Parameter(torch.randn(embed_dim // 2))
        
    def forward(self, time_steps):
        if self.max_steps > 0:
            time_normalized = time_steps.float().unsqueeze(-1) / self.max_steps
        else: 
            time_normalized = time_steps.float().unsqueeze(-1)

        linear_part = self.linear_proj(time_normalized)
        time_expanded = time_normalized * self.w + self.b # Broadcasting time_normalized
        periodic_part = torch.sin(time_expanded)
        return torch.cat([linear_part, periodic_part], dim=-1)

class PitchEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, pitch_values):
        pitch_float = pitch_values.float()
        octave = torch.floor(pitch_float / 12.0)
        semitone = pitch_float % 12.0
        
        embeddings = torch.zeros(pitch_float.size(0), self.embed_dim, device=pitch_float.device)
        
        for i in range(self.embed_dim // 4):
            div_term = 2.0 ** i
            embeddings[:, i*4] = torch.sin(octave / div_term)
            embeddings[:, i*4+1] = torch.cos(octave / div_term)
            embeddings[:, i*4+2] = torch.sin(semitone * math.pi / 6.0 * (i + 1))
            embeddings[:, i*4+3] = torch.cos(semitone * math.pi / 6.0 * (i + 1))
        
        return embeddings

class VelocityEmbedding(nn.Module):
    def __init__(self, embed_dim, max_bins): # Added max_bins
        super().__init__()
        self.max_bins = float(max_bins)
        self.projection = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, velocity_bins):
        if self.max_bins > 0:
            vel_continuous = velocity_bins.float() / self.max_bins
        else: 
            vel_continuous = velocity_bins.float()

        vel_normalized = vel_continuous.unsqueeze(-1)
        embedded = self.projection(vel_normalized)
        return self.layer_norm(embedded)