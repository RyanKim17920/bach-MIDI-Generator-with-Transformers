from x_transformers.x_transformers import *
import torch


# based on x_transformers
# three token types = values,time, instruments
class TransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            num_tokens_values,
            num_tokens_times,
            num_tokens_instruments,
            max_seq_len,
            pre_attn_layers: list[AttentionLayers],
            attn_layers: AttentionLayers,
            emb_dropout=0.,
            post_emb_norm=False,
            use_abs_pos_emb=True,
            scaled_sinu_pos_emb=False,
            l2norm_embed=False,
    ):
        super().__init__()

        dim = attn_layers.dim
        emb_dim = dim
        self.emb_dim = emb_dim
        self.num_tokens_values = num_tokens_values
        self.num_tokens_times = num_tokens_times
        self.num_tokens_instruments = num_tokens_instruments

        self.pre_attn_layers = pre_attn_layers

        self.max_seq_len = max_seq_len

        self.l2norm_embed = l2norm_embed

        pre_attn_token_emb_dim = emb_dim // 3
        self.token_emb_values = TokenEmbedding(pre_attn_token_emb_dim, num_tokens_values, l2norm_embed=l2norm_embed)
        self.token_emb_times = TokenEmbedding(pre_attn_token_emb_dim, num_tokens_times, l2norm_embed=l2norm_embed)
        self.token_emb_instruments = TokenEmbedding(pre_attn_token_emb_dim, num_tokens_instruments, l2norm_embed=l2norm_embed)

        no_abs_pos_emb = max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.has_pos_emb)

        if no_abs_pos_emb:
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(pre_attn_token_emb_dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(pre_attn_token_emb_dim, max_seq_len, l2norm_embed=l2norm_embed)

        self.post_emb_norm = nn.LayerNorm(pre_attn_token_emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers

        self.init_()

        self.to_logits_values = nn.Linear(dim, num_tokens_values)
        self.to_logits_times = nn.Linear(dim, num_tokens_times)
        self.to_logits_instruments = nn.Linear(dim, num_tokens_instruments)

        # whether can do cached kv decoding

        self.can_cache_kv_outside_max_seq_len = no_abs_pos_emb

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.token_emb_times.emb.weight, std=1e-5)
            nn.init.normal_(self.token_emb_values.emb.weight, std=1e-5)
            nn.init.normal_(self.token_emb_instruments.emb.weight, std=1e-5)
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb_times.emb.weight, std=1e-5)
                nn.init.normal_(self.pos_emb_values.emb.weight, std=1e-5)
                nn.init.normal_(self.pos_emb_instruments.emb.weight, std=1e-5)
            return

        nn.init.kaiming_normal_(self.token_emb_times.emb.weight)
        nn.init.kaiming_normal_(self.token_emb_values.emb.weight)
        nn.init.kaiming_normal_(self.token_emb_instruments.emb.weight)

    def forward(
            self,
            x,
            return_embeddings=False,
            return_logits_and_embeddings=False,
            return_intermediates=False,
            mask=None,
            return_attn=False,
            mems=None,
            return_attn_z_loss=False,
            attn_z_loss_weight=1e-4,
            seq_start_pos=None,
            cache: Optional[LayerIntermediates] = None,
            **kwargs
    ):

        # absolute positional embedding

        # external_pos_emb = exists(pos) and pos.dtype != torch.long
        # pos_emb = self.pos_emb(x, pos=pos, seq_start_pos=seq_start_pos) if not external_pos_emb else pos
        x_values = x[:, :, 0]
        x_times = x[:, :, 1]
        x_instruments = x[:, :, 2]
        x_values = self.token_emb_values(x_values)
        x_times = self.token_emb_times(x_times)
        x_instruments = self.token_emb_instruments(x_instruments)

        # post embedding norm, purportedly leads to greater stabilization

        x_values = self.post_emb_norm(x_values)
        x_times = self.post_emb_norm(x_times)
        x_instruments = self.post_emb_norm(x_instruments)

        # embedding dropout

        x_values = self.emb_dropout(x_values)
        x_times = self.emb_dropout(x_times)
        x_instruments = self.emb_dropout(x_instruments)

        # pre attention layers
        x_values = self.pre_attn_layers[0](x_values, mask=mask, mems=mems, cache=cache, return_hiddens=False,
                                           seq_start_pos=seq_start_pos, **kwargs)
        x_times = self.pre_attn_layers[1](x_times, mask=mask, mems=mems, cache=cache, return_hiddens=False,
                                          seq_start_pos=seq_start_pos, **kwargs)
        x_instruments = self.pre_attn_layers[2](x_instruments, mask=mask, mems=mems, cache=cache, return_hiddens=False,
                                                seq_start_pos=seq_start_pos, **kwargs)

        x = torch.cat([x_values,x_times,x_instruments], dim=-1)
        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, cache=cache, return_hiddens=True,
                                            seq_start_pos=seq_start_pos, **kwargs)

        if return_logits_and_embeddings:
            out = (self.to_logits_values(x),
                   self.to_logits_times(x),
                   self.to_logits_instruments(x), x)
        elif return_embeddings:
            out = x
        else:
            out = (self.to_logits_values(x),
                   self.to_logits_times(x),
                   self.to_logits_instruments(x))

        if return_attn_z_loss:
            pre_softmax_attns = list(map(lambda t: t.pre_softmax_attn, intermediates.attn_intermediates))
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight=attn_z_loss_weight)
            return_intermediates = True

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out
