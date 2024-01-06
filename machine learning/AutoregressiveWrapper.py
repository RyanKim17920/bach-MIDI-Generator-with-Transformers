from x_transformers.x_transformers import *
from x_transformers.autoregressive_wrapper import *

# based on x_transformers
# three token types = values,time, instruments
class AutoregressiveWrapper(Module):
    def __init__(
            self,
            net,
            ignore_index=-100,
            pad_value=0,
            mask_prob=0.,
            add_attn_z_loss=False
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big
        # improvements https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.
        self.mask_prob = mask_prob

        # whether to add router z-loss
        self.add_attn_z_loss = add_attn_z_loss

    @torch.no_grad()
    @eval_decorator
    def generate(
            self,
            prompts,
            seq_len,
            eos_token=None,
            eos_first_token=None,
            temperature=1.,
            prompt_lens: Optional[Tensor] = None,
            filter_logits_fn: Callable = top_k,
            restrict_to_max_seq_len=True,
            filter_kwargs: dict = dict(),
            cache_kv=True,
            **kwargs
    ):
        max_seq_len, greedy, device = self.max_seq_len, temperature == 0., prompts.device

        # prompts, ps = pack([prompts], '* n')
        # print(prompts.shape)
        b, t, _ = prompts.shape

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id=self.pad_value)
            seq_start_pos = t - prompt_lens

        # output from which sampled tokens appended to

        out = prompts

        # kv caches

        cache = None

        # sampling up to seq_len
        is_eos_tokens = None
        for _ in range(seq_len):

            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[1] > max_seq_len

                assert not (
                        cache_kv and max_len_exceeded and not self.net.can_cache_kv_outside_max_seq_len), 'the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embeeding. you can switch to rotary embeddings to resolve this issue'

                x = out[:, -max_seq_len:, :]

                if exists(cache):
                    for inter in cache.attn_intermediates:
                        inter.cached_kv = [t[..., -(max_seq_len - 1):, :] for t in inter.cached_kv]

            (logits_values, logits_times, logits_instruments), new_cache = self.net(
                x,
                return_intermediates=True,
                cache=cache,
                seq_start_pos=seq_start_pos,
                **kwargs
            )

            logits_values = logits_values[:, -1]
            logits_times = logits_times[:, -1]
            logits_instruments = logits_instruments[:, -1]
            # handle contrastive decoding, Li et al.
            # https://arxiv.org/abs/2210.15097

            # filter by top_k, top_p (nucleus), top_a, or custom

            if greedy:
                sample = torch.Tensor([logits_values.argmax(dim=-1, keepdim=True),
                                       logits_times.argmax(dim=-1, keepdim=True),
                                       logits_instruments.argmax(dim=-1, keepdim=True)])
            else:
                filtered_logits_values = filter_logits_fn(logits_values, **filter_kwargs)
                filtered_logits_times = filter_logits_fn(logits_times, **filter_kwargs)
                filtered_logits_instruments = filter_logits_fn(logits_instruments, **filter_kwargs)
                probs_values = F.softmax(filtered_logits_values / temperature, dim=-1)
                probs_times = F.softmax(filtered_logits_times / temperature, dim=-1)
                probs_instruments = F.softmax(filtered_logits_instruments / temperature, dim=-1)
                sample = torch.Tensor([[[torch.multinomial(probs_values, 1),
                                         torch.multinomial(probs_times, 1),
                                         torch.multinomial(probs_instruments, 1)]]])

            # concat sample

            out = torch.cat((out, sample), dim=1)

            if not exists(eos_token) and not exists(eos_first_token):
                continue

            if exists(eos_token):
                is_eos_tokens = torch.all(torch.eq(out[:, :, :], eos_token), dim=-1)
                if torch.any(is_eos_tokens, dim=-1):
                    break

            if exists(eos_first_token):
                is_eos_tokens = (out[:, :, 0] == eos_first_token)
                if torch.any(is_eos_tokens, dim=-1):
                    break
        if exists(eos_token) or exists(eos_first_token) or (is_eos_tokens is not None):
            # mask out everything after the eos tokens
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
            out = torch.where(mask.unsqueeze(-1), Tensor([1, 0, 0]), out)
        out = out[:, t:]

        # out, = unpack(out, ps, '* n')

        return out

    def forward(self, x, return_outputs=False, **kwargs):
        seq, ignore_index, add_attn_z_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss

        inp, target = x[:, :-1], x[:, 1:]
        target_values = target[:, :, 0]
        target_times = target[:, :, 1]
        target_instruments = target[:, :, 2]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        if self.mask_prob > 0.:
            rand = torch.randn(inp.shape, device=x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max  # first token should not be masked out
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim=-1).indices
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.).bool()
            kwargs.update(self_attn_kv_mask=mask)

        (logits_values, logits_times, logits_instruments), cache = self.net(
            inp,
            return_intermediates=True,
            return_attn_z_loss=add_attn_z_loss,
            **kwargs
        )

        loss_values = F.cross_entropy(
            rearrange(logits_values, 'b n c -> b c n'),
            target_values,
            ignore_index=ignore_index
        )
        loss_times = F.cross_entropy(
            rearrange(logits_times, 'b n c -> b c n'),
            target_times,
            ignore_index=ignore_index
        )
        loss_instruments = F.cross_entropy(
            rearrange(logits_instruments, 'b n c -> b c n'),
            target_instruments,
            ignore_index=ignore_index
        )
        loss = loss_values + loss_times + loss_instruments
        if add_attn_z_loss:
            loss = loss + cache.attn_z_loss

        if not return_outputs:
            return loss

        return loss, (logits_values, logits_times, logits_instruments, cache)
