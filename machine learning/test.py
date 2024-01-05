from torch.nn import functional as F
from torch import randint, all, eq, Tensor, where

"""
out = randint(0, 3, (2, 10))
print(out)
is_eos_tokens = (out == 0)
print(is_eos_tokens)
shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
print(shifted_is_eos_tokens)
mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
print(mask)
out = out.masked_fill(mask, -1)
print(out)
print("\n\n\n")"""

out2 = randint(0, 2, (1, 10, 3))
print(out2)
is_eos_tokens = all(eq(out2[:, :, :], Tensor([1, 0, 0])), dim=-1)
print(is_eos_tokens)
shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
print(shifted_is_eos_tokens)
mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
print(mask)
out2 = where(mask.unsqueeze(-1), Tensor([1, 0, 0]), out2)
print(out2)
