import torch
import torch.nn as nn


class AdaptiveLoRA(nn.Module):
    """LoRA module supporting in-place rank resizing for Seesaw reallocation."""

    def __init__(self, original_linear, rank, alpha):
        super().__init__()

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha

        self.register_buffer(
            "weight_orig",
            original_linear.weight.detach().clone().float(),
        )

        if original_linear.bias is not None:
            self.register_buffer(
                "bias_orig",
                original_linear.bias.detach().clone().float(),
            )
        else:
            self.bias_orig = None

        self.A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32))

        nn.init.normal_(self.A, std=0.02)

    def forward(self, x):
        x = x.float()
        base = torch.nn.functional.linear(x, self.weight_orig, self.bias_orig)
        update = torch.nn.functional.linear(x, self.A)
        update = torch.nn.functional.linear(update, self.B)
        update = update * (self.alpha / float(self.rank))
        return base + update

    def resize_rank(self, new_rank):
        """Resize A/B in place, preserving existing weights (Section 3.5)."""
        if new_rank == self.rank:
            return

        old_A = self.A.data
        old_B = self.B.data
        old_rank = self.rank

        new_A = torch.empty(new_rank, self.in_features, device=old_A.device, dtype=old_A.dtype)
        new_B = torch.zeros(self.out_features, new_rank, device=old_B.device, dtype=old_B.dtype)

        keep = min(old_rank, new_rank)
        new_A[:keep].copy_(old_A[:keep])
        new_B[:, :keep].copy_(old_B[:, :keep])

        if new_rank > old_rank:
            nn.init.normal_(new_A[old_rank:], std=0.02)

        self.A = nn.Parameter(new_A)
        self.B = nn.Parameter(new_B)
        self.rank = new_rank
