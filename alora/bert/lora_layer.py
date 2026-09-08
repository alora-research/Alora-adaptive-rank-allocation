import torch
import torch.nn as nn


class AdaptiveLoRALinear(nn.Module):

    def __init__(
        self,
        original_linear,
        rank=8,
        alpha=16
    ):

        super().__init__()

        self.rank = rank
        self.alpha = alpha

        self.in_features = (
            original_linear.in_features
        )

        self.out_features = (
            original_linear.out_features
        )

        # ----------------------------------------------------
        # FROZEN BASE WEIGHT
        # ----------------------------------------------------

        self.weight = (
            original_linear.weight
        )

        self.weight.requires_grad = False

        # ----------------------------------------------------
        # FROZEN BIAS
        # ----------------------------------------------------

        self.bias = (
            original_linear.bias
        )

        if self.bias is not None:
            self.bias.requires_grad = False

        # ----------------------------------------------------
        # LoRA A
        # ----------------------------------------------------

        self.A = nn.Parameter(

            torch.randn(
                rank,
                self.in_features
            ) * 0.01
        )

        # ----------------------------------------------------
        # LoRA B
        # ----------------------------------------------------

        self.B = nn.Parameter(

            torch.zeros(
                self.out_features,
                rank
            )
        )

        self.scale = (
            alpha / rank
        )

        self.importance = 0.0

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        base = (
            x @ self.weight.T
        )

        if self.bias is not None:

            base = (
                base + self.bias
            )

        lora = (
            x @ self.A.T
        )

        lora = (
            lora @ self.B.T
        )

        return (
            base
            +
            self.scale * lora
        )

    # ========================================================
    # IMPORTANCE
    # ========================================================

    def compute_importance(self):

        if self.B.grad is None:

            self.importance = 0.0

        else:

            self.importance = (

                self.B.grad
                .detach()
                .abs()
                .mean()
                .item()
            )

        return self.importance

    # ========================================================
    # RESIZE RANK
    # ========================================================

    def resize_rank(
        self,
        new_rank
    ):

        # AG News global limits
        MIN_RANK = 2
        MAX_RANK = 16

        new_rank = int(

            max(
                MIN_RANK,
                min(
                    MAX_RANK,
                    new_rank
                )
            )
        )

        if new_rank == self.rank:

            return False

        old_rank = self.rank

        old_A = (
            self.A.detach()
            .clone()
        )

        old_B = (
            self.B.detach()
            .clone()
        )

        new_A = torch.zeros(

            new_rank,

            self.in_features,

            device=old_A.device,

            dtype=old_A.dtype
        )

        new_B = torch.zeros(

            self.out_features,

            new_rank,

            device=old_B.device,

            dtype=old_B.dtype
        )

        keep = min(
            old_rank,
            new_rank
        )

        # Preserve learned components
        new_A[:keep] = (
            old_A[:keep]
        )

        new_B[:, :keep] = (
            old_B[:, :keep]
        )

        # Initialize new components
        if new_rank > old_rank:

            nn.init.normal_(

                new_A[old_rank:],

                mean=0.0,

                std=0.01
            )

        self.rank = new_rank

        self.A = nn.Parameter(
            new_A
        )

        self.B = nn.Parameter(
            new_B
        )

        self.scale = (
            self.alpha / new_rank
        )

        return True
