import math
import torch
import torch.nn as nn


class AdaptiveLoRALinear(nn.Module):
    """
    Adaptive LoRA linear layer for DeBERTa-v3.

    Base weight and bias are frozen.
    Only LoRA A and B are trainable.

    Importance:
        mean(abs(B.grad))

    Importance EMA:
        importance_ema = beta * old + (1-beta) * raw

    Rank can be increased or decreased while preserving
    the learned LoRA components.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float = 32.0,
        name: str = "",
        ema_beta: float = 0.85,
    ):
        super().__init__()

        self.name = name
        self.rank = rank
        self.alpha = alpha
        self.ema_beta = ema_beta

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Frozen base parameters.
        self.weight = nn.Parameter(
            base_layer.weight.detach().clone().float(),
            requires_grad=False,
        )

        if base_layer.bias is not None:
            self.bias = nn.Parameter(
                base_layer.bias.detach().clone().float(),
                requires_grad=False,
            )
        else:
            self.bias = None

        # Trainable LoRA parameters.
        self.A = nn.Parameter(
            torch.randn(
                rank,
                self.in_features,
                dtype=torch.float32,
                device=self.weight.device,
            ) * 0.02
        )

        self.B = nn.Parameter(
            torch.zeros(
                self.out_features,
                rank,
                dtype=torch.float32,
                device=self.weight.device,
            )
        )

        # EMA importance.
        self.importance_ema = 0.0

    @property
    def scaling(self):
        return self.alpha / self.rank

    def forward(self, x):
        # Frozen base projection.
        base_output = torch.nn.functional.linear(
            x,
            self.weight,
            self.bias,
        )

        # LoRA projection.
        lora_output = (
            torch.nn.functional.linear(
                torch.nn.functional.linear(x, self.A),
                self.B,
            )
            * self.scaling
        )

        return base_output + lora_output

    @torch.no_grad()
    def update_importance(self):
        """
        Update gradient-based importance using mean absolute
        B gradient followed by EMA.

        B starts at zero, so we intentionally do NOT normalize
        by B magnitude.
        """

        if self.B.grad is None:
            return self.importance_ema

        raw_importance = self.B.grad.abs().mean().item()

        self.importance_ema = (
            self.ema_beta * self.importance_ema
            + (1.0 - self.ema_beta) * raw_importance
        )

        return self.importance_ema

    @torch.no_grad()
    def get_importance(self):
        return self.importance_ema

    @torch.no_grad()
    def resize_rank(self, new_rank: int):
        """
        Resize LoRA rank while preserving existing learned
        components.

        Increasing rank:
            - preserve existing A/B
            - initialize new A rows with std=0.02
            - initialize new B columns with zeros

        Decreasing rank:
            - keep the first new_rank components
        """

        if new_rank == self.rank:
            return

        old_rank = self.rank

        # Always use the current LoRA parameter device/dtype.
        target_device = self.A.device
        target_dtype = self.A.dtype

        new_A = torch.empty(
            new_rank,
            self.in_features,
            device=target_device,
            dtype=target_dtype,
        )

        new_B = torch.empty(
            self.out_features,
            new_rank,
            device=target_device,
            dtype=target_dtype,
        )

        if new_rank > old_rank:
            # Preserve existing components.
            new_A[:old_rank].copy_(self.A.data)
            new_B[:, :old_rank].copy_(self.B.data)

            # Initialize additional components.
            nn.init.normal_(
                new_A[old_rank:],
                mean=0.0,
                std=0.02,
            )

            new_B[:, old_rank:].zero_()

        else:
            # Preserve the first new_rank components.
            new_A.copy_(self.A.data[:new_rank])
            new_B.copy_(self.B.data[:, :new_rank])

        # Replace trainable parameters.
        self.A = nn.Parameter(new_A)
        self.B = nn.Parameter(new_B)

        self.rank = new_rank

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}"
        )