import numpy as np


class SeesawController:
    """
    Implements the threshold-gated rank transfer mechanism (Section 3.4).

    Tracks an importance score per adapter (epoch-averaged or EMA-smoothed),
    normalizes scores, and transfers a single rank unit from the lowest-
    scoring donor to the highest-scoring receiver when their gap clears
    a threshold tau.
    """

    def __init__(
        self,
        adapters,
        aggregation="ema",       # "ema" (Seesaw-v2) or "epoch_average" (Seesaw-v1)
        ema_decay=0.85,
        normalization="zscore",  # "none" | "mean" | "mean_centered" | "zscore"
        importance_gap=0.25,
        min_rank=2,
        max_rank=64,
        target_total_rank=None,
    ):
        self.adapters = adapters
        self.aggregation = aggregation
        self.ema_decay = ema_decay
        self.normalization = normalization
        self.importance_gap = importance_gap
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.target_total_rank = target_total_rank or sum(a.rank for a in adapters.values())

        self.importance_score = {key: 0.0 for key in adapters}
        self._epoch_batch_scores = {key: [] for key in adapters}
        self.transfer_count = 0

    def accumulate_batch(self):
        """Call after loss.backward(), before optimizer.step(), every batch."""
        for key, adapter in self.adapters.items():
            if adapter.B.grad is not None:
                importance = adapter.B.grad.detach().float().abs().mean().item()

                if self.aggregation == "ema":
                    self.importance_score[key] = (
                        self.ema_decay * self.importance_score[key]
                        + (1.0 - self.ema_decay) * importance
                    )
                elif self.aggregation == "epoch_average":
                    self._epoch_batch_scores[key].append(importance)
                else:
                    raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def end_epoch(self):
        """Call at the end of each epoch. Only relevant for epoch_average aggregation."""
        if self.aggregation == "epoch_average":
            for key in self.adapters:
                batch_scores = self._epoch_batch_scores[key]
                self.importance_score[key] = (
                    float(np.mean(batch_scores)) if batch_scores else 0.0
                )
                self._epoch_batch_scores[key] = []

    def step(self, make_optimizer_fn):
        """
        Attempt a single rank transfer. Returns a dict describing the outcome,
        or None if no transfer occurred. `make_optimizer_fn` rebuilds the
        optimizer after resizing (Section 3.5).
        """
        from src.controller.normalization import normalize_scores

        keys = list(self.adapters.keys())
        scores = np.asarray([self.importance_score[k] for k in keys], dtype=np.float64)

        if np.all(scores <= 0):
            return {"status": "no_gradient_signal"}

        normalized = normalize_scores(scores, self.normalization)

        receiver_idx = int(np.argmax(normalized))
        donor_idx = int(np.argmin(normalized))

        receiver_key = keys[receiver_idx]
        donor_key = keys[donor_idx]

        receiver_adapter = self.adapters[receiver_key]
        donor_adapter = self.adapters[donor_key]

        gap = normalized[receiver_idx] - normalized[donor_idx]

        if gap < self.importance_gap:
            return {"status": "no_transfer", "gap": gap}

        if donor_adapter.rank <= self.min_rank:
            return {"status": "donor_at_min_rank", "donor": donor_key}

        if receiver_adapter.rank >= self.max_rank:
            return {"status": "receiver_at_max_rank", "receiver": receiver_key}

        old_receiver_rank = receiver_adapter.rank
        old_donor_rank = donor_adapter.rank

        receiver_adapter.resize_rank(old_receiver_rank + 1)
        donor_adapter.resize_rank(old_donor_rank - 1)

        new_optimizer = make_optimizer_fn()

        self.transfer_count += 1

        current_total = sum(a.rank for a in self.adapters.values())
        assert current_total == self.target_total_rank, (
            f"Rank budget violated: {current_total} != {self.target_total_rank}"
        )

        return {
            "status": "transfer",
            "donor": donor_key,
            "donor_old_rank": old_donor_rank,
            "donor_new_rank": donor_adapter.rank,
            "receiver": receiver_key,
            "receiver_old_rank": old_receiver_rank,
            "receiver_new_rank": receiver_adapter.rank,
            "gap": gap,
            "optimizer": new_optimizer,
        }
