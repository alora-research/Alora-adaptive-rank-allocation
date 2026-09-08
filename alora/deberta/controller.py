import math


class DeBERTaRankController:
    """
    A-LoRA rank controller for DeBERTa-v3.

    Rank is redistributed between LoRA modules while preserving
    a fixed global rank budget.

    Receiver score:
        normalized_importance / sqrt(rank / initial_rank)

    Donor score:
        normalized_importance * sqrt(rank / initial_rank)

    Receiver:
        module with maximum receiver score

    Donor:
        module with minimum donor score

    A rank transfer occurs when the receiver/donor score gap
    exceeds the configured threshold.
    """

    def __init__(
        self,
        modules,
        initial_rank=8,
        min_rank=2,
        max_rank=16,
        gap_threshold=0.25,
        max_rank_move=1,
    ):
        self.modules = list(modules)

        self.initial_rank = int(initial_rank)
        self.min_rank = int(min_rank)
        self.max_rank = int(max_rank)
        self.gap_threshold = float(gap_threshold)
        self.max_rank_move = int(max_rank_move)

        # Fixed global rank budget.
        self.global_rank_budget = sum(
            module.rank for module in self.modules
        )

    def update_importance(self):
        """
        Update EMA importance for every LoRA module.

        Importance is based on:
            mean(abs(B.grad))
        """

        for module in self.modules:
            module.update_importance()

    def _get_normalized_importance(self):
        """
        Normalize each module's EMA importance by
        the mean importance across all modules.

        Returns a list aligned exactly with self.modules.
        """

        importances = [
            module.get_importance()
            for module in self.modules
        ]

        if not importances:
            return []

        mean_importance = (
            sum(importances) / len(importances)
        )

        return [
            importance / (mean_importance + 1e-12)
            for importance in importances
        ]

    def _get_receiver_scores(self, normalized_importance):
        """
        Receiver score:

            normalized importance
            ----------------------
            sqrt(rank / initial_rank)

        High importance and low rank favor receiving rank.
        """

        scores = []

        for module, importance in zip(
            self.modules,
            normalized_importance,
        ):
            rank_factor = math.sqrt(
                module.rank / self.initial_rank
            )

            scores.append(
                importance / rank_factor
            )

        return scores

    def _get_donor_scores(self, normalized_importance):
        """
        Donor score:

            normalized importance
            *
            sqrt(rank / initial_rank)

        Low importance and high rank favor donating rank.
        """

        scores = []

        for module, importance in zip(
            self.modules,
            normalized_importance,
        ):
            rank_factor = math.sqrt(
                module.rank / self.initial_rank
            )

            scores.append(
                importance * rank_factor
            )

        return scores

    def step(self):
        """
        Perform one adaptive rank redistribution step.

        Returns:
            Dictionary describing the rank transfer,
            or None if no transfer is performed.
        """

        if not self.modules:
            return None

        # Match the QNLI implementation:
        # update EMA importance at the controller step.
        self.update_importance()

        normalized_importance = (
            self._get_normalized_importance()
        )

        receiver_scores = (
            self._get_receiver_scores(
                normalized_importance
            )
        )

        donor_scores = (
            self._get_donor_scores(
                normalized_importance
            )
        )

        # Select indices rather than names.
        receiver_idx = max(
            range(len(self.modules)),
            key=lambda i: receiver_scores[i],
        )

        donor_idx = min(
            range(len(self.modules)),
            key=lambda i: donor_scores[i],
        )

        # A module cannot donate rank to itself.
        if receiver_idx == donor_idx:
            return None

        # Directly use the selected module objects.
        receiver = self.modules[receiver_idx]
        donor = self.modules[donor_idx]

        receiver_score = receiver_scores[receiver_idx]
        donor_score = donor_scores[donor_idx]

        # QNLI implementation:
        # gap is calculated using rank-aware
        # receiver and donor scores.
        gap = (
            receiver_score - donor_score
        ) / (
            abs(donor_score) + 1e-12
        )

        # Do not transfer unless the gap is sufficiently large.
        if gap < self.gap_threshold:
            return None

        # Receiver must be below maximum rank.
        if receiver.rank >= self.max_rank:
            return None

        # Donor must be above minimum rank.
        if donor.rank <= self.min_rank:
            return None

        # Transfer one rank at a time.
        delta = min(
            self.max_rank_move,
            1,
            self.max_rank - receiver.rank,
            donor.rank - self.min_rank,
        )

        if delta <= 0:
            return None

        old_receiver_rank = receiver.rank
        old_donor_rank = donor.rank

        # ---------------------------------------------------------
        # Physically resize BOTH modules.
        # ---------------------------------------------------------

        receiver.resize_rank(
            old_receiver_rank + delta
        )

        donor.resize_rank(
            old_donor_rank - delta
        )

        # ---------------------------------------------------------
        # Verify fixed global rank budget.
        # ---------------------------------------------------------

        current_budget = self.get_total_rank()

        assert current_budget == self.global_rank_budget, (
            f"Global rank budget violated: "
            f"{current_budget} != "
            f"{self.global_rank_budget}"
        )

        return {
            "receiver": receiver.name,
            "donor": donor.name,
            "receiver_old_rank": old_receiver_rank,
            "receiver_new_rank": receiver.rank,
            "donor_old_rank": old_donor_rank,
            "donor_new_rank": donor.rank,
            "receiver_score": receiver_score,
            "donor_score": donor_score,
            "gap": gap,
            "global_rank_budget": current_budget,
        }

    def get_rank_allocation(self):
        """Return the current rank allocation."""

        return {
            module.name: module.rank
            for module in self.modules
        }

    def get_total_rank(self):
        """Return the current total rank."""

        return sum(
            module.rank
            for module in self.modules
        )

    def verify_budget(self):
        """Verify preservation of the fixed global rank budget."""

        current_budget = self.get_total_rank()

        assert current_budget == self.global_rank_budget, (
            f"Global rank budget violated: "
            f"{current_budget} != "
            f"{self.global_rank_budget}"
        )

        return True