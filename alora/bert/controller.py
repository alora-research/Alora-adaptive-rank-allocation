import numpy as np


class BERTRankController:

    def __init__(
        self,
        query_layers,
        min_rank=2,
        max_rank=16,
        max_rank_move=1,
        importance_gap=0.10
    ):

        self.query_layers = query_layers

        self.min_rank = min_rank
        self.max_rank = max_rank
        self.max_rank_move = max_rank_move
        self.importance_gap = importance_gap

        self.num_layers = len(
            query_layers
        )

        # Fixed global rank budget
        self.total_rank_budget = sum(
            q.rank
            for q in query_layers
        )

    # ========================================================
    # GET IMPORTANCES
    # ========================================================

    def get_importances(self):

        return np.array(

            [
                q.importance
                for q in self.query_layers
            ],

            dtype=np.float64
        )

    # ========================================================
    # GET RANKS
    # ========================================================

    def get_ranks(self):

        return [
            q.rank
            for q in self.query_layers
        ]

    # ========================================================
    # REDISTRIBUTE RANK BUDGET
    # ========================================================

    def redistribute_rank_budget(self):

        current_ranks = [

            q.rank
            for q in self.query_layers
        ]

        importances = self.get_importances()

        if np.all(
            importances <= 0
        ):

            print(
                "No positive importance."
            )

            return False

        # ----------------------------------------------------
        # Highest importance first
        # ----------------------------------------------------

        high_order = np.argsort(
            -importances
        )

        # ----------------------------------------------------
        # Lowest importance first
        # ----------------------------------------------------

        low_order = np.argsort(
            importances
        )

        receiver_idx = None
        donor_idx = None

        # ----------------------------------------------------
        # FIND RECEIVER
        # ----------------------------------------------------

        for idx in high_order:

            if current_ranks[idx] < self.max_rank:

                receiver_idx = int(idx)

                break

        # ----------------------------------------------------
        # FIND DONOR
        # ----------------------------------------------------

        for idx in low_order:

            if current_ranks[idx] > self.min_rank:

                donor_idx = int(idx)

                break

        if (
            receiver_idx is None
            or donor_idx is None
        ):

            return False

        if receiver_idx == donor_idx:

            return False

        high_importance = (
            importances[
                receiver_idx
            ]
        )

        low_importance = (
            importances[
                donor_idx
            ]
        )

        # ----------------------------------------------------
        # IMPORTANCE GAP
        # ----------------------------------------------------

        relative_gap = (

            high_importance
            -
            low_importance

        ) / (

            high_importance
            +
            1e-12
        )

        if (
            relative_gap
            <
            self.importance_gap
        ):

            return False

        # ----------------------------------------------------
        # TRANSFER RANK
        # ----------------------------------------------------

        move = min(

            self.max_rank_move,

            self.max_rank
            -
            current_ranks[receiver_idx],

            current_ranks[donor_idx]
            -
            self.min_rank
        )

        if move <= 0:

            return False

        old_receiver_rank = (
            current_ranks[
                receiver_idx
            ]
        )

        old_donor_rank = (
            current_ranks[
                donor_idx
            ]
        )

        self.query_layers[
            donor_idx
        ].resize_rank(

            old_donor_rank
            -
            move
        )

        self.query_layers[
            receiver_idx
        ].resize_rank(

            old_receiver_rank
            +
            move
        )

        # ----------------------------------------------------
        # VERIFY FIXED GLOBAL BUDGET
        # ----------------------------------------------------

        new_ranks = [

            q.rank
            for q in self.query_layers
        ]

        if sum(new_ranks) != self.total_rank_budget:

            raise RuntimeError(

                f"Rank budget violated: "
                f"{sum(new_ranks)} != "
                f"{self.total_rank_budget}"
            )

        print()
        print("=" * 60)
        print("RANK TRANSFER")
        print("=" * 60)

        print(

            f"Layer {donor_idx + 1}: "
            f"{old_donor_rank} -> "
            f"{self.query_layers[donor_idx].rank}"
        )

        print(

            f"Layer {receiver_idx + 1}: "
            f"{old_receiver_rank} -> "
            f"{self.query_layers[receiver_idx].rank}"
        )

        print(
            "Total rank:",
            sum(new_ranks)
        )

        print("=" * 60)

        return True

    # ========================================================
    # VERIFY BUDGET
    # ========================================================

    def verify_budget(self):

        total = sum(
            q.rank
            for q in self.query_layers
        )

        if total != self.total_rank_budget:

            raise RuntimeError(

                f"GLOBAL RANK BUDGET VIOLATION: "
                f"{total} != "
                f"{self.total_rank_budget}"
            )

        return True
