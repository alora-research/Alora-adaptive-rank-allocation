import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from alora.bert.lora_layer import AdaptiveLoRALinear


class BERTALoRA(nn.Module):

    def __init__(
        self,
        model_name="bert-base-uncased",
        num_labels=4,
        initial_rank=8,
        alpha=16,
    ):

        super().__init__()

        # ====================================================
        # LOAD BERT
        # ====================================================

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        )

        # ====================================================
        # FREEZE BERT
        # ====================================================

        for param in self.model.bert.parameters():

            param.requires_grad = False

        # ====================================================
        # CLASSIFIER TRAINABLE
        # ====================================================

        for param in self.model.classifier.parameters():

            param.requires_grad = True

        # ====================================================
        # INJECT A-LoRA INTO QUERY ONLY
        # ====================================================

        for layer in self.model.bert.encoder.layer:

            original_query = (
                layer.attention.self.query
            )

            layer.attention.self.query = (
                AdaptiveLoRALinear(
                    original_linear=original_query,
                    rank=initial_rank,
                    alpha=alpha
                )
            )

        # ====================================================
        # STORE QUERY LAYERS
        # ====================================================

        self.query_layers = [

            layer.attention.self.query

            for layer in self.model.bert.encoder.layer
        ]

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, **kwargs):

        return self.model(**kwargs)

    # ========================================================
    # GET QUERY LAYERS
    # ========================================================

    def get_query_layers(self):

        return self.query_layers

    # ========================================================
    # GET RANKS
    # ========================================================

    def get_ranks(self):

        return [
            layer.rank
            for layer in self.query_layers
        ]

    # ========================================================
    # GET TOTAL RANK
    # ========================================================

    def get_total_rank(self):

        return sum(
            self.get_ranks()
        )

    # ========================================================
    # COMPUTE IMPORTANCE
    # ========================================================

    def compute_importances(self):

        return [
            layer.compute_importance()
            for layer in self.query_layers
        ]

    # ========================================================
    # TRAINABLE PARAMETERS
    # ========================================================

    def get_trainable_parameters(self):

        return [
            p
            for p in self.parameters()
            if p.requires_grad
        ]

    # ========================================================
    # LoRA PARAMETERS
    # ========================================================

    def get_lora_parameters(self):

        parameters = []

        for layer in self.query_layers:

            parameters.append(layer.A)
            parameters.append(layer.B)

        return parameters

    # ========================================================
    # PARAMETER COUNTS
    # ========================================================

    def count_total_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
        )

    def count_trainable_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def count_lora_parameters(self):

        return sum(
            p.numel()
            for p in self.get_lora_parameters()
        )
