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

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

        # Freeze the pretrained BERT backbone.
        for param in self.model.bert.parameters():
            param.requires_grad = False

        # Keep the classification head trainable.
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # Inject A-LoRA into BERT self-attention query projections.
        for layer in self.model.bert.encoder.layer:
            original_query = layer.attention.self.query

            layer.attention.self.query = AdaptiveLoRALinear(
                original_linear=original_query,
                rank=initial_rank,
                alpha=alpha,
            )

        self.query_layers = [
            layer.attention.self.query
            for layer in self.model.bert.encoder.layer
        ]

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def get_query_layers(self):
        return self.query_layers

    def get_ranks(self):
        return [layer.rank for layer in self.query_layers]

    def get_rank_allocation(self):
        return self.get_ranks()

    def get_total_rank(self):
        return sum(self.get_ranks())

    def compute_importances(self):
        return [
            layer.compute_importance()
            for layer in self.query_layers
        ]

    def get_trainable_parameters(self):
        return [
            p for p in self.parameters()
            if p.requires_grad
        ]

    def get_classifier_parameters(self):
        return [
            p
            for p in self.model.classifier.parameters()
            if p.requires_grad
        ]

    def get_lora_parameters(self):
        parameters = []

        for layer in self.query_layers:
            parameters.append(layer.A)
            parameters.append(layer.B)

        return parameters

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
