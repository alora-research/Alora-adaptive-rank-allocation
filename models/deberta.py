import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from alora.deberta.lora_layer import AdaptiveLoRALinear


class DeBERTaALoRA(nn.Module):
    """
    DeBERTa-v3-base with A-LoRA applied to Query and Value
    projections in every encoder layer.

    The DeBERTa backbone is frozen.
    LoRA A/B parameters are trainable.
    The classification head can be enabled for downstream
    classification tasks.
    """

    def __init__(
        self,
        model_name="microsoft/deberta-v3-base",
        num_labels=2,
        rank=8,
        alpha=32.0,
        ema_beta=0.85,
        train_classifier=True,
    ):
        super().__init__()

        self.model_name = model_name
        self.rank = rank
        self.alpha = alpha
        self.ema_beta = ema_beta

        # Load the pretrained DeBERTa sequence-classification model.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

        # Keep the model in FP32.
        self.model.float()

        # Insert A-LoRA into Query + Value projections.
        self.lora_modules = []

        for layer_idx, layer in enumerate(
            self.model.deberta.encoder.layer
        ):
            # Original DeBERTa attention projections.
            query_proj = layer.attention.self.query_proj
            value_proj = layer.attention.self.value_proj

            # A-LoRA Query.
            query_lora = AdaptiveLoRALinear(
                base_layer=query_proj,
                rank=rank,
                alpha=alpha,
                name=f"L{layer_idx}_query_proj",
                ema_beta=ema_beta,
            )

            # A-LoRA Value.
            value_lora = AdaptiveLoRALinear(
                base_layer=value_proj,
                rank=rank,
                alpha=alpha,
                name=f"L{layer_idx}_value_proj",
                ema_beta=ema_beta,
            )

            layer.attention.self.query_proj = query_lora
            layer.attention.self.value_proj = value_lora

            self.lora_modules.append(query_lora)
            self.lora_modules.append(value_lora)

        # Freeze everything first.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Enable only LoRA parameters.
        for module in self.lora_modules:
            module.A.requires_grad = True
            module.B.requires_grad = True

        # For downstream classification, train the newly initialized
        # classification head.
        if train_classifier:
            for parameter in self.model.classifier.parameters():
                parameter.requires_grad = True

            # DeBERTa-v3 sequence classification also contains a pooler.
            if hasattr(self.model, "pooler"):
                for parameter in self.model.pooler.parameters():
                    parameter.requires_grad = True

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        """
        Forward pass through DeBERTa-v3.
        """

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

    def get_lora_modules(self):
        """Return all Query + Value A-LoRA modules."""

        return self.lora_modules

    def get_lora_parameters(self):
        """
        Return trainable LoRA A/B parameters.
        """

        parameters = []

        for module in self.lora_modules:
            parameters.append(module.A)
            parameters.append(module.B)

        return parameters

    def get_classifier_parameters(self):
        """Return classification-head parameters."""

        parameters = list(
            self.model.classifier.parameters()
        )

        if hasattr(self.model, "pooler"):
            parameters.extend(
                self.model.pooler.parameters()
            )

        return parameters

    def get_trainable_parameters(self):
        """Return all currently trainable parameters."""

        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

    def count_lora_parameters(self):
        """Count trainable LoRA parameters."""

        return sum(
            parameter.numel()
            for parameter in self.get_lora_parameters()
        )

    def count_trainable_parameters(self):
        """Count all trainable parameters."""

        return sum(
            parameter.numel()
            for parameter in self.get_trainable_parameters()
        )

    def get_rank_allocation(self):
        """Return the current rank of every LoRA module."""

        return {
            module.name: module.rank
            for module in self.lora_modules
        }

    def get_total_rank(self):
        """Return the current total LoRA rank."""

        return sum(
            module.rank
            for module in self.lora_modules
        )

    def update_importance(self):
        """
        Update gradient-based EMA importance for all
        LoRA modules.
        """

        for module in self.lora_modules:
            module.update_importance()

    def resize_module_rank(
        self,
        module_name,
        new_rank,
    ):
        """
        Resize one LoRA module by name.
        """

        for module in self.lora_modules:
            if module.name == module_name:
                module.resize_rank(new_rank)
                return

        raise ValueError(
            f"LoRA module not found: {module_name}"
        )

    def trainable_parameter_summary(self):
        """Return a compact parameter summary."""

        total = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        trainable = self.count_trainable_parameters()
        lora = self.count_lora_parameters()

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "lora_parameters": lora,
            "trainable_percentage": (
                100.0 * trainable / total
                if total > 0
                else 0.0
            ),
            "num_lora_modules": len(self.lora_modules),
            "total_rank": self.get_total_rank(),
        }