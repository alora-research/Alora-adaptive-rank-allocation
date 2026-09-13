import torch
import torch.nn as nn

from src.models.adaptive_lora import AdaptiveLoRA


def get_target_modules(model, backbone, target_projections):
    """
    Extract (layer_idx, proj_name, parent, module_name, original_module)
    tuples for the requested target projections.

    backbone: "bert" or "deberta"
    target_projections: list like ["query"] or ["query", "key", "value", "output"]
    """
    target_modules = []

    if backbone == "bert":
        layers = model.bert.encoder.layer
        for layer_idx, layer in enumerate(layers):
            if "query" in target_projections:
                target_modules.append(
                    (layer_idx, "query", layer.attention.self, "query", layer.attention.self.query)
                )
            if "key" in target_projections:
                target_modules.append(
                    (layer_idx, "key", layer.attention.self, "key", layer.attention.self.key)
                )
            if "value" in target_projections:
                target_modules.append(
                    (layer_idx, "value", layer.attention.self, "value", layer.attention.self.value)
                )
            if "output" in target_projections:
                target_modules.append(
                    (layer_idx, "output", layer.attention.output, "dense", layer.attention.output.dense)
                )

    elif backbone == "deberta":
        layers = model.deberta.encoder.layer
        for layer_idx, layer in enumerate(layers):
            if "query" in target_projections:
                target_modules.append(
                    (layer_idx, "query", layer.attention.self, "query_proj", layer.attention.self.query_proj)
                )
            if "key" in target_projections:
                target_modules.append(
                    (layer_idx, "key", layer.attention.self, "key_proj", layer.attention.self.key_proj)
                )
            if "value" in target_projections:
                target_modules.append(
                    (layer_idx, "value", layer.attention.self, "value_proj", layer.attention.self.value_proj)
                )
            if "output" in target_projections:
                target_modules.append(
                    (layer_idx, "output", layer.attention.output, "dense", layer.attention.output.dense)
                )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return target_modules


def allocate_ranks(total_rank, num_modules):
    """Distribute total_rank as evenly as possible across num_modules."""
    base_rank = total_rank // num_modules
    extra = total_rank % num_modules
    ranks = [base_rank + (1 if i < extra else 0) for i in range(num_modules)]
    assert sum(ranks) == total_rank
    return ranks


def insert_lora_adapters(model, backbone, target_projections, total_rank, alpha):
    """Insert AdaptiveLoRA modules into the model and freeze the backbone."""
    target_modules = get_target_modules(model, backbone, target_projections)
    ranks = allocate_ranks(total_rank, len(target_modules))

    adapters = {}
    for i, (layer_idx, proj_name, parent, module_name, original_module) in enumerate(target_modules):
        adapter = AdaptiveLoRA(original_module, rank=ranks[i], alpha=alpha)
        setattr(parent, module_name, adapter)
        key = f"layer{layer_idx}.{proj_name}"
        adapters[key] = adapter

    for param in model.parameters():
        param.requires_grad = False

    for adapter in adapters.values():
        adapter.A.requires_grad = True
        adapter.B.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    return adapters


def make_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8
    )
