import argparse
import os
import random

import numpy as np
import torch
import yaml
from datasets import Value, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from models.deberta import DeBERTaALoRA
from alora.deberta.controller import DeBERTaRankController


SUPPORTED_TASKS = {"qqp", "qnli", "sst2", "stsb"}
REGRESSION_TASKS = {"stsb"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_task_config(config, task):
    defaults = dict(config.get("training", {}))
    task_config = dict(config.get("tasks", {}).get(task, {}))
    defaults.update(task_config)
    return defaults


def build_glue_dataloaders(task, training_config, tokenizer):
    dataset = load_dataset("glue", task)

    max_length = int(training_config.get("max_length", 128))
    batch_size = int(training_config.get("batch_size", 32))
    eval_batch_size = int(
        training_config.get("eval_batch_size", batch_size)
    )

    sentence_keys = {
        "qqp": ("question1", "question2"),
        "qnli": ("question", "sentence"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = sentence_keys[task]

    def tokenize(batch):
        if sentence2_key is None:
            return tokenizer(
                batch[sentence1_key],
                truncation=True,
                max_length=max_length,
            )
        return tokenizer(
            batch[sentence1_key],
            batch[sentence2_key],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
    )

    tokenized = tokenized.rename_column("label", "labels")

    if task == "stsb":
        tokenized = tokenized.cast_column(
            "labels",
            Value("float32"),
        )

    keep_columns = {
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "labels",
    }

    remove_columns = [
        column
        for column in tokenized["train"].column_names
        if column not in keep_columns
    ]

    tokenized = tokenized.remove_columns(remove_columns)
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    eval_split = "validation" if "validation" in tokenized else "test"

    eval_loader = DataLoader(
        tokenized[eval_split],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    return train_loader, eval_loader


def build_optimizer(model, training_config):
    lr = float(training_config.get("learning_rate", 5e-4))
    classifier_lr = float(
        training_config.get("classifier_learning_rate", lr)
    )
    weight_decay = float(
        training_config.get("weight_decay", 0.01)
    )

    lora_params = [
        p for p in model.get_lora_parameters()
        if p.requires_grad
    ]

    classifier_params = [
        p for p in model.get_classifier_parameters()
        if p.requires_grad
    ]

    groups = [
        {
            "params": lora_params,
            "lr": lr,
            "weight_decay": weight_decay,
        }
    ]

    if classifier_params:
        groups.append(
            {
                "params": classifier_params,
                "lr": classifier_lr,
                "weight_decay": weight_decay,
            }
        )

    return AdamW(groups)


def build_scheduler(optimizer, training_config, total_steps):
    warmup_ratio = float(
        training_config.get("warmup_ratio", 0.10)
    )
    warmup_steps = int(total_steps * warmup_ratio)

    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def classification_metrics(predictions, labels):
    accuracy = (predictions == labels).float().mean().item()
    return {"accuracy": accuracy, "selection_metric": accuracy}


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1

        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j

    return ranks


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


@torch.no_grad()
def evaluate(model, loader, device, task):
    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_predictions = []
    all_labels = []

    for batch in loader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)
        labels = batch["labels"]
        n = labels.size(0)

        total_loss += outputs.loss.item() * n
        total_examples += n

        if task in REGRESSION_TASKS:
            predictions = outputs.logits.squeeze(-1)
            all_predictions.extend(
                predictions.float().cpu().numpy().tolist()
            )
            all_labels.extend(
                labels.float().cpu().numpy().tolist()
            )
        else:
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    loss = total_loss / max(total_examples, 1)

    if task in REGRESSION_TASKS:
        p = pearson(all_predictions, all_labels)
        s = spearman(all_predictions, all_labels)
        return {
            "loss": loss,
            "pearson": p,
            "spearman": s,
            "selection_metric": p,
            "selection_name": "pearson",
        }

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    metrics = classification_metrics(predictions, labels)

    return {
        "loss": loss,
        "accuracy": metrics["accuracy"],
        "selection_metric": metrics["selection_metric"],
        "selection_name": "accuracy",
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    max_grad_norm,
):
    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        outputs = model(**batch)
        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected.")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

        # Gradient-based importance is measured after backward.
        model.update_importance()

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        n = batch["labels"].size(0)
        total_loss += loss.item() * n
        total_examples += n

    return total_loss / max(total_examples, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Universal DeBERTa-v3 A-LoRA training runner."
    )
    parser.add_argument(
        "--task",
        choices=sorted(SUPPORTED_TASKS),
        required=True,
    )
    parser.add_argument(
        "--config",
        default="configs/deberta.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
    )
    args = parser.parse_args()

    config = load_config(args.config)
    task_config = build_task_config(config, args.task)

    set_seed(args.seed)
    device = get_device()

    model_config = config["model"]
    alora_config = config["alora"]

    model_name = model_config["name"]
    num_labels = 1 if args.task == "stsb" else int(
        task_config.get("num_labels", 2)
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            "outputs",
            args.task,
            f"seed_{args.seed}",
        )

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 64)
    print("A-LoRA: Adaptive Rank Allocation")
    print("=" * 64)
    print(f"Model:   {model_name}")
    print(f"Task:    {args.task}")
    print(f"Device:  {device}")
    print(f"Seed:    {args.seed}")
    print("=" * 64)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = DeBERTaALoRA(
        model_name=model_name,
        num_labels=num_labels,
        rank=int(model_config["rank"]),
        alpha=float(model_config["alpha"]),
        ema_beta=float(alora_config["ema_beta"]),
        train_classifier=bool(
            model_config.get("train_classifier", True)
        ),
    )

    train_loader, eval_loader = build_glue_dataloaders(
        args.task,
        task_config,
        tokenizer,
    )

    controller = DeBERTaRankController(
        modules=model.get_lora_modules(),
        initial_rank=int(model_config["rank"]),
        min_rank=int(alora_config["min_rank"]),
        max_rank=int(alora_config["max_rank"]),
        gap_threshold=float(alora_config["gap_threshold"]),
        max_rank_move=int(alora_config["max_rank_move"]),
    )

    model.to(device)

    print(
        f"Total parameters:     "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    print(
        f"Trainable parameters:  "
        f"{model.count_trainable_parameters():,}"
    )
    print(
        f"LoRA parameters:       "
        f"{model.count_lora_parameters():,}"
    )
    print(
        f"Initial rank budget:   "
        f"{model.get_total_rank()}"
    )

    epochs = int(task_config["epochs"])
    total_steps = len(train_loader) * epochs

    optimizer = build_optimizer(model, task_config)
    scheduler = build_scheduler(
        optimizer,
        task_config,
        total_steps,
    )

    max_grad_norm = float(
        task_config.get("max_grad_norm", 1.0)
    )

    best_metric = -float("inf")
    best_epoch = None

    start_epoch = int(
        alora_config.get("controller_start_epoch", 2)
    )
    controller_interval = int(
        alora_config.get("controller_interval", 2)
    )

    for epoch in range(1, epochs + 1):
        print()
        print(f"===== Epoch {epoch}/{epochs} =====")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=max_grad_norm,
        )

        metrics = evaluate(
            model,
            eval_loader,
            device,
            args.task,
        )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Eval loss:  {metrics['loss']:.4f}")

        if args.task == "stsb":
            print(f"Pearson:    {metrics['pearson']:.4f}")
            print(f"Spearman:   {metrics['spearman']:.4f}")
        else:
            print(
                f"Accuracy:   "
                f"{metrics['accuracy'] * 100:.2f}%"
            )

        # --------------------------------------------------------
        # A-LoRA controller
        # --------------------------------------------------------

        rank_changed = False

        if (
            epoch >= start_epoch
            and epoch % controller_interval == 0
        ):
            print()
            print("Running A-LoRA rank controller...")

            result = controller.step()

            if result is not None:
                rank_changed = True

                print(
                    f"Receiver: {result['receiver']} "
                    f"{result['receiver_old_rank']} -> "
                    f"{result['receiver_new_rank']}"
                )
                print(
                    f"Donor:    {result['donor']} "
                    f"{result['donor_old_rank']} -> "
                    f"{result['donor_new_rank']}"
                )
                print(f"Gap:      {result['gap']:.6f}")
            else:
                print("No rank transfer performed.")

        controller.verify_budget()

        print(
            f"Current rank budget: {model.get_total_rank()}"
        )

        # --------------------------------------------------------
        # Rebuild optimizer after physical A/B replacement.
        # --------------------------------------------------------

        if rank_changed:
            optimizer = build_optimizer(
                model,
                task_config,
            )

            scheduler = build_scheduler(
                optimizer,
                task_config,
                total_steps,
            )

            # Rebuild starts a fresh optimizer state.
            # The following restores the scheduler's LR position.
            completed_steps = min(
                epoch * len(train_loader),
                total_steps,
            )

            if completed_steps > 0:
                scheduler.step(completed_steps)

        # --------------------------------------------------------
        # Save best checkpoint.
        # --------------------------------------------------------

        current_metric = metrics["selection_metric"]

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "rank_allocation": model.get_rank_allocation(),
                    "best_metric": best_metric,
                    "best_metric_name": metrics["selection_name"],
                    "epoch": epoch,
                    "task": args.task,
                    "model": "deberta-v3-base",
                    "seed": args.seed,
                },
                os.path.join(output_dir, "best_model.pt"),
            )

            print("Saved best_model.pt")

    controller.verify_budget()

    final_path = os.path.join(
        output_dir,
        "final_model.pt",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "rank_allocation": model.get_rank_allocation(),
            "best_metric": best_metric,
            "best_metric_name": (
                "pearson"
                if args.task == "stsb"
                else "accuracy"
            ),
            "best_epoch": best_epoch,
            "final_rank_budget": model.get_total_rank(),
            "task": args.task,
            "model": "deberta-v3-base",
            "seed": args.seed,
        },
        final_path,
    )

    print()
    print("=" * 64)
    print("Training complete")
    print("=" * 64)

    if args.task == "stsb":
        print(f"Best Pearson: {best_metric:.4f}")
    else:
        print(f"Best accuracy: {best_metric * 100:.2f}%")

    print(f"Final rank budget: {model.get_total_rank()}")
    print("Final rank allocation:")

    for name, rank in model.get_rank_allocation().items():
        print(f"  {name}: {rank}")

    print("=" * 64)


if __name__ == "__main__":
    main()
