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

from models.bert import BERTALoRA
from models.deberta import DeBERTaALoRA

from alora.bert.controller import BERTRankController
from alora.deberta.controller import DeBERTaRankController


# ============================================================
# SUPPORTED TASKS
# ============================================================

BERT_TASKS = {
    "mrpc",
    "sst2",
    "stsb",
    "rte",
    "qnli",
    "ag_news",
}

DEBERTA_TASKS = {
    "qqp",
    "qnli",
    "sst2",
    "stsb",
}

REGRESSION_TASKS = {"stsb"}


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# CONFIG
# ============================================================

def load_config(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return yaml.safe_load(f)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# TASK CONFIG
# ============================================================

def build_task_config(
    config,
    task
):

    defaults = dict(
        config.get(
            "training",
            {}
        )
    )

    task_config = dict(
        config.get(
            "tasks",
            {}
        ).get(
            task,
            {}
        )
    )

    defaults.update(
        task_config
    )

    return defaults


# ============================================================
# AG NEWS DATALOADERS
# ============================================================

def build_agnews_dataloaders(
    training_config,
    tokenizer
):

    dataset = load_dataset(
        "ag_news"
    )

    max_length = int(
        training_config.get(
            "max_length",
            128
        )
    )

    batch_size = int(
        training_config.get(
            "batch_size",
            8
        )
    )

    eval_batch_size = int(
        training_config.get(
            "eval_batch_size",
            batch_size
        )
    )

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    tokenized = dataset.map(
        tokenize,
        batched=True
    )

    keep_columns = {
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "label",
    }

    remove_columns = [

        column

        for column
        in tokenized["train"].column_names

        if column not in keep_columns
    ]

    tokenized = tokenized.remove_columns(
        remove_columns
    )

    tokenized.set_format(
        "torch"
    )

    def collate_fn(batch):

        result = {

            "input_ids": torch.tensor(
                [
                    x["input_ids"]
                    for x in batch
                ],
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                [
                    x["attention_mask"]
                    for x in batch
                ],
                dtype=torch.long
            ),

            "labels": torch.tensor(
                [
                    x["label"]
                    for x in batch
                ],
                dtype=torch.long
            )
        }

        if "token_type_ids" in batch[0]:

            result["token_type_ids"] = torch.tensor(

                [
                    x["token_type_ids"]
                    for x in batch
                ],

                dtype=torch.long
            )

        return result

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        tokenized["test"],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return (
        train_loader,
        test_loader
    )


# ============================================================
# GLUE DATALOADERS
# ============================================================

def build_glue_dataloaders(
    task,
    training_config,
    tokenizer
):

    dataset = load_dataset(
        "glue",
        task
    )

    max_length = int(
        training_config.get(
            "max_length",
            128
        )
    )

    batch_size = int(
        training_config.get(
            "batch_size",
            32
        )
    )

    eval_batch_size = int(
        training_config.get(
            "eval_batch_size",
            batch_size
        )
    )

    sentence_keys = {

        "qqp":
            ("question1", "question2"),

        "qnli":
            ("question", "sentence"),

        "sst2":
            ("sentence", None),

        "stsb":
            ("sentence1", "sentence2"),

        "mrpc":
            ("sentence1", "sentence2"),

        "rte":
            ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = (
        sentence_keys[task]
    )

    def tokenize(batch):

        if sentence2_key is None:

            return tokenizer(

                batch[sentence1_key],

                truncation=True,

                max_length=max_length
            )

        return tokenizer(

            batch[sentence1_key],

            batch[sentence2_key],

            truncation=True,

            max_length=max_length
        )

    tokenized = dataset.map(
        tokenize,
        batched=True
    )

    tokenized = tokenized.rename_column(
        "label",
        "labels"
    )

    if task == "stsb":

        tokenized = tokenized.cast_column(
            "labels",
            Value("float32")
        )

    keep_columns = {

        "input_ids",
        "token_type_ids",
        "attention_mask",
        "labels",
    }

    remove_columns = [

        column

        for column
        in tokenized["train"].column_names

        if column not in keep_columns
    ]

    tokenized = tokenized.remove_columns(
        remove_columns
    )

    tokenized.set_format(
        "torch"
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    train_loader = DataLoader(

        tokenized["train"],

        batch_size=batch_size,

        shuffle=True,

        collate_fn=collator
    )

    eval_split = (

        "validation"
        if "validation" in tokenized
        else "test"
    )

    eval_loader = DataLoader(

        tokenized[eval_split],

        batch_size=eval_batch_size,

        shuffle=False,

        collate_fn=collator
    )

    return (
        train_loader,
        eval_loader
    )


# ============================================================
# OPTIMIZER
# ============================================================

def build_optimizer(
    model,
    training_config
):

    lr = float(
        training_config.get(
            "learning_rate",
            2e-5
        )
    )

    classifier_lr = float(
        training_config.get(
            "classifier_learning_rate",
            lr
        )
    )

    weight_decay = float(
        training_config.get(
            "weight_decay",
            0.0
        )
    )

    lora_params = [

        p

        for p in model.get_lora_parameters()

        if p.requires_grad
    ]

    classifier_params = [

        p

        for p in model.get_classifier_parameters()

        if p.requires_grad
    ]

    groups = [

        {
            "params":
                lora_params,

            "lr":
                lr,

            "weight_decay":
                weight_decay,
        }
    ]

    if classifier_params:

        groups.append(

            {
                "params":
                    classifier_params,

                "lr":
                    classifier_lr,

                "weight_decay":
                    weight_decay,
            }
        )

    return AdamW(
        groups
    )


# ============================================================
# SCHEDULER
# ============================================================

def build_scheduler(
    optimizer,
    training_config,
    total_steps
):

    warmup_ratio = float(
        training_config.get(
            "warmup_ratio",
            0.0
        )
    )

    warmup_steps = int(
        total_steps *
        warmup_ratio
    )

    return get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=
            warmup_steps,

        num_training_steps=
            total_steps
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    predictions,
    labels
):

    accuracy = (
        predictions == labels
    ).float().mean().item()

    return {

        "accuracy":
            accuracy,

        "selection_metric":
            accuracy,

        "selection_name":
            "accuracy"
    }


# ============================================================
# CORRELATION
# ============================================================

def pearson(x, y):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    y = np.asarray(
        y,
        dtype=np.float64
    )

    if (
        len(x) < 2
        or np.std(x) == 0
        or np.std(y) == 0
    ):

        return 0.0

    return float(
        np.corrcoef(
            x,
            y
        )[0, 1]
    )


def rankdata(values):

    values = np.asarray(
        values
    )

    order = np.argsort(
        values
    )

    ranks = np.empty(
        len(values),
        dtype=np.float64
    )

    i = 0

    while i < len(values):

        j = i + 1

        while (
            j < len(values)
            and
            values[order[j]]
            ==
            values[order[i]]
        ):

            j += 1

        ranks[order[i:j]] = (
            i + 1 + j
        ) / 2.0

        i = j

    return ranks


def spearman(x, y):

    return pearson(
        rankdata(x),
        rankdata(y)
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    task
):

    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_predictions = []
    all_labels = []

    for batch in loader:

        batch = {

            key:
                value.to(device)

            for key, value
            in batch.items()
        }

        outputs = model(
            **batch
        )

        labels = batch[
            "labels"
        ]

        n = labels.size(0)

        total_loss += (
            outputs.loss.item()
            * n
        )

        total_examples += n

        if task in REGRESSION_TASKS:

            predictions = (
                outputs.logits
                .squeeze(-1)
            )

            all_predictions.extend(
                predictions
                .float()
                .cpu()
                .numpy()
                .tolist()
            )

            all_labels.extend(
                labels
                .float()
                .cpu()
                .numpy()
                .tolist()
            )

        else:

            predictions = (
                outputs.logits
                .argmax(
                    dim=-1
                )
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    loss = (
        total_loss /
        max(
            total_examples,
            1
        )
    )

    if task in REGRESSION_TASKS:

        p = pearson(
            all_predictions,
            all_labels
        )

        s = spearman(
            all_predictions,
            all_labels
        )

        return {

            "loss":
                loss,

            "pearson":
                p,

            "spearman":
                s,

            "selection_metric":
                p,

            "selection_name":
                "pearson"
        }

    predictions = torch.cat(
        all_predictions
    )

    labels = torch.cat(
        all_labels
    )

    metrics = classification_metrics(
        predictions,
        labels
    )

    return {

        "loss":
            loss,

        "accuracy":
            metrics["accuracy"],

        "selection_metric":
            metrics["selection_metric"],

        "selection_name":
            "accuracy"
    }


# ============================================================
# BERT TRAINING EPOCH
# ============================================================

def train_bert_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    max_grad_norm
):

    model.train()

    total_loss = 0.0
    total_examples = 0

    num_layers = len(
        model.query_layers
    )

    importance_sum = np.zeros(
        num_layers,
        dtype=np.float64
    )

    importance_count = 0

    for batch in loader:

        batch = {

            key:
                value.to(device)

            for key, value
            in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(
            **batch
        )

        loss = outputs.loss

        if not torch.isfinite(loss):

            raise RuntimeError(
                "Non-finite loss detected."
            )

        loss.backward()

        if max_grad_norm > 0:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm
            )

        # ----------------------------------------------------
        # BERT A-LoRA IMPORTANCE
        # ----------------------------------------------------

        batch_importances = []

        for query in model.query_layers:

            query.compute_importance()

            batch_importances.append(
                query.importance
            )

        importance_sum += np.asarray(
            batch_importances
        )

        importance_count += 1

        optimizer.step()

        if scheduler is not None:

            scheduler.step()

        n = batch[
            "labels"
        ].size(0)

        total_loss += (
            loss.item() * n
        )

        total_examples += n

    # --------------------------------------------------------
    # EPOCH-LEVEL IMPORTANCE
    # --------------------------------------------------------

    if importance_count > 0:

        epoch_importances = (

            importance_sum
            /
            importance_count
        )

    else:

        epoch_importances = np.zeros(
            num_layers
        )

    for i, query in enumerate(
        model.query_layers
    ):

        query.importance = float(
            epoch_importances[i]
        )

    return (
        total_loss /
        max(
            total_examples,
            1
        ),
        epoch_importances
    )


# ============================================================
# DEBERTA TRAINING EPOCH
# ============================================================

def train_deberta_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    max_grad_norm
):

    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in loader:

        batch = {

            key:
                value.to(device)

            for key, value
            in batch.items()
        }

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            **batch
        )

        loss = outputs.loss

        if not torch.isfinite(loss):

            raise RuntimeError(
                "Non-finite loss detected."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm
        )

        model.update_importance()

        optimizer.step()

        if scheduler is not None:

            scheduler.step()

        n = batch[
            "labels"
        ].size(0)

        total_loss += (
            loss.item() * n
        )

        total_examples += n

    return (
        total_loss /
        max(
            total_examples,
            1
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=
        "Universal A-LoRA training runner."
    )

    parser.add_argument(

        "--task",

        choices=sorted(
            BERT_TASKS |
            DEBERTA_TASKS
        ),

        required=True
    )

    parser.add_argument(

        "--config",

        default=None
    )

    parser.add_argument(

        "--seed",

        type=int,

        default=42
    )

    parser.add_argument(

        "--output_dir",

        default=None
    )

    args = parser.parse_args()

    # ========================================================
    # DETERMINE MODEL FAMILY
    # ========================================================

    if args.config is not None:

        config_path = args.config

    elif args.task in BERT_TASKS:

        config_path = (
            "configs/bert.yaml"
        )

    else:

        config_path = (
            "configs/deberta.yaml"
        )

    if args.task in BERT_TASKS:

        model_family = "bert"

    else:

        model_family = "deberta"

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    config = load_config(
        config_path
    )

    task_config = build_task_config(
        config,
        args.task
    )

    model_config = config[
        "model"
    ]

    alora_config = config[
        "alora"
    ]

    # ========================================================
    # SETUP
    # ========================================================

    set_seed(
        args.seed
    )

    device = get_device()

    model_name = model_config[
        "name"
    ]

    if args.task == "stsb":

        num_labels = 1

    else:

        num_labels = int(
            task_config.get(
                "num_labels",
                2
            )
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    if args.output_dir is None:

        output_dir = os.path.join(

            "outputs",

            model_family,

            args.task,

            f"seed_{args.seed}"
        )

    else:

        output_dir = args.output_dir

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print("=" * 70)
    print("A-LoRA: Adaptive Rank Allocation")
    print("=" * 70)

    print(
        f"Model:       {model_name}"
    )

    print(
        f"Model family: {model_family}"
    )

    print(
        f"Task:        {args.task}"
    )

    print(
        f"Device:      {device}"
    )

    print(
        f"Seed:        {args.seed}"
    )

    print("=" * 70)

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name
        )
    )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    if model_family == "bert":

        model = BERTALoRA(

            model_name=model_name,

            num_labels=num_labels,

            initial_rank=int(
                model_config["rank"]
            ),

            alpha=float(
                model_config["alpha"]
            )
        )

    else:

        model = DeBERTaALoRA(

            model_name=model_name,

            num_labels=num_labels,

            rank=int(
                model_config["rank"]
            ),

            alpha=float(
                model_config["alpha"]
            ),

            ema_beta=float(
                alora_config["ema_beta"]
            ),

            train_classifier=bool(
                model_config.get(
                    "train_classifier",
                    True
                )
            )
        )

    # ========================================================
    # DATA
    # ========================================================

    if (
        model_family == "bert"
        and
        args.task == "ag_news"
    ):

        train_loader, eval_loader = (
            build_agnews_dataloaders(
                task_config,
                tokenizer
            )
        )

    else:

        train_loader, eval_loader = (
            build_glue_dataloaders(
                args.task,
                task_config,
                tokenizer
            )
        )

    # ========================================================
    # CONTROLLER
    # ========================================================

    if model_family == "bert":

        controller = BERTRankController(

            query_layers=
                model.get_query_layers(),

            min_rank=int(
                alora_config["min_rank"]
            ),

            max_rank=int(
                alora_config["max_rank"]
            ),

            max_rank_move=int(
                alora_config[
                    "max_rank_move"
                ]
            ),

            importance_gap=float(
                alora_config[
                    "importance_gap"
                ]
            )
        )

    else:

        controller = DeBERTaRankController(

            modules=
                model.get_lora_modules(),

            initial_rank=int(
                model_config["rank"]
            ),

            min_rank=int(
                alora_config["min_rank"]
            ),

            max_rank=int(
                alora_config["max_rank"]
            ),

            gap_threshold=float(
                alora_config[
                    "gap_threshold"
                ]
            ),

            max_rank_move=int(
                alora_config[
                    "max_rank_move"
                ]
            )
        )

    # ========================================================
    # DEVICE
    # ========================================================

    model.to(
        device
    )

    print(
        f"Total parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print(
        f"Trainable parameters: "
        f"{model.count_trainable_parameters():,}"
    )

    print(
        f"LoRA parameters: "
        f"{model.count_lora_parameters():,}"
    )

    print(
        f"Initial rank budget: "
        f"{model.get_total_rank()}"
    )

    # ========================================================
    # TRAINING SETTINGS
    # ========================================================

    epochs = int(
        task_config[
            "epochs"
        ]
    )

    total_steps = (
        len(train_loader)
        *
        epochs
    )

    optimizer = build_optimizer(
        model,
        task_config
    )

    scheduler = build_scheduler(
        optimizer,
        task_config,
        total_steps
    )

    max_grad_norm = float(
        task_config.get(
            "max_grad_norm",
            0.0
        )
    )

    best_metric = -float(
        "inf"
    )

    best_epoch = None

    if model_family == "bert":

        start_epoch = int(
            alora_config.get(
                "warmup_epochs",
                task_config.get(
                    "warmup_epochs",
                    1
                )
            )
        )

        # AG News reference adapts after
        # one warm-up epoch.
        if start_epoch <= 0:

            start_epoch = 1

        controller_interval = 1

    else:

        start_epoch = int(
            alora_config.get(
                "controller_start_epoch",
                2
            )
        )

        controller_interval = int(
            alora_config.get(
                "controller_interval",
                2
            )
        )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        print()
        print(
            f"===== Epoch "
            f"{epoch}/{epochs} ====="
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        if model_family == "bert":

            train_loss, epoch_importances = (
                train_bert_one_epoch(

                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    device,
                    max_grad_norm
                )
            )

        else:

            train_loss = (
                train_deberta_one_epoch(

                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    device,
                    max_grad_norm
                )
            )

        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        metrics = evaluate(

            model,

            eval_loader,

            device,

            args.task
        )

        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Eval loss:  "
            f"{metrics['loss']:.4f}"
        )

        if args.task == "stsb":

            print(
                f"Pearson:    "
                f"{metrics['pearson']:.4f}"
            )

            print(
                f"Spearman:   "
                f"{metrics['spearman']:.4f}"
            )

        else:

            print(
                f"Accuracy:   "
                f"{metrics['accuracy'] * 100:.2f}%"
            )

        # ----------------------------------------------------
        # A-LoRA CONTROLLER
        # ----------------------------------------------------

        rank_changed = False

        if (
            model_family == "bert"
            and
            epoch >= start_epoch
            and
            epoch <= epochs
        ):

            print()
            print(
                "Applying epoch-level "
                "BERT A-LoRA rank adaptation..."
            )

            result = (
                controller
                .redistribute_rank_budget()
            )

            if result:

                rank_changed = True

                print(
                    f"Receiver: "
                    f"{result['receiver']} "
                    f"{result['receiver_old_rank']} "
                    f"-> "
                    f"{result['receiver_new_rank']}"
                )

                print(
                    f"Donor:    "
                    f"{result['donor']} "
                    f"{result['donor_old_rank']} "
                    f"-> "
                    f"{result['donor_new_rank']}"
                )

                print(
                    f"Relative gap: "
                    f"{result['relative_gap']:.6f}"
                )

            else:

                print(
                    "No rank transfer performed."
                )

        elif (
            model_family == "deberta"
            and
            epoch >= start_epoch
            and
            epoch % controller_interval == 0
        ):

            print()
            print(
                "Running A-LoRA "
                "rank controller..."
            )

            result = (
                controller.step()
            )

            if result is not None:

                rank_changed = True

                print(
                    f"Receiver: "
                    f"{result['receiver']} "
                    f"{result['receiver_old_rank']} "
                    f"-> "
                    f"{result['receiver_new_rank']}"
                )

                print(
                    f"Donor:    "
                    f"{result['donor']} "
                    f"{result['donor_old_rank']} "
                    f"-> "
                    f"{result['donor_new_rank']}"
                )

                print(
                    f"Gap:      "
                    f"{result['gap']:.6f}"
                )

            else:

                print(
                    "No rank transfer performed."
                )

        # ----------------------------------------------------
        # VERIFY FIXED GLOBAL BUDGET
        # ----------------------------------------------------

        controller.verify_budget()

        print(
            f"Current rank budget: "
            f"{model.get_total_rank()}"
        )

        print(
            "Current rank allocation:"
        )

        if model_family == "bert":

            print(
                model.get_ranks()
            )

        else:

            for name, rank in (
                model.get_rank_allocation()
                .items()
            ):

                print(
                    f"  {name}: {rank}"
                )

        # ----------------------------------------------------
        # REBUILD OPTIMIZER AFTER RESIZE
        # ----------------------------------------------------

        if rank_changed:

            optimizer = build_optimizer(
                model,
                task_config
            )

            scheduler = build_scheduler(
                optimizer,
                task_config,
                total_steps
            )

            completed_steps = min(

                epoch *
                len(train_loader),

                total_steps
            )

            if completed_steps > 0:

                scheduler.step(
                    completed_steps
                )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        current_metric = (
            metrics[
                "selection_metric"
            ]
        )

        if (
            current_metric
            >
            best_metric
        ):

            best_metric = (
                current_metric
            )

            best_epoch = epoch

            torch.save(

                {
                    "model_state_dict":
                        model.state_dict(),

                    "rank_allocation":
                        model.get_rank_allocation()
                        if hasattr(
                            model,
                            "get_rank_allocation"
                        )
                        else model.get_ranks(),

                    "best_metric":
                        best_metric,

                    "best_metric_name":
                        metrics[
                            "selection_name"
                        ],

                    "epoch":
                        epoch,

                    "task":
                        args.task,

                    "model":
                        model_name,

                    "model_family":
                        model_family,

                    "seed":
                        args.seed,
                },

                os.path.join(
                    output_dir,
                    "best_model.pt"
                )
            )

            print(
                "Saved best_model.pt"
            )

    # ========================================================
    # FINAL BUDGET CHECK
    # ========================================================

    controller.verify_budget()

    # ========================================================
    # FINAL MODEL
    # ========================================================

    final_path = os.path.join(

        output_dir,

        "final_model.pt"
    )

    torch.save(

        {
            "model_state_dict":
                model.state_dict(),

            "rank_allocation":
                model.get_rank_allocation()
                if hasattr(
                    model,
                    "get_rank_allocation"
                )
                else model.get_ranks(),

            "best_metric":
                best_metric,

            "best_epoch":
                best_epoch,

            "final_rank_budget":
                model.get_total_rank(),

            "task":
                args.task,

            "model":
                model_name,

            "model_family":
                model_family,

            "seed":
                args.seed,
        },

        final_path
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("Training complete")
    print("=" * 70)

    if args.task == "stsb":

        print(
            f"Best Pearson: "
            f"{best_metric:.4f}"
        )

    else:

        print(
            f"Best accuracy: "
            f"{best_metric * 100:.2f}%"
        )

    print(
        f"Final rank budget: "
        f"{model.get_total_rank()}"
    )

    print(
        "Final rank allocation:"
    )

    if model_family == "bert":

        for i, rank in enumerate(
            model.get_ranks()
        ):

            print(
                f"  L{i + 1}_query: "
                f"{rank}"
            )

    else:

        for name, rank in (
            model.get_rank_allocation()
            .items()
        ):

            print(
                f"  {name}: {rank}"
            )

    print(
        f"Saved: {final_path}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
