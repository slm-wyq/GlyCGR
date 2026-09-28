
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
import time
import uuid

import benchmark_glycgr_raw as common


HERE = Path(__file__).resolve().parent
MODELS = ("GlyCGR-T", "GlyCGR-TC", "GlyCGR-FF")
SEEDS = (0, 1, 2)


def make_model(model_name, num_classes, lib_size, device, config_only=False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import (TopKPooling, TransformerConv, global_add_pool,
                                    global_max_pool, global_mean_pool)

    if model_name == "GlyCGR-FF":
        train_config = {"virtual": True, "optimizer": "AdamW", "lr": 1e-3,
                        "weight_decay": 1e-4, "t_max": 50}
        if config_only:
            return train_config
        return common.make_model(num_classes, lib_size, device, fusion_mode="all")

    class GlyCGRT(nn.Module):
        """Graph-only control from the recorded architecture."""
        train_config = {"virtual": False, "optimizer": "Adam", "lr": 5e-6,
                        "weight_decay": 1e-3, "t_max": 50}

        def __init__(self):
            super().__init__()
            self.item_embedding = nn.Embedding(lib_size + 1, 256)
            self.conv1 = TransformerConv(256, 32, heads=8, concat=True)
            self.pool1 = TopKPooling(256, ratio=0.8)
            self.conv2 = TransformerConv(256, 32, heads=8, concat=True)
            self.pool2 = TopKPooling(256, ratio=0.8)
            self.conv3 = TransformerConv(256, 32, heads=8, concat=True)
            self.pool3 = TopKPooling(256, ratio=0.8)
            self.lin1 = nn.Linear(512, 1024)
            self.lin2 = nn.Linear(1024, 64)
            self.lin3 = nn.Linear(64, num_classes)
            self.bn1 = nn.BatchNorm1d(1024)
            self.bn2 = nn.BatchNorm1d(64)
            self.act1 = nn.LeakyReLU()
            self.act2 = nn.LeakyReLU()

        def forward(self, x, edge_index, batch):
            x = self.item_embedding(x).squeeze(1)
            pooled = []
            for conv, pool in ((self.conv1, self.pool1), (self.conv2, self.pool2),
                               (self.conv3, self.pool3)):
                x = F.leaky_relu(conv(x, edge_index))
                x, edge_index, _, batch, _, _ = pool(x, edge_index, None, batch)
                pooled.append(torch.cat(
                    [global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1))
            x = pooled[0] + pooled[1] + pooled[2]
            x = self.act1(self.bn1(self.lin1(x)))
            x = self.act2(self.bn2(self.lin2(x)))
            x = F.dropout(x, p=0.5, training=self.training)
            return self.lin3(x)

    class GlyCGRTC(nn.Module):
        """Topology readout and projected fingerprint concatenated at the head."""
        train_config = {"virtual": False, "optimizer": "AdamW", "lr": 1e-3,
                        "weight_decay": 1e-4, "t_max": 50}

        def __init__(self):
            super().__init__()
            self.item_embedding = nn.Embedding(lib_size + 1, 256)
            self.fp_emb = nn.Sequential(
                nn.Linear(common.FP_DIM, 128), nn.ReLU(), nn.BatchNorm1d(128),
                nn.Dropout(0.5), nn.Linear(128, 256), nn.ReLU())
            conv_args = dict(in_channels=256, out_channels=32, heads=8,
                             concat=True, dropout=0.5)
            self.conv1 = TransformerConv(**conv_args)
            self.conv2 = TransformerConv(**conv_args)
            self.conv3 = TransformerConv(**conv_args)
            self.conv4 = TransformerConv(**conv_args)
            self.fc = nn.Sequential(
                nn.Linear(1280, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
                nn.Dropout(0.5), nn.Linear(1024, 64), nn.ReLU(),
                nn.BatchNorm1d(64), nn.Dropout(0.25),
                nn.Linear(64, num_classes))

        def forward(self, x, edge_index, batch, fp):
            x = self.item_embedding(x)
            pooled = []
            for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
                x = F.leaky_relu(conv(x, edge_index))
                pooled.append(global_add_pool(x, batch))
            joint = torch.cat([*pooled, self.fp_emb(fp)], dim=1)
            return self.fc(joint)

    model_type = GlyCGRT if model_name == "GlyCGR-T" else GlyCGRTC
    if config_only:
        return model_type.train_config
    model = model_type().to(device)
    model.apply(lambda layer: torch.nn.init.sparse_(layer.weight, sparsity=0.1)
                if type(layer) is nn.Linear else None)
    return model


def model_config(model_name):
    return make_model(model_name, None, None, None, config_only=True)


def forward_batch(model_name, model, batch):
    if model_name == "GlyCGR-T":
        return model(batch.x, batch.edge_index_ori, batch.batch)
    fp = batch.fp.view(batch.y.numel(), common.FP_DIM)
    if model_name == "GlyCGR-TC":
        return model(batch.x, batch.edge_index_ori, batch.batch, fp)
    return model(batch.x, batch.edge_index_ori, batch.edge_index_full, batch.batch, fp)


def predict(model_name, model, loader, device, positive_id):
    import torch
    model.eval()
    truth, predicted, probabilities = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = forward_batch(model_name, model, batch)
            truth.extend(batch.y.cpu().tolist())
            predicted.extend(logits.argmax(dim=-1).cpu().tolist())
            if positive_id is not None:
                probabilities.extend(
                    torch.softmax(logits, dim=-1)[:, positive_id].cpu().tolist())
    if not truth:
        raise ValueError("Cannot evaluate an empty split.")
    return truth, predicted, probabilities if positive_id is not None else None


def save_aggregate(out_dir, results):
    fields = ["model", "task", "seed", "split", "best_epoch", "accuracy",
              "macro_f1", "macro_f1_present", "mcc", "auprc", "auroc",
              "n_samples", "n_correct"]
    common.write_csv(out_dir / "run_metrics.csv", fields, results)
    summary = []
    for model, task in dict.fromkeys((r["model"], r["task"]) for r in results):
        for split in ("valid", "test"):
            chosen = [r for r in results if r["model"] == model and
                      r["task"] == task and r["split"] == split]
            if not chosen:
                continue
            metrics = (common.BINARY_METRICS if task == common.IMMUNO_TASK
                       else common.METRICS)
            for metric in metrics:
                values = [r[metric] for r in chosen]
                summary.append({
                    "model": model, "task": task, "split": split, "metric": metric,
                    "n_runs": len(values), "mean": statistics.mean(values),
                    "sd": statistics.stdev(values) if len(values) > 1 else "",
                    "sd_ddof": 1,
                })
    common.write_csv(out_dir / "summary.csv",
                     ["model", "task", "split", "metric", "n_runs", "mean",
                      "sd", "sd_ddof"], summary)


def train_one(model_name, task, seed, rows, classes, sets, lib_size, args, out_dir):
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader

    spec = model_config(model_name)
    common.set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(sets["train"], batch_size=args.batch_size,
                              shuffle=True, generator=generator)
    valid_loader = DataLoader(sets["valid"], batch_size=64, shuffle=False)
    test_loader = DataLoader(sets["test"], batch_size=64, shuffle=False)
    model = make_model(model_name, len(classes), lib_size, args.device)
    optimizer_type = torch.optim.Adam if spec["optimizer"] == "Adam" else torch.optim.AdamW
    optimizer = optimizer_type(model.parameters(), lr=spec["lr"],
                               weight_decay=spec["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=spec["t_max"])
    criterion = torch.nn.CrossEntropyLoss()
    positive_id = classes.index("1") if task == common.IMMUNO_TASK else None
    selection_metric = "auprc" if positive_id is not None else "macro_f1"
    best_score, best_epoch, best_state = -math.inf, None, None
    prefix = f"{model_name}_{task}_seed{seed}"
    history_fields = ["epoch", "learning_rate", "train_loss", "valid_accuracy",
                      "valid_macro_f1", "valid_auprc", "best_epoch",
                      "best_valid_score"]
    started = time.perf_counter()
    with (out_dir / f"{prefix}_history.csv").open("w", encoding="utf-8",
                                                  newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            learning_rate = optimizer.param_groups[0]["lr"]
            losses = []
            for batch in train_loader:
                batch = batch.to(args.device)
                optimizer.zero_grad()
                logits = forward_batch(model_name, model, batch)
                loss = criterion(logits, batch.y)
                if not torch.isfinite(loss):
                    raise ValueError(f"{prefix}, epoch {epoch}: non-finite loss")
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()
            truth, predicted, probs = predict(
                model_name, model, valid_loader, args.device, positive_id)
            scores = common.compute_metrics(truth, predicted, probs, positive_id)
            if scores[selection_metric] > best_score:
                best_score, best_epoch = scores[selection_metric], epoch
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
            writer.writerow({
                "epoch": epoch, "learning_rate": learning_rate,
                "train_loss": float(np.mean(losses)),
                "valid_accuracy": scores["accuracy"],
                "valid_macro_f1": scores["macro_f1"],
                "valid_auprc": scores.get("auprc", ""),
                "best_epoch": best_epoch, "best_valid_score": best_score,
            })
            handle.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"  {prefix} epoch {epoch}/{args.epochs} "
                      f"valid {selection_metric}={scores[selection_metric]:.6f} "
                      f"best={best_score:.6f}@{best_epoch}", flush=True)
    if best_state is None:
        raise RuntimeError(f"{prefix}: no validation checkpoint was selected")
    model.load_state_dict(best_state)
    results = []
    for split, loader in (("valid", valid_loader), ("test", test_loader)):
        truth, predicted, probs = predict(
            model_name, model, loader, args.device, positive_id)
        scores = common.compute_metrics(truth, predicted, probs, positive_id)
        records = [row for row in rows if row["split"] == split]
        if len(records) != len(truth):
            raise RuntimeError(f"{prefix}: prediction count differs from CSV split")
        prediction_rows = [{
            "row_index": row["row_index"], "split": split, "target": row["target"],
            "true_id": actual, "predicted_id": predicted_id,
            "true_label": classes[actual], "predicted_label": classes[predicted_id],
            "correct": int(actual == predicted_id),
            "positive_score": probs[i] if probs is not None else "",
        } for i, (row, actual, predicted_id)
           in enumerate(zip(records, truth, predicted))]
        common.write_csv(
            out_dir / f"{prefix}_{split}_predictions.csv",
            ["row_index", "split", "target", "true_id", "predicted_id",
             "true_label", "predicted_label", "correct", "positive_score"],
            prediction_rows)
        results.append({
            "model": model_name, "task": task, "seed": seed, "split": split,
            "best_epoch": best_epoch, **scores, "n_samples": len(truth),
            "n_correct": sum(a == b for a, b in zip(truth, predicted)),
        })
    if args.save_checkpoints:
        torch.save({
            "state_dict": best_state, "model": model_name, "task": task,
            "seed": seed, "best_epoch": best_epoch, "class_labels": classes,
        }, out_dir / f"{prefix}_best.pt")
    print(f"  test {model_name}/{task}/seed{seed}: "
          f"ACC={results[-1]['accuracy']:.6f}, "
          f"correct={results[-1]['n_correct']}/{results[-1]['n_samples']}; "
          f"{(time.perf_counter()-started)/60:.1f} min", flush=True)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=common.DEFAULT_DATA)
    parser.add_argument("--immunogenicity-data", type=Path,
                        default=common.DEFAULT_IMMUNO_DATA)
    parser.add_argument("--out", type=Path, default=HERE / "results_glycgr_ablations")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", choices=common.TASKS,
                        default=list(common.TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--check-data", action="store_true",
                        help="Validate original CSV splits and graph parsing; write nothing")
    args = parser.parse_args(argv)
    if (len(set(args.models)) != len(args.models) or
            len(set(args.tasks)) != len(args.tasks) or
            len(set(args.seeds)) != len(args.seeds)):
        parser.error("Models, tasks and seeds must not contain duplicates.")
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("epochs must be positive and batch-size must be >= 2.")
    if any(seed < 0 or seed > 2**32 - 1 for seed in args.seeds):
        parser.error("Seeds must lie in [0, 2**32 - 1].")
    return args


def main(argv=None):
    args = parse_args(argv)
    taxonomy = [task for task in args.tasks if task in common.TAXONOMY_TASKS]
    datasets = []
    if taxonomy:
        datasets.append(("taxonomy", args.data.resolve(), taxonomy))
    if common.IMMUNO_TASK in args.tasks:
        datasets.append((common.IMMUNO_TASK, args.immunogenicity_data.resolve(),
                         [common.IMMUNO_TASK]))
    for name, path, _ in datasets:
        if not path.is_file():
            raise FileNotFoundError(f"{name} input file not found: {path}")
    input_hashes = {name: common.sha256_file(path) for name, path, _ in datasets}
    if not args.check_data:
        try:
            import numpy as np
            import sklearn
            import torch
            import torch_geometric
        except ImportError as exc:
            raise SystemExit(f"Missing training dependency: {exc}") from exc
        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        if device.type not in ("cpu", "cuda"):
            raise ValueError("This benchmark supports cpu or cuda devices.")
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        out_dir = args.out.resolve() / run_id
        out_dir.mkdir(parents=True, exist_ok=False)
        common.write_json(out_dir / "run_config.json", {
            "models": args.models, "tasks": args.tasks, "seeds": args.seeds,
            "model_configs": {m: model_config(m) for m in args.models},
            "epochs": args.epochs, "batch_size": args.batch_size,
            "input_csv": {name: str(path) for name, path, _ in datasets},
            "input_sha256": input_hashes,
            "script_sha256": common.sha256_file(__file__),
            "common_script_sha256": common.sha256_file(common.__file__),
            "selection": {"taxonomy": "earliest maximum validation macro-F1",
                          "immunogenicity": "earliest maximum validation AUPRC"},
            "sd": "sample standard deviation, ddof=1",
            "device": str(device),
            "versions": {"python": sys.version, "numpy": np.__version__,
                         "torch": torch.__version__,
                         "torch_geometric": torch_geometric.__version__,
                         "sklearn": sklearn.__version__},
        })
        print(f"Output: {out_dir}\nDevice: {device}", flush=True)
    results = []
    for name, path, tasks in datasets:
        rows = common.read_rows(path, tasks)
        libr, topology, prepared, audit = common.prepare_data(rows, tasks)
        print(f"[{name}] {path}\n{json.dumps(audit, ensure_ascii=False, indent=2)}",
              flush=True)
        if args.check_data:
            continue
        n_train = audit["split_counts"]["train"]
        if n_train < 2 or n_train % args.batch_size == 1:
            raise ValueError(f"{name}: one-sample final training batch is incompatible "
                             "with BatchNorm; choose another batch size.")
        common.write_json(out_dir / f"{name}_data_audit.json", audit)
        common.write_json(out_dir / f"{name}_glycoletter_vocabulary.json", libr)
        for task in tasks:
            classes, labels = prepared[task]
            common.write_json(out_dir / f"{task}_classes.json", classes)
            for model_name in args.models:
                sets = common.make_graphs(
                    rows, labels, topology, len(libr),
                    virtual=model_config(model_name)["virtual"])
                for seed in args.seeds:
                    results.extend(train_one(
                        model_name, task, seed, rows, classes, sets, len(libr),
                        args, out_dir))
                    save_aggregate(out_dir, results)
                del sets
        if common.sha256_file(path) != input_hashes[name]:
            raise RuntimeError(f"{name} input file changed during the run.")
    if args.check_data:
        print("CHECK PASSED: original rows and splits retained; no files written.")
    else:
        print(f"Finished. Metrics and sample SD: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
