

import argparse
from array import array
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
import uuid

# ---------------- 可直接修改的默认配置 ----------------
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data" / "MPM-fingerprint.csv"  # 分类学输入表
DEFAULT_IMMUNO_DATA = HERE / "data" / "fingerprint-immunogenicity.csv"  # 免疫输入表
if not DEFAULT_IMMUNO_DATA.is_file():
    DEFAULT_IMMUNO_DATA = HERE.parent / "data" / "fingerprint-immunogenicity.csv"
DEFAULT_OUT = HERE / "results_glycgr_raw"                  # 结果输出根目录
TAXONOMY_TASKS = ("species", "genus", "family", "order", "class",
                  "phylum", "kingdom", "domain")
IMMUNO_TASK = "immunogenicity"
TASKS = TAXONOMY_TASKS + (IMMUNO_TASK,)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)                           # 5个独立随机种子
DEFAULT_LR = 1e-3                                         # 学习率
DEFAULT_WEIGHT_DECAY = 1e-4                               # 权重衰减
DEFAULT_T_MAX = 100                                       # 余弦调度周期
DEFAULT_EPOCHS = 100                                      # 完整训练轮数
DEFAULT_BATCH_SIZE = 64                                   # 训练批大小
FP_DIM = 2072                                             # 使用已保存的指纹
SPLITS = ("train", "valid", "test")
METRICS = ("accuracy", "macro_f1", "macro_f1_present", "mcc")
BINARY_METRICS = METRICS + ("auprc", "auroc")
# -----------------------------------------------------


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path, tasks):
    """Validate every row, preserving its order and multiplicity."""
    rows = []
    csv.field_size_limit(10 ** 8)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        needed = {"target", "fingerprint", "split", *tasks}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        for row_index, raw in enumerate(reader):
            if None in raw:
                raise ValueError(f"CSV record {row_index + 1}: excess columns")
            if any(raw.get(k) is None or not raw[k].strip() for k in needed):
                raise ValueError(f"CSV record {row_index + 1}: missing required value")
            split = raw["split"].strip().lower()
            split = "valid" if split in ("val", "validation") else split
            if split not in SPLITS:
                raise ValueError(f"CSV record {row_index + 1}: invalid split {raw['split']!r}")
            try:
                fp = json.loads(raw["fingerprint"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV record {row_index + 1}: invalid fingerprint list") from exc
            if (not isinstance(fp, list) or len(fp) != FP_DIM
                    or any(type(v) not in (int, float) or v not in (0, 1) for v in fp)):
                raise ValueError(f"CSV record {row_index + 1}: expected {FP_DIM} binary fingerprint values")
            rows.append({
                "row_index": row_index,
                "target": raw["target"],
                "split": split,
                "fp": array("f", fp),
                "labels": {task: " ".join(raw[task].split()) for task in tasks},
            })
    if not rows:
        raise ValueError("The input CSV is empty.")
    counts = Counter(row["split"] for row in rows)
    if any(counts[s] == 0 for s in SPLITS):
        raise ValueError(f"All three splits must be nonempty; got {dict(counts)}")
    return rows


def unwrap(nested_list):
    return [item for sublist in nested_list for item in sublist]


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start


def small_motif_find(s):
    b = s.split('(')
    b = [k.split(')') for k in b]
    b = [item for sublist in b for item in sublist]
    b = [k.strip('[') for k in b]
    b = [k.strip(']') for k in b]
    b = [k.replace('[', '') for k in b]
    b = [k.replace(']', '') for k in b]
    return '*'.join(b)


def min_process_glycans(glycan_list):
    glycan_motifs = [small_motif_find(k) for k in glycan_list]
    return [i.split('*') for i in glycan_motifs]


def motif_find(s, exhaustive=False):
    b = s.split('(')
    b = [k.split(')') for k in b]
    b = [item for sublist in b for item in sublist]
    b = [k.strip('[') for k in b]
    b = [k.strip(']') for k in b]
    b = [k.replace('[', '') for k in b]
    b = [k.replace(']', '') for k in b]
    if exhaustive and len(b) < 5:
        return ['*'.join(b)]
    return ['*'.join(b[i:i + 5]) for i in range(0, len(b) - 4, 2)]


def process_glycans(glycan_list, exhaustive=False):
    glycan_motifs = [motif_find(k, exhaustive=exhaustive) for k in glycan_list]
    return [[i.split('*') for i in k] for k in glycan_motifs]


def get_lib(glycan_list, mode='letter', exhaustive=True):
    """Sorted glycoletter vocabulary, using the reference implementation."""
    proc = process_glycans(glycan_list, exhaustive=exhaustive)
    words = unwrap(proc)
    words = [list(k) for k in set(tuple(k) for k in words)]
    if mode == 'letter':
        return sorted(set(unwrap(words)))
    return sorted(set(tuple(k) for k in words))


def string_to_labels(character_string, libr):
    return list(map(lambda character: libr.index(character), character_string))


def glycan_to_graph(glycan, libr):
    """Convert an IUPAC-condensed glycan into glycoletter nodes and graph edges.

    Both monosaccharides and linkages are nodes, matching the GlyCGR encoder.
    """
    bracket_count = glycan.count('[')
    parts = []
    branchbranch = []
    branchbranch2 = []
    position_bb = []
    b_counts = []
    bb_count = 0
    if bool(re.search(r'\[[^\]]+\[', glycan)):
        double_pos = [(k.start(), k.end()) for k in re.finditer(r'\[[^\]]+\[', glycan)]
        for spos, pos in double_pos:
            bracket_count -= 1
            glycan_part = glycan[spos + 1:]
            glycan_part = glycan_part[glycan_part.find('['):]
            idx = [k.end() for k in re.finditer(r'\][^\(]+\(', glycan_part)][0]
            branchbranch.append(glycan_part[:idx - 1].replace(']', '').replace('[', ''))
            branchbranch2.append(glycan[pos - 1:])
            glycan_part = glycan[:pos - 1]
            b_counts.append(glycan_part.count('[') - bb_count)
            glycan_part = glycan_part[glycan_part.rfind('[') + 1:]
            position_bb.append(glycan_part.count('(') * 2)
            bb_count += 1
        for b in branchbranch2:
            glycan = glycan.replace(b, ']'.join(b.split(']')[1:]))
    main = re.sub(r"[\[].*?[\]]", "", glycan)
    position = []
    branch_points = [x.start() for x in re.finditer(r'\]', glycan)]
    for i in branch_points:
        glycan_part = glycan[:i + 1]
        glycan_part = re.sub(r"[\[].*?[\]]", "", glycan_part)
        position.append(glycan_part.count('(') * 2)
    parts.append(main)

    for k in range(1, bracket_count + 1):
        start = find_nth(glycan, '[', k) + 1
        if bool(re.search(r"[\]][^\[]+[\(]", glycan[start:])):
            if bool(re.search(r'\]\[', glycan[start:])):
                glycan_part = re.sub(r"[\[].*?[\]]", "", glycan[start:])
                end = re.search(r"[\]].*?[\(]", glycan_part).span()[1] - 1
                parts.append(glycan_part[:end].replace(']', ''))
            else:
                end = re.search(r"[\]].*?[\(]", glycan[start:]).span()[1] + start - 1
                parts.append(glycan[start:end].replace(']', ''))
        else:
            if bool(re.search(r'\]\[', glycan[start:])):
                glycan_part = re.sub(r"[\[].*?[\]]", "", glycan[start:])
                parts.append(glycan_part.replace(']', ''))
            else:
                parts.append(glycan[start:].replace(']', ''))

    try:
        for bb in branchbranch:
            parts.append(bb)
    except Exception:
        pass

    parts = min_process_glycans(parts)
    parts_lengths = [len(j) for j in parts]
    parts_tokenized = [string_to_labels(k, libr) for k in parts]
    parts_tokenized = [parts_tokenized[0]] + [parts_tokenized[k][:-1] for k in range(1, len(parts_tokenized))]
    parts_tokenized = [item for sublist in parts_tokenized for item in sublist]

    range_list = list(range(len([item for sublist in parts for item in sublist])))
    init = 0
    parts_positions = []
    for k in parts_lengths:
        parts_positions.append(range_list[init:init + k])
        init += k

    for j in range(1, len(parts_positions) - len(branchbranch)):
        parts_positions[j][-1] = position[j - 1]
    for j in range(1, len(parts_positions)):
        try:
            for z in range(j + 1, len(parts_positions)):
                parts_positions[z][:-1] = [o - 1 for o in parts_positions[z][:-1]]
        except Exception:
            pass
    try:
        for i, j in enumerate(range(len(parts_positions) - len(branchbranch), len(parts_positions))):
            parts_positions[j][-1] = parts_positions[b_counts[i]][position_bb[i]]
    except Exception:
        pass

    pairs = []
    for i in parts_positions:
        pairs.append([(i[m], i[m + 1]) for m in range(0, len(i) - 1)])
    pairs = list(zip(*[item for sublist in pairs for item in sublist]))
    return parts_tokenized, pairs



def prepare_data(rows, tasks):
    """Check graph parsing and report overlaps; never filter or repartition."""
    targets = [row["target"] for row in rows]
    libr = get_lib(targets)
    topology = {}
    for row in rows:
        seq = row["target"]
        if seq in topology:
            continue
        try:
            nodes, edges = glycan_to_graph(seq, libr)
            if not nodes:
                raise ValueError("empty graph")
            if not edges:
                if len(nodes) != 1:
                    raise ValueError("multi-node graph without edges")
                edges = ([], [])
            if len(edges) != 2 or len(edges[0]) != len(edges[1]):
                raise ValueError("invalid edge array")
            if any(i < 0 or i >= len(nodes) for part in edges for i in part):
                raise ValueError("edge index out of range")
            topology[seq] = (nodes, edges)
        except Exception as exc:
            raise ValueError(
                f"Graph parsing failed at CSV record {row['row_index'] + 1}, "
                f"split={row['split']}: {exc}; no rows were removed."
            ) from exc
    split_counts = Counter(row["split"] for row in rows)
    split_targets = {s: {r["target"] for r in rows if r["split"] == s} for s in SPLITS}
    audit = {
        "n_rows": len(rows), "n_unique_targets": len(topology),
        "split_counts": dict(split_counts), "vocabulary_size": len(libr),
        "fingerprint_dimension": FP_DIM, "rows_removed": 0,
        "cross_split_exact_target_overlap": {
            f"{a}__{b}": len(split_targets[a] & split_targets[b])
            for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
        },
        "tasks": {},
    }
    prepared = {}
    for task in tasks:
        classes = list(dict.fromkeys(row["labels"][task] for row in rows))
        if len(classes) < 2:
            raise ValueError(f"{task}: fewer than two classes")
        converter = {label: index for index, label in enumerate(classes)}
        labels = [converter[row["labels"][task]] for row in rows]
        train_classes = {r["labels"][task] for r in rows if r["split"] == "train"}
        absent = set(classes) - train_classes
        pairs = Counter((r["labels"][task], r["target"]) for r in rows)
        audit["tasks"][task] = {
            "n_classes": len(classes),
            "duplicate_label_target_rows_retained": sum(n - 1 for n in pairs.values()),
            "classes_absent_from_train": len(absent),
            "test_rows_with_class_absent_from_train": sum(
                r["split"] == "test" and r["labels"][task] in absent for r in rows),
        }
        prepared[task] = (classes, labels)
    return libr, topology, prepared, audit


def make_graphs(rows, labels, topology, lib_size, virtual=True):
    import torch
    from torch_geometric.data import Data
    sets = {split: [] for split in SPLITS}
    for row, label in zip(rows, labels):
        nodes, edges = topology[row["target"]]
        n = len(nodes)
        original = torch.tensor([list(edges[0]), list(edges[1])], dtype=torch.long)
        virtual_edges = torch.tensor(
            [list(range(n)) + [n] * n, [n] * n + list(range(n))], dtype=torch.long)
        graph = Data(
            x=torch.tensor(list(nodes) + ([lib_size] if virtual else []), dtype=torch.long),
            edge_index_ori=original,
            edge_index_full=(torch.cat([original, virtual_edges], dim=1)
                             if virtual else original),
            y=torch.tensor([label], dtype=torch.long),
            fp=torch.tensor(row["fp"], dtype=torch.float32),
        )
        sets[row["split"]].append(graph)
    return sets


def make_model(num_classes, lib_size, device, fusion_mode="delayed"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import TransformerConv, global_add_pool

    if fusion_mode not in ("delayed", "all"):
        raise ValueError(f"Unknown fusion mode: {fusion_mode}")

    class GlyCGR(nn.Module):
        """Reference architecture; virtual-node indexing uses ordered graph boundaries."""

        def __init__(self, num_classes, lib_size, hidden_dim=256, fp_dim=FP_DIM,
                     heads=8, fp_emb_dim=128, dropout=0.5):
            super().__init__()
            assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
            self.hidden_dim = hidden_dim
            self.dropout = dropout

            self.item_embedding = nn.Embedding(num_embeddings=lib_size + 1, embedding_dim=hidden_dim)
            self.fp_emb = nn.Sequential(
                nn.Linear(fp_dim, fp_emb_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(fp_emb_dim),
                nn.Dropout(dropout),
                nn.Linear(fp_emb_dim, hidden_dim),
                nn.LeakyReLU(),
            )
            conv = lambda: TransformerConv(in_channels=hidden_dim, out_channels=hidden_dim // heads,
                                           heads=heads, concat=True, dropout=0.1)
            self.conv1, self.conv2, self.conv3, self.conv4 = conv(), conv(), conv(), conv()
            self.fc = nn.Sequential(
                nn.Linear(4 * hidden_dim, 512),
                nn.LeakyReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(dropout),
                nn.Linear(512, 64),
                nn.LeakyReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(dropout / 2),
                nn.Linear(64, num_classes),
            )

        def forward(self, x, edge_index_ori, edge_index_full, batch, fp, inference=False):
            h = self.item_embedding(x)

            # the virtual node is the last node of every graph; its embedding is the fingerprint
            virtual_mask = torch.zeros_like(batch, dtype=torch.bool)
            # PyG batches keep each graph's nodes contiguous; avoid duplicate-index scatter.
            virtual_mask[:-1] = batch[:-1] != batch[1:]
            virtual_mask[-1] = True
            h = h.clone()
            h[virtual_mask] = self.fp_emb(fp)

            pooled = []
            edge_schedule = ((edge_index_ori,) * 3 + (edge_index_full,)
                             if fusion_mode == "delayed" else (edge_index_full,) * 4)
            for conv, edge_index in zip((self.conv1, self.conv2, self.conv3, self.conv4),
                                        edge_schedule):
                h = F.leaky_relu(conv(h, edge_index))
                if conv is self.conv4:
                    pooled.append(global_add_pool(h, batch))
                else:
                    masked = h.clone()
                    masked[virtual_mask] = 0.
                    pooled.append(global_add_pool(masked, batch))

            graph_repr = torch.cat(pooled, dim=1)
            out = self.fc(graph_repr)
            if inference:
                return out, graph_repr
            return out


    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.sparse_(m.weight, sparsity=0.1)


    # Match the recorded runner: transfer first, then initialize on the selected device.
    model = GlyCGR(num_classes=num_classes, lib_size=lib_size).to(device)
    model.apply(init_weights)
    return model


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict(model, loader, device, positive_id=None):
    import torch
    model.eval()
    truth, prediction, positive_scores = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index_ori, batch.edge_index_full,
                           batch.batch, batch.fp.view(batch.y.numel(), FP_DIM))
            truth.extend(batch.y.cpu().tolist())
            prediction.extend(logits.argmax(dim=-1).cpu().tolist())
            if positive_id is not None:
                positive_scores.extend(
                    torch.softmax(logits, dim=-1)[:, positive_id].cpu().tolist())
    if not truth:
        raise ValueError("Cannot evaluate an empty split.")
    return truth, prediction, positive_scores if positive_id is not None else None


def compute_metrics(truth, prediction, positive_scores=None, positive_id=None):
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                 matthews_corrcoef, roc_auc_score)
    scores = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "macro_f1_present": float(f1_score(
            truth, prediction, labels=sorted(set(truth)), average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, prediction)),
    }
    if positive_id is not None:
        binary_truth = [int(label == positive_id) for label in truth]
        if len(set(binary_truth)) != 2:
            raise ValueError("AUPRC/AUROC require both classes in this split.")
        scores["auprc"] = float(average_precision_score(binary_truth, positive_scores))
        scores["auroc"] = float(roc_auc_score(binary_truth, positive_scores))
    if not all(math.isfinite(v) for v in scores.values()):
        raise ValueError(f"Non-finite evaluation metric: {scores}")
    return scores


def write_csv(path, fields, records):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def save_aggregate(out_dir, results):
    fields = ["task", "seed", "split", "best_epoch", "accuracy", "macro_f1",
              "macro_f1_present", "mcc", "auprc", "auroc", "n_samples", "n_correct"]
    write_csv(out_dir / "run_metrics.csv", fields, results)
    summary = []
    for task in dict.fromkeys(row["task"] for row in results):
        for split in ("valid", "test"):
            selected = [r for r in results if r["task"] == task and r["split"] == split]
            for metric in (BINARY_METRICS if task == IMMUNO_TASK else METRICS):
                values = [r[metric] for r in selected]
                summary.append({
                    "task": task, "split": split, "metric": metric,
                    "n_runs": len(values), "mean": statistics.mean(values),
                    "sd": statistics.stdev(values) if len(values) > 1 else "",
                    "sd_ddof": 1,
                })
    write_csv(out_dir / "summary.csv",
              ["task", "split", "metric", "n_runs", "mean", "sd", "sd_ddof"], summary)


def train_one(task, seed, rows, classes, sets, libr, args, out_dir):
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    positive_id = classes.index("1") if task == IMMUNO_TASK else None
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        sets["train"], batch_size=args.batch_size, shuffle=True, generator=generator)
    valid_loader = DataLoader(sets["valid"], batch_size=64, shuffle=False)
    test_loader = DataLoader(sets["test"], batch_size=64, shuffle=False)
    model = make_model(len(classes), len(libr), args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
    criterion = torch.nn.CrossEntropyLoss()
    best_score, best_epoch, best_state = -math.inf, None, None
    selection_metric = "auprc" if task == IMMUNO_TASK else "macro_f1"
    prefix = f"{task}_seed{seed}"
    history_fields = ["epoch", "learning_rate", "train_loss", "valid_accuracy",
                      "valid_macro_f1", "valid_auprc", "best_epoch", "best_valid_score"]
    started = time.perf_counter()
    with (out_dir / f"{prefix}_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            lr_used = optimizer.param_groups[0]["lr"]
            losses = []
            for batch in train_loader:
                batch = batch.to(args.device)
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index_ori, batch.edge_index_full,
                               batch.batch, batch.fp.view(batch.y.numel(), FP_DIM))
                loss = criterion(logits, batch.y)
                if not torch.isfinite(loss):
                    raise ValueError(f"{task}, seed {seed}, epoch {epoch}: non-finite loss")
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()  # 每轮一次，不在验证阶段重复更新。
            truth, prediction, probabilities = predict(
                model, valid_loader, args.device, positive_id)
            scores = compute_metrics(truth, prediction, probabilities, positive_id)
            # 并列时保留最早一轮，与实验记录中的严格 > 规则一致。
            if scores[selection_metric] > best_score:
                best_score, best_epoch = scores[selection_metric], epoch
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
            writer.writerow({
                "epoch": epoch, "learning_rate": lr_used,
                "train_loss": float(np.mean(losses)),
                "valid_accuracy": scores["accuracy"],
                "valid_macro_f1": scores["macro_f1"],
                "valid_auprc": scores.get("auprc", ""), "best_epoch": best_epoch,
                "best_valid_score": best_score,
            })
            handle.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"  epoch {epoch:3d}/{args.epochs}  valid {selection_metric} "
                      f"{scores[selection_metric]:.6f}  best {best_score:.6f} @ {best_epoch}",
                      flush=True)
    if best_state is None:
        raise RuntimeError("No valid checkpoint was selected.")
    model.load_state_dict(best_state)
    results = []
    for split, loader in (("valid", valid_loader), ("test", test_loader)):
        truth, prediction, probabilities = predict(model, loader, args.device, positive_id)
        scores = compute_metrics(truth, prediction, probabilities, positive_id)
        records = [row for row in rows if row["split"] == split]
        if len(records) != len(truth):
            raise RuntimeError("Prediction count differs from the original split size.")
        save_rows = [{
            "row_index": row["row_index"], "split": split, "target": row["target"],
            "true_id": true_id, "predicted_id": predicted_id,
            "true_label": classes[true_id], "predicted_label": classes[predicted_id],
            "correct": int(true_id == predicted_id),
            "positive_score": (probabilities[index] if probabilities is not None else ""),
        } for index, (row, true_id, predicted_id) in enumerate(zip(records, truth, prediction))]
        write_csv(
            out_dir / f"{prefix}_{split}_predictions.csv",
            ["row_index", "split", "target", "true_id", "predicted_id",
              "true_label", "predicted_label", "correct", "positive_score"], save_rows)
        results.append({
            "task": task, "seed": seed, "split": split, "best_epoch": best_epoch,
            **scores, "n_samples": len(truth),
            "n_correct": sum(a == b for a, b in zip(truth, prediction)),
        })
    if args.save_checkpoints:
        torch.save({
            "state_dict": best_state, "task": task, "seed": seed, "best_epoch": best_epoch,
            "class_labels": classes, "glycoletter_vocabulary": libr, "fp_dim": FP_DIM,
        }, out_dir / f"{prefix}_best.pt")
    print(f"  test: ACC={results[-1]['accuracy']:.6f}, "
          f"Macro-F1={results[-1]['macro_f1']:.6f}, "
          f"correct={results[-1]['n_correct']}/{results[-1]['n_samples']}; "
          f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--immunogenicity-data", type=Path, default=DEFAULT_IMMUNO_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--t-max", type=int, default=DEFAULT_T_MAX)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--save-checkpoints", action="store_true", help="另外保存每次最佳模型权重")
    parser.add_argument("--check-data", action="store_true",
                        help="只检查数据、划分及图解析，不训练、不写任何输出文件")
    args = parser.parse_args(argv)
    if len(args.tasks) != len(set(args.tasks)) or len(args.seeds) != len(set(args.seeds)):
        parser.error("Tasks and seeds must not contain duplicates.")
    if args.epochs < 1 or args.t_max < 1 or args.batch_size < 2:
        parser.error("epochs and t-max must be positive; batch-size must be >= 2.")
    if (not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("lr must be positive and weight-decay nonnegative (both finite).")
    if any(seed < 0 or seed > 2 ** 32 - 1 for seed in args.seeds):
        parser.error("Seeds must lie in [0, 2**32 - 1].")
    return args


def main(argv=None):
    args = parse_args(argv)
    dataset_specs = []
    taxonomy = [task for task in args.tasks if task in TAXONOMY_TASKS]
    if taxonomy:
        dataset_specs.append(("taxonomy", args.data.resolve(), taxonomy))
    if IMMUNO_TASK in args.tasks:
        dataset_specs.append((IMMUNO_TASK, args.immunogenicity_data.resolve(), [IMMUNO_TASK]))
    for name, path, _ in dataset_specs:
        if not path.is_file():
            raise FileNotFoundError(f"{name} input file not found: {path}")
    input_hashes = {name: sha256_file(path) for name, path, _ in dataset_specs}
    if args.check_data:
        for name, path, tasks in dataset_specs:
            rows = read_rows(path, tasks)
            _, _, _, audit = prepare_data(rows, tasks)
            print(f"[{name}] {path}\n{json.dumps(audit, ensure_ascii=False, indent=2)}")
        print("CHECK PASSED: original rows and splits retained; no files written.")
        return
    try:
        import numpy as np
        import sklearn
        import torch
        import torch_geometric
    except ImportError as exc:
        raise SystemExit(
            f"Missing training dependency: {exc}. Use an environment with numpy, "
            "scikit-learn, torch and torch-geometric. --check-data needs none of these."
        ) from exc
    args.device = (
        "cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This benchmark supports cpu or cuda devices.")
    # 每次创建新目录，避免覆盖以前的运行结果或把不同配置混入同一份均值。
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = args.out.resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "model": "GlyCGR", "scope": "eight taxonomy levels and immunogenicity",
        "input_csv": {name: str(path) for name, path, _ in dataset_specs},
        "input_sha256": input_hashes,
        "script_sha256": sha256_file(__file__),
        "tasks": args.tasks, "seeds": args.seeds,
        "hparams": {"lr": args.lr, "weight_decay": args.weight_decay,
                    "t_max": args.t_max, "epochs": args.epochs, "batch_size": args.batch_size},
        "device": str(device), "best_epoch_indexing": "one-based",
        "selection": {"taxonomy": "earliest maximum validation macro-F1",
                      "immunogenicity": "earliest maximum validation AUPRC"},
        "sd": "sample standard deviation, ddof=1; blank for only one run",
        "label_normalization": "collapse whitespace; retain first-occurrence label order",
        "vocabulary": "sorted glycoletters from all input structures, label-free",
        "graph_cache": "unique strings parsed once; all original dataset rows retained",
        "architecture": {"hidden_dim": 256, "heads": 8, "n_layers": 4,
                         "fp_projection": [FP_DIM, 128, 256],
                         "head": [1024, 512, 64, "n_classes"],
                         "fusion_layers": ["original", "original", "original", "full"],
                         "virtual_node_indexing": "last node using ordered graph boundaries"},
        "versions": {"python": sys.version, "numpy": np.__version__,
                     "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
                     "sklearn": sklearn.__version__, "cuda": torch.version.cuda},
        "bitwise_reproducibility": "not guaranteed for GPU scatter reductions",
    }
    write_json(out_dir / "run_config.json", manifest)
    results = []
    print(f"Output: {out_dir}\nDevice: {device}", flush=True)
    for name, path, tasks in dataset_specs:
        rows = read_rows(path, tasks)
        libr, topology, prepared, audit = prepare_data(rows, tasks)
        n_train = audit["split_counts"]["train"]
        if n_train < 2 or n_train % args.batch_size == 1:
            raise ValueError(
                f"{name}: training would contain a one-sample batch, incompatible with "
                "BatchNorm. Choose a batch size without this remainder.")
        print(f"[{name}] {path}\n{json.dumps(audit, ensure_ascii=False, indent=2)}", flush=True)
        write_json(out_dir / f"{name}_data_audit.json", audit)
        write_json(out_dir / f"{name}_glycoletter_vocabulary.json", libr)
        for task in tasks:
            classes, labels = prepared[task]
            write_json(out_dir / f"{task}_classes.json", classes)
            sets = make_graphs(rows, labels, topology, len(libr))
            for seed in args.seeds:
                print(f"\n[{task}] seed={seed}, classes={len(classes)}", flush=True)
                results.extend(train_one(task, seed, rows, classes, sets, libr, args, out_dir))
                save_aggregate(out_dir, results)
            del sets
        if sha256_file(path) != input_hashes[name]:
            raise RuntimeError(f"{name} input file changed during the run.")
    print(f"\nFinished. Metrics and sample SD: {out_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
