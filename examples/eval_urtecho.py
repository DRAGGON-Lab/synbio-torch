"""Evaluate trained Urtecho promoter runs on the authors' fixed held-out test split.

Loads each run's resolved config and best checkpoint, rebuilds the model, and
scores it on the exact genome-chunk test partition (``split == "test"``, 30,392
sequences) that Urtecho et al. report against, so the metrics are directly
comparable to their Figure 7.

Every regression cell is scored against the raw ``expression`` column, read from
the prepared CSV rather than from the batch labels. That holds the target fixed
no matter what a cell trained on: the floored target reproduces the authors'
convention, and the log-space cell has its transform inverted first. Ground
truth is keyed by record id, not row order.

The authors' own archived DragoNN CNN predictions are scored through this same
code, so the reproduction is literal rather than a quotation of their text. They
appear in the table as reference rows, so every cell is measured against the
published result on identical terms.

Prints a per-run table, archives the held-out test metrics and predictions, and
writes a measured-vs-predicted figure for the strongest regression cell.

Run after the training runs finish (not concurrently, to avoid re-syncing the
shared environment under a live run)::

    env -u VIRTUAL_ENV uv run --with matplotlib python examples/eval_urtecho.py

Pass ``--authors-only`` to score just the archived baselines, which needs no
trained runs and verifies the metric code before any of our numbers are trusted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from synbiotorch.config import RunConfig
from synbiotorch.datasets.dataset import Collator, EncodedDataset
from synbiotorch.encoders.base import build_encoder
from synbiotorch.engine import select_device
from synbiotorch.models import build_model, requires_fixed_length
from synbiotorch.pipeline import prepare_data
from synbiotorch.tasks import SupervisedTask, build_task
from synbiotorch.tokenize.base import build_tokenizer

CORPUS_CSV = Path("data/urtecho_promoters/urtecho_promoters.csv")
AUTHORS_DIR = Path("data/urtecho_promoters")
AUTHORS_REGRESSION = "20191111_tss_scramble_peak_regression_hyperparam_tuned_predictions.txt"
AUTHORS_CLASSIFICATION = "20191118_tss_scramble_peak_classification_hyperparam_tuned_predictions.txt"

REGRESSION_RUNS = {
    "CNN (Urtecho architecture)": "runs/urtecho_reg_cnn",
    "char transformer (scratch)": "runs/urtecho_reg_char",
    "k-mer transformer (scratch)": "runs/urtecho_reg_kmer",
    "DNABERT-2 (fine-tuned)": "runs/urtecho_reg_dnabert2",
    "DNABERT-2 (log target)": "runs/urtecho_reg_dnabert2_log",
}
CLASSIFICATION_RUNS = {
    "CNN (Urtecho architecture)": "runs/urtecho_clf_cnn",
    "char transformer (scratch)": "runs/urtecho_clf_char",
    "k-mer transformer (scratch)": "runs/urtecho_clf_kmer",
    "DNABERT-2 (fine-tuned)": "runs/urtecho_clf_dnabert2",
}

# Urtecho et al. Figure 7, for the table's reference column.
#
# Their R^2 is `summary(lm(observed ~ predicted))$r.squared`, which refits a line
# and so is the squared Pearson correlation, not the coefficient of determination.
# On their archived CNN predictions that is 0.3131, matching the reported 0.31;
# the coefficient of determination over the same rows is 0.3020. Both are
# reported below so the comparison can be read either way.
#
# Their reported RMSE of 3.12 corresponds to a different field of the same call.
# `pracma::rmserr(predicted, observed)` returns a list carrying both `rmse` =
# 6.5307 and `nmse` = sum((y-x)^2)/sum((x-mean(x))^2) = 3.1223, the reported
# figure. The published bar chart plots `rmse`, so the comparable error on this
# scale is 6.53.
PUBLISHED = {
    "regression": {
        "metric": "R^2 (squared Pearson, as the paper defines it)",
        "value": 0.31,
        "source": "Urtecho 2020 Fig 7B (tuned DragoNN CNN)",
    },
    "classification": {"metric": "AUPRC", "value": 0.44, "source": "Urtecho 2020 Fig 7A (tuned DragoNN CNN)"},
}

BOOTSTRAP_SAMPLES = 2000


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.std() == 0 or y_pred.std() == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve, as ``sum((R_n - R_n-1) * P_n)``.

    Matches ``sklearn.metrics.average_precision_score``: the step-wise estimate
    that makes no interpolation between operating points. Implemented here so
    the example keeps the numpy/torch-only dependency footprint of its sibling
    evaluation scripts.
    """
    order = np.argsort(-scores, kind="mergesort")
    labels, scores = labels[order], scores[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    # Collapse tied scores onto a single operating point, keeping the last.
    last = np.r_[np.where(np.diff(scores))[0], labels.size - 1]
    tp, fp = tp[last], fp[last]
    total_positive = tp[-1]
    if total_positive == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / total_positive
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum identity, averaging tied ranks."""
    n_pos = float(labels.sum())
    n_neg = float(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile 95% CI for a metric by resampling the test set with replacement."""
    rng = np.random.default_rng(seed)
    size = y_true.size
    stats = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, size, size)
        stats[i] = metric(y_true[idx], y_pred[idx])
    stats.sort()
    return float(stats[int(0.025 * n)]), float(stats[int(0.975 * n)])


def _ground_truth() -> dict[str, tuple[float, int]]:
    """Map record id to its raw expression and activity call on the test split."""
    truth: dict[str, tuple[float, int]] = {}
    with CORPUS_CSV.open() as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                truth[row["id"]] = (float(row["expression"]), int(row["active"]))
    if not truth:
        raise SystemExit(f"no test rows in {CORPUS_CSV}; run examples/prepare_urtecho_promoters.py first")
    return truth


@torch.no_grad()
def evaluate(run_dir: Path, device: torch.device) -> tuple[list[str], np.ndarray, str]:
    """Return the test record ids, this run's predictions, and its objective.

    Regression predictions come back on the raw expression scale: a cell trained
    through ``task.target_transform`` has that transform inverted here, so every
    cell is scored on one scale. Classification predictions are P(active).
    """
    config = RunConfig.from_yaml(str(run_dir / "config.resolved.yaml"))
    objective = config.task.objective
    data = prepare_data(config)
    test = [data.objects[i] for i in data.split.test]

    tokenizer = build_tokenizer(config.tokenizer)
    encoder = build_encoder(config.encoder, tokenizer)
    spec = encoder.output_spec
    model = build_model(
        config.model,
        config.task,
        vocab_size=spec.vocab_size,
        pad_token_id=spec.pad_token_id,
        max_length=spec.max_length,
    )
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    dataset = EncodedDataset(test, encoder)
    # `label_dtype` is the task's string name, not a torch dtype, and fixed-length
    # padding has to match what the run trained with or a flatten head mis-shapes.
    collate = Collator(
        tokenizer.pad_token_id,
        with_labels=True,
        label_dtype="long" if objective == "classification" else "float",
        pad_to=spec.max_length if requires_fixed_length(config.model) else None,
    )
    preds: list[torch.Tensor] = []
    batch_size = 512
    for start in range(0, len(dataset), batch_size):
        batch = collate([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels" and torch.is_tensor(v)}
        out = model(**inputs).float().cpu()
        # The regression head already returns [B]; reshape rather than squeeze so
        # a trailing batch of one does not collapse to a scalar.
        preds.append(out.softmax(dim=-1)[:, 1] if objective == "classification" else out.reshape(-1))

    prediction = torch.cat(preds).numpy()
    # Undo whatever target transform the cell trained under, so every regression
    # cell is returned on the one raw scale the metrics are computed against.
    task = build_task(config.task)
    if isinstance(task, SupervisedTask):
        prediction = task.inverse_transform(prediction)
    return [design.iri for design in test], prediction, objective


def _score_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    lo, hi = _bootstrap_ci(_r2, y_true, y_pred)
    pearson = _pearson(y_true, y_pred)
    return {
        "n_test": int(y_true.size),
        "test_r2": _r2(y_true, y_pred),
        "test_r2_ci95": [lo, hi],
        # The quantity the paper calls R^2: it refits observed ~ predicted, so it
        # is the squared correlation and ignores calibration error. Kept beside
        # the coefficient of determination so the published 0.31 is comparable
        # against like for like.
        "test_r2_squared_pearson": pearson**2,
        "test_rmse": _rmse(y_true, y_pred),
        "test_mae": _mae(y_true, y_pred),
        "test_pearson_r": pearson,
        # Reported alongside the raw-scale metrics because the target is heavy
        # tailed: raw R^2 is dominated by a thin tail of strong promoters.
        "test_r2_log1p": _r2(np.log1p(np.maximum(y_true, 0.0)), np.log1p(np.maximum(y_pred, 0.0))),
    }


def _score_classification(labels: np.ndarray, scores: np.ndarray) -> dict:
    lo, hi = _bootstrap_ci(_average_precision, labels, scores)
    return {
        "n_test": int(labels.size),
        "n_positive": int(labels.sum()),
        "test_auprc": _average_precision(labels, scores),
        "test_auprc_ci95": [lo, hi],
        "test_auroc": _auroc(labels, scores),
    }


def _authors_regression() -> tuple[np.ndarray, np.ndarray] | None:
    path = AUTHORS_DIR / AUTHORS_REGRESSION
    if not path.exists():
        return None
    arr = np.genfromtxt(path, delimiter="\t", usecols=(1, 2))
    return arr[:, 1], arr[:, 0]


def _authors_classification() -> tuple[np.ndarray, np.ndarray] | None:
    path = AUTHORS_DIR / AUTHORS_CLASSIFICATION
    if not path.exists():
        return None
    arr = np.genfromtxt(path, delimiter="\t")
    return arr[:, 1], arr[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="urtecho_scatter.pdf", help="scatter figure output path")
    parser.add_argument(
        "--metrics-out",
        default="examples/urtecho_test_metrics.json",
        help="combined held-out test summary the results table is read from",
    )
    parser.add_argument(
        "--authors-only",
        action="store_true",
        help="score only the authors' archived predictions, to verify the metric code",
    )
    args = parser.parse_args()

    metrics_out = Path(args.metrics_out)
    preds_dir = metrics_out.parent / "urtecho_predictions"
    summary: dict[str, list[dict]] = {"regression": [], "classification": []}

    authors_reg = _authors_regression()
    if authors_reg is not None:
        record = {"configuration": "Urtecho 2020 CNN (authors' archived predictions)", "run": None}
        record.update(_score_regression(*authors_reg))
        summary["regression"].append(record)
    authors_clf = _authors_classification()
    if authors_clf is not None:
        record = {"configuration": "Urtecho 2020 CNN (authors' archived predictions)", "run": None}
        record.update(_score_classification(*authors_clf))
        summary["classification"].append(record)

    best: tuple[float, str, np.ndarray, np.ndarray] | None = None
    if not args.authors_only:
        truth = _ground_truth()
        device = select_device()
        print(f"device: {device}\n")
        preds_dir.mkdir(parents=True, exist_ok=True)

        for task, runs in (("regression", REGRESSION_RUNS), ("classification", CLASSIFICATION_RUNS)):
            for label, path in runs.items():
                run_dir = Path(path)
                if not (run_dir / "best.pt").exists():
                    print(f"skipping {label} ({task}): no run at {run_dir}")
                    continue
                ids, prediction, objective = evaluate(run_dir, device)
                missing = [i for i in ids if i not in truth]
                if missing:
                    raise SystemExit(f"{run_dir}: {len(missing)} test ids absent from {CORPUS_CSV}")
                y_true = np.array([truth[i][0 if objective == "regression" else 1] for i in ids], dtype=np.float64)

                record = {"configuration": label, "run": run_dir.name, "checkpoint": "best.pt"}
                record.update(
                    _score_regression(y_true, prediction)
                    if objective == "regression"
                    else _score_classification(y_true, prediction)
                )
                np.savez_compressed(preds_dir / f"{run_dir.name}.npz", y_true=y_true, y_pred=prediction)
                record["predictions"] = str(preds_dir / f"{run_dir.name}.npz")
                summary[task].append(record)
                if objective == "regression" and (best is None or record["test_r2"] > best[0]):
                    best = (float(record["test_r2"]), label, y_true, prediction)

    _report(summary)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps({"published": PUBLISHED, "runs": summary}, indent=2) + "\n")
    print(f"\nwrote {metrics_out}")

    if best is not None:
        _scatter(best[2], best[3], best[1], best[0], Path(args.out))


def _report(summary: dict[str, list[dict]]) -> None:
    print(f"\nregression  (target: raw expression; published {PUBLISHED['regression']['value']} R^2)")
    print("  R^2    = coefficient of determination;  r^2 = squared Pearson, the paper's definition")
    print(f"{'configuration':<46}{'R^2':>8}{'95% CI':>20}{'r^2':>8}{'RMSE':>9}{'MAE':>8}")
    print("-" * 99)
    for row in summary["regression"]:
        ci = f"[{row['test_r2_ci95'][0]:.3f}, {row['test_r2_ci95'][1]:.3f}]"
        print(
            f"{row['configuration']:<46}{row['test_r2']:>8.4f}{ci:>20}"
            f"{row['test_r2_squared_pearson']:>8.4f}{row['test_rmse']:>9.3f}{row['test_mae']:>8.3f}"
        )

    print(f"\nclassification  (published {PUBLISHED['classification']['value']} AUPRC)")
    print(f"{'configuration':<46}{'AUPRC':>8}{'95% CI':>20}{'AUROC':>9}")
    print("-" * 83)
    for row in summary["classification"]:
        ci = f"[{row['test_auprc_ci95'][0]:.3f}, {row['test_auprc_ci95'][1]:.3f}]"
        print(f"{row['configuration']:<46}{row['test_auprc']:>8.4f}{ci:>20}{row['test_auroc']:>9.4f}")


def _scatter(y_true: np.ndarray, y_pred: np.ndarray, label: str, r2: float, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available; skipping figure. Re-run with `uv run --with matplotlib`.")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    # The target spans four orders of magnitude, so plot on log axes with the
    # measurement floor as the lower bound.
    floor = 0.1
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    ax.hexbin(
        np.maximum(y_true, floor),
        np.maximum(y_pred, floor),
        gridsize=60,
        bins="log",
        xscale="log",
        yscale="log",
        cmap="Blues",
        mincnt=1,
    )
    limits = (floor, max(y_true.max(), y_pred.max()))
    ax.plot(limits, limits, color="0.3", lw=1, ls="--")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("measured promoter activity")
    ax.set_ylabel("predicted promoter activity")
    ax.set_title(f"{label}\n$R^2 = {r2:.3f}$", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
