"""Prepare the Urtecho et al. E. coli genome-wide promoter MPRA for synbio-torch.

Urtecho, Insigne, Tripp, Brinck, Lubock, Acree, Kim, Chan & Kosuri, "Genome-wide
Functional Characterization of Escherichia coli Promoters and Sequence Elements
Encoding Their Regulation", eLife reviewed preprint 92558,
doi:10.1101/2020.01.04.894907. Each record is a 150 bp E. coli genomic sequence
whose label is autonomous promoter activity measured by a genomically-integrated
massively parallel reporter assay, pooled across three libraries: reported TSS
regions, tiles across called promoter peaks, and 10 bp scanning scrambles.

The processed tables are published in the authors' repository
(KosuriLab/ecoli_promoter_mpra). The repository carries no license file and the
preprint is CC BY-NC 4.0, so the tables are downloaded on demand and are NOT
redistributed with synbio-torch. This script only reshapes them locally into one
CSV the ``table`` source reads.

Rather than consume the authors' pre-split text files, this script ports their
genome-chunk partitioning (scripts/modeling/define_genome_splits.py) and then
asserts the rebuilt partitions match those published files exactly: 87,164 train
and 30,392 test rows over identical sequence sets. A silently-changed upstream,
or a drifting port, fails the run rather than quietly shifting the baseline.

Two upstream conventions are reproduced deliberately:

* The regression floor is applied to the fitting partitions only. The authors
  clamp training expression below 1.0 up to 1.0 to "decrease the importance of
  the inactive/noisy values" and score against a raw, unclamped test target.
  ``expression_train_floored`` carries that asymmetry; ``expression`` stays raw
  everywhere and is the only column the held-out metrics are computed against.
* The 0.75-1.25 classification buffer described in the paper's Methods is absent
  from both the published split files and the authors' archived predictions, and
  scoring the full unbuffered test set is what reproduces their AUPRC. The
  ``active`` column therefore keeps every row.

Validation is one interior training chunk, so it is leakage-free by genome
position; it drives early stopping only and is never reported.

Usage::

    python examples/prepare_urtecho_promoters.py
"""

from __future__ import annotations

import csv
import math
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "urtecho_promoters"
OUTPUT_CSV = DATA_DIR / "urtecho_promoters.csv"

# Pinned to a commit rather than a moving branch so the corpus is reproducible.
KOSURI_COMMIT = "484abdb6ba2d398501009ee6eb25ca7f190d3b4c"
BASE_URL = f"https://raw.githubusercontent.com/KosuriLab/ecoli_promoter_mpra/{KOSURI_COMMIT}"
COMBINED = "processed_data/combined"

MASTER = "tss_scramble_peak_expression_model_format.txt"
ORACLE_TRAIN = "tss_scramble_peak_expression_model_format_floored_train_genome_split.txt"
ORACLE_TEST = "tss_scramble_peak_expression_model_format_floored_test_genome_split.txt"
ORACLE_TRAIN_CLF = "tss_scramble_peak_expression_model_format_train_genome_split_classification.txt"
ORACLE_TEST_CLF = "tss_scramble_peak_expression_model_format_test_genome_split_classification.txt"
# The authors' archived DragoNN CNN predictions, scored by eval_urtecho.py through
# the same metric code as our own runs so the comparison is literal.
AUTHORS_REGRESSION = "20191111_tss_scramble_peak_regression_hyperparam_tuned_predictions.txt"
AUTHORS_CLASSIFICATION = "20191118_tss_scramble_peak_classification_hyperparam_tuned_predictions.txt"

DOWNLOADS = (
    MASTER,
    ORACLE_TRAIN,
    ORACLE_TEST,
    ORACLE_TRAIN_CLF,
    ORACLE_TEST_CLF,
    AUTHORS_REGRESSION,
    AUTHORS_CLASSIFICATION,
)

# E. coli MG1655 U00096.2, and the train fraction the authors passed to their
# splitting script. Together these fix the eight genome chunks.
GENOME_SIZE = 4_639_675
TRAIN_FRACTION = 0.75
# Chunk index the authors assign to test; its origin-equidistant partner joins it.
TEST_CHUNK_INDEX = 1
# One interior training chunk, adjacent to neither test chunk, held out for
# early stopping. Not part of the authors' design, and never reported.
VAL_CHUNK_INDEX = 4

EXPECTED_TRAIN_TEST = {"train": 87_164, "test": 30_392}
# The authors clamp sub-threshold training expression to this value.
FLOOR = 1.0

Chunk = tuple[int, int]


def create_splits(train_fraction: float, genome_size: int) -> list[Chunk]:
    """Partition the genome into equal chunks, two of which make up the test fraction.

    Ported from the authors' ``define_genome_splits.py`` so the resulting
    partition is theirs, not an approximation of it. Coordinates are 1-based and
    the trailing short chunk is absorbed into the last full one.
    """
    split_size = int(math.floor(genome_size * ((1 - train_fraction) / 2)))
    splits = [(i, i + split_size - 1) for i in range(1, genome_size + 1, split_size)]
    splits.pop()
    last = splits.pop()
    splits.append((last[0], genome_size))
    return splits


def find_equidistant_pair(splits: list[Chunk], split1: Chunk) -> Chunk:
    """Return the chunk as far from the origin going backwards as ``split1`` is going forwards.

    The chromosome is circular with the origin at coordinate 1, so pairing the
    test chunks this way holds replication-associated copy number roughly equal
    between train and test.
    """
    start, end = splits[0][0], splits[-1][1]
    split_size = splits[0][1] - splits[0][0] + 1
    remainder = end % split_size
    distance1 = split1[0] - start
    for split2 in splits:
        if distance1 == abs(split2[1] - end + remainder):
            return split2
    raise ValueError(f"no chunk equidistant from the origin to {split1}")


def _in_range(chunk: Chunk, start: int, end: int) -> bool:
    return start >= chunk[0] and end <= chunk[1]


def _ensure_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in DOWNLOADS:
        dest = DATA_DIR / Path(name).name
        if dest.exists():
            continue
        url = f"{BASE_URL}/{COMBINED}/{name}"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)


def _classify_library(name: str) -> str:
    """Name the source library from the authors' variant-name encoding.

    TSS records read ``TSS_<id>_<sources>,<pos>,<strand>``; scanning scrambles
    carry a ``_scrambled<a>-<b>`` suffix on that strand field; peak tiles read
    ``<start>_<end>_<strand>_pos<a>-<b>``; the rest are assay controls.
    """
    if "_scrambled" in name:
        return "scramble"
    if name.startswith("TSS_"):
        return "tss"
    if name.lstrip(">").startswith(("neg_control", "pos_control", "random")):
        return "control"
    return "peak_tile"


def _strand(name: str, library: str) -> str:
    if library in ("tss", "scramble"):
        fields = name.split(",")
        if len(fields) >= 3 and fields[2][:1] in ("+", "-"):
            return fields[2][0]
    elif library == "peak_tile":
        fields = name.split("_")
        if len(fields) >= 3 and fields[2] in ("+", "-"):
            return fields[2]
    return ""


def _sequences(path: Path) -> set[str]:
    with path.open() as handle:
        return {line.split("\t", 1)[0] for line in handle if line.strip()}


def main() -> None:
    _ensure_files()

    splits = create_splits(TRAIN_FRACTION, GENOME_SIZE)
    test_chunks = (splits[TEST_CHUNK_INDEX], find_equidistant_pair(splits, splits[TEST_CHUNK_INDEX]))
    val_chunk = splits[VAL_CHUNK_INDEX]
    if val_chunk in test_chunks:
        raise ValueError(f"validation chunk {val_chunk} collides with the authors' test chunks")

    chunk_split = {
        chunk: "test" if chunk in test_chunks else "val" if chunk == val_chunk else "train" for chunk in splits
    }
    print(f"{len(splits)} genome chunks of {splits[0][1] - splits[0][0] + 1} bp")
    print(f"  test: {test_chunks[0]} and {test_chunks[1]}")
    print(f"  val:  {val_chunk}")

    rows = []
    # The authors' partition is train/test only; `authors_split` reproduces it for
    # the verification gate, while `split` additionally carves out validation.
    by_authors_split: dict[str, set[str]] = {"train": set(), "test": set()}
    counts = {"train": 0, "val": 0, "test": 0}
    authors_counts = {"train": 0, "test": 0}
    dropped = 0

    with (DATA_DIR / MASTER).open() as handle:
        for index, line in enumerate(csv.reader(handle, delimiter="\t")):
            if not line:
                continue
            sequence, expression_raw, start_raw, end_raw, name, active = line[:6]
            # Controls carry no genomic coordinates and so fall into no chunk, as
            # do the handful of records straddling a chunk boundary.
            if start_raw == "NA" or end_raw == "NA":
                dropped += 1
                continue
            start, end = int(start_raw), int(end_raw)
            chunk = next((c for c in splits if _in_range(c, start, end)), None)
            if chunk is None:
                dropped += 1
                continue

            split = chunk_split[chunk]
            authors_split = "test" if split == "test" else "train"
            by_authors_split[authors_split].add(sequence)
            authors_counts[authors_split] += 1
            counts[split] += 1

            expression = float(expression_raw)
            # Floor the fitting partitions only; the reported test target stays raw.
            fitted = max(expression, FLOOR) if split != "test" else expression
            library = _classify_library(name)
            rows.append(
                {
                    "id": f"urtecho_{index}",
                    "sequence": sequence,
                    "expression": expression,
                    "expression_train_floored": fitted,
                    "active": 1 if active == "active" else 0,
                    "library": library,
                    "start": start,
                    "end": end,
                    "strand": _strand(name, library),
                    "split": split,
                }
            )

    _verify(by_authors_split, authors_counts)

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    positives = sum(row["active"] for row in rows)
    libraries: dict[str, int] = {}
    for row in rows:
        libraries[str(row["library"])] = libraries.get(str(row["library"]), 0) + 1

    print(f"\nwrote {len(rows)} rows to {OUTPUT_CSV} ({dropped} unplaced records dropped)")
    print(f"  train={counts['train']}  val={counts['val']}  test={counts['test']}")
    print(f"  active={positives} ({positives / len(rows):.1%})")
    print("  libraries: " + "  ".join(f"{k}={v}" for k, v in sorted(libraries.items())))


def _verify(by_authors_split: dict[str, set[str]], authors_counts: dict[str, int]) -> None:
    """Fail unless the ported partition matches the authors' published split files."""
    if authors_counts != EXPECTED_TRAIN_TEST:
        raise ValueError(f"rebuilt partition {authors_counts} != published {EXPECTED_TRAIN_TEST}")

    for split, filename in (("train", ORACLE_TRAIN), ("test", ORACLE_TEST)):
        published = _sequences(DATA_DIR / filename)
        if published != by_authors_split[split]:
            raise ValueError(
                f"{split} sequences diverge from {filename}: "
                f"{len(by_authors_split[split])} rebuilt vs {len(published)} published"
            )
        print(f"verified {split}: {authors_counts[split]} rows over {len(published)} unique sequences")


if __name__ == "__main__":
    main()
