# Examples

Runnable example configs and a Python quickstart for synbio-torch.

## Quickstart (offline)

Train a graph transformer on synthetic SBOL data — no external services or
downloads, runs in seconds:

```bash
python examples/quickstart.py
```

## Example configs

Run any config with the CLI:

```bash
synbiotorch train examples/configs/<name>.yaml
```

| Config | Task | Data | Prerequisites |
|--------|------|------|---------------|
| [`train_graph.yaml`](configs/train_graph.yaml) | Graph transformer regression | synthetic | `WANDB_API_KEY` (or set `wandb.enabled: false`) |
| [`finetune_structure_aware.yaml`](configs/finetune_structure_aware.yaml) | Structure-aware regression (from scratch) | synthetic | `WANDB_API_KEY` (or set `wandb.enabled: false`) |
| [`pretrain_mlm.yaml`](configs/pretrain_mlm.yaml) | From-scratch MLM pretraining | sbol-db | a running sbol-db at `base_url` |
| [`finetune_expression.yaml`](configs/finetune_expression.yaml) | Frozen DNABERT-2 → regression | sbol-db | a running sbol-db, plus DNABERT-2 (Linux/GPU — see [backbones](../docs/backbones.md)) |
| [`finetune_protein.yaml`](configs/finetune_protein.yaml) | Protein regression (from scratch) | [`data/protein_activity.csv`](data/protein_activity.csv) | none — runs offline |
| [`benchmark_dna_classification.yaml`](configs/benchmark_dna_classification.yaml) | DNA sequence classification | [`data/promoter_strength.csv`](data/promoter_strength.csv) | DNABERT-2 from the hub (network) |
| [`ingest_genbank.yaml`](configs/ingest_genbank.yaml) | Import GenBank to the Parquet cache | [`data/demo_tu.gb`](data/demo_tu.gb) | none — runs offline |

The `synthetic` and CSV configs are the quickest way to see the full pipeline end
to end. To point a config at your own data, change the `corpus` section (see
[configuration](../docs/configuration.md) and [data sources](../docs/data.md)).

## GenBank and SBOL

GenBank and SBOL are parsed in-process by the native sbol-rs binding — no external
tool. Import a GenBank file to the Parquet cache with:

```bash
synbiotorch ingest examples/configs/ingest_genbank.yaml
```

Point `corpus.path` at a directory of `.gb`/`.gbk` (or `.ttl`/`.xml` for SBOL)
files to build a real corpus; `corpus.namespace` roots the imported identities.

## Höllerer RBS demonstration (paper Table 1)

The configuration sweep on the Höllerer et al. *E. coli* RBS dataset. All three
variants share one corpus, task, and the published held-out split; they differ
only in the tokenizer and model blocks.

```bash
# 1. download the SAPIENs arrays (pinned commit) and build the split CSV
python examples/prepare_hollerer_rbs.py
# 2. build a Triton-free DNABERT-2 copy (needed on Apple Silicon / CPU)
python examples/prepare_dnabert2.py
# 3. train the three sweep variants
synbiotorch train examples/configs/hollerer_scratch_char.yaml
synbiotorch train examples/configs/hollerer_scratch_kmer.yaml
synbiotorch train examples/configs/hollerer_finetune_dnabert2.yaml
# 4. score on the fixed test split, archive metrics + predictions, write the figure
env -u VIRTUAL_ENV uv run --with matplotlib python examples/eval_hollerer.py \
    --out ../research/synbio-torch/figures/rbs_scatter.pdf
```

Step 4 scores each run on the exact SAPIENs test partition (`split == "test"`,
27,654 variants) and writes:

- [`hollerer_test_metrics.json`](hollerer_test_metrics.json) — the held-out test
  R², MAE, and bootstrap CI for each variant (the source of the paper's Table 1),
- `hollerer_predictions/<run>.npz` — per-variant measured and predicted values.

These held-out **test** metrics are distinct from the **validation** metrics in
each run's `runs/<run>/final_metrics.json`.

## Kosuri composability demonstration (design-native encoders)

An encoder ablation on the Kosuri et al. 2013 promoter × RBS composability
library, where each construct is a real composition of two annotated parts. The
constructs are ingested as SBOL 3 through the native binding, and the same corpus,
task, and split are held fixed while only the encoder varies — flat `sequence`,
`structure_aware`, or `graph` — so the design-native modalities are exercised on
real data. Two splits run for each encoder: `random` (in-distribution) and
`partout`, a held-out-parts split whose test constructs use a promoter or RBS
never seen in training, which measures generalization to novel part combinations.

The three PNAS Supporting Information files are not open-access and are not
redistributed. Download them once from
[PMC3752251](https://pmc.ncbi.nlm.nih.gov/articles/PMC3752251/) (Supplementary
Materials) into `data/kosuri/` (`sd01.xls`, `sd02.xls`, `sd03.xls`), then:

```bash
# 1. build the SBOL 3 corpus (log10 protein label + both split annotations)
env -u VIRTUAL_ENV uv run --with pandas --with xlrd python examples/prepare_kosuri.py
# 2. train the six-cell sweep (3 encoders x 2 splits)
for enc in seq structure graph; do for split in random partout; do
  synbiotorch train examples/configs/kosuri_${enc}_${split}.yaml
done; done
# 3. score each run on its test split, archive metrics + predictions
env -u VIRTUAL_ENV uv run python examples/eval_kosuri.py
```

Step 3 writes [`kosuri_test_metrics.json`](kosuri_test_metrics.json) (test R²,
MAE, and bootstrap CI per encoder × split) and `kosuri_predictions/<run>.npz`.

## Urtecho genome-wide promoter demonstration (faithful reproduction)

A reproduction of the machine-learning section of Urtecho et al., *"Genome-wide
Functional Characterization of Escherichia coli Promoters and Sequence Elements
Encoding Their Regulation"* ([eLife reviewed preprint
92558](https://elifesciences.org/reviewed-preprints/92558v1),
doi:10.1101/2020.01.04.894907), whose abstract concludes that "despite these
large datasets and modern machine learning algorithms, predicting endogenous
promoter activity from primary sequence is still challenging."

The corpus is 117,556 unique 150 bp E. coli genomic sequences with promoter
activity measured by a genomically-integrated MPRA, pooled from three libraries
(reported TSS regions, peak tiles, scanning scrambles). The paper evaluates two
tasks on a 75/25 split by genomic location — active/inactive **classification**
(AUPRC) and quantitative activity **regression** (R²) — and the sweep runs both
across four architectures, holding corpus, task, and split fixed: the authors'
own 3-layer CNN (`family: conv`, ~52k parameters, their searched
hyperparameters verbatim), two from-scratch transformers, and a fine-tuned
DNABERT-2. The CNN cells are the architectural reproduction; the rest measure
what a generic modern architecture, and a pretrained backbone, bring to the
same fixed problem.

The aim is fidelity: the same data, the same split, the same architecture, and
the same metrics as the paper, with every published number re-derived rather
than quoted. The additional architectures are context for that result, not an
attempt to unseat it.

The authors' processed tables are not redistributed here. `prepare_urtecho_promoters.py`
downloads them from the authors' repository at a pinned commit, **ports their
genome-chunk split** (`scripts/modeling/define_genome_splits.py`) rather than
consuming their pre-split files, and then asserts the rebuilt partitions match
those files exactly — 87,164 train and 30,392 test rows over identical sequence
sets — so a changed upstream fails the run instead of quietly moving the baseline.
Validation is one interior training chunk, leakage-free by genome position, used
for early stopping only.

```bash
# 1. download the authors' tables (pinned commit) and build the corpus CSV
python examples/prepare_urtecho_promoters.py
# 2. build a Triton-free DNABERT-2 copy (needed on Apple Silicon / CPU)
python examples/prepare_dnabert2.py
# 3. verify the metric code reproduces the paper from the authors' own predictions
env -u VIRTUAL_ENV uv run python examples/eval_urtecho.py --authors-only
# 4. train the sweep cells
for cfg in urtecho_clf_cnn urtecho_clf_char urtecho_clf_kmer urtecho_clf_dnabert2 \
           urtecho_reg_cnn urtecho_reg_char urtecho_reg_kmer urtecho_reg_dnabert2 \
           urtecho_reg_dnabert2_log; do
  synbiotorch train examples/configs/${cfg}.yaml
done
# 5. score on the fixed test split, archive metrics + predictions, write the figure
env -u VIRTUAL_ENV uv run --with matplotlib python examples/eval_urtecho.py
```

Step 5 scores every cell on the exact test partition and writes:

- [`urtecho_test_metrics.json`](urtecho_test_metrics.json) — held-out test AUPRC /
  R² with bootstrap CIs per cell,
- `urtecho_predictions/<run>.npz` — per-sequence measured and predicted values.

Regression cells are always scored against the raw `expression` column read from
the corpus CSV and keyed by record id, never against the batch labels, so the
target is identical no matter what a cell trained on. The authors' own archived
DragoNN CNN predictions are scored through the same code and appear in the table
as reference rows, making the reproduction literal rather than a quotation.

### Reproducing the published numbers

Scoring the authors' archived predictions through this harness recovers their
headline figures:

| Task | Paper | Quantity it corresponds to | This harness |
|------|-------|----------------------------|--------------|
| Classification AUPRC | 0.44 | average precision | **0.4392** |
| Regression "R²" | 0.31 | squared Pearson correlation | **0.3131** |
| Regression "RMSE" | 3.12 | `pracma::rmserr(...)$nmse` | **3.1223** |
| — | — | coefficient of determination | 0.3020 |
| — | — | root mean squared error | 6.5307 |

All three published numbers are accounted for exactly, once the metric
definitions are read off the figure script
(`scripts/figs/models_regression.Rmd`) rather than assumed from their names.
Two are worth knowing about before comparing anything to them:

- **Their R² is a squared correlation, not a coefficient of determination.**
  It comes from `summary(lm(observed ~ predicted))$r.squared`, which refits a
  line and is therefore invariant to scale and offset — it ignores calibration
  error. For these predictions the two happen to be close (0.3131 vs 0.3020)
  because the CNN is nearly calibrated, but they are different quantities, and
  `urtecho_test_metrics.json` carries both.
- **Their RMSE = 3.12 is not an RMSE.** The script calls
  `pracma::rmserr(predicted, observed)`, which returns a six-field list. Its
  `rmse` field is 6.5307; its `nmse` field — `sum((y-x)²)/sum((x-mean(x))²)` —
  is 3.1223, the reported value. The published bar chart plots the correct
  `rmse` (the axis runs past 6), so the slip is confined to the prose. pracma's
  own summary printer labels that field "LMSE", which is an easy field to pick
  by mistake when transcribing.

A detail that falls out of scoring every archived model the same way: among
their regression predictions the 6-mer random forest edges the CNN on both
their R² (0.3170 vs 0.3131) and true RMSE (6.485 vs 6.531), so the two are
closer than "the CNN performs best" suggests. The archived files are dated
November 2019 and may not be the exact runs behind the final figure. The
classification result is unaffected, and neither changes the paper's
conclusion.

### Results

Held-out test split (30,392 sequences), scored by `eval_urtecho.py`. The
authors' own archived predictions are scored by the same code, so the first row
is a measurement rather than a quotation. `R²` is the coefficient of
determination; `r²` is the squared Pearson correlation, which is what the paper
reports as R².

| Regression | R² | r² | RMSE |
|------------|-----|-----|------|
| **Urtecho 2020 CNN** (authors' predictions) | **0.3020** | **0.3131** | 6.531 |
| DNABERT-2, log target | 0.2406 | 0.2587 | 6.812 |
| CNN, Urtecho architecture (this library) | 0.2333 | 0.2352 | 6.844 |
| k-mer transformer, from scratch | 0.2267 | 0.2310 | 6.874 |
| DNABERT-2, fine-tuned | 0.0957 | 0.2407 | 7.433 |
| char transformer, from scratch | 0.0885 | 0.0927 | 7.463 |

| Classification | AUPRC | AUROC |
|----------------|-------|-------|
| **Urtecho 2020 CNN** (authors' predictions) | **0.4392** | 0.7373 |
| CNN, Urtecho architecture (this library) | 0.4192 | 0.7259 |
| DNABERT-2, fine-tuned | 0.3645 | 0.6914 |
| k-mer transformer, from scratch | 0.3241 | 0.6750 |
| char transformer, from scratch | 0.2998 | 0.6441 |

**The reproduction lands where it should.** Running the authors' own
architecture through this library recovers AUPRC 0.4192 against their 0.4392,
and R² 0.2333 against their 0.3020, from a single untuned configuration against
the winner of their 100-combination random hyperparameter search. The remaining
gap is the tuning budget, not a disagreement about the result: the published
model sits inside the range this architecture reaches, and the conclusion the
paper draws from it — that promoter activity is hard to predict from primary
sequence — reproduces cleanly.

What the sweep adds, by holding the corpus, task, and split fixed while varying
only the architecture:

- **Architecture dominates scale.** The 52k-parameter conv cell matches or beats
  from-scratch transformers 40–60× its size, and reaches in two epochs what they
  do not reach in twenty. Convolution plus pooling is a position-invariant motif
  scanner, which is what σ70 promoter structure actually looks like; a
  transformer has to learn that equivariance from 74k examples. On
  classification, where no target conditioning is involved, the conv cell leads
  every other cell by a wide margin.
- **Pretraining substitutes for architectural bias.** DNABERT-2 with a log
  target is the strongest regression cell here (R² 0.2406), edging out the
  conv cell, and plain DNABERT-2 is the best non-conv classifier (0.3645). A
  pretrained backbone arrives already knowing what DNA motifs look like, which
  is the one way a transformer competes without a 100-run search.
- **DNABERT-2's regression weakness is calibration, not ranking.** Its R² of
  0.0957 against an r² of 0.2407 is a 2.5× gap, far wider than any other cell:
  the predictions order the targets reasonably but sit at the wrong scale, which
  the coefficient of determination punishes and squared correlation ignores.
  Fitting the same backbone on a `log1p` target fixes it (R² 0.2406), which is
  the opposite of what the heavy-tail analysis below predicts — the conditioning
  gain outweighs the compressed tail.

Validation scores on this dataset run well above test: the conv cell finishes at
val R² 0.419 against a test R² of 0.233. Validation is one interior genome
chunk while test is two origin-equidistant chunks, so the gap is generalization
across genomic position, not a leak — the test partition is verified identical
to the authors'. Treat per-run `final_metrics.json` as optimistic and read the
held-out numbers from `urtecho_test_metrics.json`.

### What the regression metric actually measures

The raw-scale R² is dominated by a handful of sequences:

| Slice of the test set | Share of total sum of squares |
|---|---|
| top 0.5% (activity ≥ 67.0) | 57.3% |
| top 1% (≥ 45.4; 304 of 30,392) | 80.5% |
| top 5% (≥ 6.15) | 96.1% |

Predicting the bottom 99% perfectly and the top 1% as the mean would score
R² = 0.195. So "R² = 0.31" is largely a measure of predicting a few hundred very
strong promoters, not of general sequence-to-activity understanding. Two
consequences worth knowing before optimizing against it: a log-space target
necessarily looks worse on it, because it compresses exactly the tail the metric
rewards — though in practice the conditioning gain outweighed that for
DNABERT-2; and the raw target is badly conditioned for gradient training, with
76% of fitting rows at the floor and the informative residuals large enough that
gradient clipping discards them. The regression cells use
`task.target_transform: standardize` for that reason — an affine
reparameterization that leaves the objective and the reported metrics unchanged.
Without it, the from-scratch cells converge to predicting the training mean
(R² = 0.000).

### Two upstream conventions reproduced deliberately

These are matched rather than corrected, so the comparison stays fair:

- **The regression floor is one-sided.** The authors clamp training expression
  below 1.0 up to 1.0 to "decrease the importance of the inactive/noisy values"
  and score against a raw, unclamped test target. Three quarters of the fitting
  rows (66,643 of 87,164) therefore sit exactly at the floor while the test
  target still spans −0.50 to 226.9. Both of their output files carry a
  `_floored` suffix, but only the training one is floored. The
  `expression_train_floored` column carries that asymmetry in its name.
- **The classification buffer is not in the evaluated artifact.** The paper's
  Methods describe dropping sequences between 0.75 and 1.25, reducing the split
  to 53,326/18,567. Neither the published split files nor the archived
  predictions apply it, and scoring the full unbuffered test set is what
  reproduces AUPRC 0.44. The `active` column keeps every row.

## Weights & Biases

The two synthetic configs have `wandb.enabled: true`. Put `WANDB_API_KEY` in a
`.env` at the repo root and run both:

```bash
python examples/run_wandb_examples.py
```

The runner loads the key from `.env`, trains each config online, and prints the
project workspace URL. To run without an account, set `wandb.enabled: false` (or
`wandb.mode: offline`) in the config.

To refresh the screenshots in the top-level README, capture these panels from
each run and save them as `docs/images/wandb_train_graph.png` and
`docs/images/wandb_structure_aware.png`:

- `train/step_loss` (per-step training loss)
- `val/loss` and `val/r2` (per-epoch validation)
- optionally a `gradients/*` histogram (logged by `watch_model`)
