# Experiments

An experiment is an ordinary `RunConfig` plus the values to vary:

```yaml
name: titans-memory-inquiry

run:
  # Any valid RunConfig.
  seed: 42
  output_dir: runs/titans-memory-inquiry
  corpus: {source: synthetic}
  # ...

variables:
  - name: memory
    path: stream.memory_mode
    values: [adaptive, frozen, none]
  - name: seed
    path: seed
    values: [42, 43]
```

`experiment.resolve()` substitutes the Cartesian product into the template and
returns a `RunSet`. Each `Run` contains a fully validated `RunConfig` and can be
executed directly:

```python
from synbiotorch import Experiment

experiment = Experiment.from_yaml("experiment.yaml")
runs = experiment.resolve()

metrics = runs.runs[0].execute()
# Or execute every run:
results = runs.execute()
```

There is no separate method config, execution profile, plan, compiler, or config
resolver.

## Contrasts

Contrasts name the values that differ. All other variables are matched
automatically, so this comparison pairs adaptive and frozen memory at the same
seed:

```yaml
contrasts:
  - name: adaptive_vs_frozen
    treatment: {memory: adaptive}
    control: {memory: frozen}
    metric: bits_per_base
    direction: lower
```

The resolved contrast contains explicit treatment/control run IDs.

## Qualification

Most experiments need no qualification section. To vary execution machinery,
put its reference value in `reference`. Other values must have evidence covering
the experiment's invariants:

```yaml
variables:
  - name: precision
    path: train.precision
    values: [fp32, bf16]

reference:
  precision: fp32

invariants:
  - stream_ownership
  - higher_order_autograd

qualifications:
  - when: {precision: bf16}
    invariants: [stream_ownership, higher_order_autograd]
    evidence: [artifacts/bf16-qualification.json]
```

Qualification does not bypass `RunConfig` validation. Unsupported combinations
still fail before they can run.

## CLI

The bundled TITANS experiment is small and uses the synthetic corpus:

```bash
# Show its six runs and two matched contrasts.
synbiotorch experiment show \
  examples/experiments/titans_methodological_inquiry.yaml

# Execute every run, or select IDs printed by `show`.
synbiotorch experiment run \
  examples/experiments/titans_methodological_inquiry.yaml

synbiotorch experiment run \
  examples/experiments/titans_methodological_inquiry.yaml \
  --run-id run-...
```

Each run writes to a stable subdirectory beneath `run.output_dir`. The ordinary
training pipeline writes `config.resolved.yaml`, metrics, and checkpoints;
experiment execution adds `run.json` and `result.json`.
