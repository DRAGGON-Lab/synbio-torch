# TITANS Memory-as-Context

synbio-torch contains a minimal, state-explicit implementation of the
TITANS Memory-as-Context (MAC) architecture. It is
selected with `model.kind: titans_mac`; it does not reuse a HuggingFace backbone.

## Architecture

Every 32-token segment is embedded and passed through a stack of MAC blocks. A
block first reads its incoming neural-memory state, then integrates
`[persistent tokens; retrieved memory; current segment]` with causal
multi-head attention, and finally writes the integrated segment to a functional
two-layer residual MLP. The fast-memory update is sequential within a segment:

```text
surprise_t = eta_t * surprise_(t-1) - theta_t * grad_W loss(W_(t-1); k_t, v_t)
W_t        = (1 - alpha_t) * W_(t-1) + surprise_t
```

`alpha`, `eta`, and `theta` are input-dependent, per-channel gates. Query, key,
and value projections use causal depthwise convolutions with history carried in
the state. The associative objective is half the summed squared error between
the functional memory's reconstruction of the normalized key and its value.

The language model itself is deliberately pure: `forward_segment` receives one
`TitansState` per batch row and returns logits plus replacement states. It does
not know stream ids, mutate a hidden cache, compute the language-model loss, or
own checkpoint policy.

## Ordered execution

TITANS memory is meaningful only when sequence order and identity are preserved.
The integration therefore separates three responsibilities:

1. `NonOverlappingKmerTokenizer` losslessly maps canonical full k-mers to fixed
   ids and falls back to individual bases for ambiguous or trailing content.
2. `OrderedStreamWindowDataset` assigns complete logical streams to batch slots
   and emits `[batch, gradient horizon, 32]` windows without interleaving
   identities within a row.
3. `StatefulCausalBatchAdapter` owns the `stream_id -> TitansState` table,
   carries state between windows, and detaches it at each gradient-horizon
   boundary.

Rolling checkpoints include both the stream cursor and active fast-memory
states. Resume therefore continues at the next unseen token with the exact
state that produced it, rather than restarting memory at an epoch boundary.
Validation has a separate adapter and resets it before every pass, so it cannot
change training memory.

## Configuration

[`pretrain_titans_mac.yaml`](../examples/configs/pretrain_titans_mac.yaml) is the
paper-shaped configuration. Replace its FASTA path with a corpus containing
long genomic sequences, then run:

```bash
synbiotorch train examples/configs/pretrain_titans_mac.yaml
```

Training reports both token-level causal cross-entropy and exact bits per
represented base. The latter is computed additively as total NLL divided by
total represented-base count and `ln(2)`, including across gradient-accumulation
microbatches and distributed ranks. It is not an average of per-batch ratios.

For corpora larger than RAM, `materialize_encoded_streams` atomically publishes
domain-neutral pre-tokenized streams as checksummed NumPy shards plus a versioned
JSON manifest. `TokenStreamStore` reopens those shards read-only with mmap, and
`OrderedStreamWindowDataset.from_store` schedules the logical views without
copying a complete corpus into Python tuples. `materialize_token_streams` is the
sequence/tokenizer convenience layer over the same storage format.

The complete adaptive training path still rejects DDP, AMP, `torch.compile`,
gradient checkpointing, generic token packing, and DataLoader workers at config
validation. Some prerequisites are now independently validated, but the gates
remain until the end-to-end state and checkpoint contracts in the readiness
table below are satisfied.

## Domain-neutral systems probe

The offline probe creates deterministic weighted token streams, materializes and
reopens them through mmap, trains adaptive memory, validates on separate stream
identities, and prints exact loss/BPB plus observed throughput:

```bash
uv run python examples/titans_systems_probe.py \
  --tokens-per-stream 4096 \
  --steps 10
```

The manifest directory is content-addressed by the probe specification and is
reused only after checksum and index validation.

## Execution readiness

| Concern | Validated now | Remaining gate |
|---|---|---|
| Long-stream I/O | Immutable manifests, checksums, mmap views, exact cursor resume | Wire store selection into the top-level run config |
| Metrics | Additive token NLL and represented-base BPB; two-process global ratio reduction | Exercise at production rank and corpus scale |
| DDP ownership | Deterministic, disjoint rank partitions whose union is the selected split | Equalize optimizer-step schedules, route stateful forward through DDP, and checkpoint every rank's cursor and fast state |
| AMP | Attention may autocast while fast-memory state/update arithmetic stays parameter-dtype; CPU BF16 higher-order gradients are finite | CUDA BF16/FP16 accuracy, scaler, and resume qualification |
| Compilation | No-memory inference passes a full-graph compile probe | Adaptive `autograd.grad` update semantics and graph-break policy |
| Workers | Mmap storage is separate from scheduling and can be reopened independently | Central ownership and resumable prefetch; worker-local cursor copies remain rejected |
| Packing | Windows preserve next-token labels inside a stream and never cross identities | A boundary-aware stateful packer; generic concatenating packers remain invalid |

## Direct use

For architecture experiments, state can be controlled explicitly:

```python
output = model.forward_segment(
    input_ids, (model.initial_state(),), valid_mask=valid_mask,
    memory_mode="adaptive",
)
next_state = output.states[0]
```

`memory_mode` is `adaptive` (read and write), `frozen` (read without changing
fast weights), or `none` (zero retrieval and no memory update). Segment-aware
sampling is available as `synbiotorch.generate_titans`; it commits memory only
when a complete segment has been consumed, avoiding repeated writes from
overlapping prefixes.

## Validation boundary

The tests cover causal masking, no future-token leakage, the one-position
fast-weight recurrence, higher-order gradients through memory updates, stream
isolation, boundary-spanning next-token labels, deterministic generation,
additive BPB, mmap integrity and cursor resume, rank ownership and reduction,
the precision/compile boundaries, an end-to-end pipeline run, and bit-for-bit
interrupted/resumed training. Architecture and systems invariants are separate
from any domain-corpus reproduction claim.
