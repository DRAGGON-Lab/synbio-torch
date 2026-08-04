# TITANS Memory-as-Context

synbio-torch contains a minimal, state-explicit implementation of the
Memory-as-Context (MAC) architecture used by the Stage C c16 experiments. It is
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
2. `OrderedStreamWindowDataset` assigns complete biological sequences to batch
   slots and emits `[batch, gradient horizon, 32]` windows without interleaving
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

The initial implementation intentionally rejects generic packed/Parquet
streaming, DDP, AMP, `torch.compile`, gradient checkpointing, and DataLoader
workers. Those features require explicit stream-ownership or higher-order
autograd validation; silently accepting them would make the memory semantics
ambiguous. Training is step-budgeted and currently reports token-level causal
cross-entropy. Represented-base counts are preserved in every batch for a
future bits-per-base metric.

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
isolation, boundary-spanning next-token labels, deterministic generation, an
end-to-end pipeline run, and bit-for-bit interrupted/resumed training. The
implementation matches the Stage C architecture and execution contract; it does
not claim reproduction of the paper's genomic corpus or reported learning
curves without running that experiment.
