# Context Code Stream

This record entry explores a representation-side change instead of a new
backbone. The model keeps the fast recurrent trainer shape, but with
`NUM_RECURRENT_PASSES=1` it effectively becomes a compact dense GPT plus a
causal hashed local-context embedding stream.

## Idea

Each input token gets an auxiliary context code computed from:

- the current token id
- a short causal window of preceding token ids

That code is hashed into a fixed learned codebook and embedded into the model
dimension, then added to the token embedding before the main network runs.

The goal is to make the token representation richer without paying the compute
cost of a more radical backbone.

## Suggested first run

```bash
RUN_ID=context_code_stream \
MODEL_FAMILY=recurrent \
NUM_RECURRENT_PASSES=1 \
DELTA_MODE=none \
TBPTT_CHUNK=0 \
MLP_STYLE=dense \
MODEL_DIM=768 \
NUM_HEADS=12 \
NUM_KV_HEADS=4 \
MLP_MULT=4 \
CACHE_SIZE=0 \
PROTOTYPE_SIZE=0 \
CONTEXT_CODEBOOK_SIZE=4096 \
CONTEXT_NGRAM=3 \
CONTEXT_MIX_INIT=0.1 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=2048 \
VAL_LOSS_EVERY=250 \
TRAIN_LOG_EVERY=50 \
python train_gpt.py
```

## Follow-up sweeps

- `CONTEXT_CODEBOOK_SIZE=2048, 4096, 8192`
- `CONTEXT_NGRAM=2, 3, 4`
- `CONTEXT_MIX_INIT=0.05, 0.1, 0.2`

## Multi-stream context codes

The trainer also supports multiple parallel context-code streams via:

- `CONTEXT_CODEBOOK_SIZES`
- `CONTEXT_NGRAMS`
- `CONTEXT_MIX_INITS`

Example:

```bash
CONTEXT_CODEBOOK_SIZES=4096,4096 \
CONTEXT_NGRAMS=2,4 \
CONTEXT_MIX_INITS=0.08,0.08
```

The clean comparison baseline is the same config with `CONTEXT_CODEBOOK_SIZE=0`.
