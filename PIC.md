# Position-Independent Caching (PIC) — Engine Internals

PIC enables KV cache reuse for content blocks that appear at different positions across requests. Standard transformer KV caches store RoPE-encoded K tensors, making cache entries position-dependent. PIC stores un-rotated K for designated "Plus" blocks and applies RoPE at attention time, making those cache entries relocatable. This is the engine-level implementation; see the root `README_PIC.md` for spnl query syntax, benchmarking, and usage.

## Deferred RoPE

Standard attention applies RoPE before caching:

```
Q, K projections  →  RoPE(Q, K)  →  store K in cache  →  attention
```

PIC Plus blocks use deferred RoPE:

```
Q, K projections  →  store raw K in cache  →  RoPE(Q, all cached K)  →  attention
```

By storing un-rotated K, cache entries become position-independent. RoPE is applied at attention time using each token's *current* position assignment, which may differ from when it was first cached.

On cache reuse, Plus blocks store **pre-RoPE'd K** (computed at save time for positions `0..block_len`) so the forward pass only applies RoPE to new Cross tokens, not the entire cache.

## Block attention masking

When PIC blocks are present, the standard causal mask is replaced with a structured block-attention mask (`PicContext::make_pic_mask`):

| Query → Key | Rule |
|-------------|------|
| Cross → Cross | Standard causal (each token sees all prior cross tokens) |
| Plus → Cross | Each Plus block sees all cross tokens that precede it |
| Plus → Plus (same block) | Causal within the block |
| Plus → Plus (different block) | Blocked — no cross-block attention |
| Cross → Plus | Cross tokens after a Plus block can attend to that block |

This prevents information leakage between independent Plus blocks while maintaining correct attention patterns.

## Content-based cache hashing

Cache keys use **text-based content hashing** (`content_hash_text`), not token-based. This is a deliberate design choice: BPE tokenizers produce different token sequences for the same text depending on surrounding context (the token at the boundary between the chat template and the document content changes). Text hashing is position-independent by construction.

For paged attention, `BlockHash::new_position_independent` provides content-only block hashes that ignore parent chain position, enabling the block engine to match cached blocks by content alone.

## Cross-request cache reuse

### Detection

Plus blocks are identified by one of three mechanisms (checked in priority order):

1. **In-band message markers** (spnl path): The spnl backend tags Plus messages with a `\0PIC_PLUS\0` content prefix. The engine strips the marker before tokenization and records which messages are Plus. Token boundaries are resolved by tokenizing each message individually and finding the token subsequence in the full prompt via `find_subsequence`.

2. **Sentinel token env vars**: When `SPNL_PIC_PLUS_TOKEN` and `SPNL_PIC_CROSS_TOKEN` are set, `detect_pic_blocks` scans the tokenized sequence for those token IDs. This path is for raw-token APIs.

3. **Explicit `PicContext`**: A `PicContext` attached directly to a `NormalRequest` for callers who have already computed token-level block boundaries.

### Lookup and assembly

PIC lookup runs *instead of* (not after) normal prefix caching when Plus blocks are present, because normal prefix caching can only match the shared prefix and would mask the PIC benefit.

The strategy is **all-or-nothing**: if ALL Plus blocks hit the content-based cache (`PrefixCacheManagerV2::search_for_pic_block`), a composite KV cache is assembled from the cached blocks and only Cross tokens need prefill. If any Plus block misses, the sequence falls through to full prefill with within-request PIC (deferred RoPE).

### Save

After generation completes (`sampling.rs`), each Plus block's KV cache is extracted via `KvCache::narrow_range` and saved to the PIC cache (`PrefixCacheManagerV2::add_pic_block`) keyed by text content hash. Pre-RoPE'd K is computed at save time via `Pipeline::pic_pre_rope_k` and stored alongside the raw KV.

### Flow

```
New request with Plus blocks
  │
  ├─ Resolve PIC blocks (3 sources: in-band markers, env var sentinels, explicit PicContext)
  │
  ├─ search_for_pic_block() for each Plus block (text-hash lookup)
  │     │                          │
  │     │ (all hit)                │ (any miss)
  │     ▼                          ▼
  │   Assemble composite KV    Full prefill with PicContext
  │   Prefill Cross tokens     (deferred RoPE for Plus blocks)
  │   only                         │
  │     │                          │
  │     └──── generation ──────────┘
  │                │
  │     narrow_range() per Plus block → extract block KV
  │                │
  │     pic_pre_rope_k() per layer → compute pre-RoPE'd K
  │                │
  │     add_pic_block(text_hash, cache, roped_k) → save for future reuse
```

## PicRope trait and shared attention helper

`PicRope` is the trait that all RoPE implementations must implement for PIC support. It provides:

- `pic_forward_per_token(q, k, q_positions, k_positions)` — per-token Q/K RoPE
- `pic_forward_k_only(k, k_positions)` — K-only RoPE for cache pre-computation

Implementations exist for: `RotaryEmbedding`, `Llama3RotaryEmbedding`, `SmolLm3RotaryEmbedding`, `PhiRotaryEmbedding`, `DeepSeekV2RotaryEmbedding`, `GptOssRotaryEmbedding`.

`pic_sdpa_attention` is the shared helper that all models call in their PIC attention path. It handles both the initial-fill case (raw K, RoPE full cache) and the cache-reuse case (pre-RoPE'd K, RoPE only new Cross tokens). Models don't duplicate deferred-RoPE logic — they just call this helper.

## Supported models

| Model | RoPE type | PIC support |
|-------|-----------|-------------|
| Llama (all variants) | Llama3RotaryEmbedding | Full |
| Qwen2 | RotaryEmbedding | Full |
| Qwen3 | RotaryEmbedding | Full |
| Qwen3 MoE | RotaryEmbedding | Full |
| Qwen3 Next | RotaryEmbedding (partial) | Full (attention layers only; GDN layers unaffected) |
| Mistral | RotaryEmbedding | Full |
| Mixtral (MoE) | RotaryEmbedding | Full |
| Gemma | RotaryEmbedding | Full |
| Gemma 2 | RotaryEmbedding | Full (sliding window layers use PIC mask) |
| StarCoder2 | RotaryEmbedding | Full |
| Phi-2 | RotaryEmbedding (partial) | Full |
| Phi-3 / Phi-4 mini | PhiRotaryEmbedding | Full (partial rotary + short/long tables) |
| Phi-3.5 MoE | PhiRotaryEmbedding | Full |
| SmolLM3 | SmolLm3RotaryEmbedding | Full (no-rope layers handled gracefully) |
| Granite (MoE Hybrid) | RotaryEmbedding (optional) | Full (nope layers skipped; Mamba layers unaffected) |
| GPT-OSS | GptOssRotaryEmbeddingVariant | Full (Standard + YARN variants) |
| DeepSeek V2/V3 | DeepSeekV2RotaryEmbedding | PicRope implemented; model-level wiring pending (MLA attention is structurally different) |
| **GGUF Llama** | RotaryEmbedding | Full |
| **GGUF Qwen2** | RotaryEmbedding | Full |
| **GGUF Qwen3** | RotaryEmbedding | Full |
| **GGUF Qwen3 MoE** | RotaryEmbedding | Full |
| **GGUF StarCoder2** | RotaryEmbedding | Full |

Models without explicit PIC support work unchanged — `NormalModel` provides default no-op `forward_pic` (delegates to `forward`) and `pic_pre_rope_k` (returns `None`).

### Cache reuse without deferred RoPE (unsupported models)

The cache save/lookup/assembly infrastructure (`add_request.rs`, `sampling.rs`, `prefix_cacher.rs`) is pipeline-agnostic — it operates on sequences and KV caches, not model internals. Models without deferred RoPE support still benefit from PIC cache reuse: cached Plus block KV entries are loaded and only Cross tokens go through prefill, producing real TTFT speedups.

However, the cached K tensors retain **position-dependent RoPE encoding** from their original positions. When reused at different positions, the RoPE is incorrect — producing **approximate** results (the same trade-off as [CacheBlend](https://arxiv.org/pdf/2405.16444)). For many RAG workloads the positional error is small enough to be unnoticeable, but results are not mathematically identical to a full prefill.

### Not yet supported (deferred RoPE)

The following model types lack deferred RoPE, so PIC cache reuse is approximate (see above) rather than exact:

- **GGUF Phi-2**: Custom partial RoPE with raw cos/sin tensors (not `RotaryEmbedding`)
- **GGUF Phi-3**: Custom long/short RoPE implementation (not `PhiRotaryEmbedding`)
- **GLM-4 / GLM-4 MoE**: Custom local RoPE implementation
- **XLoRA models**: Different forward path
- **Vision model text backbones**: Separate model implementations in `vision_models/`

## Key types

| Type | Location | Role |
|------|----------|------|
| `PicContext` | `pic.rs` | Block layout + position IDs flowing through the forward pass |
| `PicBlock` | `pic.rs` | Contiguous token range with `is_plus` flag and optional `content_hash` |
| `DeferredRopeMap` | `pic.rs` | Per-layer tracking of which cache entries have deferred RoPE |
| `PicRope` | `pic.rs` | Trait for RoPE types that support per-token position IDs |
| `PicCacheElement` | `prefix_cacher.rs` | Stored block KV + optional pre-RoPE'd K |
| `PicBlockSearchResult` | `prefix_cacher.rs` | Return type for cache lookup |

## Key methods

| Method | Location | Role |
|--------|----------|------|
| `pic_sdpa_attention` | `pic.rs` | Shared SDPA with deferred RoPE (all models call this) |
| `PicContext::new` | `pic.rs` | Build context from blocks, auto-computing position IDs |
| `PicContext::make_pic_mask` | `pic.rs` | Generate block-diagonal attention mask |
| `compute_pic_rope_positions` | `pic.rs` | Q/K position mapping for PIC-aware RoPE |
| `detect_pic_blocks` | `pic.rs` | Scan tokens for Plus/Cross sentinels |
| `content_hash_text` | `pic.rs` | Position-independent content hash from text |
| `find_subsequence` | `add_request.rs` | Token subsequence matching for marker resolution |
| `KvCache::narrow_range` | `kv_cache/mod.rs` | Extract sub-range of KV cache along seq dimension |
| `Sequence::prefill_v2_pic` | `sequence.rs` | Set up PIC-accelerated prefill with pre-loaded KV |
| `PrefixCacheManagerV2::search_for_pic_block` | `prefix_cacher.rs` | Text-hash cache lookup |
| `PrefixCacheManagerV2::add_pic_block` | `prefix_cacher.rs` | Save block KV to content-based cache |
| `Pipeline::pic_pre_rope_k` | `pipeline/mod.rs` | Pre-compute RoPE'd K at cache save time |

## Key optimization decisions

### Pre-RoPE'd K storage (d002fd4b)

The naive approach re-applies RoPE to the entire cached K sequence on every forward step during cache reuse. Since Plus blocks always use positions `0..block_len`, RoPE can be computed once at cache save time and stored alongside the raw K. On reuse, the forward pass only applies RoPE to new Cross tokens. This makes reuse latency nearly constant regardless of cached document size.

### index_select for position gathering (d002fd4b)

The initial implementation gathered per-token positions with N `Tensor::narrow` + `Tensor::cat` operations — one per position. This was replaced with a single `Tensor::index_select` call, reducing the operation from O(n) individual GPU ops to one batched op.

### Pre-allocated slice_set for cache assembly (d002fd4b)

Assembling the composite KV cache from multiple cached blocks initially used iterative `Tensor::cat`, which allocates a new tensor on each concatenation. This was replaced with pre-allocating the output tensor (`Tensor::zeros` with the known total size) and filling it with `slice_set` — one write per block, no intermediate allocations.

### Text-based content hashing (3a011a8a)

BPE tokenizers produce different token sequences for identical text depending on surrounding context (the token at a boundary between chat template and document content changes). Token-based hashing would be position-dependent, defeating the purpose. Text-based hashing (`content_hash_text`) is inherently position-independent.

### Token subsequence matching for in-band markers (3a011a8a)

The in-band `\0PIC_PLUS\0` marker path needs to map text-level message boundaries to token-level block boundaries. Since BPE boundaries shift with context, the engine tokenizes each message individually, then finds the token subsequence in the full prompt. For messages ≥ 3 tokens, it searches for interior tokens (skipping first/last which may have boundary effects) to find the approximate position.

### All-or-nothing cache strategy (3a011a8a)

On cache miss for any Plus block, the entire sequence falls through to full prefill rather than attempting partial reuse. This avoids the complexity of assembling a hybrid cache from some cached and some freshly-computed blocks, which would require tracking per-block RoPE state. A future optimization could support partial hits.

## Performance considerations

- **Mask materialization**: The PIC block-attention mask is materialized as a dense 2D tensor. For very long sequences, a sparse or block-sparse representation would be more efficient.
- **No paged attention for cross-request caching**: Cross-request PIC caching currently only works with eager (non-paged) attention. Extending to paged attention requires integrating with the block engine's copy-on-write semantics.
