//! Position-Independent Caching (PIC) support for spnl Plus blocks.
//!
//! PIC enables KV cache reuse for content blocks that can appear at different
//! positions in a sequence (e.g., RAG documents). The key idea: store un-RoPE'd K
//! in the cache for PIC blocks, and apply RoPE at attention time with the correct
//! current positions.
//!
//! # Block types
//! - **Cross** blocks: Normal position-dependent tokens (system prompt, user query).
//!   These use standard RoPE-before-cache.
//! - **Plus** blocks: Position-independent tokens (RAG documents). These store raw K
//!   in the cache and apply RoPE on the fly.
//!
//! # Attention masking
//! When PIC blocks are present, the standard causal mask is replaced with a structured
//! block-attention mask:
//! - Cross tokens attend to all prior cross tokens (standard causal)
//! - Plus block tokens attend to all cross tokens and tokens within their own block
//! - Plus blocks do NOT attend to each other

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Device, Result, Tensor};

use crate::attention::SdpaParams;
use crate::kv_cache::KvCache;
use crate::layers::Sdpa;
use crate::pipeline::text_models_inputs_processor::FlashParams;

// ---------------------------------------------------------------------------
// PicRope trait — implemented by all RoPE types that support PIC
// ---------------------------------------------------------------------------

/// Trait for RoPE implementations that support per-token position IDs,
/// required for Position-Independent Caching.
pub trait PicRope: Send + Sync {
    /// Apply RoPE to Q and K with per-token position IDs.
    ///
    /// Batch size must be 1. `q_positions` has length = Q seq_len,
    /// `k_positions` has length = K seq_len.
    fn pic_forward_per_token(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_positions: &[usize],
        k_positions: &[usize],
    ) -> Result<(Tensor, Tensor)>;

    /// Apply RoPE to K only at the given positions.
    /// Used to pre-compute RoPE'd K at cache save time.
    fn pic_forward_k_only(&self, k: &Tensor, k_positions: &[usize]) -> Result<Tensor>;
}

// ---------------------------------------------------------------------------
// Shared PIC attention helper
// ---------------------------------------------------------------------------

/// Run SDPA attention with PIC-aware deferred RoPE.
///
/// This is the shared helper that all models call in their PIC attention path.
/// It handles both the initial-fill case (raw K, RoPE full cache) and the
/// cache-reuse case (pre-RoPE'd K, RoPE only new tokens).
///
/// `q`, `k`, `v` should already be projected and reshaped to
/// `(batch, heads, seq_len, head_dim)` but NOT yet RoPE'd.
pub fn pic_sdpa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    pic_ctx: &PicContext,
    rope: &dyn PicRope,
    kv_cache: &mut KvCache,
    attention_mask: Option<&Tensor>,
    flash_params: &FlashParams,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let seq_len = q.dim(2)?;

    if pic_ctx.has_pre_roped_k {
        // Cached Plus blocks already have RoPE'd K. Only apply RoPE
        // to the new (Cross) tokens, then append to cache.
        let past_kv_len = kv_cache.current_seq_len();
        let (q_positions, _) = compute_pic_rope_positions(pic_ctx, seq_len, past_kv_len);
        let k_new_positions: Vec<usize> = (past_kv_len..past_kv_len + seq_len)
            .map(|i| {
                if i < pic_ctx.position_ids.len() {
                    pic_ctx.position_ids[i]
                } else {
                    i
                }
            })
            .collect();

        let (q_roped, k_roped) =
            rope.pic_forward_per_token(q, k, &q_positions, &k_new_positions)?;

        let (k_all, v_all) = kv_cache.append(&k_roped, v)?;

        Sdpa.run_attention(
            &q_roped,
            &k_all,
            &v_all,
            attention_mask,
            Some(flash_params),
            sdpa_params,
        )
    } else {
        // Initial PIC: append raw K and RoPE the full cache
        let (k_all, v_all) = kv_cache.append(k, v)?;

        let past_kv_len = k_all.dim(2)? - seq_len;
        let (q_positions, k_positions) =
            compute_pic_rope_positions(pic_ctx, seq_len, past_kv_len);

        let (q_roped, k_roped) =
            rope.pic_forward_per_token(q, &k_all, &q_positions, &k_positions)?;

        Sdpa.run_attention(
            &q_roped,
            &k_roped,
            &v_all,
            attention_mask,
            Some(flash_params),
            sdpa_params,
        )
    }
}

// ---------------------------------------------------------------------------
// Global PIC cache hit counter (for benchmarking / diagnostics)
// ---------------------------------------------------------------------------

static PIC_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static PIC_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

/// Increment the global PIC cache hit counter.
pub fn record_cache_hit() {
    PIC_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
}

/// Increment the global PIC cache miss counter.
pub fn record_cache_miss() {
    PIC_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
}

/// Read and reset the global PIC cache hit/miss counters.
/// Returns `(hits, misses)`.
pub fn take_cache_stats() -> (u64, u64) {
    let hits = PIC_CACHE_HITS.swap(0, Ordering::Relaxed);
    let misses = PIC_CACHE_MISSES.swap(0, Ordering::Relaxed);
    (hits, misses)
}

/// Describes a contiguous range of tokens and whether it is position-independent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PicBlock {
    /// Start index of this block in the full token sequence.
    pub start: usize,
    /// Number of tokens in this block.
    pub len: usize,
    /// Whether this block is position-independent (Plus) or position-dependent (Cross).
    pub is_plus: bool,
    /// Unique block ID for cache lookup. Plus blocks with the same content_hash
    /// can share cached KV regardless of position.
    pub content_hash: Option<u64>,
}

/// PIC metadata that flows through the forward pass.
///
/// When `Some`, the model should use deferred RoPE for Plus blocks.
/// When `None`, the model uses the standard RoPE-before-cache path.
#[derive(Debug, Clone)]
pub struct PicContext {
    /// The blocks that make up the current sequence, in order.
    pub blocks: Vec<PicBlock>,
    /// Position assignments for each token in the sequence.
    /// For Cross tokens: absolute positions (standard).
    /// For Plus tokens: positions relative to the start of their containing Plus block.
    pub position_ids: Vec<usize>,
    /// Total sequence length covered by these blocks.
    pub total_len: usize,
    /// When true, the K tensors for cached Plus blocks are already RoPE'd
    /// (positions 0..block_len). The forward pass should only apply RoPE to
    /// new (non-cached) tokens, not the entire cache.
    pub has_pre_roped_k: bool,
}

impl PicContext {
    /// Create a new PIC context from a list of blocks.
    ///
    /// Computes position IDs automatically:
    /// - Cross blocks get sequential absolute positions
    /// - Plus blocks get positions starting from 0 within each block
    pub fn new(blocks: Vec<PicBlock>) -> Self {
        // Total length must cover all positions including gaps between blocks
        // (e.g., chat template header tokens not belonging to any message block).
        let total_len = blocks
            .iter()
            .map(|b| b.start + b.len)
            .max()
            .unwrap_or(0);
        let mut position_ids = vec![0usize; total_len];
        let mut cross_pos = 0usize;

        // First pass: assign positions to all gap tokens (not in any block) as Cross
        // by filling sequentially, then overwrite with block-specific positions.
        for i in 0..total_len {
            position_ids[i] = cross_pos;
            cross_pos += 1;
        }

        // Second pass: overwrite Plus block positions with block-local positions.
        // Cross blocks keep their sequential positions from the first pass.
        // Reset cross_pos to recount correctly.
        cross_pos = 0;
        for i in 0..total_len {
            let in_plus = blocks
                .iter()
                .any(|b| b.is_plus && i >= b.start && i < b.start + b.len);
            if in_plus {
                let block = blocks
                    .iter()
                    .find(|b| b.is_plus && i >= b.start && i < b.start + b.len)
                    .unwrap();
                position_ids[i] = i - block.start;
            } else {
                position_ids[i] = cross_pos;
                cross_pos += 1;
            }
        }

        Self {
            blocks,
            position_ids,
            total_len,
            has_pre_roped_k: false,
        }
    }

    /// Returns whether a given token index belongs to a Plus (position-independent) block.
    pub fn is_plus_token(&self, idx: usize) -> bool {
        self.blocks
            .iter()
            .any(|b| b.is_plus && idx >= b.start && idx < b.start + b.len)
    }

    /// Returns which block a token index belongs to (block index in self.blocks).
    pub fn block_for_token(&self, idx: usize) -> Option<usize> {
        self.blocks
            .iter()
            .position(|b| idx >= b.start && idx < b.start + b.len)
    }

    /// Build the block-attention mask for PIC.
    ///
    /// Returns a 2D mask tensor of shape (total_len, total_len + past_kv_len)
    /// where 0.0 means "attend" and -inf means "don't attend".
    ///
    /// Rules:
    /// - Cross tokens attend to all previous cross tokens (causal within cross)
    /// - Plus tokens attend to all cross tokens that precede them AND
    ///   tokens within their own Plus block (causal within block)
    /// - Plus tokens do NOT attend to tokens in other Plus blocks
    pub fn make_pic_mask(
        &self,
        tgt_len: usize,
        past_kv_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let full_len = tgt_len + past_kv_len;

        // For the initial implementation, tgt_len should equal total_len
        // (we're processing the full prompt in one shot for the PoC)
        let mask_data: Vec<f32> = (0..tgt_len)
            .flat_map(|i| {
                let q_idx = past_kv_len + i;
                (0..full_len).map(move |j| {
                    // Default: don't attend
                    let q_is_plus = self.is_plus_token(q_idx);
                    let k_is_plus = self.is_plus_token(j);

                    let attend = if !q_is_plus && !k_is_plus {
                        // Cross -> Cross: standard causal
                        j <= q_idx
                    } else if q_is_plus && !k_is_plus {
                        // Plus -> Cross: attend to all cross tokens that appear
                        // before this Plus block
                        let q_block_idx = self.block_for_token(q_idx);
                        if let Some(qb) = q_block_idx {
                            let q_block = &self.blocks[qb];
                            // Attend to cross tokens before the start of this Plus block
                            j < q_block.start
                        } else {
                            false
                        }
                    } else if q_is_plus && k_is_plus {
                        // Plus -> Plus: only attend within same block, causally
                        let q_block = self.block_for_token(q_idx);
                        let k_block = self.block_for_token(j);
                        q_block == k_block && j <= q_idx
                    } else {
                        // Cross -> Plus: cross tokens after a Plus block can attend
                        // to the Plus block's tokens
                        j <= q_idx
                    };

                    if attend {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        let mask = Tensor::from_vec(mask_data, (tgt_len, full_len), device)?;
        mask.to_dtype(dtype)
    }
}

/// Describes which tokens in a cached KV sequence have deferred RoPE.
///
/// This is stored alongside the KV cache to know which entries need RoPE
/// applied at attention time.
#[derive(Debug, Clone)]
pub struct DeferredRopeMap {
    /// For each token position in the cache, `true` if the K at that position
    /// is raw (un-RoPE'd) and needs RoPE applied at attention time.
    pub is_deferred: Vec<bool>,
    /// For each token position, the RoPE position to use.
    /// For deferred entries, this should be recomputed based on current arrangement.
    /// For non-deferred entries, this is the position that was already applied.
    pub rope_positions: Vec<usize>,
}

impl DeferredRopeMap {
    pub fn new() -> Self {
        Self {
            is_deferred: Vec::new(),
            rope_positions: Vec::new(),
        }
    }

    /// Extend the map with new entries.
    pub fn extend(&mut self, deferred: &[bool], positions: &[usize]) {
        debug_assert_eq!(deferred.len(), positions.len());
        self.is_deferred.extend_from_slice(deferred);
        self.rope_positions.extend_from_slice(positions);
    }

    /// Returns true if any entries need deferred RoPE.
    pub fn has_deferred(&self) -> bool {
        self.is_deferred.iter().any(|&d| d)
    }

    /// Current length (number of cached tokens tracked).
    pub fn len(&self) -> usize {
        self.is_deferred.len()
    }

    pub fn is_empty(&self) -> bool {
        self.is_deferred.is_empty()
    }

    /// Reset the map (when cache is cleared).
    pub fn reset(&mut self) {
        self.is_deferred.clear();
        self.rope_positions.clear();
    }
}

impl Default for DeferredRopeMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute Q and K position IDs for PIC-aware RoPE application.
///
/// Given a PIC context, the current sequence length, and the past KV cache length,
/// returns `(q_positions, k_positions)` suitable for per-token RoPE.
///
/// This is the shared helper that models call in their PIC attention path.
pub fn compute_pic_rope_positions(
    pic_ctx: &PicContext,
    seq_len: usize,
    past_kv_len: usize,
) -> (Vec<usize>, Vec<usize>) {
    let total_kv_len = past_kv_len + seq_len;

    let q_positions: Vec<usize> = (past_kv_len..total_kv_len)
        .map(|i| {
            if i < pic_ctx.position_ids.len() {
                pic_ctx.position_ids[i]
            } else {
                i
            }
        })
        .collect();

    let k_positions: Vec<usize> = (0..total_kv_len)
        .map(|i| {
            if i < pic_ctx.position_ids.len() {
                pic_ctx.position_ids[i]
            } else {
                i
            }
        })
        .collect();

    (q_positions, k_positions)
}

/// Read the Plus sentinel token ID from the `SPNL_PIC_PLUS_TOKEN` env var.
pub fn pic_plus_token() -> Option<u32> {
    std::env::var("SPNL_PIC_PLUS_TOKEN")
        .ok()?
        .parse()
        .ok()
}

/// Read the Cross sentinel token ID from the `SPNL_PIC_CROSS_TOKEN` env var.
pub fn pic_cross_token() -> Option<u32> {
    std::env::var("SPNL_PIC_CROSS_TOKEN")
        .ok()?
        .parse()
        .ok()
}

/// Compute a content hash for a slice of token IDs.
pub fn content_hash_tokens(tokens: &[u32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

/// Compute a content hash for a text string.
/// This is position-independent — the same text always produces the same hash
/// regardless of where it appears in the token sequence.
pub fn content_hash_text(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

/// Scan a token sequence for Plus/Cross sentinel tokens and return PIC blocks.
///
/// Returns `None` if no sentinel tokens are found (i.e., the sequence has no PIC structure).
/// Otherwise returns blocks covering the full token range, with sentinel tokens excluded
/// from the block content ranges.
pub fn detect_pic_blocks(tokens: &[u32], plus_token: u32, cross_token: u32) -> Option<Vec<PicBlock>> {
    // Check if any sentinel tokens exist
    if !tokens.iter().any(|&t| t == plus_token || t == cross_token) {
        return None;
    }

    let mut blocks = Vec::new();
    let mut i = 0;
    let mut current_start = 0;
    let mut current_is_plus = false;
    let mut in_block = false;

    while i < tokens.len() {
        if tokens[i] == plus_token || tokens[i] == cross_token {
            // Close the current block if we have one
            if in_block && i > current_start {
                let block_toks = &tokens[current_start..i];
                blocks.push(PicBlock {
                    start: current_start,
                    len: i - current_start,
                    is_plus: current_is_plus,
                    content_hash: if current_is_plus {
                        Some(content_hash_tokens(block_toks))
                    } else {
                        None
                    },
                });
            }
            // Start a new block after this sentinel
            current_is_plus = tokens[i] == plus_token;
            i += 1;
            current_start = i;
            in_block = true;
        } else {
            if !in_block {
                // Tokens before any sentinel: treat as Cross
                current_is_plus = false;
                current_start = i;
                in_block = true;
            }
            i += 1;
        }
    }

    // Close the final block
    if in_block && i > current_start {
        let block_toks = &tokens[current_start..i];
        blocks.push(PicBlock {
            start: current_start,
            len: i - current_start,
            is_plus: current_is_plus,
            content_hash: if current_is_plus {
                Some(content_hash_tokens(block_toks))
            } else {
                None
            },
        });
    }

    if blocks.is_empty() {
        None
    } else {
        Some(blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper to build blocks concisely
    // -----------------------------------------------------------------------

    fn cross(start: usize, len: usize) -> PicBlock {
        PicBlock {
            start,
            len,
            is_plus: false,
            content_hash: None,
        }
    }

    fn plus(start: usize, len: usize) -> PicBlock {
        PicBlock {
            start,
            len,
            is_plus: true,
            content_hash: Some(start as u64 * 1000 + len as u64),
        }
    }

    // -----------------------------------------------------------------------
    // PicContext::new — position ID assignment
    // -----------------------------------------------------------------------

    #[test]
    fn test_pic_context_positions() {
        let ctx = PicContext::new(vec![cross(0, 5), plus(5, 3), cross(8, 2)]);

        // Cross block: positions 0..5
        assert_eq!(ctx.position_ids[0], 0);
        assert_eq!(ctx.position_ids[4], 4);

        // Plus block: local positions 0..3
        assert_eq!(ctx.position_ids[5], 0);
        assert_eq!(ctx.position_ids[6], 1);
        assert_eq!(ctx.position_ids[7], 2);

        // Second cross block: continues from 5 (skipping Plus tokens)
        assert_eq!(ctx.position_ids[8], 5);
        assert_eq!(ctx.position_ids[9], 6);
    }

    #[test]
    fn test_positions_multiple_plus_blocks() {
        // Cross(3) Plus(2) Cross(1) Plus(4) Cross(2)
        let ctx = PicContext::new(vec![
            cross(0, 3),
            plus(3, 2),
            cross(5, 1),
            plus(6, 4),
            cross(10, 2),
        ]);

        // First cross: 0,1,2
        assert_eq!(&ctx.position_ids[0..3], &[0, 1, 2]);
        // First plus: local 0,1
        assert_eq!(&ctx.position_ids[3..5], &[0, 1]);
        // Middle cross: continues at 3
        assert_eq!(ctx.position_ids[5], 3);
        // Second plus: local 0,1,2,3
        assert_eq!(&ctx.position_ids[6..10], &[0, 1, 2, 3]);
        // Final cross: continues at 4,5
        assert_eq!(&ctx.position_ids[10..12], &[4, 5]);
    }

    #[test]
    fn test_positions_plus_at_start() {
        let ctx = PicContext::new(vec![plus(0, 4), cross(4, 3)]);

        // Plus block: local 0..4
        assert_eq!(&ctx.position_ids[0..4], &[0, 1, 2, 3]);
        // Cross starts at 0 (no prior cross tokens)
        assert_eq!(&ctx.position_ids[4..7], &[0, 1, 2]);
    }

    #[test]
    fn test_positions_gap_between_blocks() {
        // Blocks don't cover indices 3,4 — simulates chat template tokens
        let ctx = PicContext::new(vec![cross(0, 3), plus(5, 3), cross(8, 2)]);

        assert_eq!(ctx.total_len, 10);
        // Gap tokens (3,4) are treated as cross
        assert_eq!(&ctx.position_ids[0..3], &[0, 1, 2]);
        assert_eq!(&ctx.position_ids[3..5], &[3, 4]); // gap = cross
        assert_eq!(&ctx.position_ids[5..8], &[0, 1, 2]); // plus = local
        assert_eq!(&ctx.position_ids[8..10], &[5, 6]); // cross continues
    }

    #[test]
    fn test_positions_empty_blocks() {
        let ctx = PicContext::new(vec![]);
        assert_eq!(ctx.total_len, 0);
        assert!(ctx.position_ids.is_empty());
    }

    #[test]
    fn test_positions_only_plus() {
        let ctx = PicContext::new(vec![plus(0, 5)]);
        assert_eq!(&ctx.position_ids[..], &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_positions_adjacent_plus_blocks() {
        // Two plus blocks back to back — each gets local positions
        let ctx = PicContext::new(vec![plus(0, 3), plus(3, 2)]);

        assert_eq!(&ctx.position_ids[0..3], &[0, 1, 2]);
        assert_eq!(&ctx.position_ids[3..5], &[0, 1]);
    }

    // -----------------------------------------------------------------------
    // PicContext::is_plus_token
    // -----------------------------------------------------------------------

    #[test]
    fn test_pic_context_is_plus() {
        let ctx = PicContext::new(vec![cross(0, 3), plus(3, 4)]);

        assert!(!ctx.is_plus_token(0));
        assert!(!ctx.is_plus_token(2));
        assert!(ctx.is_plus_token(3));
        assert!(ctx.is_plus_token(6));
    }

    #[test]
    fn test_is_plus_token_beyond_blocks() {
        let ctx = PicContext::new(vec![cross(0, 3)]);
        // Index beyond all blocks should not be Plus
        assert!(!ctx.is_plus_token(5));
    }

    // -----------------------------------------------------------------------
    // PicContext::block_for_token
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_for_token() {
        let ctx = PicContext::new(vec![cross(0, 3), plus(3, 2), cross(5, 1)]);

        assert_eq!(ctx.block_for_token(0), Some(0));
        assert_eq!(ctx.block_for_token(2), Some(0));
        assert_eq!(ctx.block_for_token(3), Some(1));
        assert_eq!(ctx.block_for_token(4), Some(1));
        assert_eq!(ctx.block_for_token(5), Some(2));
    }

    #[test]
    fn test_block_for_token_in_gap() {
        // Gap at index 3,4
        let ctx = PicContext::new(vec![cross(0, 3), plus(5, 2)]);
        assert_eq!(ctx.block_for_token(3), None);
        assert_eq!(ctx.block_for_token(4), None);
    }

    #[test]
    fn test_block_for_token_beyond_all() {
        let ctx = PicContext::new(vec![cross(0, 3)]);
        assert_eq!(ctx.block_for_token(10), None);
    }

    // -----------------------------------------------------------------------
    // PicContext::make_pic_mask
    // -----------------------------------------------------------------------

    /// Extract the raw f32 mask values from a mask tensor for inspection.
    fn mask_to_vec(mask: &Tensor) -> Vec<f32> {
        mask.to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
            .into_iter()
            .flatten()
            .collect()
    }

    fn attends(val: f32) -> bool {
        val == 0.0
    }

    fn blocked(val: f32) -> bool {
        val.is_infinite() && val.is_sign_negative()
    }

    #[test]
    fn test_make_pic_mask_basic_rules() {
        // Layout: Cross(3) Plus_A(2) Plus_B(2) Cross(1)
        // Indices: 0 1 2 | 3 4 | 5 6 | 7
        let ctx = PicContext::new(vec![cross(0, 3), plus(3, 2), plus(5, 2), cross(7, 1)]);

        let mask = ctx
            .make_pic_mask(8, 0, &Device::Cpu, DType::F32)
            .unwrap();
        let vals = mask_to_vec(&mask);
        let n = 8;
        let at = |q: usize, k: usize| vals[q * n + k];

        // Cross -> Cross: standard causal
        assert!(attends(at(0, 0))); // token 0 sees itself
        assert!(blocked(at(0, 1))); // token 0 doesn't see future
        assert!(attends(at(2, 0))); // token 2 sees token 0
        assert!(attends(at(2, 2))); // token 2 sees itself

        // Plus_A -> Cross: sees cross tokens before Plus_A's block start (0,1,2)
        assert!(attends(at(3, 0)));
        assert!(attends(at(3, 2)));
        assert!(attends(at(4, 0)));

        // Plus_A -> Plus_A: causal within block
        assert!(attends(at(3, 3))); // first token of A sees itself
        assert!(attends(at(4, 3))); // second token of A sees first
        assert!(attends(at(4, 4))); // second token sees itself

        // Plus_A -> Plus_B: blocked (different Plus blocks)
        assert!(blocked(at(3, 5)));
        assert!(blocked(at(4, 6)));

        // Plus_B -> Plus_A: blocked
        assert!(blocked(at(5, 3)));
        assert!(blocked(at(6, 4)));

        // Plus_B -> Plus_B: causal within block
        assert!(attends(at(5, 5)));
        assert!(attends(at(6, 5)));
        assert!(attends(at(6, 6)));

        // Plus_B -> Cross: sees cross tokens before Plus_B's start (0,1,2)
        assert!(attends(at(5, 0)));
        assert!(attends(at(5, 2)));
        // Plus_B should NOT see the cross tokens at Plus_A's positions
        // (those are Plus, not Cross, so this is Plus->Plus cross-block)
        assert!(blocked(at(5, 3)));

        // Cross(7) -> Plus: cross after Plus can see Plus tokens
        assert!(attends(at(7, 3))); // sees Plus_A
        assert!(attends(at(7, 5))); // sees Plus_B
        assert!(attends(at(7, 7))); // sees itself

        // Cross(7) -> Cross: causal
        assert!(attends(at(7, 0)));
        assert!(attends(at(7, 2)));
    }

    #[test]
    fn test_make_pic_mask_with_past_kv() {
        // past_kv_len=2, so q indices start at 2 in the context
        // Layout: Cross(2) [past] | Plus(2) Cross(1) [new]
        let ctx = PicContext::new(vec![cross(0, 2), plus(2, 2), cross(4, 1)]);

        let tgt_len = 3; // tokens 2,3,4 are new
        let past_kv_len = 2;
        let mask = ctx
            .make_pic_mask(tgt_len, past_kv_len, &Device::Cpu, DType::F32)
            .unwrap();
        let vals = mask_to_vec(&mask);
        let full_len = tgt_len + past_kv_len; // 5
        let at = |q: usize, k: usize| vals[q * full_len + k];

        // q=0 is token index 2 (Plus), q=1 is token 3 (Plus), q=2 is token 4 (Cross)
        // Plus(idx=2) -> Cross(idx=0): should attend (cross before plus block)
        assert!(attends(at(0, 0)));
        assert!(attends(at(0, 1)));
        // Plus(idx=2) -> Plus(idx=2): attend (same block, self)
        assert!(attends(at(0, 2)));

        // Cross(idx=4) -> all prior: attends
        assert!(attends(at(2, 0)));
        assert!(attends(at(2, 4)));
    }

    #[test]
    fn test_make_pic_mask_shape() {
        let ctx = PicContext::new(vec![cross(0, 4), plus(4, 3)]);
        let mask = ctx
            .make_pic_mask(7, 0, &Device::Cpu, DType::F32)
            .unwrap();
        assert_eq!(mask.dims(), &[7, 7]);

        let mask_past = ctx
            .make_pic_mask(3, 4, &Device::Cpu, DType::F32)
            .unwrap();
        assert_eq!(mask_past.dims(), &[3, 7]);
    }

    // -----------------------------------------------------------------------
    // compute_pic_rope_positions
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_pic_rope_positions_no_past() {
        let ctx = PicContext::new(vec![cross(0, 3), plus(3, 2), cross(5, 2)]);
        let (q_pos, k_pos) = compute_pic_rope_positions(&ctx, 7, 0);

        // Q positions = position_ids[0..7]
        assert_eq!(q_pos, ctx.position_ids);
        // K positions = same (no past)
        assert_eq!(k_pos, ctx.position_ids);
    }

    #[test]
    fn test_compute_pic_rope_positions_with_past() {
        let ctx = PicContext::new(vec![cross(0, 3), plus(3, 2), cross(5, 2)]);

        // Simulating: past_kv_len=3 (cross tokens cached), seq_len=4 (processing rest)
        let (q_pos, k_pos) = compute_pic_rope_positions(&ctx, 4, 3);

        // Q covers indices 3..7 → position_ids[3..7]
        assert_eq!(q_pos, &ctx.position_ids[3..7]);
        // K covers indices 0..7 → position_ids[0..7]
        assert_eq!(k_pos, ctx.position_ids);
    }

    #[test]
    fn test_compute_pic_rope_positions_beyond_context() {
        // When indices exceed position_ids, falls back to identity
        let ctx = PicContext::new(vec![cross(0, 3)]);

        let (q_pos, k_pos) = compute_pic_rope_positions(&ctx, 2, 3);

        // Q covers indices 3,4 — both beyond position_ids (len=3), so fallback
        assert_eq!(q_pos, vec![3, 4]);
        // K covers 0..5 — first 3 from position_ids, rest fallback
        assert_eq!(k_pos, vec![0, 1, 2, 3, 4]);
    }

    // -----------------------------------------------------------------------
    // content_hash_text
    // -----------------------------------------------------------------------

    #[test]
    fn test_content_hash_text_deterministic() {
        let h1 = content_hash_text("The capital of France is Paris");
        let h2 = content_hash_text("The capital of France is Paris");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_text_differs_for_different_content() {
        let h1 = content_hash_text("Document A");
        let h2 = content_hash_text("Document B");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_content_hash_text_empty() {
        // Empty string should still produce a valid hash
        let h = content_hash_text("");
        // Just verify it doesn't panic and returns something
        let _ = h;
    }

    // -----------------------------------------------------------------------
    // content_hash_tokens
    // -----------------------------------------------------------------------

    #[test]
    fn test_content_hash_tokens_deterministic() {
        let h1 = content_hash_tokens(&[10, 20, 30]);
        let h2 = content_hash_tokens(&[10, 20, 30]);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_tokens_differs_for_different_content() {
        let h1 = content_hash_tokens(&[10, 20, 30]);
        let h2 = content_hash_tokens(&[10, 20, 31]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_content_hash_tokens_order_matters() {
        let h1 = content_hash_tokens(&[10, 20]);
        let h2 = content_hash_tokens(&[20, 10]);
        assert_ne!(h1, h2);
    }

    // -----------------------------------------------------------------------
    // DeferredRopeMap
    // -----------------------------------------------------------------------

    #[test]
    fn test_deferred_rope_map() {
        let mut map = DeferredRopeMap::new();
        assert!(!map.has_deferred());

        map.extend(&[false, false, true, true], &[0, 1, 0, 1]);
        assert!(map.has_deferred());
        assert_eq!(map.len(), 4);

        map.reset();
        assert!(map.is_empty());
    }

    #[test]
    fn test_deferred_rope_map_no_deferred() {
        let mut map = DeferredRopeMap::new();
        map.extend(&[false, false, false], &[0, 1, 2]);
        assert!(!map.has_deferred());
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_deferred_rope_map_multiple_extends() {
        let mut map = DeferredRopeMap::new();
        map.extend(&[false], &[0]);
        map.extend(&[true, true], &[0, 1]);
        assert!(map.has_deferred());
        assert_eq!(map.len(), 3);
    }

    // -----------------------------------------------------------------------
    // detect_pic_blocks — edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_pic_blocks_no_sentinels() {
        assert!(detect_pic_blocks(&[1, 2, 3, 4], 100, 101).is_none());
    }

    #[test]
    fn test_detect_pic_blocks_simple() {
        // cross_token(101), 10, 11, plus_token(100), 20, 21, 22, cross_token(101), 30
        let tokens = vec![101, 10, 11, 100, 20, 21, 22, 101, 30];
        let blocks = detect_pic_blocks(&tokens, 100, 101).unwrap();

        assert_eq!(blocks.len(), 3);

        assert_eq!(blocks[0].start, 1);
        assert_eq!(blocks[0].len, 2);
        assert!(!blocks[0].is_plus);
        assert!(blocks[0].content_hash.is_none());

        assert_eq!(blocks[1].start, 4);
        assert_eq!(blocks[1].len, 3);
        assert!(blocks[1].is_plus);
        assert!(blocks[1].content_hash.is_some());

        assert_eq!(blocks[2].start, 8);
        assert_eq!(blocks[2].len, 1);
        assert!(!blocks[2].is_plus);
    }

    #[test]
    fn test_detect_pic_blocks_content_hash_stable() {
        let tokens_a = vec![100, 20, 21, 22];
        let tokens_b = vec![100, 20, 21, 22];
        let blocks_a = detect_pic_blocks(&tokens_a, 100, 101).unwrap();
        let blocks_b = detect_pic_blocks(&tokens_b, 100, 101).unwrap();
        assert_eq!(blocks_a[0].content_hash, blocks_b[0].content_hash);
    }

    #[test]
    fn test_detect_pic_blocks_tokens_before_sentinel() {
        // Tokens 1,2,3 appear before any sentinel — should be treated as Cross
        let tokens = vec![1, 2, 3, 100, 10, 11];
        let blocks = detect_pic_blocks(&tokens, 100, 101).unwrap();

        assert_eq!(blocks.len(), 2);
        // Pre-sentinel tokens: Cross
        assert_eq!(blocks[0].start, 0);
        assert_eq!(blocks[0].len, 3);
        assert!(!blocks[0].is_plus);
        // After plus sentinel
        assert_eq!(blocks[1].start, 4);
        assert_eq!(blocks[1].len, 2);
        assert!(blocks[1].is_plus);
    }

    #[test]
    fn test_detect_pic_blocks_consecutive_sentinels() {
        // Two sentinels in a row: plus then cross — empty plus block is skipped
        let tokens = vec![100, 101, 10, 11];
        let blocks = detect_pic_blocks(&tokens, 100, 101).unwrap();

        // The plus sentinel at 0 starts a block at 1, but cross sentinel at 1
        // closes it immediately (len=0), so it's skipped.
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].start, 2);
        assert!(!blocks[0].is_plus);
    }

    #[test]
    fn test_detect_pic_blocks_sentinel_at_end() {
        // Sentinel as the very last token — no content after it
        let tokens = vec![101, 10, 11, 100];
        let blocks = detect_pic_blocks(&tokens, 100, 101).unwrap();

        // Only the cross block [10, 11] should appear
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].start, 1);
        assert_eq!(blocks[0].len, 2);
        assert!(!blocks[0].is_plus);
    }

    #[test]
    fn test_detect_pic_blocks_only_plus() {
        let tokens = vec![100, 10, 20, 30];
        let blocks = detect_pic_blocks(&tokens, 100, 101).unwrap();

        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].is_plus);
        assert_eq!(blocks[0].start, 1);
        assert_eq!(blocks[0].len, 3);
    }

    #[test]
    fn test_detect_pic_blocks_multiple_plus() {
        // plus(10,11) plus(20,21)
        let tokens = vec![100, 10, 11, 100, 20, 21];
        let blocks = detect_pic_blocks(&tokens, 100, 101).unwrap();

        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].is_plus);
        assert_eq!(blocks[0].start, 1);
        assert_eq!(blocks[0].len, 2);
        assert!(blocks[1].is_plus);
        assert_eq!(blocks[1].start, 4);
        assert_eq!(blocks[1].len, 2);

        // Different content → different hashes
        assert_ne!(blocks[0].content_hash, blocks[1].content_hash);
    }

    #[test]
    fn test_detect_pic_blocks_empty_input() {
        assert!(detect_pic_blocks(&[], 100, 101).is_none());
    }

    #[test]
    fn test_detect_pic_blocks_only_sentinel() {
        // A single sentinel with no content produces no blocks
        let tokens = vec![100];
        let blocks = detect_pic_blocks(&tokens, 100, 101);
        assert!(blocks.is_none());
    }

    // -----------------------------------------------------------------------
    // Cache stats counters
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_stats_record_and_reset() {
        // Reset any prior state
        let _ = take_cache_stats();

        record_cache_hit();
        record_cache_hit();
        record_cache_miss();

        let (hits, misses) = take_cache_stats();
        assert_eq!(hits, 2);
        assert_eq!(misses, 1);

        // After take, counters should be reset
        let (hits, misses) = take_cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
    }

    // -----------------------------------------------------------------------
    // pic_sdpa_attention integration tests
    // -----------------------------------------------------------------------

    use std::collections::HashMap;
    use crate::layers::RotaryEmbedding;

    const HEAD_DIM: usize = 4;
    const NUM_HEADS: usize = 2;
    const MAX_POS: usize = 64;
    const EPS: f64 = 1e-4;

    fn cpu_flash_params() -> FlashParams {
        FlashParams {
            max_q: 0,
            max_k: 0,
            cumulative_seqlens_q: HashMap::new(),
            cumulative_seqlens_k: HashMap::new(),
            causal: false,
        }
    }

    fn test_sdpa_params() -> SdpaParams {
        SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        }
    }

    fn test_rope() -> RotaryEmbedding {
        RotaryEmbedding::new(10000.0, HEAD_DIM, MAX_POS, &Device::Cpu, true, DType::F32).unwrap()
    }

    /// Create a tensor of shape (1, NUM_HEADS, seq_len, HEAD_DIM) filled with `val`.
    fn make_tensor(seq_len: usize, val: f32) -> Tensor {
        Tensor::full(val, (1, NUM_HEADS, seq_len, HEAD_DIM), &Device::Cpu).unwrap()
    }

    fn make_kv_cache() -> KvCache {
        KvCache::new_normal(2, MAX_POS, 16)
    }

    #[test]
    fn test_pic_sdpa_output_shape() {
        let rope = test_rope();
        let flash = cpu_flash_params();
        let sdpa = test_sdpa_params();
        let seq_len = 6;

        let q = make_tensor(seq_len, 1.0);
        let k = make_tensor(seq_len, 1.0);
        let v = make_tensor(seq_len, 1.0);
        let mut cache = make_kv_cache();

        let ctx = PicContext::new(vec![cross(0, 3), plus(3, 3)]);
        let mask = ctx.make_pic_mask(seq_len, 0, &Device::Cpu, DType::F32).unwrap();

        let out = pic_sdpa_attention(
            &q, &k, &v, &ctx, &rope, &mut cache, Some(&mask), &flash, &sdpa,
        )
        .unwrap();

        assert_eq!(out.dims(), &[1, NUM_HEADS, seq_len, HEAD_DIM]);
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()), "Output must be finite");
    }

    #[test]
    fn test_pic_sdpa_populates_cache() {
        let rope = test_rope();
        let flash = cpu_flash_params();
        let sdpa = test_sdpa_params();
        let seq_len = 6;

        let q = make_tensor(seq_len, 1.0);
        let k = make_tensor(seq_len, 1.0);
        let v = make_tensor(seq_len, 1.0);
        let mut cache = make_kv_cache();

        let ctx = PicContext::new(vec![cross(0, 3), plus(3, 3)]);
        let mask = ctx.make_pic_mask(seq_len, 0, &Device::Cpu, DType::F32).unwrap();

        let _ = pic_sdpa_attention(
            &q, &k, &v, &ctx, &rope, &mut cache, Some(&mask), &flash, &sdpa,
        )
        .unwrap();

        assert_eq!(cache.current_seq_len(), seq_len);
    }

    #[test]
    fn test_pic_sdpa_cross_only_matches_standard() {
        let rope = test_rope();
        let flash = cpu_flash_params();
        let sdpa = test_sdpa_params();
        let seq_len = 4;

        let q = make_tensor(seq_len, 1.0);
        let k = make_tensor(seq_len, 0.5);
        let v = make_tensor(seq_len, 0.3);

        // All-cross PIC context (no Plus blocks)
        let ctx = PicContext::new(vec![cross(0, seq_len)]);
        let mask = ctx.make_pic_mask(seq_len, 0, &Device::Cpu, DType::F32).unwrap();

        let mut cache_pic = make_kv_cache();
        let out_pic = pic_sdpa_attention(
            &q, &k, &v, &ctx, &rope, &mut cache_pic, Some(&mask), &flash, &sdpa,
        )
        .unwrap();

        // Standard path: apply RoPE manually, then SDPA
        let mut cache_std = make_kv_cache();
        let positions: Vec<usize> = (0..seq_len).collect();
        let (q_roped, k_roped) = rope.forward_per_token(&q, &k, &positions, &positions).unwrap();
        let (k_all, v_all) = cache_std.append(&k_roped, &v).unwrap();
        let out_std = crate::layers::Sdpa.run_attention(
            &q_roped,
            &k_all,
            &v_all,
            Some(&mask),
            Some(&flash),
            &sdpa,
        )
        .unwrap();

        let diff = (&out_pic - &out_std)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64;
        assert!(
            diff < EPS,
            "All-cross PIC should match standard SDPA, diff={diff}"
        );
    }

    #[test]
    fn test_pic_sdpa_plus_blocks_isolated() {
        let rope = test_rope();
        let flash = cpu_flash_params();
        let sdpa = test_sdpa_params();

        // Layout: Cross(2) Plus_A(2) Plus_B(2)
        let seq_len = 6;
        let q = make_tensor(seq_len, 1.0);
        let k = make_tensor(seq_len, 1.0);
        let v_a = Tensor::from_vec(
            vec![1.0f32; NUM_HEADS * seq_len * HEAD_DIM],
            (1, NUM_HEADS, seq_len, HEAD_DIM),
            &Device::Cpu,
        )
        .unwrap();

        let ctx = PicContext::new(vec![cross(0, 2), plus(2, 2), plus(4, 2)]);
        let mask = ctx.make_pic_mask(seq_len, 0, &Device::Cpu, DType::F32).unwrap();

        let mut cache_a = make_kv_cache();
        let out_a = pic_sdpa_attention(
            &q, &k, &v_a, &ctx, &rope, &mut cache_a, Some(&mask), &flash, &sdpa,
        )
        .unwrap();

        // Change Plus_B's V values (indices 4,5) but keep everything else the same
        let mut v_b_data = vec![1.0f32; NUM_HEADS * seq_len * HEAD_DIM];
        // Modify Plus_B slice: for each head, positions 4 and 5
        for h in 0..NUM_HEADS {
            for s in 4..6 {
                for d in 0..HEAD_DIM {
                    v_b_data[h * seq_len * HEAD_DIM + s * HEAD_DIM + d] = 99.0;
                }
            }
        }
        let v_b = Tensor::from_vec(
            v_b_data,
            (1, NUM_HEADS, seq_len, HEAD_DIM),
            &Device::Cpu,
        )
        .unwrap();

        let mut cache_b = make_kv_cache();
        let out_b = pic_sdpa_attention(
            &q, &k, &v_b, &ctx, &rope, &mut cache_b, Some(&mask), &flash, &sdpa,
        )
        .unwrap();

        // Plus_A tokens (indices 2,3) should produce the same output regardless of Plus_B's V
        let out_a_plus_a = out_a.narrow(2, 2, 2).unwrap();
        let out_b_plus_a = out_b.narrow(2, 2, 2).unwrap();
        let diff = (&out_a_plus_a - &out_b_plus_a)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64;
        assert!(
            diff < EPS,
            "Plus_A output should be isolated from Plus_B content changes, diff={diff}"
        );
    }

    #[test]
    fn test_pic_sdpa_decode_step() {
        let rope = test_rope();
        let flash = cpu_flash_params();
        let sdpa = test_sdpa_params();

        // First: fill 6 tokens
        let fill_len = 6;
        let q = make_tensor(fill_len, 1.0);
        let k = make_tensor(fill_len, 1.0);
        let v = make_tensor(fill_len, 1.0);
        let mut cache = make_kv_cache();

        let ctx = PicContext::new(vec![cross(0, 3), plus(3, 3)]);
        let mask = ctx.make_pic_mask(fill_len, 0, &Device::Cpu, DType::F32).unwrap();

        let _ = pic_sdpa_attention(
            &q, &k, &v, &ctx, &rope, &mut cache, Some(&mask), &flash, &sdpa,
        )
        .unwrap();
        assert_eq!(cache.current_seq_len(), fill_len);

        // Decode: 1 new token
        let q_dec = make_tensor(1, 0.5);
        let k_dec = make_tensor(1, 0.5);
        let v_dec = make_tensor(1, 0.5);

        // For decode, extend context: the new token is cross at position 6
        let mut ctx_dec = PicContext::new(vec![cross(0, 3), plus(3, 3), cross(6, 1)]);
        ctx_dec.has_pre_roped_k = true;

        let out_dec = pic_sdpa_attention(
            &q_dec,
            &k_dec,
            &v_dec,
            &ctx_dec,
            &rope,
            &mut cache,
            None,
            &flash,
            &sdpa,
        )
        .unwrap();

        assert_eq!(out_dec.dims(), &[1, NUM_HEADS, 1, HEAD_DIM]);
        assert_eq!(cache.current_seq_len(), fill_len + 1);
        let vals: Vec<f32> = out_dec.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()), "Decode output must be finite");
    }

    #[test]
    fn test_pic_sdpa_pre_roped_k_path() {
        let rope = test_rope();
        let flash = cpu_flash_params();
        let sdpa = test_sdpa_params();
        let seq_len = 4;

        let q = make_tensor(seq_len, 1.0);
        let k = make_tensor(seq_len, 1.0);
        let v = make_tensor(seq_len, 1.0);
        let mut cache = make_kv_cache();

        let mut ctx = PicContext::new(vec![cross(0, 2), plus(2, 2)]);
        ctx.has_pre_roped_k = true;

        let mask = ctx.make_pic_mask(seq_len, 0, &Device::Cpu, DType::F32).unwrap();

        let out = pic_sdpa_attention(
            &q, &k, &v, &ctx, &rope, &mut cache, Some(&mask), &flash, &sdpa,
        )
        .unwrap();

        assert_eq!(out.dims(), &[1, NUM_HEADS, seq_len, HEAD_DIM]);
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()), "Pre-roped K path output must be finite");
    }
}
