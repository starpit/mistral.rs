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

    #[test]
    fn test_pic_context_positions() {
        let blocks = vec![
            PicBlock {
                start: 0,
                len: 5,
                is_plus: false,
                content_hash: None,
            },
            PicBlock {
                start: 5,
                len: 3,
                is_plus: true,
                content_hash: Some(123),
            },
            PicBlock {
                start: 8,
                len: 2,
                is_plus: false,
                content_hash: None,
            },
        ];

        let ctx = PicContext::new(blocks);

        // Cross block: positions 0..5
        assert_eq!(ctx.position_ids[0], 0);
        assert_eq!(ctx.position_ids[4], 4);

        // Plus block: local positions 0..3
        assert_eq!(ctx.position_ids[5], 0);
        assert_eq!(ctx.position_ids[6], 1);
        assert_eq!(ctx.position_ids[7], 2);

        // Second cross block: continues from 5
        assert_eq!(ctx.position_ids[8], 5);
        assert_eq!(ctx.position_ids[9], 6);
    }

    #[test]
    fn test_pic_context_is_plus() {
        let blocks = vec![
            PicBlock {
                start: 0,
                len: 3,
                is_plus: false,
                content_hash: None,
            },
            PicBlock {
                start: 3,
                len: 4,
                is_plus: true,
                content_hash: Some(42),
            },
        ];

        let ctx = PicContext::new(blocks);

        assert!(!ctx.is_plus_token(0));
        assert!(!ctx.is_plus_token(2));
        assert!(ctx.is_plus_token(3));
        assert!(ctx.is_plus_token(6));
    }

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
    fn test_detect_pic_blocks_no_sentinels() {
        // No sentinel tokens -> None
        assert!(detect_pic_blocks(&[1, 2, 3, 4], 100, 101).is_none());
    }

    #[test]
    fn test_detect_pic_blocks_simple() {
        // cross_token(101), 10, 11, plus_token(100), 20, 21, 22, cross_token(101), 30
        let tokens = vec![101, 10, 11, 100, 20, 21, 22, 101, 30];
        let blocks = detect_pic_blocks(&tokens, 100, 101).unwrap();

        assert_eq!(blocks.len(), 3);

        // First cross block: tokens [10, 11] at positions 1..3
        assert_eq!(blocks[0].start, 1);
        assert_eq!(blocks[0].len, 2);
        assert!(!blocks[0].is_plus);
        assert!(blocks[0].content_hash.is_none());

        // Plus block: tokens [20, 21, 22] at positions 4..7
        assert_eq!(blocks[1].start, 4);
        assert_eq!(blocks[1].len, 3);
        assert!(blocks[1].is_plus);
        assert!(blocks[1].content_hash.is_some());

        // Second cross block: token [30] at position 8
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

        // Same content should produce the same hash
        assert_eq!(blocks_a[0].content_hash, blocks_b[0].content_hash);
    }
}
