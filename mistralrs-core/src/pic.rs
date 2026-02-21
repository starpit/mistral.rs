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

use candle_core::{DType, Device, Result, Tensor};

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
}

impl PicContext {
    /// Create a new PIC context from a list of blocks.
    ///
    /// Computes position IDs automatically:
    /// - Cross blocks get sequential absolute positions
    /// - Plus blocks get positions starting from 0 within each block
    pub fn new(blocks: Vec<PicBlock>) -> Self {
        let total_len: usize = blocks.iter().map(|b| b.len).sum();
        let mut position_ids = vec![0usize; total_len];
        let mut cross_pos = 0usize;

        for block in &blocks {
            if block.is_plus {
                // Plus blocks: local positions starting from 0
                for i in 0..block.len {
                    position_ids[block.start + i] = i;
                }
            } else {
                // Cross blocks: sequential absolute positions
                for i in 0..block.len {
                    position_ids[block.start + i] = cross_pos;
                    cross_pos += 1;
                }
            }
        }

        Self {
            blocks,
            position_ids,
            total_len,
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
}
