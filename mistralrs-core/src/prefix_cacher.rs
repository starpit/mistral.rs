use std::hash::{DefaultHasher, Hash, Hasher};
use candle_core::{Device, Result};
use indexmap::IndexMap;
use itertools::Itertools;
use tracing::info;

use crate::{pipeline::KvCache, sequence::Sequence};

/// Result of a PIC block cache lookup: (raw KV cache, optional pre-RoPE'd K per layer).
pub type PicBlockSearchResult = (Vec<Option<KvCache>>, Option<Vec<Option<candle_core::Tensor>>>);

#[derive(PartialEq, Eq, Debug, Hash)]
struct Tokens(Vec<u32>);

impl Tokens {
    /// Returns the length of the common prefix shared with `other`.
    fn shared_prefix_len(&self, other: &Self) -> usize {
        self.0
            .iter()
            .zip(other.0.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }
}

impl From<Vec<u32>> for Tokens {
    fn from(value: Vec<u32>) -> Self {
        Self(value)
    }
}

#[derive(Clone)]
struct CacheElement {
    cache: Vec<Option<KvCache>>,
    audio_hashes: Option<Vec<u64>>,
    image_hashes: Option<Vec<u64>>,
}

/// PIC block cache entry storing raw KV cache plus optional pre-RoPE'd K.
#[derive(Clone)]
struct PicCacheElement {
    cache: Vec<Option<KvCache>>,
    /// Pre-RoPE'd K tensors (one per layer). When present, the K tensors
    /// have RoPE pre-applied for positions 0..block_len, allowing the
    /// forward pass to skip re-RoPE-ing cached Plus tokens.
    roped_k: Option<Vec<Option<candle_core::Tensor>>>,
}

pub struct PrefixCacheManagerV2 {
    caches: IndexMap<Tokens, CacheElement>,
    /// PIC block cache keyed by text content hash (position-independent).
    pic_blocks: IndexMap<u64, PicCacheElement>,
    n_on_device: usize,
    no_prefix_cache: bool,
    has_paged_attention: bool,
}

#[derive(Clone)]
pub enum MatchingCache {
    Normal {
        normal: Vec<Option<KvCache>>,
        images_to_keep: usize,
        audios_to_keep: usize,
        toks: Vec<u32>,
        offset: usize,
    },
}

impl PrefixCacheManagerV2 {
    pub fn new(n_on_device: usize, no_prefix_cache: bool, has_paged_attention: bool) -> Self {
        if !no_prefix_cache && !has_paged_attention {
            info!("Prefix caching enabled (sequence-level, non-paged attention). Expect higher multi-turn throughput for both text and multimodal.");
        }
        PrefixCacheManagerV2 {
            caches: IndexMap::new(),
            pic_blocks: IndexMap::new(),
            n_on_device,
            no_prefix_cache,
            has_paged_attention,
        }
    }

    /// This always keeps the cache on the device.
    pub fn add_sequence(&mut self, seq: &mut Sequence) {
        // Do not cache if prefix caching disabled
        if self.no_prefix_cache {
            return;
        }

        // For paged attention, prefix caching is handled by the KVCacheManager.
        // PrefixCacheManagerV2 only handles non-paged attention caching.
        if !self.has_paged_attention {
            let cache = seq.normal_cache().to_vec();

            self.caches.insert(
                seq.get_toks().to_vec().into(),
                CacheElement {
                    cache,
                    image_hashes: seq.image_hashes().map(|x| x.to_vec()),
                    audio_hashes: seq.audio_hashes().map(|x| x.to_vec()),
                },
            );
        }
    }

    /// Evict the caches. This will evict the first k seqs such that the number of sequences on device after the copy is
    /// the maximum allowed. Returns the number of evicted sequences.
    pub fn evict_caches(&mut self) -> Result<usize> {
        if self.no_prefix_cache {
            return Ok(0);
        }
        let mut n_on_device = 0;
        for cache in self.caches.values() {
            let first_non_none = cache.cache.iter().find_or_first(|x| x.is_some());
            let Some(Some(first_non_none)) = first_non_none else {
                continue;
            };

            let cache_device = match first_non_none {
                KvCache::Normal { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
                KvCache::Rotating { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
            };

            if !matches!(cache_device, Device::Cpu) {
                n_on_device += 1;
            }
        }
        let mut n_evicted = 0;
        // Intentionally evict the first ones first, as they are the oldest
        for cache in self.caches.values_mut() {
            if n_on_device - n_evicted <= self.n_on_device {
                break;
            }
            let first_non_none = cache.cache.iter().find_or_first(|x| x.is_some());
            let Some(Some(first_non_none)) = first_non_none else {
                continue;
            };

            let cache_device = match first_non_none {
                KvCache::Normal { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
                KvCache::Rotating { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
            };

            if !matches!(cache_device, Device::Cpu) {
                cache.cache.clear();
                n_evicted += 1;
            }
        }

        self.caches.retain(|_tokens, cache| !cache.cache.is_empty());

        Ok(n_evicted)
    }

    /// Evict all the caches.
    pub fn evict_all_caches(&mut self) -> Result<usize> {
        let len = self.caches.len();
        self.caches.clear();
        Ok(len)
    }

    /// Search for a matching cache given some tokens. Image-containing sequences are now cached too.
    pub fn search_for_matching_cache(
        &mut self,
        toks: &[u32],
        image_hashes: Option<&[u64]>,
        audio_hashes: Option<&[u64]>,
    ) -> Result<Option<MatchingCache>> {
        // Do not search if prefix caching disabled or no tokens
        if self.no_prefix_cache || toks.is_empty() {
            return Ok(None);
        }

        if self.has_paged_attention {
            // For paged attention, prefix caching is handled by the KVCacheManager.
            // PrefixCacheManagerV2 only handles non-paged attention caching.
            return Ok(None);
        }

        let toks = Tokens(toks.to_vec());

        let mut best_match: Option<(usize, &CacheElement, usize, usize)> = None;
        for (k, v) in &self.caches {
            let match_len = toks.shared_prefix_len(k);
            if match_len == 0 {
                continue;
            }

            let images_match_until = match image_hashes {
                Some(input_hashes) => match &v.image_hashes {
                    Some(cached_hashes) => input_hashes
                        .iter()
                        .zip(cached_hashes)
                        .take_while(|(a, b)| a == b)
                        .count(),
                    None => 0,
                },
                None => 0,
            };

            let audios_match_until = match audio_hashes {
                Some(input_hashes) => match &v.audio_hashes {
                    Some(cached_hashes) => input_hashes
                        .iter()
                        .zip(cached_hashes)
                        .take_while(|(a, b)| a == b)
                        .count(),
                    None => 0,
                },
                None => 0,
            };

            // Vision/audio models can use repeated placeholder tokens (e.g. <image_soft_token>)
            // that are identical regardless of the actual image/audio content. This means
            // token-level matching can extend through image/audio regions even when the
            // underlying content differs, leaving stale vision/audio encoder outputs in the
            // KV cache. Skip entries where any overlapping images or audios diverge.
            let cached_image_count = v.image_hashes.as_ref().map_or(0, |h| h.len());
            let input_image_count = image_hashes.map_or(0, |h| h.len());
            if images_match_until < input_image_count.min(cached_image_count) {
                continue;
            }

            let cached_audio_count = v.audio_hashes.as_ref().map_or(0, |h| h.len());
            let input_audio_count = audio_hashes.map_or(0, |h| h.len());
            if audios_match_until < input_audio_count.min(cached_audio_count) {
                continue;
            }

            if best_match
                .as_ref()
                .is_none_or(|(len, _, _, _)| match_len > *len)
            {
                best_match = Some((match_len, v, images_match_until, audios_match_until));
            }
        }

        if let Some((match_len, cache_element, images_match_until, audios_match_until)) = best_match
        {
            let new_toks = toks.0[match_len..].to_vec();
            if new_toks.is_empty() {
                return Ok(None);
            }

            let mut cache = cache_element.clone();
            // Count how many input images are not already cached
            let images_to_keep = if let Some(input_hashes) = image_hashes {
                input_hashes.len().saturating_sub(images_match_until)
            } else {
                0
            };
            let audios_to_keep = if let Some(input_hashes) = audio_hashes {
                input_hashes.len().saturating_sub(audios_match_until)
            } else {
                0
            };
            for layer in cache.cache.iter_mut().flatten() {
                if layer.try_set_len(match_len).is_err() {
                    return Ok(None);
                }
            }
            for layer in cache.cache.iter_mut().flatten() {
                layer.set_len(match_len)?;
            }
            return Ok(Some(MatchingCache::Normal {
                normal: cache.cache,
                images_to_keep,
                audios_to_keep,
                toks: new_toks,
                offset: match_len,
            }));
        }

        Ok(None)
    }

    /// Search for a cached PIC (Plus) block by text content hash.
    ///
    /// Unlike `search_for_matching_cache` which matches by prefix position,
    /// this matches by text content hash alone, enabling position-independent reuse.
    /// The hash should be computed via `pic::content_hash_text()` from the original
    /// message text — NOT from token IDs (which are position-dependent due to BPE).
    ///
    /// Returns `(cache, roped_k)` where `roped_k` contains pre-RoPE'd K tensors
    /// if available (one per layer).
    pub fn search_for_pic_block(
        &self,
        _block_tokens: &[u32],
        content_hash: u64,
    ) -> Result<Option<PicBlockSearchResult>> {
        if self.no_prefix_cache {
            return Ok(None);
        }

        if let Some(entry) = self.pic_blocks.get(&content_hash) {
            return Ok(Some((entry.cache.clone(), entry.roped_k.clone())));
        }

        Ok(None)
    }

    /// Add a PIC block's KV cache keyed by text content hash.
    ///
    /// The `content_hash` should be computed via `pic::content_hash_text()` from
    /// the original message text. `roped_k` contains optional pre-RoPE'd K tensors
    /// (one per layer) for positions 0..block_len.
    pub fn add_pic_block(
        &mut self,
        content_hash: u64,
        cache: Vec<Option<KvCache>>,
        roped_k: Option<Vec<Option<candle_core::Tensor>>>,
    ) {
        if self.no_prefix_cache {
            return;
        }

        self.pic_blocks.insert(
            content_hash,
            PicCacheElement { cache, roped_k },
        );
    }
}
