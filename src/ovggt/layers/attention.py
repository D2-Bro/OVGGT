import logging
import json
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional

XFORMERS_AVAILABLE = False



class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.num_anchor_tokens = 0
        self.cache_debug_name = "unlabeled"
        self.cache_debug_records = []
        self.cache_debug_step = 0
        self.cache_tsne_frame_ids = None
        self.cache_tsne_token_ids = None
        self.cache_tsne_step = 0

    def _reset_cache_state(self):
        self.num_anchor_tokens = 0
        self.cache_tsne_frame_ids = None
        self.cache_tsne_token_ids = None
        self.cache_tsne_step = 0

    def _cache_tsne_enabled(self) -> bool:
        dump_dir = os.environ.get("OVGGT_CACHE_TSNE_DIR")
        if not dump_dir:
            return False

        layer_filter = os.environ.get("OVGGT_CACHE_TSNE_LAYERS")
        if not layer_filter:
            return True

        layer_name = getattr(self, "cache_debug_name", "unlabeled")
        allowed = [item.strip() for item in layer_filter.split(",") if item.strip()]
        return any(layer_name.startswith(item) for item in allowed)

    def _update_cache_tsne_debug(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        num_new_tokens: int,
        current_frame_idx: Optional[int] = None,
        kept_indices: Optional[torch.Tensor] = None,
        current_token_indices: Optional[torch.Tensor] = None,
    ) -> None:
        if current_frame_idx is None or not self._cache_tsne_enabled():
            return

        B, H, N, D = k.shape
        device = k.device
        num_new_tokens = min(max(int(num_new_tokens), 0), N)

        prev_frame_ids = self.cache_tsne_frame_ids
        prev_token_ids = self.cache_tsne_token_ids
        prev_valid = (
            prev_frame_ids is not None
            and prev_token_ids is not None
            and prev_frame_ids.shape[0] == B
            and prev_token_ids.shape[0] == B
        )

        if prev_valid:
            prev_frame_ids = prev_frame_ids.to(device=device)
            prev_token_ids = prev_token_ids.to(device=device)
        else:
            prev_frame_ids = torch.empty(B, 0, dtype=torch.long, device=device)
            prev_token_ids = torch.empty(B, 0, dtype=torch.long, device=device)

        if current_token_indices is None:
            current_token_ids = torch.arange(num_new_tokens, device=device).unsqueeze(0).expand(B, -1)
        else:
            current_token_ids = current_token_indices.to(device=device, dtype=torch.long)

        current_frame_ids = torch.full(
            (B, num_new_tokens),
            int(current_frame_idx),
            dtype=torch.long,
            device=device,
        )

        pre_frame_ids = torch.cat([prev_frame_ids, current_frame_ids], dim=1)
        pre_token_ids = torch.cat([prev_token_ids, current_token_ids], dim=1)

        if kept_indices is not None:
            if kept_indices.dim() == 3:
                kept_indices = kept_indices[:, 0, :]
            kept_indices = kept_indices.to(device=device, dtype=torch.long)
            frame_ids = torch.gather(pre_frame_ids, 1, kept_indices)
            token_ids = torch.gather(pre_token_ids, 1, kept_indices)
        else:
            frame_ids = pre_frame_ids
            token_ids = pre_token_ids

        if frame_ids.shape[1] != N:
            # Keep the main model path untouched if debug metadata falls out of sync.
            self.cache_tsne_frame_ids = None
            self.cache_tsne_token_ids = None
            return

        self.cache_tsne_frame_ids = frame_ids.detach()
        self.cache_tsne_token_ids = token_ids.detach()
        self._dump_cache_tsne_debug(k)

    def _dump_cache_tsne_debug(self, k: torch.Tensor) -> None:
        dump_dir = os.environ.get("OVGGT_CACHE_TSNE_DIR")
        if not dump_dir:
            return

        every = int(os.environ.get("OVGGT_CACHE_TSNE_EVERY", "1"))
        if every > 1 and self.cache_tsne_step % every != 0:
            self.cache_tsne_step += 1
            return

        layer_name = getattr(self, "cache_debug_name", "unlabeled")
        safe_layer_name = layer_name.replace("/", "_")
        output_dir = os.path.join(dump_dir, safe_layer_name)
        os.makedirs(output_dir, exist_ok=True)

        features = k.transpose(1, 2).reshape(k.shape[0], k.shape[2], -1)
        frame_ids = self.cache_tsne_frame_ids
        token_ids = self.cache_tsne_token_ids

        max_tokens = int(os.environ.get("OVGGT_CACHE_TSNE_MAX_TOKENS", "4000"))
        if max_tokens > 0 and features.shape[1] > max_tokens:
            sample_idx = torch.linspace(
                0,
                features.shape[1] - 1,
                steps=max_tokens,
                device=features.device,
            ).long()
            features = features[:, sample_idx, :]
            frame_ids = frame_ids[:, sample_idx]
            token_ids = token_ids[:, sample_idx]
        else:
            sample_idx = None

        output_path = os.path.join(output_dir, f"step_{self.cache_tsne_step:06d}.pt")
        torch.save(
            {
                "layer": layer_name,
                "step": self.cache_tsne_step,
                "features": features.detach().float().cpu(),
                "frame_ids": frame_ids.detach().cpu(),
                "token_ids": token_ids.detach().cpu(),
                "sample_indices": None if sample_idx is None else sample_idx.detach().cpu(),
            },
            output_path,
        )
        self.cache_tsne_step += 1

    def intra_frame_prune(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        importance: torch.Tensor,
        keep_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prune current frame tokens based on repr_shift importance.

        This is Stage 1 of the two-stage eviction strategy:
        - Intra-frame pruning: prune current frame tokens BEFORE storing to KV cache
        - Only affects what gets stored, not what's used for current layer attention

        Args:
            k (torch.Tensor): Current frame K [B, H, N, D]
            v (torch.Tensor): Current frame V [B, H, N, D]
            importance (torch.Tensor): Importance scores [B, N] from repr_shift
                Higher = more important = KEEP
            keep_count (int): Number of tokens to keep

        Returns:
            (pruned_k, pruned_v, kept_indices)
            pruned_k: [B, H, keep_count, D]
            pruned_v: [B, H, keep_count, D]
            kept_indices: [B, keep_count] indices of kept tokens, or None if no pruning
        """
        B, H, N, D = k.shape

        if keep_count >= N:
            return k, v, None

        # Keep at least 1 token
        keep_count = max(keep_count, 1)

        # Keep top-k by importance (higher = more important)
        _, top_indices = torch.topk(importance, k=keep_count, dim=-1)
        top_indices = top_indices.sort(dim=-1).values  # Maintain temporal order

        # Gather pruned K, V
        expanded_indices = top_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, keep_count, D)
        pruned_k = torch.gather(k, 2, expanded_indices)
        pruned_v = torch.gather(v, 2, expanded_indices)

        return pruned_k, pruned_v, top_indices

    def _history_anchor_layout(self, history_anchor_layout: str) -> str:
        if history_anchor_layout not in ("front", "tail"):
            raise ValueError(f"Unknown history_anchor_layout: {history_anchor_layout}")
        return history_anchor_layout

    def _anchor_split_counts(
        self,
        total_anchor_tokens: int,
        anchor_base_token_count: Optional[int],
        history_anchor_layout: str,
    ) -> Tuple[int, int]:
        total_anchor_tokens = int(total_anchor_tokens)
        if history_anchor_layout == "front":
            return total_anchor_tokens, 0

        base_count = self.num_anchor_tokens if anchor_base_token_count is None else int(anchor_base_token_count)
        base_count = max(min(base_count, total_anchor_tokens), 0)
        history_count = max(total_anchor_tokens - base_count, 0)
        return base_count, history_count

    def _move_current_before_tail_anchors(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        history_anchor_tokens: int,
        num_new_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if history_anchor_tokens <= 0 or num_new_tokens <= 0:
            return k, v, None

        B, _, N, _ = k.shape
        current_start = N - num_new_tokens
        history_start = current_start - history_anchor_tokens
        if history_start < 0:
            return k, v, None

        order = torch.cat(
            [
                torch.arange(0, history_start, device=k.device),
                torch.arange(current_start, N, device=k.device),
                torch.arange(history_start, current_start, device=k.device),
            ],
            dim=0,
        )
        k = k.index_select(2, order)
        v = v.index_select(2, order)
        kept_indices = order.unsqueeze(0).expand(B, -1)
        return k, v, kept_indices

    def _eviction_front(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_budget: int,
        num_anchor_tokens: int,
        importance_scores: torch.Tensor = None,
        num_new_tokens: int = 0,
        importance_weight: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:
        return self.eviction(
            k,
            v,
            cache_budget,
            num_anchor_tokens,
            importance_scores=importance_scores,
            num_new_tokens=num_new_tokens,
            importance_weight=importance_weight,
        )

    def _eviction_tail(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_budget: int,
        global_anchor_tokens: int,
        history_anchor_tokens: int,
        importance_scores: torch.Tensor = None,
        num_new_tokens: int = 0,
        importance_weight: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:
        B, H, N, D = k.shape

        if N <= cache_budget:
            return k, v, None, None

        protected_tokens = global_anchor_tokens + history_anchor_tokens
        candidate_end = N - history_anchor_tokens
        if global_anchor_tokens < 0 or candidate_end < global_anchor_tokens:
            return self.eviction(
                k,
                v,
                cache_budget,
                protected_tokens,
                importance_scores=importance_scores,
                num_new_tokens=num_new_tokens,
                importance_weight=importance_weight,
            )

        global_k = k[:, :, :global_anchor_tokens, :]
        global_v = v[:, :, :global_anchor_tokens, :]
        candidate_k = k[:, :, global_anchor_tokens:candidate_end, :]
        candidate_v = v[:, :, global_anchor_tokens:candidate_end, :]
        history_k = k[:, :, candidate_end:, :]
        history_v = v[:, :, candidate_end:, :]

        num_candidates = candidate_k.shape[2]
        num_to_keep_from_candidates = min(max(cache_budget - protected_tokens, 0), num_candidates)
        num_old_candidates = max(num_candidates - num_new_tokens, 0)

        if num_to_keep_from_candidates >= num_candidates:
            return k, v, None, None

        protected_indices = torch.cat(
            [
                torch.arange(0, global_anchor_tokens, device=k.device),
                torch.arange(candidate_end, N, device=k.device),
            ],
            dim=0,
        ).unsqueeze(0).expand(B, -1)

        if num_to_keep_from_candidates <= 0:
            final_k = torch.cat([global_k, history_k], dim=2)
            final_v = torch.cat([global_v, history_v], dim=2)
            return final_k, final_v, None, protected_indices

        candidate_k_norm = F.normalize(candidate_k, p=2, dim=-1)
        mean_vector = torch.mean(candidate_k_norm, dim=2, keepdim=True)
        baseline_scores = torch.sum(candidate_k_norm * mean_vector, dim=-1)
        baseline_diversity = 1.0 - baseline_scores
        baseline_diversity_avg = baseline_diversity.mean(dim=1)

        use_hybrid = (
            importance_scores is not None
            and importance_scores.shape[1] == num_new_tokens
            and num_new_tokens > 0
            and num_old_candidates > 0
        )

        if use_hybrid:
            old_scores = baseline_diversity_avg[:, :num_old_candidates]
            new_importance = importance_scores

            old_min, old_max = old_scores.min(dim=-1, keepdim=True)[0], old_scores.max(dim=-1, keepdim=True)[0]
            old_normalized = (old_scores - old_min) / (old_max - old_min + 1e-8)

            new_min, new_max = new_importance.min(dim=-1, keepdim=True)[0], new_importance.max(dim=-1, keepdim=True)[0]
            new_normalized = (new_importance - new_min) / (new_max - new_min + 1e-8)

            weighted_old = (1.0 - importance_weight) * old_normalized
            weighted_new = importance_weight * new_normalized
            combined_scores = torch.cat([weighted_old, weighted_new], dim=-1)
            avg_scores = combined_scores.mean().item()

            _, top_indices = torch.topk(combined_scores, k=num_to_keep_from_candidates, dim=-1)
            top_indices_sorted = top_indices.sort(dim=-1).values
            kept_current_counts = (top_indices >= num_old_candidates).sum(dim=-1)
            evicted_current_counts = num_new_tokens - kept_current_counts
            debug_name = getattr(self, "cache_debug_name", "unlabeled")
            debug_record = {
                "layer": debug_name,
                "step": self.cache_debug_step,
                "evicted_current": evicted_current_counts.detach().cpu().tolist(),
                "kept_current": kept_current_counts.detach().cpu().tolist(),
                "total_current": num_new_tokens,
                "cache_budget": cache_budget,
                "num_anchor_tokens": protected_tokens,
                "num_old_candidates": num_old_candidates,
                "history_anchor_layout": "tail",
            }
            self.cache_debug_step += 1
            self.cache_debug_records.append(debug_record)
            eviction_log_path = os.environ.get("OVGGT_EVICTION_LOG_PATH", "eviction_debug.jsonl")
            with open(eviction_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(debug_record) + "\n")
            print(
                f"[eviction][{debug_name}] current-frame evicted tokens per batch:",
                debug_record["evicted_current"],
                f"(kept={debug_record['kept_current']}, total_current={num_new_tokens})",
            )

            expanded_indices = top_indices_sorted.unsqueeze(1).unsqueeze(-1).expand(B, H, num_to_keep_from_candidates, D)
            kept_candidate_indices = top_indices_sorted + global_anchor_tokens
        else:
            avg_scores = baseline_scores.mean().item()
            _, top_indices = torch.topk(-baseline_scores, k=num_to_keep_from_candidates, dim=-1)
            top_indices_sorted = top_indices.sort(dim=-1).values
            expanded_indices = top_indices_sorted.unsqueeze(-1).expand(B, H, num_to_keep_from_candidates, D)
            kept_candidate_indices = top_indices_sorted[:, 0, :] + global_anchor_tokens

        kept_candidate_k = torch.gather(candidate_k, 2, expanded_indices)
        kept_candidate_v = torch.gather(candidate_v, 2, expanded_indices)

        final_k = torch.cat([global_k, kept_candidate_k, history_k], dim=2)
        final_v = torch.cat([global_v, kept_candidate_v, history_v], dim=2)
        kept_indices = torch.cat(
            [
                torch.arange(0, global_anchor_tokens, device=k.device).unsqueeze(0).expand(B, -1),
                kept_candidate_indices,
                torch.arange(candidate_end, N, device=k.device).unsqueeze(0).expand(B, -1),
            ],
            dim=-1,
        )

        return final_k, final_v, avg_scores, kept_indices

    def evict_with_layout(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_budget: int,
        num_anchor_tokens: int,
        importance_scores: torch.Tensor = None,
        num_new_tokens: int = 0,
        importance_weight: float = 0.5,
        history_anchor_layout: str = "front",
        anchor_base_token_count: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:
        history_anchor_layout = self._history_anchor_layout(history_anchor_layout)
        if history_anchor_layout == "front":
            return self._eviction_front(
                k,
                v,
                cache_budget,
                num_anchor_tokens,
                importance_scores=importance_scores,
                num_new_tokens=num_new_tokens,
                importance_weight=importance_weight,
            )

        global_anchor_tokens, history_anchor_tokens = self._anchor_split_counts(
            num_anchor_tokens,
            anchor_base_token_count,
            history_anchor_layout,
        )
        return self._eviction_tail(
            k,
            v,
            cache_budget,
            global_anchor_tokens,
            history_anchor_tokens,
            importance_scores=importance_scores,
            num_new_tokens=num_new_tokens,
            importance_weight=importance_weight,
        )

    def eviction(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_budget: int,
        num_anchor_tokens: int,
        importance_scores: torch.Tensor = None,
        num_new_tokens: int = 0,
        importance_weight: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:
        """
        Evicts tokens from the key-value cache using hybrid scoring strategy.

        Hybrid strategy:
        - Old tokens (from past frames): use baseline (cosine diversity) weighted by (1 - importance_weight)
        - New tokens (from current frame): use repr_shift importance weighted by importance_weight

        Args:
            k (torch.Tensor): The key tensor of shape [B, H, N, D].
            v (torch.Tensor): The value tensor of shape [B, H, N, D].
            cache_budget (int): The maximum number of tokens to retain.
            num_anchor_tokens (int): The number of initial tokens to preserve.
            importance_scores (torch.Tensor): Optional [B, N_new] tensor for new tokens.
                Higher = more important = KEEP.
            num_new_tokens (int): Number of new tokens in the current frame.
            importance_weight (float): Weight for current frame importance scores (default=0.5).
                New token scores = importance_weight * normalized_importance
                Old token scores = (1 - importance_weight) * normalized_diversity

        Returns:
            A tuple of (pruned_k, pruned_v, avg_scores, kept_indices).
            kept_indices: [B, cache_budget] global indices of kept tokens, or None if no eviction.
        """
        B, H, N, D = k.shape

        if N <= cache_budget:
            return k, v, None, None

        anchor_k, candidate_k = k.split([num_anchor_tokens, N - num_anchor_tokens], dim=2)
        anchor_v, candidate_v = v.split([num_anchor_tokens, N - num_anchor_tokens], dim=2)

        num_candidates = N - num_anchor_tokens
        num_to_keep_from_candidates = min(max(cache_budget - num_anchor_tokens, 0), num_candidates)
        num_old_candidates = max(num_candidates - num_new_tokens, 0)

        # Edge case: if we can keep all candidates
        if num_to_keep_from_candidates >= num_candidates:
            return k, v, None, None
        
        # Edge case: budget is too small (occupied by anchors), keep only anchors
        if num_to_keep_from_candidates <= 0:
            anchor_indices = torch.arange(num_anchor_tokens, device=k.device).unsqueeze(0).expand(B, -1)
            return anchor_k, anchor_v, None, anchor_indices

        # Compute baseline scores (cosine diversity) for ALL candidates
        candidate_k_norm = F.normalize(candidate_k, p=2, dim=-1)
        mean_vector = torch.mean(candidate_k_norm, dim=2, keepdim=True)
        baseline_scores = torch.sum(candidate_k_norm * mean_vector, dim=-1)  # [B, H, N_cand]
        # Convert to diversity: lower similarity = higher diversity = keep
        baseline_diversity = 1.0 - baseline_scores  # [B, H, N_cand]
        # Average across heads for unified scoring
        baseline_diversity_avg = baseline_diversity.mean(dim=1)  # [B, N_cand]

        # Check if we can use hybrid scoring
        use_hybrid = (
            importance_scores is not None
            and importance_scores.shape[1] == num_new_tokens
            and num_new_tokens > 0
            and num_old_candidates > 0
        )

        if use_hybrid:
            # Hybrid strategy: baseline for old, repr_shift for new
            old_scores = baseline_diversity_avg[:, :num_old_candidates]  # [B, N_old]
            new_importance = importance_scores  # [B, N_new]

            # Normalize both to [0, 1] range for fair comparison
            old_min, old_max = old_scores.min(dim=-1, keepdim=True)[0], old_scores.max(dim=-1, keepdim=True)[0]
            old_normalized = (old_scores - old_min) / (old_max - old_min + 1e-8)

            new_min, new_max = new_importance.min(dim=-1, keepdim=True)[0], new_importance.max(dim=-1, keepdim=True)[0]
            new_normalized = (new_importance - new_min) / (new_max - new_min + 1e-8)

            # Apply importance_weight: current frame weighted by importance_weight, past frames by (1 - importance_weight)
            weighted_old = (1.0 - importance_weight) * old_normalized
            weighted_new = importance_weight * new_normalized

            # Combine: [B, N_cand]
            combined_scores = torch.cat([weighted_old, weighted_new], dim=-1)
            avg_scores = combined_scores.mean().item()

            # Keep tokens with HIGHEST combined score
            _, top_indices = torch.topk(combined_scores, k=num_to_keep_from_candidates, dim=-1)
            top_indices_sorted = top_indices.sort(dim=-1).values  # Maintain temporal order
            kept_current_counts = (top_indices >= num_old_candidates).sum(dim=-1)
            evicted_current_counts = num_new_tokens - kept_current_counts
            debug_name = getattr(self, "cache_debug_name", "unlabeled")
            debug_record = {
                "layer": debug_name,
                "step": self.cache_debug_step,
                "evicted_current": evicted_current_counts.detach().cpu().tolist(),
                "kept_current": kept_current_counts.detach().cpu().tolist(),
                "total_current": num_new_tokens,
                "cache_budget": cache_budget,
                "num_anchor_tokens": num_anchor_tokens,
                "num_old_candidates": num_old_candidates,
            }
            self.cache_debug_step += 1
            self.cache_debug_records.append(debug_record)
            eviction_log_path = os.environ.get("OVGGT_EVICTION_LOG_PATH", "eviction_debug.jsonl")
            with open(eviction_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(debug_record) + "\n")
            print(
                f"[eviction][{debug_name}] current-frame evicted tokens per batch:",
                debug_record["evicted_current"],
                f"(kept={debug_record['kept_current']}, total_current={num_new_tokens})",
            )

            # Expand for gather across heads: [B, H, num_to_keep, D]
            expanded_indices = top_indices_sorted.unsqueeze(1).unsqueeze(-1).expand(B, H, num_to_keep_from_candidates, D)

            # Build kept_indices
            anchor_indices = torch.arange(num_anchor_tokens, device=k.device).unsqueeze(0).expand(B, -1)
            kept_candidate_indices = top_indices_sorted + num_anchor_tokens
            kept_indices = torch.cat([anchor_indices, kept_candidate_indices], dim=-1)
        else:
            # Fallback: pure baseline (cosine diversity)
            avg_scores = baseline_scores.mean().item()

            # Keep tokens with LOWEST similarity (most diverse)
            _, top_indices = torch.topk(-baseline_scores, k=num_to_keep_from_candidates, dim=-1)
            top_indices_sorted = top_indices.sort(dim=-1).values

            expanded_indices = top_indices_sorted.unsqueeze(-1).expand(B, H, num_to_keep_from_candidates, D)

            # Build kept_indices (use first head's indices)
            anchor_indices = torch.arange(num_anchor_tokens, device=k.device).unsqueeze(0).expand(B, -1)
            kept_candidate_indices = top_indices_sorted[:, 0, :] + num_anchor_tokens
            kept_indices = torch.cat([anchor_indices, kept_candidate_indices], dim=-1)

        kept_candidate_k = torch.gather(candidate_k, 2, expanded_indices)
        kept_candidate_v = torch.gather(candidate_v, 2, expanded_indices)

        # randomize order of kept candidates, but keep anchors in front
        # random_order = torch.randperm(kept_candidate_k.shape[2], device=k.device)
        # # print(random_order)
        # kept_candidate_k = kept_candidate_k[:, :, random_order, :]
        # kept_candidate_v = kept_candidate_v[:, :, random_order, :]

        final_k = torch.cat([anchor_k, kept_candidate_k], dim=2)
        final_v = torch.cat([anchor_v, kept_candidate_v], dim=2)

        return final_k, final_v, avg_scores, kept_indices

    def forward(self,
        x: torch.Tensor,
        pos=None,
        attn_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_budget=2000,
        importance_scores=None,
        intra_frame_keep_ratio=1.0,
        defer_eviction=False,
        anchor_token_count: int = None,
        anchor_base_token_count: int = None,
        history_anchor_layout: str = "front",
        importance_weight: float = 0.5,
        cache_debug_current_frame_idx: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        """
        Forward pass with optional two-stage eviction support.

        Args:
            intra_frame_keep_ratio: Ratio of current frame tokens to keep (1.0 = keep all)
            defer_eviction: If True, skip inter-frame eviction here (handled by Block)

        When defer_eviction=True, returns additional info for Block to handle eviction:
            (output, (k_full, v_full, k_current, v_current, past_kv_or_none), scores)
        """
        B, N, C = x.shape
        num_new_tokens = N  # Number of new tokens from current frame
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = None
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if use_cache and self.num_anchor_tokens == 0:
            self.num_anchor_tokens = k.shape[2]

        k_current, v_current = k.clone(), v.clone() 
        past_kv_for_block = None

        if use_cache:
            if past_key_values is not None:
                past_k, past_v = past_key_values
                past_kv_for_block = (past_k, past_v)
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            kept_indices = None
            effective_anchor_count = anchor_token_count if anchor_token_count is not None else self.num_anchor_tokens
            history_anchor_layout = self._history_anchor_layout(history_anchor_layout)
            if history_anchor_layout == "tail":
                _, history_anchor_tokens = self._anchor_split_counts(
                    effective_anchor_count,
                    anchor_base_token_count,
                    history_anchor_layout,
                )
                k, v, kept_indices = self._move_current_before_tail_anchors(
                    k,
                    v,
                    history_anchor_tokens,
                    num_new_tokens,
                )
            if not defer_eviction:
                # Legacy path: apply eviction in attention
                if cache_budget is not None and k.shape[2] > cache_budget:
                    # Use anchor_token_count if provided (History Anchor), otherwise fall back to num_anchor_tokens
                    k, v, scores, eviction_kept_indices = self.evict_with_layout(
                        k, v, cache_budget, effective_anchor_count,
                        importance_scores=importance_scores,
                        num_new_tokens=num_new_tokens,
                        importance_weight=importance_weight,
                        history_anchor_layout=history_anchor_layout,
                        anchor_base_token_count=anchor_base_token_count,
                    )
                    if kept_indices is not None and eviction_kept_indices is not None:
                        kept_indices = torch.gather(kept_indices, 1, eviction_kept_indices)
                    elif eviction_kept_indices is not None:
                        kept_indices = eviction_kept_indices
                self._update_cache_tsne_debug(
                    k,
                    v,
                    num_new_tokens=num_new_tokens,
                    current_frame_idx=cache_debug_current_frame_idx,
                    kept_indices=kept_indices,
                )
                new_kv = (k, v, kept_indices)
            else:
                # Two-stage path: return info for Block to handle eviction
                # Block will apply: 1) intra-frame pruning, 2) inter-frame eviction
                new_kv = (k, v, k_current, v_current, past_kv_for_block)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # Mask
            if attn_mask is not None:
                assert attn_mask.shape[-2:] == (N, N), f"Expected mask shape [..., {N}, {N}], got {attn_mask.shape}"
                attn = attn + attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = attn @ v

        # if use_cache:
        #     if not defer_eviction:
        #         # Legacy path: apply eviction in attention
        #         if cache_budget is not None and k.shape[2] > cache_budget:
        #             # Use anchor_token_count if provided (History Anchor), otherwise fall back to num_anchor_tokens
        #             effective_anchor_count = anchor_token_count if anchor_token_count is not None else self.num_anchor_tokens
        #             k, v, scores, kept_indices = self.eviction(
        #                 k, v, cache_budget, effective_anchor_count,
        #                 importance_scores=importance_scores,
        #                 num_new_tokens=num_new_tokens,
        #                 importance_weight=importance_weight,
        #             )
        #         new_kv = (k, v, kept_indices)
        #     else:
        #         # Two-stage path: return info for Block to handle eviction
        #         # Block will apply: 1) intra-frame pruning, 2) inter-frame eviction
        #         new_kv = (k, v, k_current, v_current, past_kv_for_block)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if use_cache:
                return x, new_kv, scores
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
