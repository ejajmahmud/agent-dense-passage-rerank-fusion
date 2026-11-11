"""Core Reciprocal Rank Fusion implementation."""
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ScoredPassage:
    doc_id: str
    fused_score: float
    channel_ranks: Dict[str, int]

class ReciprocalRankFusion:
    """Blends ranked lists from multiple search channels using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, k_smoothing: int = 60):
        self.k_smoothing = k_smoothing

    def fuse(self, ranked_channels: Dict[str, List[str]], channel_weights: Dict[str, float] = None) -> List[ScoredPassage]:
        weights = channel_weights or {ch: 1.0 for ch in ranked_channels.keys()}
        scores: Dict[str, float] = {}
        ranks: Dict[str, Dict[str, int]] = {}

        for ch_name, doc_list in ranked_channels.items():
            w = weights.get(ch_name, 1.0)
            for rank_idx, doc_id in enumerate(doc_list, start=1):
                # RRF formula: score += weight / (k + rank)
                rrf_contrib = w / (self.k_smoothing + rank_idx)
                scores[doc_id] = scores.get(doc_id, 0.0) + rrf_contrib
                if doc_id not in ranks:
                    ranks[doc_id] = {}
                ranks[doc_id][ch_name] = rank_idx

        # Sort by accumulated score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            ScoredPassage(
                doc_id=doc_id,
                fused_score=round(score, 5),
                channel_ranks=ranks[doc_id]
            )
            for doc_id, score in sorted_docs
        ]
