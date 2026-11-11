"""Unit tests for agent-dense-passage-rerank-fusion."""
import unittest
from agent_dense_passage_rerank_fusion.fusion import ReciprocalRankFusion

class TestReciprocalRankFusion(unittest.TestCase):
    def setUp(self):
        self.rrf = ReciprocalRankFusion(k_smoothing=60)

    def test_consensus_boost(self):
        # docA is rank 1 in both channels, docB is only in channel 1
        channels = {
            "ch1": ["docA", "docB"],
            "ch2": ["docA", "docC"]
        }
        res = self.rrf.fuse(channels)
        self.assertEqual(res[0].doc_id, "docA")
        self.assertGreater(res[0].fused_score, res[1].fused_score)
