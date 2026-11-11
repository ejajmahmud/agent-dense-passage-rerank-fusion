"""CLI for agent-dense-passage-rerank-fusion."""
import sys
import argparse
from .fusion import ReciprocalRankFusion

def main():
    parser = argparse.ArgumentParser(description="Agent Dense Passage Rerank Fusion CLI")
    parser.add_argument("--demo", action="store_true", help="Demonstrate multi-channel RRF fusion")
    args = parser.parse_args()

    if args.demo:
        rrf = ReciprocalRankFusion(k_smoothing=60)
        channels = {
            "bm25": ["doc_sql_err", "doc_db_intro", "doc_redis"],
            "dense": ["doc_db_intro", "doc_sql_err", "doc_postgres"]
        }
        res = rrf.fuse(channels)
        print("Fused Ranked Results:")
        for r in res:
            print(f"  - {r.doc_id}: score={r.fused_score} (ranks: {r.channel_ranks})")
    else:
        parser.print_help()
    sys.exit(0)

if __name__ == "__main__":
    main()
