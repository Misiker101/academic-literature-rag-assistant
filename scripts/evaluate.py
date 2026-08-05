"""
Evaluation script — produces the quantifiable numbers,
e.g. "Improved retrieval Hit Rate@8 by 23% and MRR by 0.18 versus a naive
dense-only baseline by adding hybrid (BM25 + dense) search and cross-encoder
re-ranking, evaluated on a N-question benchmark."

HOW TO USE
----------
1. Build the index first: python scripts/build_index.py
2. Create your own eval set at eval/sample_qa.json. For each question, put
   the paper you KNOW contains the answer (expected_paper) — a substring of
   its title is fine. Aim for 20-40 questions across your ~100 PDFs for a
   credible number. The provided file is just a 2-question example.
3. Run: python scripts/evaluate.py
4. Read the printed report (also written to eval/results.json).

METRICS
-------
* Hit Rate@k   : fraction of questions where a chunk from the expected paper
                 appears anywhere in the top-k retrieved results.
* MRR          : Mean Reciprocal Rank of the first correct-paper chunk.
Both are computed for:
  (a) BASELINE : dense-only (Chroma/MiniLM) retrieval, top k, no re-ranking.
  (b) FULL     : this project's pipeline — hybrid (dense+BM25) retrieval of
                 20 candidates, then cross-encoder re-ranking to top k.
The printed "% improvement" is exactly the number to quote on a CV/resume.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import indexing, retrieval  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def _is_hit(doc, expected_paper: str) -> bool:
    title = doc.metadata.get("paper_title", "").lower()
    fname = doc.metadata.get("filename", "").lower()
    hint = expected_paper.lower()
    return hint in title or hint in fname


def evaluate(qa_path: Path, k: int = config.TOP_K_FINAL):
    qa_pairs = json.loads(qa_path.read_text())
    vectorstore = indexing.load_vectorstore()

    baseline_hits, baseline_rr = 0, []
    full_hits, full_rr = 0, []

    for item in qa_pairs:
        question = item["question"]
        expected = item["expected_paper"]

        # --- Baseline: dense-only, no re-ranking ---
        baseline_docs = vectorstore.as_retriever(search_kwargs={"k": k}).invoke(question)
        rank = next((i + 1 for i, d in enumerate(baseline_docs) if _is_hit(d, expected)), None)
        if rank:
            baseline_hits += 1
            baseline_rr.append(1 / rank)
        else:
            baseline_rr.append(0)

        # --- Full pipeline: hybrid + cross-encoder re-rank ---
        full_docs = retrieval.retrieve(question, final_k=k)
        rank = next((i + 1 for i, d in enumerate(full_docs) if _is_hit(d, expected)), None)
        if rank:
            full_hits += 1
            full_rr.append(1 / rank)
        else:
            full_rr.append(0)

    n = len(qa_pairs)
    baseline_hit_rate = baseline_hits / n
    full_hit_rate = full_hits / n
    baseline_mrr = sum(baseline_rr) / n
    full_mrr = sum(full_rr) / n

    hit_rate_improvement = (
        ((full_hit_rate - baseline_hit_rate) / baseline_hit_rate * 100) if baseline_hit_rate > 0 else float("inf")
    )
    mrr_improvement = (
        ((full_mrr - baseline_mrr) / baseline_mrr * 100) if baseline_mrr > 0 else float("inf")
    )

    report = {
        "n_questions": n,
        "k": k,
        "baseline_dense_only": {"hit_rate": round(baseline_hit_rate, 3), "mrr": round(baseline_mrr, 3)},
        "full_hybrid_rerank": {"hit_rate": round(full_hit_rate, 3), "mrr": round(full_mrr, 3)},
        "hit_rate_improvement_pct": round(hit_rate_improvement, 1),
        "mrr_improvement_pct": round(mrr_improvement, 1),
    }

    print(json.dumps(report, indent=2))
    (config.EVAL_DIR / "results.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    qa_file = config.EVAL_DIR / "sample_qa.json"
    evaluate(qa_file)
