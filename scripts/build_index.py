"""
Run this once (and again whenever you add/remove PDFs) to (re)build the
Chroma vector index and BM25 keyword index.

Usage:
    python scripts/build_index.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import ingestion, indexing  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Reading PDFs from %s", config.PDF_DIR)
    chunks = ingestion.load_and_chunk_pdfs(config.PDF_DIR)
    if not chunks:
        logger.error(
            "No usable chunks found. Put PDFs in %s and check %s for skipped files.",
            config.PDF_DIR,
            config.SKIPPED_LOG_PATH,
        )
        sys.exit(1)
    indexing.build_all_indexes(chunks)
    logger.info("Done. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
