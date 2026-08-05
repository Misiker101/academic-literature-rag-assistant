"""
Run with:  streamlit run app.py
"""
import io
import logging
from pathlib import Path

import fitz  
import streamlit as st
from PIL import Image

import config
from src.pipeline import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="Academic Literature RAG", layout="wide")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, citations, query_type}
if "preview" not in st.session_state:
    st.session_state.preview = None  # {source, page, snippet, paper_title}


def set_preview(citation: dict):
    st.session_state.preview = citation



# PDF page rendering helper
@st.cache_data(show_spinner=False)
def render_pdf_page(source_path: str, page_number: int, snippet: str = "") -> bytes:
    """Render a single PDF page to a PNG, highlighting the cited snippet
    (a "hover-to-see" style preview: the exact passage is boxed in yellow)."""
    doc = fitz.open(source_path)
    page = doc[page_number - 1]

    if snippet:
        # Search for a short, distinctive slice of the snippet (full snippets
        # rarely match exactly due to whitespace/line-break differences).
        needle = " ".join(snippet.split())[:80]
        for rect in page.search_for(needle):
            highlight = page.add_highlight_annot(rect)
            highlight.set_colors(stroke=(1, 0.85, 0))
            highlight.update()

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes



# Layout
st.title("📚 Academic Literature RAG Assistant")
st.caption(
    "Ask questions across your entire related-work / bibliography folder. "
    "Click any citation tag to preview the exact page it came from."
)

if not config.GOOGLE_API_KEY:
    st.warning(
        "GOOGLE_API_KEY is not set. Add it to a `.env` file (see `.env.example`) "
        "before asking questions.",
        icon="⚠️",
    )

if not config.CHROMA_DIR.exists() or not any(config.CHROMA_DIR.iterdir() if config.CHROMA_DIR.exists() else []):
    st.info(
        "No index found yet. Add PDFs to `data/pdfs/` and run "
        "`python scripts/build_index.py` from a terminal, then reload this page.",
        icon="ℹ️",
    )

left, right = st.columns([3, 2], gap="large")

with left:
    chat_container = st.container(height=560)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    st.caption("Sources:")
                    cols = st.columns(min(len(msg["citations"]), 4) or 1)
                    for i, cit in enumerate(msg["citations"]):
                        label = f"[{cit['paper_title']}, p.{cit['page']}]"
                        with cols[i % len(cols)]:
                            if st.button(label, key=f"cite_{id(msg)}_{i}"):
                                set_preview(cit)

    question = st.chat_input("Ask about your papers (e.g. 'What dataset did the ViT paper use?')")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Retrieving & reasoning over your papers..."):
            try:
                result = answer_question(question)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "citations": result.citations,
                        "query_type": getattr(result, "query_type", None),
                    }
                )
                if result.citations:
                    set_preview(result.citations[0])
            except Exception as e:  # noqa: BLE001
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ Error: {e}", "citations": []}
                )
        st.rerun()

with right:
    st.subheader("📄 Source preview")
    preview = st.session_state.preview
    if preview is None:
        st.info("Click a citation tag on the left to preview the source page here.")
    else:
        st.markdown(f"**{preview['paper_title']}** — page {preview['page']}")
        source_path = preview.get("source")
        try:
            img_bytes = render_pdf_page(source_path, int(preview["page"]), preview.get("snippet", ""))
            st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not render page: {e}")
        with st.expander("Cited snippet (text)"):
            st.write(preview.get("snippet", ""))


# Sidebar: corpus status / admin
with st.sidebar:
    st.header("Corpus")
    pdf_count = len(list(config.PDF_DIR.glob("*.pdf"))) if config.PDF_DIR.exists() else 0
    st.metric("PDFs in data/pdfs/", pdf_count)

    if config.SKIPPED_LOG_PATH.exists():
        import json

        skipped = json.loads(config.SKIPPED_LOG_PATH.read_text())
        st.metric("Skipped (scanned/unreadable)", len(skipped))
        if skipped:
            with st.expander("View skipped files"):
                for s in skipped:
                    st.write(f"- **{s['file']}** — {s['reason']}")

    st.divider()
    st.caption(
        "To (re)index after adding/removing PDFs, run in a terminal:\n\n"
        "`python scripts/build_index.py`"
    )
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.preview = None
        st.rerun()
