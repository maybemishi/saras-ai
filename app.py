"""
app.py — SARAS Streamlit Research Dashboard

Run with:
  streamlit run app.py
"""

import streamlit as st
import time
import io
from config import settings
from graph import saras_graph
from state import ResearchState
from utils.pdf_parser import ingest_pdf, clear_session_collection
from utils.credibility import label_for_score
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SARAS — Autonomous Research System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.agent-step {
    padding: 8px 14px;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 14px;
    font-weight: 500;
}
.step-active  { background: #dbeafe; color: #1e3a8a; border-left: 4px solid #3b82f6; }
.step-done    { background: #dcfce7; color: #14532d; border-left: 4px solid #22c55e; }
.step-waiting { background: #f1f5f9; color: #475569; border-left: 4px solid #94a3b8; }
.step-retry   { background: #fef3c7; color: #92400e; border-left: 4px solid #f59e0b; }
.credibility-high   { color: #16a34a; font-weight: 600; }
.credibility-medium { color: #ca8a04; font-weight: 600; }
.credibility-low    { color: #dc2626; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────
def init_session():
    defaults = {
        "result": None,
        "pipeline_log": [],
        "running": False,
        "docs_ingested": False,
        "ingested_filenames": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/SARAS-Research%20AI-blue?style=for-the-badge", width="stretch") #made a change here
    st.markdown("## ⚙️ Configuration")

    st.markdown("**Model**")
    st.code(settings.GEMINI_MODEL, language=None)

    st.markdown("**Search results per task**")
    st.code(str(settings.MAX_SEARCH_RESULTS), language=None)

    st.markdown("**Max retry loops**")
    st.code(str(settings.MAX_RETRY_LOOPS), language=None)

    st.divider()

    st.markdown("## 📄 Upload Documents")
    st.caption("Upload your own PDFs to cross-reference with live web data.")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("📥 Ingest Documents", use_container_width=True):
            with st.spinner("Parsing and embedding documents..."):
                clear_session_collection()
                st.session_state.ingested_filenames = []
                total_chunks = 0
                for f in uploaded_files:
                    pdf_bytes = f.read()
                    n = ingest_pdf(pdf_bytes, f.name)
                    total_chunks += n
                    st.session_state.ingested_filenames.append(f.name)
                st.session_state.docs_ingested = True
            st.success(f"✅ Ingested {len(uploaded_files)} file(s) → {total_chunks} chunks stored")

    if st.session_state.docs_ingested:
        st.markdown("**Ingested files:**")
        for fname in st.session_state.ingested_filenames:
            st.markdown(f"- 📎 `{fname}`")

    st.divider()
    if st.button("🗑️ Clear Session & Documents", use_container_width=True):
        clear_session_collection()
        st.session_state.result = None
        st.session_state.pipeline_log = []
        st.session_state.docs_ingested = False
        st.session_state.ingested_filenames = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 🔬 SARAS")
st.markdown("#### System for Autonomous Research and Academic Synthesis")
st.divider()

col_input, col_status = st.columns([3, 1])

with col_input:
    query = st.text_area(
        "Enter your research query",
        placeholder="e.g. What are the latest advancements in multimodal large language models and their applications in education?",
        height=100,
        key="query_input",
    )

with col_status:
    st.markdown("**Session status**")
    if st.session_state.docs_ingested:
        st.success(f"📎 {len(st.session_state.ingested_filenames)} doc(s) loaded")
    else:
        st.info("No documents uploaded")

run_col, _ = st.columns([1, 4])
with run_col:
    run_button = st.button(
        "🚀 Run Research",
        use_container_width=True,
        disabled=st.session_state.running,
    )


# ── Pipeline execution ─────────────────────────────────────────────────────
AGENT_STEPS = [
    ("planner",          "Planner — decomposing query"),
    ("web_scripter",     "Web Scripter — retrieving sources"),
    ("analyst",          "Analyst — cross-referencing & verifying"),
    ("synthesizer",      "Synthesizer — writing report"),
    ("source_aggregator","Source Aggregator — building bibliography"),
]

if run_button and query.strip():
    st.session_state.running = True
    st.session_state.result = None
    st.session_state.pipeline_log = []

    # Validate keys before running
    try:
        settings.validate()
    except EnvironmentError as e:
        st.error(str(e))
        st.session_state.running = False
        st.stop()

    st.divider()
    st.markdown("### 🔄 Pipeline Progress")
    step_placeholders = {}
    for step_id, label in AGENT_STEPS:
        step_placeholders[step_id] = st.empty()
        step_placeholders[step_id].markdown(
            f'<div class="agent-step step-waiting">⏳ {label}</div>',
            unsafe_allow_html=True,
        )

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Build initial state
    initial_state = ResearchState(
        query=query.strip(),
        has_uploaded_docs=st.session_state.docs_ingested,
    )

    # Stream through the graph node by node
    final_state = None
    step_order = [s[0] for s in AGENT_STEPS]
    completed_steps = set()

    try:
        for i, (step_id, label) in enumerate(AGENT_STEPS):
            step_placeholders[step_id].markdown(
                f'<div class="agent-step step-active">🔵 {label}...</div>',
                unsafe_allow_html=True,
            )
            status_text.markdown(f"**Current:** {label}")
            progress_bar.progress((i) / len(AGENT_STEPS))

        # Run the full graph (blocking)
        final_state = saras_graph.invoke(initial_state)

        # Mark retry if it happened
        if final_state.get("retry_count", 0) > 0:
            step_placeholders["analyst"].markdown(
                f'<div class="agent-step step-retry">'
                f'🔁 Analyst — retried {final_state["retry_count"]}x, then approved'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Mark all done
        for step_id, label in AGENT_STEPS:
            if step_id != "analyst" or final_state.get("retry_count", 0) == 0:
                step_placeholders[step_id].markdown(
                    f'<div class="agent-step step-done">✅ {label}</div>',
                    unsafe_allow_html=True,
                )

        progress_bar.progress(1.0)
        status_text.markdown("**✅ Research complete!**")
        st.session_state.result = final_state

    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        st.exception(e)
    finally:
        st.session_state.running = False


# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result
    st.divider()
    st.markdown("## 📋 Research Output")

    tab_report, tab_sources, tab_analyst = st.tabs([
        "📄 Full Report", "🔗 Sources", "🔍 Analyst Notes"
    ])
    
    # ── Tab 1: Report ─────────────────────────────────────────────────────
    with tab_report:
        report_md = result.get("report_markdown", "")
        bibliography = result.get("bibliography", "")
        full_report = report_md + "\n\n---\n\n" + bibliography if bibliography else report_md

        st.markdown(full_report)

        # ── PDF Generator ─────────────────────────────────────────
        def generate_pdf(text):
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            import io

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()

            content = []
            for line in text.split("\n"):
                content.append(Paragraph(line, styles["Normal"]))
                content.append(Spacer(1, 8))  # spacing between lines

            doc.build(content)
            buffer.seek(0)
            return buffer

        # ── DOCX Generator ────────────────────────────────────────
        def generate_docx(text):
            from docx import Document
            import io

            doc = Document()
            for line in text.split("\n"):
                doc.add_paragraph(line)

            buffer = io.BytesIO()
            doc.save(buffer)
            buffer.seek(0)
            return buffer

        # ── Download Section ──────────────────────────────────────
        st.markdown("## 📥 Download")

        format_option = st.selectbox(
            "Choose format",
            ["PDF", "DOCX"]
        )

        if format_option == "PDF":
            file_data = generate_pdf(full_report)
            file_name = "saras_report.pdf"
            mime = "application/pdf"
        else:
            file_data = generate_docx(full_report)
            file_name = "saras_report.docx"
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        st.download_button(
            label="⬇️ Download File",
            data=file_data,
            file_name=file_name,
            mime=mime,
            use_container_width=True
        )

    # ── Tab 2: Sources ────────────────────────────────────────────────────
    with tab_sources:
        final_sources = result.get("final_sources", [])
        if not final_sources:
            st.info("No sources collected.")
        else:
            st.markdown(f"**{len(final_sources)} sources found** (ranked by credibility)")
            st.divider()

            for i, source in enumerate(final_sources, 1):
                label = label_for_score(source.get("credibility_score", 0.5) if isinstance(source, dict) else source.credibility_score)
                score_val = source.get("credibility_score", 0.5) if isinstance(source, dict) else source.credibility_score
                title = source.get("title", "") if isinstance(source, dict) else source.title
                url = source.get("url", "") if isinstance(source, dict) else source.url
                snippet = source.get("snippet", "") if isinstance(source, dict) else source.snippet
                origin = source.get("origin", "web") if isinstance(source, dict) else source.origin

                css_class = f"credibility-{label.lower()}"
                with st.expander(f"{i}. {title[:80]}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        if origin == "document":
                            st.markdown(f"📎 **Uploaded document:** `{url}`")
                        else:
                            st.markdown(f"🌐 **URL:** [{url}]({url})")
                        st.markdown(f"**Excerpt:** {snippet[:300]}...")
                    with col2:
                        st.markdown(
                            f'<span class="{css_class}">● {label} credibility</span><br>'
                            f'Score: {score_val:.2f}<br>'
                            f'Origin: {origin.capitalize()}',
                            unsafe_allow_html=True,
                        )

    # ── Tab 3: Analyst Notes ──────────────────────────────────────────────
    with tab_analyst:
        st.markdown("### Verification Summary")

        verdict = result.get("analyst_verdict", "unknown")
        retry_count = result.get("retry_count", 0)
        contradiction_notes = result.get("contradiction_notes", "")
        research_plan = result.get("research_plan", [])

        if verdict == "approved":
            st.success(f"✅ Analyst approved the data after {retry_count} retry loop(s)")
        else:
            st.warning("⚠️ Analyst forced approval after max retries")

        if contradiction_notes:
            st.markdown("**Contradiction / gap notes:**")
            st.info(contradiction_notes)

        if research_plan:
            st.markdown("**Research sub-tasks executed:**")
            for j, task in enumerate(research_plan, 1):
                st.markdown(f"{j}. {task}")


elif run_button and not query.strip():
    st.warning("Please enter a research query before running.")
