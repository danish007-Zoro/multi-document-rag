import os
import sys
import tempfile
import base64
import html

import streamlit as st


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

FONT_DIR = os.path.join(
    PROJECT_ROOT,
    "assets",
    "fonts"
)


# ==================================================
# IMPORT RAG PIPELINE
# ==================================================

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rag_pipeline import RAGPipeline


# ==================================================
# FONT LOADING
# ==================================================

def load_font_base64(filename):

    path = os.path.join(
        FONT_DIR,
        filename
    )

    if not os.path.exists(path):
        return ""

    with open(path, "rb") as font_file:

        return base64.b64encode(
            font_file.read()
        ).decode("utf-8")


space_regular = load_font_base64(
    "SPACEGROTESK-REGULAR.TTF"
)

space_medium = load_font_base64(
    "SPACEGROTESK-MEDIUM.TTF"
)

space_semibold = load_font_base64(
    "SPACEGROTESK-SEMIBOLD.TTF"
)

space_bold = load_font_base64(
    "SPACEGROTESK-BOLD.TTF"
)

mono_regular = load_font_base64(
    "JETBRAINSMONO-REGULAR.TTF"
)

mono_medium = load_font_base64(
    "JETBRAINSMONO-MEDIUM.TTF"
)

mono_semibold = load_font_base64(
    "JETBRAINSMONO-SEMIBOLD.TTF"
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Nexus RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CUSTOM CSS
# ==================================================

font_css = f"""
<style>

@font-face {{
    font-family: 'Nexus Space';
    src: url(data:font/ttf;base64,{space_regular})
        format('truetype');
    font-weight: 400;
}}

@font-face {{
    font-family: 'Nexus Space';
    src: url(data:font/ttf;base64,{space_medium})
        format('truetype');
    font-weight: 500;
}}

@font-face {{
    font-family: 'Nexus Space';
    src: url(data:font/ttf;base64,{space_semibold})
        format('truetype');
    font-weight: 600;
}}

@font-face {{
    font-family: 'Nexus Space';
    src: url(data:font/ttf;base64,{space_bold})
        format('truetype');
}}

@font-face {{
    font-family: 'Nexus Mono';
    src: url(data:font/ttf;base64,{mono_regular})
        format('truetype');
    font-weight: 400;
}}

@font-face {{
    font-family: 'Nexus Mono';
    src: url(data:font/ttf;base64,{mono_medium})
        format('truetype');
    font-weight: 500;
}}

@font-face {{
    font-family: 'Nexus Mono';
    src: url(data:font/ttf;base64,{mono_semibold})
        format('truetype');
    font-weight: 600;
}}


/* ==================================================
   ROOT
================================================== */

:root {{
    --bg: #090b0f;
    --surface: #101319;
    --surface-2: #141820;

    --border: #252b35;
    --border-soft: #1d222b;

    --text: #e7e9ee;
    --text-secondary: #858c9b;
    --text-muted: #5e6675;

    --accent: #8b7cff;
    --success: #5ce0a4;
}}


/* ==================================================
   GLOBAL
================================================== */

html,
body,
[class*="css"] {{
    font-family: 'Nexus Space', sans-serif;
}}

.stApp {{
    background: var(--bg);
    color: var(--text);
}}

.main {{
    background: var(--bg);
}}

.block-container {{
    max-width: 1380px;
    padding-top: 1.4rem;
    padding-bottom: 5rem;
}}

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

header {{
    background: transparent !important;
}}


/* ==================================================
   SIDEBAR
================================================== */

section[data-testid="stSidebar"] {{
    background: #0d1015;
    border-right: 1px solid var(--border-soft);
}}

section[data-testid="stSidebar"] > div {{
    padding-top: 1.5rem;
    padding-left: 1.15rem;
    padding-right: 1.15rem;
}}


/* ==================================================
   BRAND
================================================== */

.brand {{
    font-family: 'Nexus Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 1.8px;
    color: var(--text);
}}

.brand-accent {{
    color: var(--accent);
}}

.brand-subtitle {{
    margin-top: 7px;
    color: var(--text-muted);
    font-family: 'Nexus Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.5px;
}}


/* ==================================================
   SECTION LABELS
================================================== */

.section-label {{
    margin-top: 1.7rem;
    margin-bottom: 0.65rem;

    color: var(--text-muted);

    font-family: 'Nexus Mono', monospace;
    font-size: 0.61rem;
    font-weight: 500;

    letter-spacing: 1.5px;
    text-transform: uppercase;
}}


/* ==================================================
   SIDEBAR DIVIDER
================================================== */

.sidebar-divider {{
    height: 1px;
    background: var(--border-soft);
    margin: 1.4rem 0;
}}


/* ==================================================
   DOCUMENT ITEM
================================================== */

.document-item {{
    display: flex;
    align-items: center;

    padding: 8px 2px;

    border-bottom: 1px solid #171b22;
}}

.document-number {{
    width: 28px;

    color: var(--text-muted);

    font-family: 'Nexus Mono', monospace;
    font-size: 0.61rem;
}}

.document-name {{
    color: #cdd1da;

    font-size: 0.78rem;
    font-weight: 500;

    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}


/* ==================================================
   INDEX STATS
================================================== */

.index-stat {{
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 6px 0;

    border-bottom: 1px solid #171b22;
}}

.index-stat-label {{
    color: var(--text-muted);

    font-family: 'Nexus Mono', monospace;
    font-size: 0.61rem;

    letter-spacing: 0.7px;
}}

.index-stat-value {{
    color: var(--text-secondary);

    font-family: 'Nexus Mono', monospace;
    font-size: 0.67rem;
    font-weight: 500;
}}


/* ==================================================
   MAIN HEADER
================================================== */

.main-kicker {{
    color: var(--accent);

    font-family: 'Nexus Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;

    letter-spacing: 1.8px;
    text-transform: uppercase;

    margin-bottom: 7px;
}}

.main-title {{
    color: var(--text);

    font-family: 'Nexus Space', sans-serif;
    font-size: 2rem;
    font-weight: 600;

    letter-spacing: -0.8px;

    margin: 0;
}}

.main-description {{
    color: var(--text-secondary);

    font-size: 0.88rem;
    font-weight: 400;

    margin-top: 7px;
}}


/* ==================================================
   SYSTEM STATUS
================================================== */

.system-status {{
    display: flex;
    justify-content: flex-end;
    align-items: center;

    gap: 8px;

    padding-top: 10px;

    color: var(--text-muted);

    font-family: 'Nexus Mono', monospace;
    font-size: 0.62rem;

    letter-spacing: 0.8px;
    text-transform: uppercase;
}}

.status-dot {{
    width: 6px;
    height: 6px;

    border-radius: 50%;

    background: var(--success);

    box-shadow:
        0 0 8px rgba(92, 224, 164, 0.35);
}}


/* ==================================================
   EMPTY STATE
================================================== */

.empty-state {{
    margin-top: 3.5rem;

    padding: 5rem 2rem;

    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);

    text-align: center;
}}

.empty-kicker {{
    color: var(--text-muted);

    font-family: 'Nexus Mono', monospace;
    font-size: 0.62rem;

    letter-spacing: 1.5px;
    text-transform: uppercase;
}}

.empty-title {{
    margin-top: 10px;

    color: var(--text);

    font-size: 1.15rem;
    font-weight: 500;
}}

.empty-description {{
    max-width: 430px;

    margin: 8px auto 0 auto;

    color: var(--text-muted);

    font-size: 0.78rem;
    line-height: 1.7;
}}


/* ==================================================
   QUESTION
================================================== */

.question-label {{
    color: var(--text-muted);

    font-family: 'Nexus Mono', monospace;
    font-size: 0.61rem;
    font-weight: 500;

    letter-spacing: 1.5px;
    text-transform: uppercase;

    margin-top: 2.5rem;
    margin-bottom: 8px;
}}

.question-text {{
    color: var(--text);

    font-size: 1rem;
    font-weight: 500;

    line-height: 1.6;
}}


/* ==================================================
   ANSWER
================================================== */

.answer-label {{
    color: var(--accent);

    font-family: 'Nexus Mono', monospace;
    font-size: 0.61rem;
    font-weight: 500;

    letter-spacing: 1.5px;
    text-transform: uppercase;

    margin-top: 2rem;
    margin-bottom: 10px;
}}

.answer-text {{
    color: #dfe2e8;

    font-size: 0.93rem;
    font-weight: 400;

    line-height: 1.75;

    max-width: 900px;
}}


/* ==================================================
   SOURCES
================================================== */

.sources-label {{
    color: var(--text-muted);

    font-family: 'Nexus Mono', monospace;
    font-size: 0.61rem;
    font-weight: 500;

    letter-spacing: 1.5px;
    text-transform: uppercase;

    margin-top: 2rem;
    margin-bottom: 9px;
}}

.source-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 9px 0;

    border-top: 1px solid var(--border-soft);

    max-width: 900px;
}}

.source-document {{
    color: #c9cdd5;

    font-size: 0.76rem;
    font-weight: 500;
}}

.source-meta {{
    color: var(--text-muted);

    font-family: 'Nexus Mono', monospace;
    font-size: 0.59rem;

    white-space: nowrap;
}}

.source-inspection-label {{
    margin-bottom: 8px;

    color: #8b7cff;

    font-family: 'Nexus Mono', monospace;
    font-size: 0.58rem;
    font-weight: 500;

    letter-spacing: 1.4px;
}}

.retrieved-chunk {{
    padding: 14px;

    background: #0b0e13;

    border: 1px solid #202630;
    border-radius: 5px;

    color: #cfd4dd;

    font-family: 'Nexus Mono', monospace;
    font-size: 0.68rem;

    line-height: 1.7;

    white-space: pre-wrap;
}}

.source-metadata {{
    margin-top: 10px;

    color: #5e6675;

    font-family: 'Nexus Mono', monospace;
    font-size: 0.56rem;

    letter-spacing: 0.5px;
}}


/* ==================================================
   SUMMARY
================================================== */

.summary-container {{
    margin-top: 3rem;

    max-width: 950px;

    padding: 1.8rem 0 1.2rem 0;

    border-top: 1px solid var(--border);
}}

.summary-title {{
    color: var(--accent);

    font-family: 'Nexus Mono', monospace;
    font-size: 0.64rem;

    letter-spacing: 1.5px;
    text-transform: uppercase;
}}

.summary-document {{
    margin-top: 8px;

    color: var(--text);

    font-size: 1.05rem;
    font-weight: 600;
}}

.summary-content {{
    max-width: 950px;

    color: #dfe2e8;

    font-size: 0.91rem;
    line-height: 1.75;
}}


/* ==================================================
   BUTTONS
================================================== */

.stButton > button {{
    width: 100%;

    min-height: 36px;

    border-radius: 6px;

    border: 1px solid #303642;

    background: transparent;

    color: var(--text-secondary);

    font-family: 'Nexus Mono', monospace;
    font-size: 0.63rem;

    letter-spacing: 0.7px;

    transition: all 0.18s ease;
}}

.stButton > button:hover {{
    border-color: var(--accent);

    background: rgba(139, 124, 255, 0.05);

    color: var(--text);
}}


/* ==================================================
   FILE UPLOADER
================================================== */

section[data-testid="stFileUploader"] {{
    border: 1px dashed #303642;

    border-radius: 7px;

    background: #0b0e13;
}}

section[data-testid="stFileUploader"] label {{
    color: var(--text-secondary);
}}


/* ==================================================
   SELECTBOX
================================================== */

div[data-baseweb="select"] > div {{
    background: #0b0e13;

    border-color: #303642;
}}


/* ==================================================
   CHAT INPUT
================================================== */

div[data-testid="stChatInput"] {{
    border-top: 1px solid var(--border-soft);
}}

div[data-testid="stChatInput"] textarea {{
    font-family: 'Nexus Space', sans-serif;
}}


/* ==================================================
   CHAT MESSAGE
================================================== */

div[data-testid="stChatMessage"] {{
    background: transparent;

    padding-left: 0;
    padding-right: 0;
}}


/* ==================================================
   SCROLLBAR
================================================== */

::-webkit-scrollbar {{
    width: 6px;
}}

::-webkit-scrollbar-track {{
    background: var(--bg);
}}

::-webkit-scrollbar-thumb {{
    background: #282e38;

    border-radius: 3px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: #383f4c;
}}

</style>
"""

st.markdown(
    font_css,
    unsafe_allow_html=True
)


# ==================================================
# SESSION STATE
# ==================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "stats" not in st.session_state:
    st.session_state.stats = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "summary_document" not in st.session_state:
    st.session_state.summary_document = None

if "summary_result" not in st.session_state:
    st.session_state.summary_result = None

if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def clear_chat():

    st.session_state.messages = []

    st.session_state.summary_document = None

    st.session_state.summary_result = None


def clear_knowledge_base():

    st.session_state.pipeline = None
    st.session_state.documents = []
    st.session_state.stats = None
    st.session_state.messages = []

    st.session_state.summary_document = None
    st.session_state.summary_result = None
    st.session_state.uploader_version += 1

def render_sources(sources, results=None):

    if not sources:
        return

    st.html(
        '<div class="sources-label">Sources</div>'
    )

    for index, source in enumerate(
        sources,
        start=1
    ):

        document = html.escape(
            str(
                source.get(
                    "document",
                    "Unknown"
                )
            )
        )

        page = int(
            source.get(
                "page",
                0
            )
        )

        chunk_id = int(
            source.get(
                "chunk_id",
                0
            )
        )

        distance = float(
            source.get(
                "distance",
                0.0
            )
        )

        # ------------------------------------------
        # Find matching retrieved chunk
        # ------------------------------------------

        retrieved_chunk = None

        if results:

            for result in results:

                if (
                    result.get("document")
                    == source.get("document")
                    and
                    result.get("page")
                    == source.get("page")
                    and
                    result.get("chunk_id")
                    == source.get("chunk_id")
                ):

                    retrieved_chunk = result

                    break


        # ------------------------------------------
        # Source expander
        # ------------------------------------------

        with st.expander(
            f"{index:02d}  {document}    "
            f"P{page:02d} / C{chunk_id:02d} / {distance:.3f}",
            expanded=False
        ):

            st.html(
                """
                <div class="source-inspection-label">
                    RETRIEVED CHUNK
                </div>
                """
            )

            if retrieved_chunk:

                chunk_text = retrieved_chunk.get(
                    "text",
                    ""
                )

                if chunk_text:

                    safe_chunk_text = html.escape(
                        str(chunk_text)
                    )

                    st.html(
                        f"""
                        <div class="retrieved-chunk">
                            {safe_chunk_text}
                        </div>
                        """
                    )

                else:

                    st.caption(
                        "Retrieved chunk text is unavailable."
                    )

            else:

                st.caption(
                    "Retrieved chunk could not be matched."
                )


            # --------------------------------------
            # Retrieval metadata
            # --------------------------------------

            rank = (
                retrieved_chunk.get(
                    "rank"
                )
                if retrieved_chunk
                else None
            )

            faiss_index = (
                retrieved_chunk.get(
                    "faiss_index"
                )
                if retrieved_chunk
                else None
            )

            metadata_parts = [
                f"PAGE {page:02d}",
                f"CHUNK {chunk_id:02d}",
                f"DISTANCE {distance:.3f}"
            ]

            if rank is not None:
                metadata_parts.append(
                    f"RANK {int(rank):02d}"
                )

            if faiss_index is not None:
                metadata_parts.append(
                    f"FAISS {int(faiss_index):02d}"
                )

            st.html(
                f"""
                <div class="source-metadata">
                    {' &nbsp;·&nbsp; '.join(metadata_parts)}
                </div>
                """
            )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    # ----------------------------------------------
    # BRAND
    # ----------------------------------------------

    st.html(
        """
        <div class="brand">
            <span class="brand-accent">◈</span>
            NEXUS / RAG
        </div>

        <div class="brand-subtitle">
            DOCUMENT INTELLIGENCE WORKSPACE
        </div>
        """
    )

    st.html(
        '<div class="sidebar-divider"></div>'
    )


    # ----------------------------------------------
    # KNOWLEDGE BASE
    # ----------------------------------------------

    st.html(
        '<div class="section-label">Knowledge Base</div>'
    )

    uploaded_files = st.file_uploader(
        "Add documents",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"pdf_uploader_{st.session_state.uploader_version}"
    )

    # ----------------------------------------------
    # BUILD INDEX
    # ----------------------------------------------

    if uploaded_files:

        if st.button(
            "▶  BUILD INDEX",
            use_container_width=True,
            key="build_index_button"
        ):

            temp_paths = []

            try:

                with st.spinner(
                    "Building index..."
                ):

                    # ----------------------------------
                    # Save uploaded files temporarily
                    # ----------------------------------

                    for uploaded_file in uploaded_files:

                        temp_file = (
                            tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=".pdf"
                            )
                        )

                        temp_file.write(
                            uploaded_file.getbuffer()
                        )

                        temp_file.close()

                        temp_paths.append(
                            temp_file.name
                        )


                    # ----------------------------------
                    # Create RAG pipeline
                    # ----------------------------------

                    pipeline = RAGPipeline(
                        distance_threshold=1.4,
                        top_k=3
                    )


                    # ----------------------------------
                    # Build knowledge base
                    # ----------------------------------

                    stats = pipeline.load_documents(
                        temp_paths
                    )


                    # ----------------------------------
                    # Temporary filename mapping
                    # ----------------------------------

                    name_map = {}

                    for index, path in enumerate(
                        temp_paths
                    ):

                        temporary_name = (
                            os.path.basename(path)
                        )

                        original_name = (
                            uploaded_files[index].name
                        )

                        name_map[
                            temporary_name
                        ] = original_name


                    # ----------------------------------
                    # Fix document metadata
                    # ----------------------------------

                    for document in stats.get(
                        "documents",
                        []
                    ):

                        document["name"] = (
                            name_map.get(
                                document["name"],
                                document["name"]
                            )
                        )


                    # ----------------------------------
                    # Fix chunk metadata
                    # ----------------------------------

                    for chunk in pipeline.chunks:

                        chunk["document"] = (
                            name_map.get(
                                chunk.get(
                                    "document"
                                ),
                                chunk.get(
                                    "document"
                                )
                            )
                        )


                    # ----------------------------------
                    # Fix document page metadata
                    # ----------------------------------

                    if hasattr(
                        pipeline,
                        "document_pages"
                    ):

                        remapped_pages = {}

                        for (
                            temporary_name,
                            pages
                        ) in (
                            pipeline.document_pages.items()
                        ):

                            original_name = (
                                name_map.get(
                                    temporary_name,
                                    temporary_name
                                )
                            )

                            remapped_pages[
                                original_name
                            ] = pages

                        pipeline.document_pages = (
                            remapped_pages
                        )


                    # ----------------------------------
                    # Save application state
                    # ----------------------------------

                    st.session_state.pipeline = (
                        pipeline
                    )

                    st.session_state.documents = [
                        file.name
                        for file in uploaded_files
                    ]

                    st.session_state.stats = (
                        stats
                    )

                    st.session_state.messages = []

                    st.session_state.summary_document = (
                        None
                    )

                    st.session_state.summary_result = (
                        None
                    )

                st.success(
                    "Index ready."
                )

                st.rerun()


            except Exception as error:

                st.error(
                    f"Indexing failed: {error}"
                )


            finally:

                for path in temp_paths:

                    if os.path.exists(path):

                        os.remove(path)


    # ----------------------------------------------
    # DOCUMENTS
    # ----------------------------------------------

    if st.session_state.documents:

        st.html(
            '<div class="section-label">Documents</div>'
        )

        for index, document_name in enumerate(
            st.session_state.documents,
            start=1
        ):

            safe_name = html.escape(
                str(document_name)
            )

            st.html(
                f"""
                <div class="document-item">

                    <span class="document-number">
                        {index:02d}
                    </span>

                    <span class="document-name">
                        {safe_name}
                    </span>

                </div>
                """
            )


    # ----------------------------------------------
    # SUMMARY
    # ----------------------------------------------

    if (
        st.session_state.pipeline
        and st.session_state.documents
    ):

        st.html(
            '<div class="section-label">Summary</div>'
        )


        selected_document = st.selectbox(
            "Select document",
            st.session_state.documents,
            index=0,
            key="summary_document_selector",
            label_visibility="collapsed"
        )


        if st.button(
            "SUMMARY",
            use_container_width=True,
            key="generate_summary_button"
        ):

            try:

                with st.spinner(
                    "Generating summary..."
                ):

                    summary = (
                        st.session_state
                        .pipeline
                        .summarize(
                            selected_document
                        )
                    )


                st.session_state.summary_document = (
                    selected_document
                )

                st.session_state.summary_result = (
                    summary
                )

                st.rerun()


            except AttributeError:

                st.error(
                    "Summary is not available in the current RAG pipeline."
                )


            except Exception as error:

                st.error(
                    f"Summary generation failed: {error}"
                )


    # ----------------------------------------------
    # INDEX
    # ----------------------------------------------

    if st.session_state.stats:

        stats = st.session_state.stats

        st.html(
            '<div class="section-label">Index</div>'
        )

        st.html(
            f"""
            <div class="index-stat">

                <span class="index-stat-label">
                    DOCUMENTS
                </span>

                <span class="index-stat-value">
                    {int(stats.get("total_documents", 0)):02d}
                </span>

            </div>


            <div class="index-stat">

                <span class="index-stat-label">
                    PAGES
                </span>

                <span class="index-stat-value">
                    {int(stats.get("total_pages", 0)):02d}
                </span>

            </div>


            <div class="index-stat">

                <span class="index-stat-label">
                    CHUNKS
                </span>

                <span class="index-stat-value">
                    {int(stats.get("total_chunks", 0)):02d}
                </span>

            </div>


            <div class="index-stat">

                <span class="index-stat-label">
                    VECTORS
                </span>

                <span class="index-stat-value">
                    {int(stats.get("vectors", 0)):02d}
                </span>

            </div>


            <div class="index-stat">

                <span class="index-stat-label">
                    DIMENSION
                </span>

                <span class="index-stat-value">
                    {int(stats.get("embedding_dimension", 0))}
                </span>

            </div>
            """
        )


    # ----------------------------------------------
    # CONVERSATION
    # ----------------------------------------------

    st.html(
        '<div class="section-label">Conversation</div>'
    )


    has_chat_content = (
        bool(st.session_state.messages)
        or
        st.session_state.summary_result is not None
    )


    if st.button(
        "CLEAR CHAT",
        use_container_width=True,
        key="clear_chat_button",
        disabled=not has_chat_content
    ):

        clear_chat()

        st.rerun()


    # ----------------------------------------------
    # SYSTEM
    # ----------------------------------------------

    if st.session_state.pipeline:

        st.html(
            '<div class="section-label">System</div>'
        )


        if st.button(
            "CLEAR KNOWLEDGE BASE",
            use_container_width=True,
            key="clear_knowledge_button"
        ):

            clear_knowledge_base()

            st.rerun()


# ==================================================
# MAIN HEADER
# ==================================================

header_left, header_right = st.columns(
    [5, 1]
)


with header_left:

    st.html(
        """
        <div class="main-kicker">
            DOCUMENT INTELLIGENCE
        </div>

        <div class="main-title">
            Query the knowledge layer.
        </div>

        <div class="main-description">
            Ask across your documents. Get answers grounded in the source.
        </div>
        """
    )


with header_right:

    if st.session_state.pipeline:

        st.html(
            """
            <div class="system-status">

                <span class="status-dot"></span>

                INDEX READY

            </div>
            """
        )


# ==================================================
# NO KNOWLEDGE BASE
# ==================================================

if st.session_state.pipeline is None:

    st.html(
        """
        <div class="empty-state">

            <div class="empty-kicker">
                KNOWLEDGE LAYER OFFLINE
            </div>

            <div class="empty-title">
                Upload documents to begin.
            </div>

            <div class="empty-description">
                Add one or more PDFs from the sidebar,
                then build the index.
            </div>

        </div>
        """
    )


# ==================================================
# KNOWLEDGE BASE READY
# ==================================================

else:


    # ==================================================
    # SUMMARY
    # ==================================================

    if st.session_state.summary_result:

        summary_document = html.escape(
            str(
                st.session_state.summary_document
            )
        )


        st.html(
            f"""
            <div class="summary-container">

                <div class="summary-title">
                    Document summary
                </div>

                <div class="summary-document">
                    {summary_document}
                </div>

            </div>
            """
        )


        # Summary is already generated by the
        # application's summarization pipeline.
        # Markdown rendering preserves headings
        # and bullet points.

        st.markdown(
            st.session_state.summary_result
        )


    # ==================================================
    # EMPTY KNOWLEDGE LAYER
    # ==================================================

    if (
        not st.session_state.messages
        and not st.session_state.summary_result
    ):

        st.html(
            """
            <div class="empty-state">

                <div class="empty-kicker">
                    KNOWLEDGE LAYER ONLINE
                </div>

                <div class="empty-title">
                    Ask something about your documents.
                </div>

                <div class="empty-description">
                    Retrieval will ground the response
                    in your indexed sources.
                </div>

            </div>
            """
        )


    # ==================================================
    # CHAT HISTORY
    # ==================================================

    for message in st.session_state.messages:

        role = message.get(
            "role"
        )

        content = str(
            message.get(
                "content",
                ""
            )
        )


        # ----------------------------------------------
        # USER
        # ----------------------------------------------

        if role == "user":

            safe_content = html.escape(
                content
            )


            with st.chat_message(
                "user"
            ):

                st.html(
                    f"""
                    <div class="question-label">
                        QUESTION
                    </div>

                    <div class="question-text">
                        {safe_content}
                    </div>
                    """
                )


        # ----------------------------------------------
        # ASSISTANT
        # ----------------------------------------------

        elif role == "assistant":

            safe_content = html.escape(
                content
            ).replace(
                "\n",
                "<br>"
            )


            with st.chat_message(
                "assistant"
            ):

                st.html(
                    f"""
                    <div class="answer-label">
                        GROUNDED RESPONSE
                    </div>

                    <div class="answer-text">
                        {safe_content}
                    </div>
                    """
                )


                render_sources(
                    message.get(
                        "sources",
                        []
                    ),
                    message.get(
                        "results",
                        []
                    )
                )


    # ==================================================
    # CHAT INPUT
    # ==================================================

    query = st.chat_input(
        "Ask about your documents..."
    )


    if query:

        query = query.strip()


        if query:

            try:

                with st.spinner(
                    "Searching index..."
                ):

                    result = (
                        st.session_state
                        .pipeline
                        .ask(
                            query
                        )
                    )


                # ----------------------------------
                # SAVE USER MESSAGE
                # ----------------------------------

                st.session_state.messages.append(
                    {
                        "role": "user",

                        "content": query
                    }
                )


                # ----------------------------------
                # SAVE ASSISTANT MESSAGE
                # ----------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",

                        "content": result.get(
                            "answer",
                            "No answer was generated."
                        ),

                        "sources": result.get(
                            "sources",
                            []
                        ),

                        "results": result.get(
                            "results",
                            []
                        )
                    }
                )


                # A new question returns the interface
                # to conversation mode.

                st.session_state.summary_document = (
                    None
                )

                st.session_state.summary_result = (
                    None
                )


                st.rerun()


            except Exception as error:

                error_message = (
                    f"Something went wrong: {error}"
                )


                st.session_state.messages.append(
                    {
                        "role": "user",

                        "content": query
                    }
                )


                st.session_state.messages.append(
                    {
                        "role": "assistant",

                        "content": error_message,

                        "sources": []
                    }
                )


                st.rerun()