
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    DATA_DIR,
    EVALUATION_DIR,
    GRAPH_DIR,
    EMBEDDING_MODEL,
    RRF_K
)

from src.data_loader import load_news_data
from src.preprocessing import preprocess_dataframe, clean_text
from src.tfidf_search import TFIDFRetriever
from src.bm25_search import BM25Retriever
from src.semantic_search import SemanticRetriever
from src.hybrid_search import reciprocal_rank_fusion


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="BBC News Search Engine",
    page_icon="📰",
    layout="wide"
)


# --------------------------------------------------
# 2. SESSION STATE MANAGEMENT
# --------------------------------------------------

def clear_previous_search():
    st.session_state.pop("last_ranking", None)
    st.session_state.pop("last_query", None)
    st.session_state.pop("last_model", None)


# --------------------------------------------------
# 3. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.title {
    font-size: 38px;
    font-weight: 800;
    color: #172554;
}

.subtitle {
    font-size: 16px;
    color: #64748b;
}

.stButton button {
    background-color: #1d4ed8;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">📰 BBC News Search Engine</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent News Retrieval using '
    'TF-IDF, BM25, Sentence Transformer and Hybrid RRF</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# 5. DATASET LOADING
# --------------------------------------------------

@st.cache_data
def get_data(path):

    df = load_news_data(path)
    df = preprocess_dataframe(df)

    required_columns = [
        "category",
        "title",
        "text",
        "document",
        "clean_text"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns after preprocessing: {missing}"
        )

    return df


# --------------------------------------------------
# 6. BUILD RETRIEVAL INDEXES
# --------------------------------------------------

@st.cache_resource
def build_sparse_indexes(documents):

    tfidf = TFIDFRetriever(list(documents))
    bm25 = BM25Retriever(list(documents))

    return tfidf, bm25


@st.cache_resource
def build_semantic_index(documents):

    return SemanticRetriever(
        list(documents),
        EMBEDDING_MODEL
    )


# --------------------------------------------------
# 7. DATASET SELECTION
# --------------------------------------------------

csv_files = sorted([
    p for p in DATA_DIR.glob("*.csv")
    if p.name.lower() != "processed_data.csv"
])

if not csv_files:
    st.error(
        "BBC dataset not found! "
        "Place your dataset inside the data folder."
    )
    st.stop()


st.sidebar.title("Dataset Settings")

selected_file = st.sidebar.selectbox(
    "Select Dataset",
    csv_files,
    format_func=lambda x: x.name,
    key="selected_dataset",
    on_change=clear_previous_search
)

try:

    with st.spinner("Loading BBC News dataset..."):
        df = get_data(str(selected_file))

except Exception as error:

    st.error(f"Dataset loading failed: {error}")
    st.stop()


# --------------------------------------------------
# 8. PREPARE DOCUMENTS AND BUILD SPARSE INDEXES
# --------------------------------------------------

documents = tuple(
    df["clean_text"].fillna("").astype(str).tolist()
)

semantic_documents = tuple(
    df["document"].fillna("").astype(str).tolist()
)

if not documents or not any(documents):
    st.error("The dataset contains no usable articles.")
    st.stop()


with st.spinner("Building TF-IDF and BM25 indexes..."):
    tfidf, bm25 = build_sparse_indexes(documents)


# --------------------------------------------------
# 9. SEARCH SETTINGS
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.title("Search Settings")

model_choice = st.sidebar.selectbox(
    "Select Retrieval Model",
    [
        "TF-IDF",
        "BM25",
        "Sentence Transformer",
        "Hybrid RRF"
    ],
    key="model_choice",
    on_change=clear_previous_search
)

top_k = st.sidebar.slider(
    "Number of Results",
    min_value=5,
    max_value=20,
    value=10
)

categories = sorted(
    df["category"].dropna().astype(str).unique().tolist()
)

selected_category = st.sidebar.selectbox(
    "Filter by Category",
    ["All Categories"] + categories
)


# --------------------------------------------------
# 10. DATASET OVERVIEW
# --------------------------------------------------

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Articles",
    f"{len(df):,}"
)

col2.metric(
    "Categories",
    df["category"].nunique()
)

col3.metric(
    "Retrieval Models",
    "4"
)

col4.metric(
    "Search Engine",
    "Active"
)

st.divider()


# --------------------------------------------------
# 11. SEARCH INTERFACE
# --------------------------------------------------

st.subheader("Search News Articles")

with st.form("news_search_form"):

    query = st.text_input(
        "Enter your search query",
        placeholder="Example: Artificial Intelligence in Healthcare"
    )

    search_button = st.form_submit_button(
        "🔍 Search Articles",
        use_container_width=True
    )


# --------------------------------------------------
# 12. SEARCH EXECUTION
# --------------------------------------------------

if search_button:

    # Remove old results before processing a new search.
    clear_previous_search()

    if not query.strip():

        st.warning("Please enter a search query.")

    else:

        clean_query = clean_text(query)

        if not clean_query.strip():

            st.warning(
                "Your query contains no searchable terms. "
                "Please try another query."
            )

        else:

            try:

                with st.spinner(
                    f"Searching using {model_choice}..."
                ):

                    # ----------------------------------
                    # TF-IDF
                    # ----------------------------------

                    if model_choice == "TF-IDF":

                        results = tfidf.search(
                            clean_query,
                            len(df)
                        )

                    # ----------------------------------
                    # BM25
                    # ----------------------------------

                    elif model_choice == "BM25":

                        results = bm25.search(
                            clean_query,
                            len(df)
                        )

                    # ----------------------------------
                    # SENTENCE TRANSFORMER
                    # ----------------------------------

                    elif model_choice == "Sentence Transformer":

                        semantic = build_semantic_index(
                            semantic_documents
                        )

                        results = semantic.search(
                            query,
                            len(df)
                        )

                    # ----------------------------------
                    # HYBRID RRF
                    # ----------------------------------

                    elif model_choice == "Hybrid RRF":

                        semantic = build_semantic_index(
                            semantic_documents
                        )

                        tfidf_results = tfidf.search(
                            clean_query,
                            len(df)
                        )

                        bm25_results = bm25.search(
                            clean_query,
                            len(df)
                        )

                        semantic_results = semantic.search(
                            query,
                            len(df)
                        )

                        results = reciprocal_rank_fusion(
                            [
                                tfidf_results,
                                bm25_results,
                                semantic_results
                            ],
                            k=RRF_K,
                            top_k=len(df)
                        )

                    else:

                        raise ValueError(
                            "Invalid retrieval model selected."
                        )

                # Save the new search results.
                st.session_state["last_ranking"] = results
                st.session_state["last_query"] = query
                st.session_state["last_model"] = model_choice

                st.success(
                    f"Search completed successfully using {model_choice}!"
                )

            except Exception as error:

                st.error(f"Search failed: {error}")


# --------------------------------------------------
# 13. DISPLAY SEARCH RESULTS
# --------------------------------------------------

if "last_ranking" in st.session_state:

    st.divider()

    current_model = st.session_state["last_model"]

    st.subheader(
        f"Search Results using {current_model}"
    )

    st.caption(
        f"Search query: {st.session_state['last_query']}"
    )

    filtered_results = []

    for doc_id, score in st.session_state["last_ranking"]:

        if doc_id < 0 or doc_id >= len(df):
            continue

        article = df.iloc[doc_id]

        if (
            selected_category != "All Categories"
            and str(article["category"]).lower()
            != selected_category.lower()
        ):
            continue

        filtered_results.append(
            (doc_id, score)
        )

        if len(filtered_results) >= top_k:
            break

    if not filtered_results:

        st.info(
            "No articles found for the selected category. "
            "Try another category or search query."
        )

    else:

        st.success(
            f"{len(filtered_results)} relevant articles found."
        )

        for rank, (doc_id, score) in enumerate(
            filtered_results,
            start=1
        ):

            article = df.iloc[doc_id]

            title = str(article["title"]).strip()
            category = str(article["category"]).strip()
            content = str(article["text"]).strip()

            with st.container(border=True):

                st.markdown(
                    f"### {rank}. {title or 'Untitled Article'}"
                )

                st.caption(
                    f"Category: {category} | "
                    f"Relevance Score: {score:.5f}"
                )

                preview_length = 1000

                st.write(
                    content[:preview_length]
                    + (
                        "..."
                        if len(content) > preview_length
                        else ""
                    )
                )

                with st.expander("Read Full Article"):

                    st.write(content)


# --------------------------------------------------
# 14. CATEGORY DISTRIBUTION
# --------------------------------------------------

st.divider()

st.subheader("News Category Distribution")

category_counts = df["category"].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))

category_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Category")
ax.set_ylabel("Number of Articles")
ax.set_title("BBC News Dataset Distribution")

plt.xticks(rotation=30)
plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# --------------------------------------------------
# 15. EVALUATION DASHBOARD
# --------------------------------------------------

st.divider()

st.header("📊 Model Evaluation Dashboard")

results_path = EVALUATION_DIR / "results.csv"
summary_path = EVALUATION_DIR / "model_summary.csv"

if results_path.exists():

    evaluation_df = pd.read_csv(results_path)

    st.subheader("Detailed Evaluation Results")

    st.dataframe(
        evaluation_df,
        use_container_width=True,
        hide_index=True
    )

    if summary_path.exists():

        summary_df = pd.read_csv(
            summary_path,
            index_col=0
        )

        st.subheader("Average Model Performance")

        st.dataframe(
            summary_df.style.format("{:.4f}"),
            use_container_width=True
        )

        st.subheader("Performance Comparison")

        available_metrics = [
            metric for metric in [
                "Precision@10",
                "Recall@10",
                "MAP",
                "nDCG@10"
            ]
            if metric in summary_df.columns
        ]

        if available_metrics:

            metric = st.selectbox(
                "Choose Evaluation Metric",
                available_metrics
            )

            st.bar_chart(
                summary_df[metric]
            )

    # Precision-recall graph

    graph_path = GRAPH_DIR / "precision_recall.png"

    if graph_path.exists():

        st.subheader("Precision-Recall Graph")

        st.image(
            str(graph_path),
            use_container_width=True
        )

    # Other metric comparison graphs

    st.subheader("Metric Comparison Graphs")

    graph_files = [
        GRAPH_DIR / "Precision@10.png",
        GRAPH_DIR / "Recall@10.png",
        GRAPH_DIR / "MAP.png",
        GRAPH_DIR / "nDCG@10.png"
    ]

    graph_columns = st.columns(2)

    for index, graph_file in enumerate(graph_files):

        if graph_file.exists():

            with graph_columns[index % 2]:

                st.image(
                    str(graph_file),
                    caption=graph_file.stem,
                    use_container_width=True
                )

else:

    st.info(
        "Evaluation results are not available yet. "
        "Run: python evaluations/run_evaluation.py"
    )


# --------------------------------------------------
# 16. FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
### About the Project

This project implements an Intelligent Information Retrieval
system using four different retrieval approaches:

1. **TF-IDF:** Classical vector-space retrieval.
2. **BM25:** Probabilistic information retrieval.
3. **Sentence Transformer:** Dense semantic retrieval.
4. **Hybrid RRF:** Reciprocal Rank Fusion combining multiple retrieval models.

### Evaluation Metrics

- Precision@10
- Recall@10
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (nDCG)

---

**Information Retrieval Project | BBC News Dataset**
""")
