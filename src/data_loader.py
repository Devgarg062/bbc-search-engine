
import pandas as pd
import os


def load_news_data(csv_path):
    """
    Load and standardize the BBC News dataset.

    Supports:
    - Tab-separated files (.tsv or tab-separated .csv)
    - Comma-separated CSV files

    Returns a DataFrame containing:
    category, title, text, document
    """

    # Check whether the dataset exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    # Detect the delimiter
    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as file:
        first_line = file.readline()

    if first_line.count("\t") > first_line.count(","):
        delimiter = "\t"
    else:
        delimiter = ","

    # Load dataset
    df = pd.read_csv(
        csv_path,
        sep=delimiter,
        encoding="utf-8-sig",
        quotechar='"',
        on_bad_lines="error"
    )

    # Normalize column names
    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    # Identify category column
    category_candidates = [
        "category", "label", "topic", "class"
    ]

    category_col = next(
        (
            col for col in category_candidates
            if col in df.columns
        ),
        None
    )

    # Identify title column
    title_candidates = [
        "title", "headline", "name"
    ]

    title_col = next(
        (
            col for col in title_candidates
            if col in df.columns
        ),
        None
    )

    # Identify article content column
    text_candidates = [
        "content",
        "text",
        "document",
        "article",
        "body",
        "description",
        "news"
    ]

    text_col = next(
        (
            col for col in text_candidates
            if col in df.columns
        ),
        None
    )

    # Validate required columns
    if category_col is None:
        raise ValueError(
            f"Category column not found. "
            f"Available columns: {list(df.columns)}"
        )

    if text_col is None:
        raise ValueError(
            f"Article content column not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Standardize the dataset
    result = pd.DataFrame({
        "category": df[category_col].fillna("").astype(str),
        "title": (
            df[title_col].fillna("").astype(str)
            if title_col is not None
            else ""
        ),
        "text": df[text_col].fillna("").astype(str)
    })

    # Provide both column names for compatibility
    result["document"] = result["text"]

    # Remove empty articles
    result = result[
        result["text"].str.strip().ne("")
    ].reset_index(drop=True)

    # Validate dataset
    if result.empty:
        raise ValueError(
            "No valid news articles found in the dataset."
        )

    # Display loading information
    print(
        f"Dataset loaded successfully: {len(result)} articles"
    )

    print(
        "Categories:",
        result["category"].unique().tolist()
    )

    print(
        "Available columns:",
        result.columns.tolist()
    )

    return result
