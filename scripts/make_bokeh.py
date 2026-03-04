#!/usr/bin/env python3
"""Build the interactive ISEF Bokeh website from embedding data."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.layouts import column
from bokeh.models import (
    CategoricalColorMapper,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    MultiChoice,
    OpenURL,
    TabPanel,
    Tabs,
    TapTool,
    TextInput,
    Toggle,
)
from bokeh.palettes import Category20
from bokeh.plotting import figure
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from tqdm import tqdm

tqdm.pandas()

REQUIRED_CALLBACKS = {"category_filter", "winner_filter", "search_filter"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create index.html from ISEF embeddings parquet."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=repo_root / "data" / "embeddings_bge_m3.parquet",
        help="Path to embeddings parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "public" / "index.html",
        help="Output HTML path",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=100.0,
        help="t-SNE perplexity (auto-capped to sample size)",
    )
    parser.add_argument(
        "--keywords-csv",
        type=Path,
        default=repo_root / "category_keywords.csv",
        help="Where to save category keywords CSV",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root / "data" / ".cache" / "make_bokeh",
        help="Directory for hash-based cache artifacts",
    )
    parser.add_argument(
        "--callbacks-js",
        type=Path,
        default=Path(__file__).resolve().parent / "make_bokeh_callbacks.js",
        help="Path to callback JS sections used by CustomJS",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the generated HTML in a browser",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quicker local iteration",
    )
    return parser.parse_args()


def load_embeddings(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)

    fallback = Path("embeddings_bge_m3.parquet")
    if fallback.exists():
        return pd.read_parquet(fallback)

    raise FileNotFoundError(
        f"Embeddings parquet not found at '{path}' or '{fallback}'. "
        "Run `uv run python scripts/create_embeddings_bge_m3.py` first."
    )


def load_callback_codes(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Callback JS file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    matches = re.finditer(
        r"/\*\s*CALLBACK:\s*(?P<name>[a-z_]+)\s*\*/(?P<code>.*?)/\*\s*END_CALLBACK\s*\*/",
        text,
        re.DOTALL,
    )
    callbacks = {match.group("name"): match.group("code").strip() for match in matches}
    missing = sorted(REQUIRED_CALLBACKS - set(callbacks))
    if missing:
        raise ValueError(
            f"Missing callback sections in {path}: {', '.join(missing)}"
        )
    return callbacks


def ensure_mobile_viewport_meta(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if 'name="viewport"' in text:
        return

    marker = '<meta charset="utf-8">'
    viewport_meta = (
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
    )
    if marker in text:
        text = text.replace(marker, f"{marker}\n    {viewport_meta}", 1)
    else:
        text = text.replace("<head>", f"<head>\n    {viewport_meta}", 1)
    path.write_text(text, encoding="utf-8")


def normalize_category(value: object) -> str:
    if value is None:
        return "Uncategorized"
    if isinstance(value, float) and pd.isna(value):
        return "Uncategorized"

    category = str(value).strip()
    if not category or category.lower() in {"nan", "none", "null"}:
        return "Uncategorized"
    return category


def get_awards(elem: object) -> int:
    if isinstance(elem, float) and pd.isna(elem):
        return 0

    if isinstance(elem, list):
        awards = elem
    elif isinstance(elem, str):
        try:
            awards = ast.literal_eval(elem)
        except (ValueError, SyntaxError):
            return 0
    else:
        return 0

    if not awards:
        return 0

    return len([award for award in awards if str(award) != "nan"])


def awards_to_string(awards: object) -> str:
    if awards is None or (isinstance(awards, float) and pd.isna(awards)):
        return ""

    if isinstance(awards, list):
        values = awards
    else:
        awards_str = str(awards).strip()
        if not awards_str or awards_str == "nan":
            return ""
        if "[" in awards_str:
            try:
                values = ast.literal_eval(awards_str)
            except (ValueError, SyntaxError):
                return awards_str
        else:
            return awards_str

    return " | ".join(
        str(x) for x in values if x is not None and str(x).strip() and str(x) != "nan"
    )


def stable_hash_frame(frame: pd.DataFrame, *, salt: str) -> str:
    normalized = frame.fillna("").astype(str)
    hashed = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(salt.encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def stable_hash_projection(
    ids: pd.Series, matrix: np.ndarray, perplexity: float, pca_components: int
) -> str:
    digest = hashlib.sha256()
    digest.update(b"isef-projection-v1")
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(matrix).tobytes())
    digest.update(ids.fillna("").astype(str).str.cat(sep="\x1f").encode("utf-8"))
    digest.update(f"{perplexity:.8f}".encode("ascii"))
    digest.update(str(pca_components).encode("ascii"))
    return digest.hexdigest()


def ensure_xy(ds: pd.DataFrame, perplexity: float, cache_dir: Path) -> pd.DataFrame:
    if {"x", "y"}.issubset(ds.columns):
        return ds

    matrix = np.asarray(ds["embedding"].to_list(), dtype=np.float32)
    n_samples = matrix.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 rows to compute t-SNE.")

    effective_perplexity = min(perplexity, float(n_samples - 1))
    if effective_perplexity < 1:
        effective_perplexity = 1.0

    pca_components = min(50, matrix.shape[1], n_samples - 1)
    if pca_components < 1:
        pca_components = 1

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = stable_hash_projection(
        ds["id"], matrix, effective_perplexity, pca_components
    )
    pca_cache_path = cache_dir / f"pca_{cache_key}.npy"
    xy_cache_path = cache_dir / f"xy_{cache_key}.npz"

    if xy_cache_path.exists():
        with np.load(xy_cache_path) as cached:
            x = cached["x"]
            y = cached["y"]
    else:
        if pca_cache_path.exists():
            pca_matrix = np.load(pca_cache_path)
        else:
            pca = PCA(n_components=pca_components, random_state=42)
            pca_matrix = pca.fit_transform(matrix)
            np.save(pca_cache_path, pca_matrix)

        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        vis_dims = tsne.fit_transform(pca_matrix)
        x = vis_dims[:, 0]
        y = vis_dims[:, 1]
        np.savez_compressed(xy_cache_path, x=x, y=y)

    ds = ds.copy()
    ds["x"] = x
    ds["y"] = y
    return ds


def selected_topics(
    model: LatentDirichletAllocation, vectorizer: CountVectorizer, top_n: int = 3
) -> list[str]:
    feature_names = vectorizer.get_feature_names_out()
    current_words: list[str] = []
    keywords: list[tuple[str, float]] = []

    for topic in model.components_:
        words = [
            (feature_names[i], float(topic[i]))
            for i in topic.argsort()[: -top_n - 1 : -1]
        ]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])

    keywords.sort(key=lambda x: x[1], reverse=True)
    return [ii[0] for ii in keywords]


def get_keywords_from_series(frame: pd.DataFrame) -> list[str]:
    documents = (
        frame["title"].astype(str).str.strip() + " " + frame["abstract"].astype(str).str.strip()
    ).to_list()
    if not documents:
        return []

    # Keep notebook defaults for real runs, but degrade safely for tiny test slices.
    min_df = 5 if len(documents) >= 50 else 1
    vec = CountVectorizer(
        min_df=min_df,
        max_df=0.9,
        stop_words="english",
        lowercase=True,
        token_pattern=r"[a-zA-Z\-][a-zA-Z\-]{2,}",
    )
    try:
        vectorized_data = vec.fit_transform(documents)
    except ValueError:
        return []

    if vectorized_data.shape[1] == 0:
        return []

    n_components = min(5, vectorized_data.shape[1])
    lda = LatentDirichletAllocation(
        n_components=max(1, n_components),
        max_iter=10,
        learning_method="online",
        random_state=42,
    )
    lda.fit(vectorized_data)
    return selected_topics(lda, vec)


def build_category_frame(ds: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    keywords_input = ds.loc[:, ["category", "title", "abstract"]]
    cache_key = stable_hash_frame(keywords_input, salt="category-keywords-v1")
    cache_path = cache_dir / f"category_keywords_{cache_key}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    records: list[dict[str, str]] = []
    grouped = ds.groupby("category", dropna=False)
    for category, frame in tqdm(grouped, total=grouped.ngroups, desc="Category keywords"):
        records.append(
            {
                "category": str(category),
                "keywords": ", ".join(get_keywords_from_series(frame)),
            }
        )

    category_frame = pd.DataFrame(records).sort_values("category").reset_index(drop=True)
    category_frame.to_parquet(cache_path, index=False)
    return category_frame


def make_layout(
    ds: pd.DataFrame, category_frame: pd.DataFrame, callback_codes: dict[str, str]
):
    source = ColumnDataSource(
        data={
            "x": ds["x"].to_list(),
            "y": ds["y"].to_list(),
            "x_back": ds["x"].to_list(),
            "y_back": ds["y"].to_list(),
            "year": ds["year"].to_list(),
            "title": ds["title"].to_list(),
            "category": ds["category"].to_list(),
            "awards": ds["awards"].apply(awards_to_string).to_list(),
            "url": ds["id"].apply(
                lambda project_id: (
                    f"https://abstracts.societyforscience.org/Home/FullAbstract?ProjectId={project_id}"
                )
            ),
            "num_awards": ds["num_awards"].to_list(),
            "abstract": ds["abstract"].to_list(),
        }
    )

    hover = HoverTool(
        tooltips="""
<div class="project-tooltip">
  <div><strong>Title:</strong> @title</div>
  <div><strong>Year:</strong> @year</div>
  <div><strong>Category:</strong> @category</div>
  <div><strong>Awards:</strong> @awards</div>
</div>
""",
        point_policy="follow_mouse",
        attachment="vertical",
        show_arrow=False,
    )
    tap_tool = TapTool(callback=OpenURL(url="@url"))
    tools = [hover, "wheel_zoom", tap_tool]

    last_year = ds["year"].max()
    if pd.isna(last_year):
        category_options = sorted(ds["category"].astype(str).unique().tolist())
    else:
        category_options = (
            ds.loc[ds["year"] == last_year, "category"].astype(str).unique().tolist()
        )
        category_options.sort()

    category_selector = MultiChoice(
        value=[],
        options=category_options,
        title="Filter by Category",
        sizing_mode="stretch_width",
        min_height=64,
        margin=(0, 0, 0, 0),
        styles={
            "background-color": "#ffffff",
            "color": "#111827",
            "border": "1px solid #d1d5db",
            "border-radius": "8px",
            "padding": "6px",
        },
    )
    category_selector.js_on_change(
        "value",
        CustomJS(
            args=dict(source=source),
            code=callback_codes["category_filter"],
        ),
    )

    winning_projects_only = Toggle(
        label="Only Display Winning Projects?",
        button_type="success",
        sizing_mode="stretch_width",
        height=40,
        margin=(0, 0, 0, 0),
    )
    winning_projects_only.js_on_change(
        "active",
        CustomJS(
            args=dict(source=source),
            code=callback_codes["winner_filter"],
        ),
    )

    search = TextInput(
        value="",
        title="Search for Projects",
        sizing_mode="stretch_width",
        margin=(0, 0, 0, 0),
        styles={
            "background-color": "#ffffff",
            "color": "#111827",
            "border": "1px solid #d1d5db",
            "border-radius": "8px",
            "padding": "6px",
        },
    )
    search.js_on_change(
        "value",
        CustomJS(
            args=dict(source=source),
            code=callback_codes["search_filter"],
        ),
    )

    widgets = column(
        winning_projects_only,
        search,
        category_selector,
        sizing_mode="stretch_width",
        spacing=12,
        margin=(0, 0, 0, 0),
        styles={
            "padding": "14px",
            "border": "1px solid #e5e7eb",
            "border-radius": "12px",
            "background-color": "#f8fafc",
        },
    )

    factors = ds["category"].astype(str).unique().tolist()
    base = Category20[20]
    repeats = (len(factors) // len(base)) + 1
    palette = (base * repeats)[: len(factors)] if factors else base
    category_mapper = CategoricalColorMapper(factors=factors, palette=palette)

    p = figure(
        title="Science Fair Projects Visualization",
        tools=tools,
        active_scroll="wheel_zoom",
        height=820,
        sizing_mode="stretch_width",
        output_backend="webgl",
    )
    p.scatter(
        x="x",
        y="y",
        legend_field="category",
        color={"field": "category", "transform": category_mapper},
        alpha=1.0,
        size=5,
        source=source,
    )
    p.toolbar.autohide = True
    p.legend.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.grid.visible = False

    keyword_rows: list[str] = []
    for _, row in category_frame.iterrows():
        category = html.escape(str(row.get("category", "")))
        keywords = html.escape(str(row.get("keywords", "")))
        keyword_rows.append(
            "<tr>"
            f"<td style='vertical-align:top; padding:10px; border-bottom:1px solid #f1f5f9; word-break:break-word;'>{category}</td>"
            f"<td style='vertical-align:top; padding:10px; border-bottom:1px solid #f1f5f9; word-break:break-word; line-height:1.35;'>{keywords}</td>"
            "</tr>"
        )
    keywords_table_html = (
        """
<div style="overflow:auto; height:100%; border:1px solid #e5e7eb; border-radius:10px; background:#fff;">
  <table style="width:100%; border-collapse:collapse; table-layout:fixed; font-size:13px;">
    <thead>
      <tr>
        <th style="position:sticky; top:0; z-index:1; text-align:left; padding:10px; width:28%; border-bottom:1px solid #e5e7eb; background:#f8fafc;">Category</th>
        <th style="position:sticky; top:0; z-index:1; text-align:left; padding:10px; border-bottom:1px solid #e5e7eb; background:#f8fafc;">Keywords</th>
      </tr>
    </thead>
    <tbody>
"""
        + "".join(keyword_rows)
        + """
    </tbody>
  </table>
</div>
"""
    )
    category_keywords_table = Div(
        text=keywords_table_html,
        sizing_mode="stretch_width",
        height=820,
        margin=(0, 0, 0, 0),
    )

    plot_tab = TabPanel(child=p, title="t-SNE Visualization")
    keywords_tab = TabPanel(child=category_keywords_table, title="Category Keywords")
    main_tabs = Tabs(tabs=[plot_tab, keywords_tab], sizing_mode="stretch_width")
    main_tabs.margin = (0, 0, 0, 0)

    tooltip_style = Div(
        text="""
<style>
.project-tooltip {
  max-width: min(82vw, 250px);
  white-space: normal;
  word-break: break-word;
  overflow-wrap: anywhere;
  line-height: 1.3;
}

.bk-tooltip {
  max-width: min(84vw, 360px) !important;
}

@media (max-width: 700px) {
  .project-tooltip {
    max-width: calc(100vw - 24px);
    font-size: 12px;
    line-height: 1.25;
  }

  .bk-tooltip {
    max-width: calc(100vw - 20px) !important;
  }
}
</style>
""",
        margin=(0, 0, 0, 0),
    )
    title = Div(
        text="<h1 style='margin: 0;'>International Science Fair Projects Analysis</h1>",
        margin=(0, 0, 0, 0),
    )
    description = Div(
        text="""
Scraped data from the International Science and Engineering fair to identify relevant project trends at ISEF and generate insights about category selection.
<br>
The plot generated below is a 2D visualization of each projects. <strong>Closer points represents more similar projects</strong>. This association allows an intuitive visualization about the kinds and variety of projects present in each category.
""",
        margin=(0, 0, 0, 0),
    )
    disclaimer = Div(
        text="""
All data presented comes from <a href="https://abstracts.societyforscience.org">abstracts.societyforscience.org</a>. <i>Society For Science</i> is not affilated with this project. To identify how this visulization was made, check out the <a href="https://www.kaggle.com/code/wkaisertexas/international-science-fair-analysis">Kaggle Notebook</a>
""",
        margin=(0, 0, 0, 0),
    )
    return column(
        tooltip_style,
        title,
        description,
        disclaimer,
        widgets,
        main_tabs,
        sizing_mode="stretch_width",
        spacing=18,
        margin=(0, 0, 0, 0),
        styles={
            "padding": "20px clamp(14px, 3vw, 28px) 28px",
            "max-width": "1400px",
            "margin": "0 auto",
            "box-sizing": "border-box",
            "gap": "18px",
        },
    )


def main() -> None:
    args = parse_args()

    ds = load_embeddings(args.embeddings)
    if args.limit is not None:
        ds = ds.head(args.limit).copy()

    required = {"id", "title", "category", "year", "awards", "abstract", "embedding"}
    missing = sorted(required - set(ds.columns))
    if missing:
        raise ValueError(f"Embeddings parquet missing required columns: {missing}")

    if "num_awards" not in ds.columns:
        ds["num_awards"] = ds["awards"].apply(get_awards)

    ds = ds.copy()
    ds["category"] = ds["category"].map(normalize_category)
    ds["title"] = ds["title"].fillna("").astype(str)
    ds["abstract"] = ds["abstract"].fillna("").astype(str)

    callback_codes = load_callback_codes(args.callbacks_js)
    ds = ensure_xy(ds, perplexity=args.perplexity, cache_dir=args.cache_dir)
    category_frame = build_category_frame(ds, cache_dir=args.cache_dir)

    args.keywords_csv.parent.mkdir(parents=True, exist_ok=True)
    category_frame.to_csv(args.keywords_csv, index=False)

    layout = make_layout(ds, category_frame, callback_codes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_file(str(args.output), "ISEF Analysis | Top Science Fair Project Insights")
    save(layout)
    ensure_mobile_viewport_meta(args.output)
    if args.show:
        show(layout)

    print(f"Wrote interactive site: {args.output}")
    print(f"Wrote category keywords: {args.keywords_csv}")


if __name__ == "__main__":
    main()
