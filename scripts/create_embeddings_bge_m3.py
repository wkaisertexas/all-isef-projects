#!/usr/bin/env python3
"""Create data/embeddings_bge_m3.parquet for notebook analysis."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate BGE-M3 embeddings for ISEF project title+abstract text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "data" / "isef-database.csv",
        help="Path to input CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "embeddings_bge_m3.parquet",
        help="Path to output parquet",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Embedding device preference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit (useful for quick smoke tests)",
    )
    return parser.parse_args()


def is_valid_abstract(abstract: object) -> bool:
    if not isinstance(abstract, str):
        return False
    return len(abstract.strip()) > 50


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


def resolve_device(preference: str) -> str:
    def is_available(device: str) -> bool:
        if device == "cuda":
            return torch.cuda.is_available()
        if device == "mps":
            return torch.backends.mps.is_built() and torch.backends.mps.is_available()
        return device == "cpu"

    if preference != "auto":
        if not is_available(preference):
            raise RuntimeError(f"Requested device '{preference}' is not available.")
        return preference

    for candidate in ("cuda", "mps", "cpu"):
        if is_available(candidate):
            return candidate
    raise RuntimeError("No usable device found.")


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    ds = pd.read_csv(args.input)
    ds["num_awards"] = ds["awards"].apply(get_awards)

    mask = ds["abstract"].apply(is_valid_abstract)
    ds = ds[mask].copy()
    ds["abstract"] = ds["abstract"].astype(str).str.strip()
    ds["embeddings_input"] = (
        ds["title"].astype(str).str.strip() + "\n" + ds["abstract"].astype(str)
    )

    if args.limit is not None:
        ds = ds.head(args.limit).copy()

    device = resolve_device(args.device)
    model = SentenceTransformer(args.model_name, device=device)

    embeddings = model.encode(
        ds["embeddings_input"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    ds["embedding"] = embeddings.tolist()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(args.output, index=False)

    print(f"Wrote {len(ds)} rows to {args.output}")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Embedding size: {len(ds['embedding'].iloc[0])}")


if __name__ == "__main__":
    main()
